"""Model adapters.

Both dLLMs are wrapped behind one interface so that decode.py has exactly one code
path.  The adapter owns everything model-specific: how a forward is called with a
prefix KV cache, how draft blocks get their position ids, and how logits map to a
distribution over the tokens of the current block.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ops: models and datasets live in the project-owned store, immune to the shared
# cache's eviction sweeps (three eviction incidents on  alone). The shared
# cache path keeps SYMLINKS into the store, so snapshotted jobs and third-party code
# keep working; setting HF_HOME here covers fresh downloads for anything that imports
# spot.models before transformers.
HF_STORE = os.environ.get("SPOT_HF_STORE", os.path.expanduser("~/.cache/huggingface/hub"))
if os.path.isdir(HF_STORE):
    os.environ.setdefault("HF_HOME", HF_STORE)

# Full-attention targets are loaded from a local snapshot: set LLADA_PATH / DREAM_PATH
# to the directory of the downloaded model, or leave unset to resolve by repo id.
LLADA_PATH = os.environ.get("LLADA_PATH", "GSAI-ML/LLaDA-8B-Instruct")
DREAM_PATH = os.environ.get("DREAM_PATH", "Dream-org/Dream-v0-Instruct-7B")
LLADA2_PATH = os.environ.get("LLADA2_PATH", "inclusionAI/LLaDA2.0-mini")
SDAR_PATH = os.environ.get("SDAR_PATH", "JetLM/SDAR-8B-Chat-b32")

NEG = torch.finfo(torch.float32).min


@dataclass
class ForwardOut:
    logits: torch.Tensor  # (1, n_query, V) float32
    past: Optional[Any]
    n_positions: int  # query positions pushed through the network
    key_len: int  # attention context length seen by those queries
    # Final-layer hidden states, only when the caller asks (want_hidden=True).  Medusa
    # heads read these instead of running their own network, which is what makes their
    # drafter-cost column zero; decoding never requests them, so nothing is paid at
    # inference for training-time support.
    hidden: Optional[torch.Tensor] = None


class BaseAdapter:
    """Interface used by decode.py."""

    name: str
    mask_id: int
    device: torch.device
    # sampling temperature: decode sets this from its config before generation. The
    # sampler's contract (sampling.perturbed_scores) is that probs are ALREADY
    # softmax(logits/T); until nothing below 1.0 divided the logits, which was
    # invisible because every T>0 run ever requested was exactly T=1.
    sample_temp: float = 1.0

    # -- prompting -------------------------------------------------------------
    def encode_prompt(self, messages: List[dict], add_generation_prompt: bool = True) -> torch.Tensor:
        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt, tokenize=False
        )
        return self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)

    def encode_raw(self, text: str) -> torch.Tensor:
        """Tokenise a completion-style prompt with no chat template applied."""
        return self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)

    # SDAR's logit matrix is WIDER than its tokenizer -- config vocab_size 151936
    # against len(tokenizer) 151670, i.e. 266 embedding rows (151670-151935) that no token
    # string maps to.  They are untrained padding, and at T=0/T=1 the sampler effectively
    # never selects one; at T=1.5 it does, and `convert_tokens_to_string` then dies on
    # `"".join(tokens)` with `TypeError: expected str, NoneType found`.
    #
    # This is a RENDERING failure, not a decode failure: the committed id was a faithful
    # draw from the target's own distribution, which really does place mass on those rows.
    # So the fix belongs here and not in the sampling path -- dropping the id from the
    # string leaves the commit law untouched.  The count is surfaced on the instance
    # rather than swallowed, because a run that hits this is emitting tokens the model
    # cannot render and its accuracy number should be read with that in mind.
    # Counters are named for their scope because `run_sweep` loops every budget and seed
    # inside ONE process (run_sweep.py:140), so a single "count" would silently pool cells
    # together, which is a real trap for anything reading a counter per run.
    # `_last` is this decode call; `_total` is process-lifetime. Read the delta of `_total`
    # across a cell, or `_last` per sample; never treat `_total` as a per-run figure.
    _n_unrenderable_total = 0
    _n_unrenderable_last = 0
    _warned_unrenderable = False

    def decode(self, ids: Sequence[int]) -> str:
        ids = list(ids)
        limit = len(self.tokenizer)
        keep = [i for i in ids if i < limit]
        n = len(ids) - len(keep)
        cls = type(self)
        cls._n_unrenderable_last = n
        if n:
            cls._n_unrenderable_total += n
            if not cls._warned_unrenderable:
                cls._warned_unrenderable = True
                print(f"== WARNING: {cls.__name__} sampled {n} id(s) outside the tokenizer "
                      f"(vocab rows {limit}..); dropped from the string, commit law "
                      f"untouched. ", flush=True)
        return self.tokenizer.decode(keep, skip_special_tokens=True)

    # -- forward ---------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        past: Optional[Any],
        use_cache: bool) -> ForwardOut:
        raise NotImplementedError

    def trim_cache(self, past: Any, length: int) -> Any:
        raise NotImplementedError

    def cache_len(self, past: Any) -> int:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        return int(self.model.config.vocab_size if hasattr(self.model.config, "vocab_size")
                   else self.model.config.embedding_size)

    # -- logits -> block distribution ------------------------------------------
    logit_shift: int = 0  # 0: logits at n describe token n (LLaDA).  1: at n-1 (Dream).
    # True for block-diffusion checkpoints: attention is causal at block granularity,
    # block boundaries sit at absolute multiples of block_length, and the model carries no
    # implicit causality of its own -- given no mask it attends everywhere, so decode.py
    # must supply a block-causal bias on EVERY round, drafted or not.
    block_causal: bool = False

    def block_probs(self, logits: torch.Tensor, lo: int, hi: int, row: int = 0) -> torch.Tensor:
        """Distribution over the tokens at chunk positions [lo, hi) of batch row `row`.

        Only the block's own rows are upcast: casting the whole (chunk_len, 126k) logit
        tensor to fp32 would add hundreds of MB of memory traffic per round and show up as
        a speed difference between configurations that forward chunks of different length.
        """
        s = self.logit_shift
        return torch.softmax(logits[row, lo - s : hi - s, :].float() / self.sample_temp, dim=-1)

    def block_probs_piece(self, logits: torch.Tensor, start: int, block_len: int, shift: int) -> torch.Tensor:
        """Same, for an appended draft piece that begins at chunk index `start`.

        The piece holds `shift` predecessor tokens followed by the `block_len` draft tokens, so
        the distribution for draft token i sits at chunk index start + i.
        """
        return torch.softmax(logits[0, start : start + block_len, :].float() / self.sample_temp, dim=-1)


class LLaDAAdapter(BaseAdapter):
    name = "llada"

    def __init__(self, path: str = LLADA_PATH, dtype=torch.bfloat16, device="cuda"):
        from transformers import AutoTokenizer

        from vendor.llada.configuration_llada import LLaDAConfig
        from vendor.llada.modeling_llada import LLaDAModelLM

        self.device = torch.device(device)
        cfg = LLaDAConfig.from_pretrained(path)
        assert cfg.block_group_size == 1, "vendored patch only threads position_ids for block_group_size==1"
        # transformers 4.55: torch_dtype, not dtype (dtype= silently lands in the config)
        self.model = LLaDAModelLM.from_pretrained(path, config=cfg, torch_dtype=dtype).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.mask_id = int(cfg.mask_token_id)
        self.eos_id = int(cfg.eos_token_id)
        self.dtype = dtype

    def forward(self, input_ids, position_ids, attn_bias, past, use_cache):
        out = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_bias=attn_bias,
            past_key_values=past,
            use_cache=use_cache)
        return ForwardOut(
            logits=out.logits,  # left in the model dtype; block_probs upcasts the rows it needs
            past=out.past_key_values if use_cache else None,
            n_positions=int(input_ids.shape[1]),
            key_len=int(input_ids.shape[1] + (self.cache_len(past) if past is not None else 0)))

    def trim_cache(self, past, length):
        return [(k[:, :, :length, :], v[:, :, :length, :]) for (k, v) in past]

    def cache_len(self, past):
        return int(past[0][0].shape[-2])

    logit_shift = 0  # LLaDA is a masked LM: logits at a position describe that position


class DreamAdapter(BaseAdapter):
    """Dream-7B.

    Dream is Qwen2-derived and shipped with `shift_logits`: the distribution for the token at
    position n is read off the logits at position n-1 (see Dream's generation_utils.py).  Its
    remote code is pinned to transformers 4.46, so we load it, then use our own forward loop.
    """

    name = "dream"

    def __init__(self, path: str = DREAM_PATH, dtype=torch.bfloat16, device="cuda"):
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device)
        # Dream registers AutoModel (not AutoModelForCausalLM) for its remote code.
        self.model = (
            AutoModel.from_pretrained(path, torch_dtype=dtype, trust_remote_code=True).to(self.device).eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.mask_id = int(self.model.config.mask_token_id)
        self.eos_id = int(self.model.config.eos_token_id)
        self.dtype = dtype

    def forward(self, input_ids, position_ids, attn_bias, past, use_cache):
        from transformers.cache_utils import DynamicCache

        q_len = int(input_ids.shape[1])
        past_len = self.cache_len(past) if past is not None else 0
        # Dream hands the mask straight to SDPA, so it must already be (1, 1, q_len, key_len);
        # LLaDA's code slices a square bias itself.  decode.py builds the square, we slice.
        if attn_bias is not None:
            if attn_bias.shape[-2] != q_len:
                attn_bias = attn_bias[:, :, attn_bias.shape[-2] - q_len :, :]
            # SDPA rejects a mask whose dtype differs from the query's; LLaDA's own code
            # casts internally, Dream's does not.
            attn_bias = attn_bias.to(self.dtype)
        cache = past
        if cache is None and use_cache:
            cache = DynamicCache()
        out = self.model(
            input_ids=input_ids,
            position_ids=position_ids.view(1, -1) if position_ids is not None else None,
            attention_mask=attn_bias,
            past_key_values=cache,
            use_cache=cache is not None,
            cache_position=torch.arange(past_len, past_len + q_len, device=input_ids.device))
        # DreamAttention appends to whatever cache it is handed, so a read-only forward has to
        # roll the cache back to the prefix it was called with.
        if past is not None and not use_cache:
            DreamCacheOps.trim(past, past_len)
        return ForwardOut(
            logits=out.logits,
            past=cache if use_cache else None,
            n_positions=q_len,
            key_len=q_len + past_len)

    def trim_cache(self, past, length):
        return DreamCacheOps.trim(past, length)

    def cache_len(self, past):
        return DreamCacheOps.length(past)

    logit_shift = 1  # Dream reads the distribution for position n off the logits at n-1


class LLaDA2Adapter(BaseAdapter):
    """LLaDA2.0-mini -- block-diffusion, i.e. attention is causal at block granularity.

    Two consequences, both measured in tests/diag_llada2_layers.py rather than assumed:

    * Block-causality holds *exactly* at the attention level.  With the block mask applied,
      block k's hidden states are bitwise identical under arbitrary changes to block k+1,
      right up to the first MoE layer (layer 0 is dense here: first_k_dense_replace=1).
      This is what makes tree-mask draft verification exact in principle on this family.
    * It is nonetheless useless for *distributional* measurement, because it is a 256-expert
      MoE.  Expert dispatch groups tokens across the whole forward, so any change to the
      token set reshuffles accumulation order for every token.  The resulting floor is
      TV mean 0.0722 -- 36x LLaDA-8B's dense floor, and larger than the worst distortion in
      the whole the development grid.  Use this adapter for throughput questions only.

    The checkpoint wants a full square block mask of shape (batch, 1, seq, seq) and raises
    if given anything else, so unlike Dream we must not pre-slice to (1, 1, q_len, key_len).
    """

    name = "llada2"

    def __init__(self, path: str = LLADA2_PATH, dtype=torch.bfloat16, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=dtype, device_map=device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        # Relax the checkpoint's square-mask-only check so the prefix-KV cache
        # is usable (rectangular additive bias over [past | chunk]).  The square/uncached
        # path is byte-identical -- the patch touches nothing else.
        from vendor.llada2_cached_forward import attach as _attach_cached
        _attach_cached(self.model.model)
        # not exposed on the config; these are the defaults in the checkpoint's generate()
        self.mask_id = int(getattr(self.model.config, "mask_token_id", 156895))
        self.eos_id = int(getattr(self.model.config, "eos_token_id", 156892) or 156892)
        self.dtype = dtype

    def forward(self, input_ids, position_ids, attn_bias, past, use_cache,
                want_hidden: bool = False):
        from transformers.cache_utils import DynamicCache

        q_len = int(input_ids.shape[1])
        batch = int(input_ids.shape[0])
        past_len = self.cache_len(past) if past is not None else 0
        if attn_bias is not None:
            attn_bias = attn_bias.to(self.dtype)
            if attn_bias.shape[-2] != q_len:
                # decode.py hands a square K x K bias; take the query rows, like SDAR's
                # _bool_mask does -- the patched vendor forward accepts the resulting
                # rectangular (b, 1, q, past+q) additive bias
                attn_bias = attn_bias[:, :, attn_bias.shape[-2] - q_len:, :]
            if attn_bias.shape[0] != batch:
                attn_bias = attn_bias.expand(batch, -1, -1, -1).contiguous()
        # decode.py hands out 1-D position ids (LLaDA accepts them); this checkpoint's RoPE
        # indexes position_ids[:, None, :] and needs (batch, q_len).
        if position_ids is not None:
            position_ids = position_ids.view(1, -1)
            if batch > 1:
                position_ids = position_ids.expand(batch, -1)
        cache = past
        if cache is None and use_cache:
            cache = DynamicCache()
        out = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attn_bias,
            past_key_values=cache,
            use_cache=cache is not None,
            output_hidden_states=want_hidden)
        if past is not None and not use_cache:
            DreamCacheOps.trim(past, past_len)
        return ForwardOut(
            logits=out.logits,
            past=cache if use_cache else None,
            n_positions=q_len,
            key_len=q_len + past_len,
            hidden=(out.hidden_states[-1] if want_hidden else None))

    def trim_cache(self, past, length):
        return DreamCacheOps.trim(past, length)

    def cache_len(self, past):
        return DreamCacheOps.length(past)

    logit_shift = 0  # its generate() reads block logits at the block's own positions
    block_causal = True


class SDARAdapter(BaseAdapter):
    """SDAR-8B-Chat-b32 -- block-diffusion and *dense*.

    The combination that makes the exactness work possible.  Measured in
    tests/check_sdar_structure.py, not assumed:

    * block-causality is **bitwise exact** -- block k's hidden states and logits are
      identical under arbitrary changes to block k+1, at all 36 layers.  So the leakage
      path that makes tree-mask verification approximate on LLaDA-8B is absent.
    * at fixed tensor shape the model is bitwise reproducible (content changes give
      TV 0.0).  The only divergence comes from changing shapes (TV ~0.012).  Hold shape
      fixed and the measurement floor is exactly zero.

    Two model-specific hazards, both paid for in failed jobs:

    * **SDAR wants a boolean mask, True = attend** (modeling_sdar.py:300), not the additive
      bias LLaDA/LLaDA2 take.  Handing it an additive 0/NEG bias inverts the mask exactly.
      decode.py speaks additive-bias everywhere, so the conversion happens here, at the
      boundary, leaving decode.py with one code path.
    * upstream routes an all-True mask to flash_attn_func, which the vendored copy does not
      have; vendor/patch_sdar.py makes that branch conditional on the kernel existing.
      An all-True mask is normal in cached decoding, so this is not a corner case.
    """

    name = "sdar"

    def __init__(self, path: str = SDAR_PATH, dtype=torch.bfloat16, device="cuda"):
        from transformers import AutoTokenizer

        from vendor.sdar.configuration_sdar import SDARConfig
        from vendor.sdar.modeling_sdar import SDARForCausalLM

        self.device = torch.device(device)
        cfg = SDARConfig.from_pretrained(path)
        # flex_attention would need torch.compile, which cannot build its Triton CUDA
        # extension in some environments.  SDAR ignores ALL_ATTENTION_FUNCTIONS and hardcodes a
        # flash/SDPA choice anyway, so this only steers transformers' own bookkeeping.
        cfg._attn_implementation = "sdpa"
        self.model = SDARForCausalLM.from_pretrained(
            path, config=cfg, torch_dtype=dtype, device_map=device,
            attn_implementation="sdpa").eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.mask_id = 151669  # "<MASK>"; not carried on the config
        self.eos_id = 151645  # "<|im_end|>"
        self.dtype = dtype

    def _bool_mask(self, attn_bias, batch, q_len):
        """additive bias (0 attend / NEG blocked) -> bool (True attend), shaped for SDAR."""
        if attn_bias is None:
            return None
        m = attn_bias > -1e3
        if m.shape[-2] != q_len:  # decode.py hands us a square bias; take the query rows
            m = m[:, :, m.shape[-2] - q_len :, :]
        if m.shape[0] != batch:
            m = m.expand(batch, -1, -1, -1)
        return m.contiguous()

    def forward(self, input_ids, position_ids, attn_bias, past, use_cache,
                want_hidden: bool = False):
        from transformers.cache_utils import DynamicCache

        q_len = int(input_ids.shape[1])
        batch = int(input_ids.shape[0])
        past_len = self.cache_len(past) if past is not None else 0
        mask = self._bool_mask(attn_bias, batch, q_len)
        # decode.py hands out 1-D position ids (LLaDA accepts them); SDAR's RoPE indexes
        # position_ids[:, None, :] and needs (batch, q_len).
        if position_ids is not None:
            position_ids = position_ids.view(1, -1)
            if batch > 1:
                position_ids = position_ids.expand(batch, -1)
        cache = past
        if cache is None and use_cache:
            cache = DynamicCache()
        out = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=mask,
            past_key_values=cache,
            use_cache=cache is not None,
            # SDAR's own switch: store_kv=True appends to the cache, False retrieves
            # without mutating it.  That is exactly the read-only forward decode.py wants,
            # so unlike Dream no trim-back is needed.
            store_kv=bool(use_cache),
            output_hidden_states=want_hidden)
        return ForwardOut(
            logits=out.logits,
            past=cache if use_cache else None,
            n_positions=q_len,
            key_len=q_len + past_len,
            hidden=(out.hidden_states[-1] if want_hidden else None))

    def trim_cache(self, past, length):
        return DreamCacheOps.trim(past, length)

    def cache_len(self, past):
        return DreamCacheOps.length(past)

    # set from measurement in tests/check_sdar_logit_shift.py, not assumed: SDAR is
    # Qwen3-derived (like Dream, which shifts) but trained as a masked diffusion model
    # (like LLaDA, which does not).  Measured: unshifted.
    logit_shift = 0
    block_causal = True


class DreamCacheOps:
    @staticmethod
    def length(past) -> int:
        if hasattr(past, "get_seq_length"):
            return int(past.get_seq_length())
        return int(past[0][0].shape[-2])

    @staticmethod
    def trim(past, length):
        if hasattr(past, "layers"):  # transformers >= 4.54 DynamicCache
            for layer in past.layers:
                layer.keys = layer.keys[:, :, :length, :]
                layer.values = layer.values[:, :, :length, :]
            return past
        if hasattr(past, "key_cache"):
            past.key_cache = [k[:, :, :length, :] for k in past.key_cache]
            past.value_cache = [v[:, :, :length, :] for v in past.value_cache]
            return past
        return [(k[:, :, :length, :], v[:, :, :length, :]) for (k, v) in past]


def load_adapter(name: str, **kw) -> BaseAdapter:
    if name.startswith("sdar17ft:"):
        # the 1.7B sibling with finetuned weights loaded over the base
        ad = load_adapter("sdar17", **kw)
        ck = torch.load(name.split(":", 1)[1], map_location=ad.device)
        ad.model.load_state_dict(ck["model"])
        ad.model.eval()
        return ad
    if name.startswith("dflash:"):
        # trained small denoiser as an aux table source (spot/dflash.py)
        from .dflash import DFlashAdapter

        return DFlashAdapter(name.split(":", 1)[1], **kw)
    if name == "llada":
        return LLaDAAdapter(**kw)
    if name == "dream":
        return DreamAdapter(**kw)
    if name == "llada2":
        return LLaDA2Adapter(**kw)
    if name == "sdar":
        return SDARAdapter(**kw)
    if name in ("sdar_b16", "sdar_b8", "sdar_b4", "sdar17", "sdar4b"):
        # Block-size variants: separately trained checkpoints, identical architecture and
        # vendored modeling code.  b4 is the unsuffixed default repo per the model card.
        # sdar17 is the 1.7B b32 sibling -- the aux refresher, ~4.7x cheaper/forward.
        import glob
        repo = {"sdar_b16": "models--JetLM--SDAR-8B-Chat-b16",
                "sdar_b8": "models--JetLM--SDAR-8B-Chat-b8",
                "sdar_b4": "models--JetLM--SDAR-8B-Chat",
                "sdar17": "models--JetLM--SDAR-1.7B-Chat-b32",
                "sdar4b": "models--JetLM--SDAR-4B-Chat-b32"}[name]
        hits = (glob.glob(f"{HF_STORE}/hub/{repo}/snapshots/*")
                or glob.glob(os.path.expanduser(
                    f"~/.cache/huggingface/hub/{repo}/snapshots/*")))
        assert hits, f"{repo} not downloaded"
        return SDARAdapter(path=sorted(hits)[-1], **kw)
    raise ValueError(f"unknown model {name}")
