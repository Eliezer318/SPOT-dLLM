"""Generate vendor/llada/ from the HF snapshot of GSAI-ML/LLaDA-8B-Instruct.

This tree is generated, not checked in.  One patch: thread an explicit `position_ids` from LLaDAModelLM.forward down to
RotaryEmbedding.forward.

Why.  decode.py's concat layout forwards  [true tail][draft piece]xD  in ONE sequence,
and every draft piece must be rotated at the positions of the block it drafts for, not at
its offset in the concatenated chunk.  Upstream LLaDA infers positions from the key
length (`pos_sin[key_len-query_len:key_len]`), which is only right for a single
contiguous sequence.  The SDAR/Dream/LLaDA2 code paths already accept position_ids;
this makes LLaDA-8B accept the same argument.

Semantics (matches decode.py::_assemble and the upstream cache, which stores keys
UNROTATED and re-rotates the full key length every forward):
  * query rows are rotated at `position_ids`;
  * keys are the cached prefix, whose positions are 0..past_len-1, followed by the new
    rows at `position_ids`.
  * position_ids=None reproduces upstream behaviour exactly (same slice).

Every anchor is asserted, so an upstream file change fails loudly here instead of
silently producing a differently-patched model.  Rerun after any re-download:
    python vendor/patch_llada.py
"""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "llada")


class UpstreamDrift(AssertionError):
    """The snapshot no longer contains a string this patch rewrites."""


def _sub(src, old, new, tag, count=1):
    got = src.count(old)
    if got != count:
        raise UpstreamDrift(f"{tag}: anchor occurs {got}x in modeling_llada.py, expected {count}")
    return src.replace(old, new)


def find_snapshot():
    p = os.environ.get("LLADA_PATH")
    if p and os.path.isdir(p):
        return p
    roots = [os.environ.get("SPOT_HUB"),
             os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")]
    for r in roots:
        if not r:
            continue
        hits = sorted(glob.glob(os.path.join(r, "models--GSAI-ML--LLaDA-8B-Instruct", "snapshots", "*", "modeling_llada.py")))
        if hits:
            return os.path.dirname(hits[-1])
    raise SystemExit("LLaDA-8B-Instruct snapshot not found; set LLADA_PATH")


SIG_TAIL = (
    "        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,\n"
    "        use_cache: bool = False,\n"
    "    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:\n"
)
SIG_TAIL_NEW = (
    "        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,\n"
    "        use_cache: bool = False,\n"
    "        position_ids: Optional[torch.Tensor] = None,  # PATCH: explicit RoPE positions\n"
    "    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:\n"
)

ROTARY_OLD = """    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = q_.shape[-2], k_.shape[-2]  # could be different if layer_past not None
            pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)
            q_ = self.apply_rotary_pos_emb(
                pos_sin[:, :, key_len - query_len : key_len, :],
                pos_cos[:, :, key_len - query_len : key_len, :],
                q_)
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        return q_.type_as(q), k_.type_as(k)
"""
ROTARY_NEW = """    def forward(self, q: torch.Tensor, k: torch.Tensor,
                position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = q_.shape[-2], k_.shape[-2]  # could be different if layer_past not None
            if position_ids is None:
                pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
                pos_sin = pos_sin.type_as(q_)
                pos_cos = pos_cos.type_as(q_)
                q_ = self.apply_rotary_pos_emb(
                    pos_sin[:, :, key_len - query_len : key_len, :],
                    pos_cos[:, :, key_len - query_len : key_len, :],
                    q_)
                k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
            else:
                # PATCH: explicit positions.  Queries sit at position_ids; the keys are
                # the cached prefix (positions 0..past_len-1 -- upstream stores keys unrotated
                # and re-rotates the whole key length) followed by the new rows at position_ids.
                pos_q = position_ids.reshape(-1).to(device=q_.device, dtype=torch.long)
                past_len = key_len - query_len
                if past_len > 0:
                    pos_k = torch.cat([torch.arange(past_len, device=q_.device, dtype=torch.long), pos_q])
                else:
                    pos_k = pos_q
                # the table is warmed to max_sequence_length at init, so this is a cache hit
                # (no data-dependent sync)
                pos_sin, pos_cos = self.get_rotary_embedding(self.config.max_sequence_length, q_.device)
                pos_sin = pos_sin.type_as(q_)
                pos_cos = pos_cos.type_as(q_)
                q_ = self.apply_rotary_pos_emb(pos_sin[:, :, pos_q, :], pos_cos[:, :, pos_q, :], q_)
                k_ = self.apply_rotary_pos_emb(pos_sin[:, :, pos_k, :], pos_cos[:, :, pos_k, :], k_)
        return q_.type_as(q), k_.type_as(k)
"""


def patch(src):
    # P1 rotary
    src = _sub(src, ROTARY_OLD, ROTARY_NEW, "P1 rotary.forward")
    # P2 attention(): signature (distinct by its first body line) and the rotary call
    src = _sub(src, SIG_TAIL + "        B, T, C = q.size()",
               SIG_TAIL_NEW + "        B, T, C = q.size()", "P2 attention signature")
    src = _sub(src, "            q, k = self.rotary_emb(q, k)\n",
               "            q, k = self.rotary_emb(q, k, position_ids=position_ids)\n", "P2 rotary call")
    # P3 the abstract block forward + the two concrete block forwards
    src = _sub(src, SIG_TAIL, SIG_TAIL_NEW, "P3 block forward signatures", count=3)
    src = _sub(src, "self.attention, q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache\n",
               "self.attention, q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache, position_ids=position_ids\n",
               "P3 checkpointed attention calls", count=2)
    src = _sub(src, "self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache)",
               "self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache, position_ids=position_ids)",
               "P3 attention calls", count=2)
    # P4 block group
    src = _sub(src,
               "        layers_past: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,\n"
               "        use_cache: bool = False,\n"
               "    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:\n",
               "        layers_past: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,\n"
               "        use_cache: bool = False,\n"
               "        position_ids: Optional[torch.Tensor] = None,  # PATCH\n"
               "    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:\n",
               "P4 block-group signature")
    src = _sub(src, "block, x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache\n",
               "block, x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache, position_ids=position_ids\n",
               "P4 checkpointed block calls", count=2)
    src = _sub(src, "block(x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache)",
               "block(x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache, position_ids=position_ids)",
               "P4 block calls", count=2)
    src = _sub(src, "x, attention_bias=attention_bias, layers_past=layers_past, use_cache=use_cache\n",
               "x, attention_bias=attention_bias, layers_past=layers_past, use_cache=use_cache, position_ids=position_ids\n",
               "P4 block-group call")
    # P5 LLaDAModel.forward
    src = _sub(src,
               "        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,\n"
               "        use_cache: bool = False,\n"
               "        last_logits_only: bool = False,\n",
               "        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,\n"
               "        use_cache: bool = False,\n"
               "        last_logits_only: bool = False,\n"
               "        position_ids: Optional[torch.Tensor] = None,  # PATCH\n",
               "P5 model signature")
    # P7 upstream forbids the cache with a bare assertion; the code beneath it (OLMo-derived)
    # supports past_key_values fully and builds a bidirectional bias when none is passed.
    # decode.py's prefix cache on this target is the LOSSY one (dp=false by the gate).
    src = _sub(src,
               '        assert (past_key_values is None and not use_cache), "The kvcache is not suppotred for MDM."\n',
               '        # PATCH: upstream asserted (past_key_values is None and not use_cache) here; lifted --\n'
               '        # the prefix cache is approximate for a full-attention MDM and is gated as such by decode.py.\n',
               "P7 kvcache assertion")
    # P6 LLaDAModelLM.forward
    src = _sub(src, "        cache_position: Optional[Cache] = None,  # This is a hack",
               "        position_ids: Optional[torch.Tensor] = None,  # PATCH\n"
               "        cache_position: Optional[Cache] = None,  # This is a hack", "P6 LM signature")
    src = _sub(src,
               "            attention_bias=attention_bias,\n"
               "            past_key_values=past_key_values,\n"
               "            use_cache=use_cache,\n"
               "            output_hidden_states=output_hidden_states,\n",
               "            attention_bias=attention_bias,\n"
               "            past_key_values=past_key_values,\n"
               "            use_cache=use_cache,\n"
               "            output_hidden_states=output_hidden_states,\n"
               "            position_ids=position_ids,\n",
               "P6 LM forward call")
    return src


def main():
    snap = find_snapshot()
    os.makedirs(OUT, exist_ok=True)
    src = open(os.path.join(snap, "modeling_llada.py")).read()
    patched = patch(src)
    open(os.path.join(OUT, "modeling_llada.py"), "w").write(patched)
    cfg = open(os.path.join(snap, "configuration_llada.py")).read()
    open(os.path.join(OUT, "configuration_llada.py"), "w").write(cfg)
    open(os.path.join(OUT, "__init__.py"), "w").write("")
    open(os.path.join(OUT, "SOURCE.txt"), "w").write(f"generated by vendor/patch_llada.py from {snap}\n")
    print(f"vendor/llada written from {snap}; {patched.count('PATCH')} patch markers")


if __name__ == "__main__":
    main()
