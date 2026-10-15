"""Generate vendor/sdar/ from the HF snapshot of JetLM/SDAR-8B-Chat-b32.

This tree is generated, not checked in.  Two patches, both required when flash_attn is
not installed and cannot be built in the target environment:

  P1  `from flash_attn.ops.triton.layer_norm import rms_norm_fn` is an
      *unconditional* top-level import (modeling_sdar.py:49).  Without it the module
      does not import at all.  Guarded; SDARRMSNorm.forward falls back to the pure
      torch RMSNorm that upstream itself ships, commented out, directly beneath the
      kernel call -- un-commented verbatim, not rewritten.
  P2  `if torch.all(attention_mask): flash_attn_func(...)` routes the *normal*
      cached-decoding path (all-True mask) to a kernel we do not have.  Gated on the
      kernel existing; otherwise it falls through to the SDPA branch already there.
      Same maths: flash with causal=False == SDPA with an all-True bool mask.
      Also fixes `torch.all(None)` when the adapter passes no mask.

Every anchor is asserted, so an upstream file change fails loudly here instead of
silently producing a differently-patched model.  Rerun after any re-download:
    python vendor/patch_sdar.py
"""
import os
import sys
import glob

HUB = os.environ.get("SPOT_HUB") or os.path.join(
    os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sdar")


class UpstreamDrift(AssertionError):
    """The snapshot no longer contains a string this patch rewrites."""


def _sub(src, old, new, tag):
    if old not in src:
        raise UpstreamDrift(f"{tag}: anchor not found in modeling_sdar.py")
    if src.count(old) != 1:
        raise UpstreamDrift(f"{tag}: anchor occurs {src.count(old)}x, expected 1")
    return src.replace(old, new)


# ---------------------------------------------------------------- P1: rms_norm
P1_IMPORT_OLD = (
    "from flash_attn.ops.triton.layer_norm import rms_norm_fn as flash_rms_norm\n"
)
P1_IMPORT_NEW = (
    "try:  # PATCH: kernel absent in this environment -> torch fallback below\n"
    "    from flash_attn.ops.triton.layer_norm import rms_norm_fn as flash_rms_norm\n"
    "except Exception:\n"
    "    flash_rms_norm = None\n"
)

P1_FWD_OLD = """    def forward(self, hidden_states):
        return flash_rms_norm(
            hidden_states, weight=self.weight, bias=None, eps=self.variance_epsilon)
        '''
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * \\
            torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
        '''
"""
P1_FWD_NEW = """    def forward(self, hidden_states):
        if flash_rms_norm is not None:
            return flash_rms_norm(
                hidden_states, weight=self.weight, bias=None, eps=self.variance_epsilon)
        # PATCH: upstream's own torch fallback, un-commented verbatim.
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * \\
            torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
"""

# ------------------------------------- P3: transformers 4.52 -> 4.55 API drift
# The file is generated against transformers v4.52.4 (see its own header).  This env
# runs 4.55.4, where `LossKwargs` was dropped from transformers.utils.  It is used
# once, purely as an `Unpack[...]` type annotation on a forward signature -- inert at
# inference.  Restore the 4.52 definition locally rather than repoint it at
# `TransformersKwargs`, whose field set is different.
P3_IMPORT_OLD = (
    "from transformers.utils import LossKwargs, auto_docstring, can_return_tuple,"
    " is_torch_flex_attn_available, logging\n"
)
P3_IMPORT_NEW = (
    "from transformers.utils import auto_docstring, can_return_tuple,"
    " is_torch_flex_attn_available, logging\n"
    "from typing import TypedDict  # PATCH\n"
    "\n"
    "\n"
    "class LossKwargs(TypedDict, total=False):  # PATCH: transformers<4.55 def\n"
    "    num_items_in_batch: Optional[int]\n"
)


# ------------------------------------------------------------- P2: attention
P2_IMPORT_OLD = """try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
except:
    pass
"""
P2_IMPORT_NEW = """try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
    _HAS_FLASH_ATTN = True  # PATCH
except:
    _HAS_FLASH_ATTN = False  # PATCH
"""

P2_BRANCH_OLD = """        attention_mask = attention_mask.bool() if attention_mask is not None else None
        if torch.all(attention_mask):  # decoding
"""
P2_BRANCH_NEW = """        attention_mask = attention_mask.bool() if attention_mask is not None else None
        # PATCH: all-True is the NORMAL cached-decoding mask; without the kernel
        # it must fall through to SDPA, which is the same maths for a full mask.
        _all_attend = attention_mask is None or bool(torch.all(attention_mask))
        if _all_attend and _HAS_FLASH_ATTN:  # decoding
"""


def main():
    hits = sorted(glob.glob(os.path.join(
        HUB, "models--JetLM--SDAR-8B-Chat-b32", "snapshots", "*")))
    if not hits:
        sys.exit(f"SDAR-8B-Chat-b32 not in {HUB}")
    snap = hits[-1]

    src = open(os.path.join(snap, "modeling_sdar.py")).read()
    src = _sub(src, P1_IMPORT_OLD, P1_IMPORT_NEW, "P1-import")
    src = _sub(src, P1_FWD_OLD, P1_FWD_NEW, "P1-forward")
    src = _sub(src, P2_IMPORT_OLD, P2_IMPORT_NEW, "P2-import")
    src = _sub(src, P2_BRANCH_OLD, P2_BRANCH_NEW, "P2-branch")
    src = _sub(src, P3_IMPORT_OLD, P3_IMPORT_NEW, "P3-losskwargs")

    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "__init__.py"), "w").close()
    with open(os.path.join(OUT, "configuration_sdar.py"), "w") as f:
        f.write(open(os.path.join(snap, "configuration_sdar.py")).read())
    with open(os.path.join(OUT, "modeling_sdar.py"), "w") as f:
        f.write(src)
    print(f"wrote {OUT} from {snap}")


if __name__ == "__main__":
    main()
