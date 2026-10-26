"""LLaDA2.0-mini forward with ONE relaxed mask-shape branch.

Generated, not checked in. Built to the contract the vendor README states: verbatim
extraction, one relaxed branch, asserts against upstream drift.

Why this is safe rather than a reimplementation. The checkpoint's forward does:

    if attention_mask.size() == (batch_size, 1, seq_length, seq_length):
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(...)
    else:
        raise ValueError("LLaDA2.0 only support block attention mask with shape ...")

and for a 4-D mask that helper is the IDENTITY (transformers 4.55.4,
`_prepare_4d_causal_attention_mask_for_sdpa`: `if attention_mask.dim() == 4:
expanded_4d_mask = attention_mask`). So the square branch applies no transformation and
the else-branch is a pure shape assertion. Accepting a rectangular (b, 1, q, past+q)
additive bias therefore changes nothing about the computation -- it only stops refusing a
geometry the attention math already handles.

Validation is not by inspection: `attach` is gated by exact cached-vs-uncached D0 parity(gate), which is what actually proves the extraction.
"""
import inspect
import sys
import textwrap
import types

_ORIG = """        if attention_mask.size() == (batch_size, 1, seq_length, seq_length):
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_seen_tokens)
        else:
            raise ValueError(
                f"LLaDA2.0 only support block attention mask with shape: {(batch_size, 1, seq_length, seq_length)}, the input attention with shape {attention_mask.size()=}!"
            )
"""

# The same branch in revision dad945cac317 of the checkpoint (formatting changed upstream after the extraction above:
# the helper call's closing parenthesis is on its own line).  Same code, so the same relaxed branch replaces it.
_ORIG_B = '        if attention_mask.size() == (batch_size, 1, seq_length, seq_length):\n            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(\n                attention_mask,\n                (batch_size, seq_length),\n                inputs_embeds,\n                past_seen_tokens,\n            )\n        else:\n            raise ValueError(\n                f"LLaDA2.0 only support block attention mask with shape: {(batch_size, 1, seq_length, seq_length)}, the input attention with shape {attention_mask.size()=}!"\n            )\n'

_PATCHED = """        if attention_mask.size() == (batch_size, 1, seq_length, seq_length):
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_seen_tokens)
        elif (
            attention_mask.dim() == 4
            and attention_mask.shape[0] == batch_size
            and attention_mask.shape[1] == 1
            and attention_mask.shape[2] == seq_length
            and attention_mask.shape[3] == past_seen_tokens + seq_length
        ):
            # RELAXED BRANCH (the whole patch): rectangular additive bias over
            # [past | chunk] for the prefix-KV cache. The square path is a no-op for 4-D
            # masks, so this is the same computation with the shape check widened.
            attention_mask = attention_mask.to(inputs_embeds.dtype)
        else:
            raise ValueError(
                f"LLaDA2.0 cached forward: expected (b,1,S,S) or (b,1,q,past+q); got {attention_mask.size()=}"
            )
"""


class UpstreamDrift(RuntimeError):
    """The checkpoint's forward no longer matches what this patch was extracted from."""


def attach(model):
    """Rebind `model.forward` to the verbatim source with the mask branch relaxed."""
    if getattr(model, "_spot_cached_attached", False):
        return model

    src = inspect.getsource(type(model).forward)
    orig = next((o for o in (_ORIG, _ORIG_B) if o in src), None)
    if orig is None:
        raise UpstreamDrift(
            "modeling_llada2_moe.forward does not contain the expected mask-shape branch; "
            "the vendored patch was extracted against a different revision. Refusing to "
            "guess -- re-extract against this checkpoint."
        )

    # replace BEFORE dedenting: _ORIG is captured at class-body indentation
    patched = src.replace(orig, _PATCHED, 1)
    # drop decorators (docstring helpers; not needed on the rebound function), then
    # dedent -- getsource returns the method still indented inside its class
    lines = [l for l in patched.splitlines(True) if not l.lstrip().startswith("@")]
    body = textwrap.dedent("".join(lines))
    ns = dict(sys.modules[type(model).__module__].__dict__)
    exec(compile(body, "<llada2_cached_forward>", "exec"), ns)
    model.forward = types.MethodType(ns["forward"], model)
    model._spot_cached_attached = True
    return model
