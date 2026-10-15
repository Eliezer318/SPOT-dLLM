# vendor/ — pinned model-code patches
Vendored/patched modeling code. `llada2_cached_forward.py`: LLaDA2-mini's forward with ONE relaxed mask-shape branch enabling the exact prefix-KV cache; extraction is verbatim and asserts against upstream drift.

## Licenses

The files here are third-party model code distributed under their own licenses, not the repository's
MIT license: see the headers of `sdar/*.py` (Apache License 2.0) and `llada/SOURCE.txt`.
