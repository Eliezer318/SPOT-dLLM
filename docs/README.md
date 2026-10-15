# Project page

`index.html` is the built page. It loads Google Fonts and, from `assets/`, the teaser GIF
(`spot_replay.gif`), the method figure (`fig_method.png`) and the affiliation logos
(`logos/`, sources in `logos/SOURCES.md`). Edit `src/page.html` (head content, a
`<!--BODY-->` marker, then the body), then rebuild:

```
python docs/tools/build.py        # inlines assets/traces.json into index.html
python docs/tools/make_gif.py     # assets/spot_replay.gif for the top-level README
```

`assets/traces.json` holds the recorded runs the replay animates: per target forward, the
generated positions it committed, the tree depth each came from, and the measured round
time (SDAR-8B, T = 0, block 32, budget 8; one GSM8K and one HumanEval prompt, four arms,
identical outputs). GitHub Pages is not enabled; to publish, serve `docs/` from `main`.
