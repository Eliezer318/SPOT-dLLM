#!/usr/bin/env python
"""README animation: the recorded GSM8K replay as block grids, four arms, at 2x real time.
    python docs/tools/make_gif.py   ->  docs/assets/spot_replay.gif"""
import json, os
from PIL import Image, ImageDraw, ImageFont
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = json.load(open(os.path.join(D, "assets", "traces.json")))["gsm8k"]
ARMS = [("free", "Draft-free", (107, 120, 134)), ("spiffy", "Spiffy (budget 8)", (46, 109, 164)),
        ("spot", "SPOT (budget 8)", (179, 50, 42)), ("spotg", "SPOT-γ (budget 8)", (217, 102, 31))]
BG, INK, MUTED, MASK, FLASH = (246, 247, 249), (19, 26, 34), (86, 97, 110), (216, 222, 230), (255, 217, 102)
FPS, SPEED, CELL, GAP = 12, 2.0, 10, 2
fb = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"; fr = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"; fm = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
F_T, F_L, F_S, F_M = ImageFont.truetype(fb, 17), ImageFont.truetype(fb, 14), ImageFont.truetype(fr, 12), ImageFont.truetype(fm, 13)
gw = 32 * CELL + 31 * GAP; gh = 8 * CELL + 7 * GAP
PADX, LANE_W, LANE_H, TOP = 20, gw, gh + 58, 50
W = PADX * 3 + 2 * LANE_W; H = TOP + 2 * LANE_H + 16 + 26
def mix(a, b, t): return tuple(int(a[i] * (1 - t) + b[i] * t) for i in range(3))
lanes = []
for key, name, col in ARMS:
    arm = data["arms"][key]; cum, acc = [], 0.0
    for r in arm["rounds"]: acc += r["t"]; cum.append(acc)
    lanes.append(dict(key=key, name=name, col=col, arm=arm, cum=cum))
T_END = max(l["cum"][-1] for l in lanes)
frames, durations = [], []
n = int(T_END / SPEED * FPS) + 2
free_wall = data["arms"]["free"]["wall"]
for f in range(n + 1):
    t = min(f * SPEED / FPS, T_END)
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.text((PADX, 14), "SPOT replay · SDAR-8B · GSM8K · recorded wall clock (2× speed)", font=F_T, fill=INK)
    for li, l in enumerate(lanes):
        x0 = PADX + (li % 2) * (LANE_W + PADX); y0 = TOP + (li // 2) * LANE_H
        k = sum(1 for c in l["cum"] if c <= t)
        state = {}
        for r in l["arm"]["rounds"][:k]:
            for pos, depth in r["c"]:
                if 0 <= pos < 256:
                    state[pos] = depth
        now = {p for p, _ in l["arm"]["rounds"][k - 1]["c"]} if 0 < k and (t - l["cum"][k - 1]) < 0.12 and k < len(l["cum"]) + 1 else set()
        done = k >= len(l["cum"])
        d.ellipse((x0, y0 + 4, x0 + 10, y0 + 14), fill=l["col"])
        d.text((x0 + 16, y0), l["name"], font=F_L, fill=INK)
        el = min(t, l["cum"][-1]); written = len(state)
        stat = f"{k:3d} forwards  {el:4.1f} s  {written:3d} tok"
        d.text((x0, y0 + 20), stat, font=F_M, fill=MUTED)
        if done:
            sp = "1.00×" if l["key"] == "free" else f"{free_wall / l['arm']['wall']:.2f}×"
            tw = d.textlength(sp, font=F_L); d.rounded_rectangle((x0 + LANE_W - tw - 12, y0 - 2, x0 + LANE_W, y0 + 18), 4, fill=l["col"])
            d.text((x0 + LANE_W - tw - 6, y0), sp, font=F_L, fill=(255, 255, 255))
        gy = y0 + 42
        for pos in range(256):
            cx = x0 + (pos % 32) * (CELL + GAP); cy = gy + (pos // 32) * (CELL + GAP)
            if pos in now and not done: c = FLASH
            elif pos in state: c = l["col"] if state[pos] > 0 else mix(MASK, l["col"], 0.5)
            else: c = MASK
            if pos >= data["eos"] and pos in state and pos not in now: c = mix(c, BG, 0.5)
            d.rectangle((cx, cy, cx + CELL - 1, cy + CELL - 1), fill=c)
    ly = H - 24; lx = PADX
    for c, lab in ((MASK, "masked"), (FLASH, "current forward"), (mix(MASK, (107, 120, 134), 0.5), "target's own step"), ((179, 50, 42), "accepted draft")):
        d.rectangle((lx, ly + 3, lx + 10, ly + 13), fill=c); d.text((lx + 15, ly), lab, font=F_S, fill=MUTED); lx += 30 + d.textlength(lab, font=F_S) + 10
    frames.append(im); durations.append(int(1000 / FPS))
durations[-1] = 3000
# one shared palette built from every frame, so fixed colours (legend, flash) never shift between frames
strip = Image.new("RGB", (W, H * 4))
for i, fr in enumerate(frames[:: max(1, len(frames) // 4)][:4]):
    strip.paste(fr, (0, H * i))
pal = strip.quantize(colors=96, method=Image.MEDIANCUT)
frames = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
out = os.path.join(D, "assets", "spot_replay.gif")
frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
print("wrote", out, os.path.getsize(out), "bytes,", len(frames), "frames", W, "x", H)
