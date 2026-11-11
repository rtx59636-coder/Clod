#!/usr/bin/env python3
"""TikTok-сериал «23:00». Серия 1.

Все кадры рисуются процедурно (тёмная плоская иллюстрация в стиле референсов),
озвучка — edge-tts, звуки синтезируются в numpy, сборка — ffmpeg.

    pip install pillow numpy edge-tts
    python3 episode1/make_video.py            # полное видео
    python3 episode1/make_video.py --preview  # контактный лист кадров
"""
import asyncio
import math
import os
import random
import subprocess
import sys

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(ROOT, "..", "assets", "fonts")
BUILD = os.path.join(ROOT, "build")
OUTDIR = os.path.join(ROOT, "output")
W, H, FPS, K = 1080, 1920, 30, 2  # K — суперсэмплинг статичных слоёв
SR = 44100

RED = (229, 57, 53)
HOOD, SKIN, HAIR, HAND = (18, 20, 26), (122, 102, 92), (12, 13, 17), (146, 120, 106)
RIM = (50, 55, 72)

# ---------------------------------------------------------------- drawing utils
_fc = {}


def font(size, weight="Black", mono=False):
    key = (int(size), weight, mono)
    if key not in _fc:
        f = ImageFont.truetype(os.path.join(FONTS, "JetBrainsMono.ttf" if mono else "Montserrat.ttf"), int(size))
        f.set_variation_by_name(weight)
        _fc[key] = f
    return _fc[key]


class SD:
    """ImageDraw, принимающий координаты в пространстве 1080x1920."""

    def __init__(s, img, k=K):
        s.img, s.k, s.d = img, k, ImageDraw.Draw(img, "RGBA")

    def b(s, box):
        return [v * s.k for v in box]

    def rect(s, box, fill, r=0, outline=None, width=0):
        w = max(1, int(width * s.k)) if outline else 0
        if r:
            s.d.rounded_rectangle(s.b(box), radius=r * s.k, fill=fill, outline=outline, width=w)
        else:
            s.d.rectangle(s.b(box), fill=fill, outline=outline, width=w)

    def ell(s, box, fill, outline=None, width=0):
        s.d.ellipse(s.b(box), fill=fill, outline=outline, width=max(1, int(width * s.k)) if outline else 0)

    def poly(s, pts, fill):
        s.d.polygon([(x * s.k, y * s.k) for x, y in pts], fill=fill)

    def line(s, pts, fill, width):
        s.d.line([(x * s.k, y * s.k) for x, y in pts], fill=fill, width=max(1, int(width * s.k)))

    def text(s, xy, txt, size, fill, weight="Black", mono=False, anchor="mm", spacing=0.18, align="center"):
        s.d.multiline_text((xy[0] * s.k, xy[1] * s.k), txt, font=font(size * s.k, weight, mono), fill=fill,
                           anchor=anchor, align=align, spacing=size * s.k * spacing)


def grad(stops, w=W * K, h=H * K):
    ys = np.linspace(0, 1, h)
    cols = np.stack([np.interp(ys, [p for p, _ in stops], [c[i] for _, c in stops]) for i in range(3)], 1)
    return Image.fromarray(np.repeat(cols[:, None, :], w, axis=1).astype(np.uint8))


def soft(base, fn, blur, color=(0, 0, 0), div=4):
    """Рисует fn на слое пониженного разрешения, размывает и накладывает (дым, свечение)."""
    k = base.size[0] / W
    lay = Image.new("RGBA", (base.size[0] // div, base.size[1] // div), color + (0,))
    fn(SD(lay, k / div))
    lay = lay.filter(ImageFilter.GaussianBlur(blur * k / div)).resize(base.size, Image.BILINEAR)
    base.paste(lay, (0, 0), lay)
    return base


def skyline(sd, rng, y, x0, x1, hmin, hmax, wmin, wmax, col, win=None, lit=0.0, dark=None):
    x = x0
    while x < x1:
        w, h = rng.uniform(wmin, wmax), rng.uniform(hmin, hmax)
        sd.rect([x, y - h, x + w, H + 10], col)
        if rng.random() < 0.25:
            sd.rect([x + w * 0.5 - 2, y - h - rng.uniform(20, 60), x + w * 0.5 + 2, y - h], col)
        ww = max(7, w / 9)
        wh = ww * 1.35
        for cy in np.arange(y - h + ww * 1.6, H, wh * 2.1):
            for ci in range(int((w - ww) / (ww * 2))):
                cx = x + ww + ci * ww * 2
                if win and rng.random() < lit:
                    sd.rect([cx, cy, cx + ww, cy + wh], win)
                elif dark:
                    sd.rect([cx, cy, cx + ww, cy + wh], dark)
        x += w + rng.uniform(-10, 14)


def person_back(sd, x, y, s, hood=HOOD, skin=SKIN, hair=HAIR, rim=RIM, backpack=False):
    """Герой со спины, как на референсах: тёмные взъерошенные волосы, капюшон."""
    if rim:
        sd.ell([x - 0.91 * s, y - 1.17 * s, x + 0.91 * s, y + 0.56 * s], rim)
        sd.ell([x - 2.06 * s, y + 0.84 * s, x + 2.06 * s, y + 5.0 * s], rim)
    sd.ell([x - 2.0 * s, y + 0.9 * s, x + 2.0 * s, y + 5.0 * s], hood)
    sd.rect([x - 2.0 * s, y + 2.9 * s, x + 2.0 * s, y + 14 * s], hood)
    if backpack:
        bp = tuple(c + 10 for c in hood)
        sd.rect([x - 1.5 * s, y + 1.2 * s, x - 1.05 * s, y + 6 * s], bp, r=0.2 * s)
        sd.rect([x + 1.05 * s, y + 1.2 * s, x + 1.5 * s, y + 6 * s], bp, r=0.2 * s)
    sd.ell([x - 1.15 * s, y + 0.5 * s, x + 1.15 * s, y + 1.5 * s], tuple(c + 7 for c in hood))
    sd.rect([x - 0.4 * s, y + 0.1 * s, x + 0.4 * s, y + 0.95 * s], skin)
    sd.ell([x - 0.98 * s, y - 0.3 * s, x - 0.62 * s, y + 0.3 * s], skin)
    sd.ell([x + 0.62 * s, y - 0.3 * s, x + 0.98 * s, y + 0.3 * s], skin)
    sd.ell([x - 0.85 * s, y - 1.1 * s, x + 0.85 * s, y + 0.5 * s], hair)
    pts = [(x - 0.84 * s, y - 0.1 * s)]
    for i in range(8):
        pts.append((x - 0.8 * s + 1.6 * s * i / 7, y + (0.44 if i % 2 == 0 else 0.2) * s))
    pts.append((x + 0.84 * s, y - 0.1 * s))
    sd.poly(pts, hair)
    cy, rx, ry = y - 0.3 * s, 0.85 * s, 0.8 * s
    for a in (-75, -45, -15, 15, 45, 75):
        t0, t1, tt = math.radians(a - 14), math.radians(a + 14), math.radians(a + 12)
        sd.poly([(x + rx * math.sin(t0), cy - ry * math.cos(t0)),
                 (x + 1.2 * rx * math.sin(tt), cy - 1.2 * ry * math.cos(tt)),
                 (x + rx * math.sin(t1), cy - ry * math.cos(t1))], hair)


def figure(sd, x, feet, h, col, eyes=False, eye=(235, 235, 235)):
    """Безликий силуэт в капюшоне."""
    hw, hr = h * 0.16, h * 0.075
    hy = feet - h + hr * 1.3
    sd.poly([(x - hw, feet - h * 0.78), (x + hw, feet - h * 0.78), (x + hw * 1.06, feet - h * 0.08),
             (x - hw * 1.06, feet - h * 0.08)], col)
    sd.ell([x - hw, feet - h * 0.86, x + hw, feet - h * 0.7], col)
    sd.rect([x - hw * 0.8, feet - h * 0.12, x - hw * 0.12, feet], col)
    sd.rect([x + hw * 0.12, feet - h * 0.12, x + hw * 0.8, feet], col)
    sd.ell([x - hr * 1.3, hy - hr * 1.35, x + hr * 1.3, hy + hr * 1.25], col)
    if eyes:
        e = hr * 0.3
        for dx in (-0.45, 0.45):
            sd.ell([x + dx * hr - e, hy - e * 0.55, x + dx * hr + e, hy + e * 0.55], eye)
    return (x, hy)


# ---------------------------------------------------------------- phone
def phone(w, h, screen_fn, hand=True):
    m = int(w * 0.35)
    lw, lh = w + 2 * m, h + m + int(h * 1.2)
    im = Image.new("RGBA", (lw * K, lh * K), (0, 0, 0, 0))
    sd = SD(im)
    x0, y0, x1, y1 = m, m, m + w, m + h
    r = w * 0.13
    sd.rect([x0, y0, x1, y1], (28, 30, 37), r=r)
    bz = w * 0.04
    sb = (x0 + bz, y0 + bz, x1 - bz, y1 - bz)
    sd.rect(list(sb), (15, 17, 23), r=r * 0.8)
    screen_fn(sd, *sb)
    cx = (x0 + x1) / 2
    sd.rect([cx - w * 0.14, sb[1] + w * 0.03, cx + w * 0.14, sb[1] + w * 0.09], (0, 0, 0), r=w * 0.03)
    sd.rect([x0, y0, x1, y1], None, r=r, outline=(72, 76, 90), width=max(1.5, w * 0.007))
    if hand:
        for i in range(4):
            fy = y0 + h * (0.45 + 0.1 * i)
            sd.ell([x0 - w * 0.09, fy, x0 + w * 0.07, fy + h * 0.09], HAND)
        sd.poly([(x1 - w * 0.35, y1 - h * 0.02), (x1 + w * 0.12, y1 - h * 0.12), (x1 + w * 0.42, y1 + h * 1.2),
                 (x1 - w * 0.2, y1 + h * 1.2)], HAND)
        sd.ell([x1 - w * 0.42, y1 - h * 0.12, x1 + w * 0.16, y1 + h * 0.2], HAND)
        sd.ell([x1 - w * 0.07, y0 + h * 0.56, x1 + w * 0.09, y0 + h * 0.86], HAND)
    return im, (m + w / 2, m + h / 2)


def paste_phone(base, ph, cx, cy, angle):
    im, (pcx, pcy) = ph
    vx, vy = (pcx * K - im.size[0] / 2), (pcy * K - im.size[1] / 2)
    rot = im.rotate(angle, resample=Image.BICUBIC, expand=True)
    a = math.radians(angle)
    rx, ry = vx * math.cos(a) + vy * math.sin(a), -vx * math.sin(a) + vy * math.cos(a)
    base.paste(rot, (int(cx * K - rot.size[0] / 2 - rx), int(cy * K - rot.size[1] / 2 - ry)), rot)


def scr_lock(time_txt):
    def f(sd, x0, y0, x1, y1):
        w = x1 - x0
        sd.text(((x0 + x1) / 2, y0 + (y1 - y0) * 0.22), time_txt, w * 0.24, (215, 218, 228), weight="Light")
        sd.text(((x0 + x1) / 2, y0 + (y1 - y0) * 0.31), "среда, 11 ноября", w * 0.05, (160, 165, 180), "Medium")
    return f


def scr_nosignal(sd, x0, y0, x1, y1):
    w, cx, cy = x1 - x0, (x0 + x1) / 2, (y0 + y1) / 2
    bw = w * 0.08
    for i in range(4):
        bh = w * (0.08 + 0.07 * i)
        bx = cx - bw * 2.6 + i * bw * 1.4
        sd.rect([bx, cy - bh, bx + bw, cy], (150, 152, 165), r=bw * 0.2)
    sd.line([(cx - w * 0.26, cy + w * 0.03), (cx + w * 0.26, cy - w * 0.38)], RED, w * 0.035)
    sd.text((cx, cy + w * 0.18), "НЕТ СЕТИ", w * 0.11, (225, 228, 235))


def scr_alert(sd, x0, y0, x1, y1):
    w, cx, cy = x1 - x0, (x0 + x1) / 2, (y0 + y1) / 2
    ty = cy - w * 0.32
    r = w * 0.2
    pts = [(cx, ty - r), (cx + r * 1.1, ty + r * 0.75), (cx - r * 1.1, ty + r * 0.75)]
    sd.line(pts + [pts[0], pts[1]], RED, w * 0.035)
    sd.text((cx, ty + r * 0.15), "!", w * 0.2, RED)
    sd.text((cx, cy + w * 0.3), "НЕ ВЫХОДИТЕ\nНА УЛИЦУ\nПОСЛЕ 23:00", w * 0.095, RED)


def scr_chat(msgs):
    def f(sd, x0, y0, x1, y1):
        w = x1 - x0
        sd.rect([x0, y0, x1, y0 + w * 0.36], (24, 26, 33), r=w * 0.1)
        sd.ell([x0 + w * 0.08, y0 + w * 0.15, x0 + w * 0.24, y0 + w * 0.31], (90, 95, 110))
        sd.ell([x0 + w * 0.13, y0 + w * 0.175, x0 + w * 0.19, y0 + w * 0.235], (150, 155, 170))
        sd.text((x0 + w * 0.3, y0 + w * 0.23), "Неизвестный номер", w * 0.06, (220, 222, 230), "Bold", anchor="lm")
        y = y0 + w * 0.5
        for txt, col, tcol in msgs:
            n = txt.count("\n") + 1
            bh = w * (0.1 + 0.105 * n)
            sd.rect([x0 + w * 0.06, y, x1 - w * 0.12, y + bh], col, r=w * 0.06)
            sd.text((x0 + w * 0.11, y + bh / 2), txt, w * 0.064, tcol, "ExtraBold", anchor="lm", align="left")
            y += bh + w * 0.06
    return f


# ---------------------------------------------------------------- captions
_cap_cache = {}


def caption_img(markup, size, weight="Black", color=(255, 255, 255), maxw=960):
    key = (markup, size, weight, color)
    if key in _cap_cache:
        return _cap_cache[key]
    longest = max(markup.replace("*", "").split("\n"), key=lambda ln: font(size, weight).getlength(ln))
    while font(size, weight).getlength(longest) > maxw:
        size -= 2
    f = font(size, weight)
    lines, red = [], False
    for raw in markup.split("\n"):
        runs, cur = [], ""
        for ch in raw:
            if ch == "*":
                if cur:
                    runs.append((cur, red))
                cur, red = "", not red
            else:
                cur += ch
        if cur:
            runs.append((cur, red))
        lines.append(runs)
    lh = int(size * 1.13)
    widths = [sum(f.getlength(t) for t, _ in ln) for ln in lines]
    pad = int(size * 0.5)
    iw, ih = int(max(widths)) + 2 * pad, lh * len(lines) + 2 * pad
    txt = Image.new("RGBA", (iw, ih), (0, 0, 0, 0))
    d = ImageDraw.Draw(txt)
    for i, ln in enumerate(lines):
        x = (iw - widths[i]) / 2
        y = pad + i * lh + lh / 2
        for t, r in ln:
            d.text((x, y), t, font=f, fill=RED if r else color, anchor="lm",
                   stroke_width=max(2, size // 30), stroke_fill=(8, 8, 10))
            x += f.getlength(t)
    sh = Image.new("RGBA", (iw, ih), (0, 0, 0, 0))
    sh.putalpha(txt.getchannel("A").filter(ImageFilter.GaussianBlur(size * 0.18)).point(lambda v: min(255, v * 1.6)))
    out = Image.alpha_composite(sh, txt)
    _cap_cache[key] = out
    return out


def tag_img(text, size=40):
    f = font(size, "Bold", mono=True)
    tw = f.getlength(text)
    im = Image.new("RGBA", (int(tw + size * 1.4), int(size * 2.0)), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([1, 1, im.size[0] - 2, im.size[1] - 2], radius=size * 0.35, fill=(16, 18, 24, 225),
                        outline=(95, 100, 115), width=2)
    d.text((size * 0.7, im.size[1] / 2), text, font=f, fill=(222, 224, 230), anchor="lm")
    return im


def paste_center(frame, im, cx, cy, alpha=1.0, scale=1.0):
    if scale != 1.0:
        im = im.resize((max(1, int(im.size[0] * scale)), max(1, int(im.size[1] * scale))), Image.BILINEAR)
    mask = im.getchannel("A")
    if alpha < 1:
        mask = mask.point(lambda v: int(v * alpha))
    frame.paste(im, (int(cx - im.size[0] / 2), int(cy - im.size[1] / 2)), mask)


# ---------------------------------------------------------------- frame helpers
def ease(x):
    x = min(1, max(0, x))
    return x * x * (3 - 2 * x)


def kb(base, t, dur, z0=1.0, z1=1.07, cx=0.5, cy=0.5, shake=0.0, seed=0):
    z = z0 + (z1 - z0) * ease(t / max(dur, 1e-6))
    bw, bh = base.size
    cw, ch = bw / z, bh / z
    x0, y0 = (bw - cw) * cx, (bh - ch) * cy
    if shake:
        x0 += shake * K * math.sin(t * 71 + seed) * (0.6 + 0.4 * math.sin(t * 13))
        y0 += shake * K * math.cos(t * 59 + seed)
        x0, y0 = min(max(0, x0), bw - cw), min(max(0, y0), bh - ch)
    return base.resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch))


def kb_map(base, t, dur, pt, z0=1.0, z1=1.07, cx=0.5, cy=0.5):
    """Куда точка pt (координаты 1080x1920) попадёт после kb()."""
    z = z0 + (z1 - z0) * ease(t / max(dur, 1e-6))
    bw, bh = base.size
    cw, ch = bw / z, bh / z
    x0, y0 = (bw - cw) * cx, (bh - ch) * cy
    return ((pt[0] * K - x0) / cw * W, (pt[1] * K - y0) / ch * H)


def mix(a, b, f):
    return a if f <= 0 else b if f >= 1 else Image.blend(a, b, f)


BLACK = Image.new("RGB", (W, H))
WHITE = Image.new("RGB", (W, H), (255, 255, 255))


def darken(im, f):
    return mix(BLACK, im, f)


def glitch(im, amt, seed):
    rng = random.Random(seed)
    r, g, b = im.split()
    r, b = ImageChops.offset(r, int(amt), 0), ImageChops.offset(b, -int(amt), 0)
    im = Image.merge("RGB", (r, g, b))
    for _ in range(int(amt / 3) + 2):
        y, hh = rng.randint(0, H - 60), rng.randint(10, 90)
        band = im.crop((0, y, W, y + hh))
        im.paste(ImageChops.offset(band, rng.randint(-int(amt * 4), int(amt * 4)), 0), (0, y))
    return im


def glow_dots(frame, pts, r, color, alpha=1.0):
    """Небольшое свечение (глаза, мигалки) прямо на кадре."""
    for x, y in pts:
        s = int(r * 6)
        lay = Image.new("RGBA", (s, s), color + (0,))
        d = ImageDraw.Draw(lay)
        d.ellipse([s / 2 - r * 1.6, s / 2 - r * 1.6, s / 2 + r * 1.6, s / 2 + r * 1.6], fill=color + (int(150 * alpha),))
        lay = lay.filter(ImageFilter.GaussianBlur(r))
        d = ImageDraw.Draw(lay)
        d.ellipse([s / 2 - r * 0.45, s / 2 - r * 0.45, s / 2 + r * 0.45, s / 2 + r * 0.45],
                  fill=(255, 255, 255, int(255 * alpha)))
        frame.paste(lay, (int(x - s / 2), int(y - s / 2)), lay)


def impact_lines(frame, x, y, s, alpha=1.0):
    d = ImageDraw.Draw(frame, "RGBA")
    c = (235, 235, 240, int(255 * alpha))
    for i, (dx, dy) in enumerate([(0, 0), (s * 0.9, s * 0.55), (s * 1.6, s * 1.4)]):
        d.line([(x + dx, y + dy), (x + dx + s * 0.35, y + dy - s * 0.55)], fill=c, width=max(3, int(s * 0.18)))


# ---------------------------------------------------------------- backgrounds
def pylon(sd, x, top, base, col=(26, 27, 34)):
    for a, b in [((x - 70, base), (x - 8, top)), ((x + 70, base), (x + 8, top))]:
        sd.line([a, b], col, 5)
    for i, yy in enumerate(np.linspace(top + 40, base - 20, 6)):
        span = 10 + 60 * (yy - top) / (base - top)
        sd.line([(x - span, yy), (x + span, yy + 30)], col, 3)
        sd.line([(x + span, yy), (x - span, yy + 30)], col, 3)
    for yy, span in [(top + 30, 85), (top + 80, 65)]:
        sd.line([(x - span, yy), (x + span, yy)], col, 5)
    sd.line([(x + 85, top + 30), (1100, top - 40)], (30, 30, 38), 2)
    sd.line([(x - 85, top + 30), (-20, top + 10)], (30, 30, 38), 2)


def bg_city_fire():
    img = grad([(0, (17, 19, 27)), (0.18, (28, 28, 38)), (0.33, (92, 38, 42)), (0.42, (60, 32, 40)),
                (0.6, (22, 22, 30)), (1, (13, 14, 19))])
    sd, rng = SD(img), random.Random(1)
    soft(img, lambda s: s.ell([350, 380, 1250, 760], (175, 52, 46, 120)), 70, (175, 52, 46))

    def smoke(s):
        r2 = random.Random(5)
        for i in range(70):
            f = i / 70
            cx, cy = 720 + 70 * f + r2.uniform(-40, 40), 600 - 560 * f
            rad = 45 + 140 * f + r2.uniform(-15, 15)
            s.ell([cx - rad, cy - rad, cx + rad, cy + rad], (23, 23, 28, 238))
    soft(img, smoke, 9, (23, 23, 28))
    skyline(sd, rng, 600, -20, 1100, 60, 210, 50, 110, (36, 32, 42), (150, 112, 78), 0.08)
    pylon(sd, 165, 380, 640)
    skyline(sd, rng, 840, -20, 1100, 150, 420, 80, 160, (25, 26, 34), (190, 145, 95), 0.05, (29, 31, 40))
    sd.rect([0, 360, 150, H], (19, 21, 28))
    for yy in range(420, 1400, 110):
        sd.rect([30, yy, 70, yy + 70], (24, 26, 34))
        sd.rect([88, yy, 128, yy + 70], (24, 26, 34))
    sd.rect([88, 640, 128, 710], (205, 160, 105))
    sd.rect([900, 320, 1080, H], (21, 23, 30))
    for yy in range(380, 1400, 120):
        sd.rect([935, yy, 985, yy + 75], (26, 28, 36))
        sd.rect([1010, yy, 1060, yy + 75], (26, 28, 36))
    sd.rect([1010, 860, 1060, 935], (205, 160, 105))
    skyline(sd, rng, 1350, 140, 900, 80, 330, 100, 200, (16, 18, 24), (200, 150, 100), 0.03)
    soft(img, lambda s: s.rect([0, 950, 1080, 1450], (42, 42, 55, 55)), 90, (42, 42, 55))
    person_back(sd, 300, 1010, 190)
    paste_phone(img, phone(215, 380, scr_nosignal), 650, 1290, -12)
    return img


def draw_heli(frame, t):
    d = ImageDraw.Draw(frame, "RGBA")
    x, y = 330 + 18 * math.sin(t * 0.5), 230 + 6 * math.sin(t * 0.9)
    ang = math.radians(112 + 10 * math.sin(t * 0.7))
    for wdt, al in [(0.30, 14), (0.21, 18), (0.12, 26)]:
        L = 900
        a1, a2 = ang - wdt / 2, ang + wdt / 2
        d.polygon([(x + 30, y + 22), (x + 30 + L * math.cos(a1), y + 22 + L * math.sin(a1)),
                   (x + 30 + L * math.cos(a2), y + 22 + L * math.sin(a2))], fill=(235, 235, 220, al))
    c = (12, 13, 17)
    d.ellipse([x - 45, y, x + 45, y + 42], fill=c)
    d.polygon([(x - 40, y + 14), (x - 140, y + 6), (x - 140, y + 18), (x - 40, y + 28)], fill=c)
    d.rectangle([x - 150, y - 6, x - 136, y + 18], fill=c)
    blade = 95 * abs(math.sin(t * 40)) + 20
    d.line([(x - blade, y - 6), (x + blade, y - 6)], fill=c, width=4)
    d.line([(x, y - 6), (x, y + 2)], fill=c, width=5)
    d.ellipse([x + 26, y + 18, x + 38, y + 30], fill=(255, 255, 240, 255))


def bg_city_calm():
    img = grad([(0, (13, 16, 25)), (0.55, (33, 39, 56)), (0.75, (26, 30, 42)), (1, (16, 18, 25))])
    sd, rng = SD(img), random.Random(2)
    for _ in range(140):
        x, y, r = rng.uniform(0, 1080), rng.uniform(0, 900), rng.uniform(0.8, 2.2)
        sd.ell([x - r, y - r, x + r, y + r], (200, 205, 220, rng.randint(60, 200)))
    soft(img, lambda s: s.ell([600, 420, 860, 680], (200, 205, 215, 90)), 50, (200, 205, 215))
    sd.ell([680, 500, 780, 600], (226, 226, 220))
    sd.ell([705, 520, 730, 545], (205, 205, 200))
    skyline(sd, rng, 1250, -20, 1100, 100, 420, 70, 150, (30, 34, 47), (190, 150, 100), 0.16, (35, 39, 52))
    skyline(sd, rng, 1500, -20, 1100, 80, 330, 100, 200, (19, 21, 29), (214, 168, 108), 0.28, (24, 26, 35))
    return img


def window_view(w, h, lit):
    v = grad([(0, (18, 22, 34)), (0.7, (40, 44, 60) if lit else (70, 32, 36)), (1, (24, 26, 34))], w * K, h * K)
    sd, rng = SD(v), random.Random(3)
    if lit:
        sd.ell([w * 0.65, h * 0.12, w * 0.8, h * 0.12 + w * 0.15], (225, 225, 218))
        skyline(sd, rng, h * 0.72, -10, w + 10, 60, 260, 40, 90, (26, 30, 42), (214, 168, 108), 0.35, (32, 36, 48))
    else:
        soft(v, lambda s: [s.ell([w * a - 40, h * 0.5, w * a + 40, h * 0.7], (210, 90, 40, 160)) for a in (0.25, 0.7)],
             25, (210, 90, 40))
        skyline(sd, rng, h * 0.72, -10, w + 10, 60, 260, 40, 90, (14, 15, 20), None, 0, (18, 19, 25))
    return v


def bg_room(lit):
    img = grad([(0, (44, 50, 68)), (1, (28, 32, 44))] if lit else [(0, (17, 19, 27)), (1, (10, 11, 15))])
    sd = SD(img)
    wx0, wy0, wx1, wy1 = 560, 330, 1010, 1130
    img.paste(window_view(wx1 - wx0, wy1 - wy0, lit), (wx0 * K, wy0 * K))
    fr = (16, 17, 22) if lit else (8, 9, 12)
    sd.rect([wx0, wy0, wx1, wy1], None, outline=fr, width=24)
    sd.rect([(wx0 + wx1) / 2 - 7, wy0, (wx0 + wx1) / 2 + 7, wy1], fr)
    sd.rect([wx0, (wy0 + wy1) / 2 - 7, wx1, (wy0 + wy1) / 2 + 7], fr)
    sd.rect([wx0 - 30, wy1, wx1 + 30, wy1 + 26], (52, 56, 70) if lit else (20, 21, 27))
    sd.rect([470, 300, 560, 1260], (36, 40, 55) if lit else (13, 15, 20), r=10)
    if lit:
        soft(img, lambda s: s.ell([60, 300, 460, 700], (230, 190, 130, 90)), 70, (230, 190, 130))
    sd.rect([0, 1330, 1080, H], (26, 27, 34) if lit else (11, 12, 15))
    sd.rect([0, 1330, 1080, 1345], (44, 46, 56) if lit else (18, 19, 23))
    person_back(sd, 330, 1110, 150, rim=RIM if lit else (34, 38, 50))
    soft(img, lambda s: s.ell([380, 1120, 660, 1360], (120, 160, 255, 110)), 45, (120, 160, 255))
    return img


def bg_dark_blur():
    return bg_room(False).filter(ImageFilter.GaussianBlur(28)).point(lambda v: int(v * 0.6))


def bg_phone(screen, w=520, h=1000, angle=7, glow=None):
    img = bg_dark_blur().copy()
    if glow:
        soft(img, lambda s: s.ell([140, 380, 940, 1500], glow + (150,)), 90, glow)
    paste_phone(img, phone(w, h, screen), 545, 1020, angle)
    return img


def bg_clock(txt, col=(232, 234, 240)):
    img = bg_dark_blur().copy()
    sd = SD(img)
    sd.rect([200, 700, 880, 980], (20, 22, 28, 235), r=60, outline=(80, 85, 100), width=3)
    sd.text((540, 840), txt, 190, col, "Medium", mono=True)
    return img


def bg_street(mode):
    """mode: 'one' — один силуэт, 'lit' — толпа под фонарями, 'dark' — толпа во тьме."""
    lit = mode == "lit"
    img = grad([(0, (19, 21, 29)), (0.45, (17, 19, 26)), (0.46, (34, 36, 46)), (0.51, (34, 36, 46)),
                (0.512, (21, 23, 30)), (0.81, (22, 24, 31)), (0.812, (30, 32, 41)), (1, (24, 26, 33))])
    sd, rng = SD(img), random.Random(4)
    for row in range(6):
        for col in range(7):
            x, y = 40 + col * 150, 60 + row * 135
            sd.rect([x, y, x + 80, y + 95], (25, 28, 37))
            if rng.random() < 0.15:
                sd.rect([x + 6, y + 6, x + 74, y + 89], (31, 34, 44))
    for x in range(0, 1080, 160):
        sd.rect([x + 20, 1260, x + 110, 1272], (52, 56, 68))
    soft(img, lambda s: s.ell([-300, -200, 500, 600], (120, 130, 160, 50)), 120, (120, 130, 160))
    lamps = [(190, 1), (880, -1)]
    if lit:
        def pools(s):
            for lx, dd in lamps:
                hx = lx + dd * 95
                s.poly([(hx - 20, 400), (hx + 20, 400), (hx + 260, 1150), (hx - 260, 1150)], (255, 230, 170, 45))
                s.ell([hx - 300, 1000, hx + 300, 1180], (255, 225, 160, 70))
        soft(img, pools, 30, (255, 230, 170))
    crowd_eyes = []
    if mode == "one":
        crowd_eyes.append(figure(sd, 560, 1000, 330, (8, 9, 12)))
    else:
        r2 = random.Random(9)
        for row in range(5):
            feet, hh = 990 + row * 115, 250 + row * 55
            n = 9 - row
            for i in range(n):
                x = (i + 0.5) * 1080 / n + r2.uniform(-35, 35)
                c = (28 - row * 4, 30 - row * 4, 38 - row * 5) if lit else (9, 10, 13)
                crowd_eyes.append(figure(sd, x, feet, hh, c, eyes=True,
                                         eye=(240, 240, 240) if lit else (200, 210, 230)))
    for lx, dd in lamps:
        sd.rect([lx - 7, 380, lx + 7, 1010], (14, 15, 19))
        sd.line([(lx, 390), (lx + dd * 95, 400)], (14, 15, 19), 9)
        sd.ell([lx + dd * 95 - 32, 392, lx + dd * 95 + 32, 418], (250, 240, 210) if lit else (24, 25, 30))
    if lit:
        soft(img, lambda s: [s.ell([lx + dd * 95 - 70, 340, lx + dd * 95 + 70, 470], (255, 240, 200, 200))
                             for lx, dd in lamps], 25, (255, 240, 200))
    if mode != "one":
        person_back(sd, 300, 1640, 150, rim=(40, 44, 58))
    else:
        sd.rect([0, 0, 60, H], (9, 10, 13))
        sd.rect([0, 1800, 1080, H], (9, 10, 13))
        person_back(sd, 150, 1700, 165, rim=(36, 40, 52))
    return img, crowd_eyes


def bg_door(with_person=False):
    img = grad([(0, (27, 30, 40)), (1, (17, 19, 26))])
    sd = SD(img)
    sd.rect([240, 230, 840, 1700], (30, 29, 32))
    sd.rect([270, 260, 810, 1690], (44, 38, 35))
    for y0, y1 in [(330, 900), (980, 1620)]:
        sd.rect([340, y0, 740, y1], (37, 32, 30), r=6)
        sd.rect([340, y0, 740, y0 + 6], (52, 45, 41))
    sd.ell([724, 1010, 764, 1050], (110, 104, 98))
    sd.rect([700, 1022, 790, 1040], (100, 95, 90), r=8)
    sd.ell([520, 650, 560, 690], (95, 95, 100))
    sd.ell([528, 658, 552, 682], (12, 12, 14))
    sd.rect([0, 1700, 1080, H], (19, 20, 25))
    soft(img, lambda s: s.rect([280, 1676, 800, 1720], (150, 165, 195, 170)), 14, (150, 165, 195))
    sd.rect([290, 1688, 790, 1698], (175, 185, 210))
    for x in (440, 590):
        sd.rect([x, 1684, x + 55, 1704], (14, 14, 17))
    if with_person:
        person_back(sd, 230, 1430, 230, rim=(44, 48, 62))
    return img


def bg_memorial(candle):
    img = grad([(0, (25, 27, 35)), (1, (15, 16, 22))])
    sd = SD(img)
    if candle:
        soft(img, lambda s: s.ell([520, 900, 1180, 1560], (230, 150, 70, 110)), 80, (230, 150, 70))
    sd.rect([0, 1350, 1080, H], (30, 26, 24))
    sd.rect([0, 1350, 1080, 1366], (48, 41, 36))
    sd.rect([360, 600, 780, 1150], (72, 58, 46), r=6)
    sd.rect([395, 635, 745, 1115], (26, 24, 24))
    img.paste(grad([(0, (128, 108, 86)), (1, (78, 64, 50))], 310 * K, 440 * K), (415 * K, 655 * K))
    sd.ell([505, 740, 655, 910], (52, 44, 36))
    sd.ell([440, 900, 720, 1170], (52, 44, 36))
    sd.rect([415, 1095, 725, 1095 + 0.5], (26, 24, 24))
    sd.rect([395, 1095, 745, 1115], (26, 24, 24))
    sd.poly([(640, 600), (720, 600), (780, 660), (780, 740)], (6, 6, 8))
    sd.rect([360, 1150, 780, 1350], None)
    sd.rect([838, 1200, 868, 1352], (205, 195, 175), r=4)
    sd.line([(853, 1200), (853, 1186)], (20, 20, 20), 3)
    return img


def bg_peephole(lamp):
    S = 760
    c = Image.new("RGB", (S * K, S * K), (20, 22, 28))
    sd = SD(c)
    bx0, by0, bx1, by1 = 300, 300, 460, 500
    wall, floor, ceil = ((48, 52, 62), (38, 40, 46), (30, 32, 38)) if lamp else ((20, 21, 26), (16, 17, 20), (12, 13, 16))
    sd.poly([(0, 0), (bx0, by0), (bx0, by1), (0, S)], wall)
    sd.poly([(S, 0), (bx1, by0), (bx1, by1), (S, S)], tuple(v - 6 for v in wall))
    sd.poly([(0, S), (bx0, by1), (bx1, by1), (S, S)], floor)
    sd.poly([(0, 0), (bx0, by0), (bx1, by0), (S, 0)], ceil)
    sd.rect([bx0, by0, bx1, by1], tuple(v - 12 for v in wall))
    sd.rect([355, 360, 405, 500], (30, 27, 26))
    for t0, t1 in [(0.25, 0.42), (0.6, 0.72)]:
        def lerp(a, b, t):
            return a + (b - a) * t
        xa, xb = lerp(0, bx0, t0), lerp(0, bx0, t1)
        sd.poly([(xa, lerp(80, by0, t0)), (xb, lerp(80, by0, t1)), (xb, lerp(S, by1, t1)), (xa, lerp(S, by1, t0))],
                (34, 30, 30) if lamp else (14, 13, 13))
    if lamp:
        soft(c, lambda s: s.ell([240 * 2.17, 120 * 2.17, 520 * 2.17, 330 * 2.17], (230, 230, 210, 120)), 60,
             (230, 230, 210))
        sd.rect([350, 175, 410, 190], (240, 240, 225))
    a = np.asarray(c)
    n = a.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    u, v = xx / (n - 1) * 2 - 1, yy / (n - 1) * 2 - 1
    r2 = u * u + v * v
    kq = 0.55
    f = (1 + kq * r2) / (1 + kq) * 0.92
    sx = np.clip(((u * f + 1) / 2 * (n - 1)).astype(int), 0, n - 1)
    sy = np.clip(((v * f + 1) / 2 * (n - 1)).astype(int), 0, n - 1)
    fish = Image.fromarray(a[sy, sx])
    img = Image.new("RGB", (W * K, H * K), (0, 0, 0))
    mask = Image.new("L", fish.size, 0)
    ImageDraw.Draw(mask).ellipse([0, 0, n - 1, n - 1], fill=255)
    vig = np.clip(1.25 - np.sqrt(r2) * 0.95, 0, 1)
    fish = Image.fromarray((np.asarray(fish) * vig[..., None]).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2))
    img.paste(fish, ((W * K - n) // 2, (920 - S // 2) * K), mask)
    sd = SD(img)
    sd.ell([540 - S / 2 - 12, 920 - S / 2 - 12, 540 + S / 2 + 12, 920 + S / 2 + 12], None, outline=(55, 55, 60), width=14)
    return img


def bg_radio():
    img = Image.new("RGB", (W * K, H * K), (6, 6, 8))
    sd = SD(img)
    sd.line([(700, 470), (840, 250)], (60, 60, 64), 6)
    sd.rect([300, 460, 780, 770], (44, 37, 32), r=26)
    sd.rect([300, 460, 780, 480], (60, 51, 44), r=10)
    sd.ell([340, 510, 560, 730], (26, 22, 20))
    for i in range(7):
        y = 535 + i * 28
        sd.line([(355, y), (545, y)], (40, 34, 30), 6)
    sd.rect([600, 520, 750, 600], (20, 12, 10), r=8)
    sd.ell([620, 650, 670, 700], (80, 72, 64))
    sd.ell([690, 650, 740, 700], (80, 72, 64))
    return img


# ---------------------------------------------------------------- audio synth
rngA = np.random.default_rng(7)


def tt(d):
    return np.arange(int(d * SR)) / SR


def noise(d):
    return rngA.standard_normal(int(d * SR))


def filt(x, lo=None, hi=None):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR) + 1e-6
    if hi:
        X *= 1 / np.sqrt(1 + (f / hi) ** 4)
    if lo:
        X *= 1 / np.sqrt(1 + (lo / f) ** 4)
    return np.fft.irfft(X, len(x))


def reverb(x, decay=1.2, wet=0.35):
    ir = noise(decay) * np.exp(-tt(decay) / (decay / 5))
    ir = filt(ir, hi=4000)
    n = len(x) + len(ir)
    y = np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(ir, n), n)
    y /= np.max(np.abs(y)) + 1e-9
    out = np.zeros(n)
    out[:len(x)] += x * (1 - wet)
    return out + y * wet * np.max(np.abs(x))


def s_bang():
    t = tt(2.5)
    x = filt(noise(2.5), hi=900) * np.exp(-t / 0.35) * 1.2 + np.sin(2 * np.pi * 42 * t) * np.exp(-t / 0.6)
    return reverb(x / np.max(np.abs(x)), 2.0, 0.4)


def s_boom():
    t = tt(3)
    x = np.sin(2 * np.pi * (38 + 30 * np.exp(-t * 6)) * t) * np.exp(-t / 0.9)
    x += filt(noise(3), hi=300) * np.exp(-t / 0.4) * 0.4
    return x / np.max(np.abs(x))


def s_stinger():
    t = tt(3.5)
    env = np.minimum(1, t / 0.02) * np.exp(-t / 1.2)
    x = sum(np.sin(2 * np.pi * f * t) * a for f, a in [(466, .3), (494, .3), (523, .25), (932, .15), (1047, .1)])
    x = x * env
    x[:int(3 * SR)] += s_boom() * 0.8
    return reverb(x / np.max(np.abs(x)), 2.5, 0.45)


def s_siren(d):
    t = tt(d)
    ph = 2 * np.pi * np.cumsum(700 + 260 * np.sin(2 * np.pi * 0.23 * t)) / SR
    x = np.sin(ph) + 0.3 * np.sin(2 * ph)
    x *= np.minimum(1, t / 1.5) * np.minimum(1, (d - t) / 1.5)
    return reverb(filt(x, hi=1600), 1.5, 0.6)


def s_buzz(d=0.9):
    t = tt(d)
    gate = ((t % 0.45) < 0.32).astype(float)
    x = np.sign(np.sin(2 * np.pi * 150 * t)) * gate
    return filt(x, hi=500) * 0.8


def s_blip():
    t = tt(0.35)
    return (np.sin(2 * np.pi * 1320 * t) * (t < 0.12) + np.sin(2 * np.pi * 990 * t) * (t >= 0.14)) * np.exp(-t / 0.2)


def s_alert(d=1.6):
    t = tt(d)
    x = np.sin(2 * np.pi * 853 * t) + np.sin(2 * np.pi * 960 * t)
    gate = ((t % 0.5) < 0.4).astype(float)
    return x * gate * 0.5


def s_tick():
    t = tt(0.06)
    return filt(noise(0.06), lo=1500) * np.exp(-t / 0.006)


def s_knock():
    t = tt(0.5)
    x = np.sin(2 * np.pi * 95 * t) * np.exp(-t / 0.06) + 0.5 * np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.035)
    x += filt(noise(0.5), hi=2500) * np.exp(-t / 0.012) * 0.9
    return reverb(x / np.max(np.abs(x)), 0.8, 0.3)


def s_heart(d, bpm):
    out = np.zeros(int(d * SR))
    beat = 60 / bpm
    for b in np.arange(0, d - 0.5, beat):
        for off, a in [(0, 1), (0.22, 0.6)]:
            t = tt(0.25)
            th = np.sin(2 * np.pi * 52 * t) * np.exp(-t / 0.07) * a
            i = int((b + off) * SR)
            out[i:i + len(th)] += th[:len(out) - i]
    return out


def s_static(d):
    x = filt(noise(d), lo=1200, hi=7000)
    mod = filt(np.abs(noise(d)), hi=12)
    return x * (0.4 + 2 * np.abs(mod))


def s_hum(d, flick=False):
    t = tt(d)
    x = sum(np.sin(2 * np.pi * 100 * k * t) / k for k in range(1, 6))
    if flick:
        g = np.repeat(rngA.random(int(d * 20) + 1) > 0.4, SR // 20)[:len(t)].astype(float)
        x = x * g + filt(noise(d), lo=2000) * g * 0.3
    return x * 0.3


def s_glitch(d=0.4):
    out = []
    while sum(map(len, out)) < d * SR:
        seg = tt(rngA.uniform(0.015, 0.05))
        out.append(np.sign(np.sin(2 * np.pi * rngA.uniform(200, 3000) * seg)) * rngA.uniform(0.3, 1))
    return np.concatenate(out)[:int(d * SR)] * 0.6


def s_riser(d):
    t = tt(d)
    f = 120 * (6 ** (t / d))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.4 + filt(noise(d), lo=800) * 0.5
    return x * (t / d) ** 2


def s_whine(d):
    t = tt(d)
    return np.sin(2 * np.pi * 2637 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)) * np.minimum(1, t / 1)


def s_rotor(d):
    t = tt(d)
    return filt(noise(d), hi=250) * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 11 * t))) * 2


def s_drone(d, intensity):
    t = tt(d)
    x = (0.5 * np.sin(2 * np.pi * 55 * t) + 0.3 * np.sin(2 * np.pi * 55.35 * t) +
         0.22 * np.sin(2 * np.pi * 82.4 * t) + 0.15 * np.sin(2 * np.pi * 77.8 * t))
    x *= 0.75 + 0.25 * np.sin(2 * np.pi * 0.11 * t)
    wind = filt(noise(d), hi=380) * 0.9 * (0.6 + 0.4 * np.sin(2 * np.pi * 0.07 * t + 1))
    pad = (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 261.6 * t) * 0.8 + np.sin(2 * np.pi * 311.1 * t) * 0.7)
    pad = filt(pad * (0.6 + 0.4 * np.sin(2 * np.pi * 0.25 * t)), hi=900) * 0.25
    return (x * 0.5 + wind * 0.4) * (0.4 + 0.6 * intensity) + pad * intensity ** 2


# ---------------------------------------------------------------- TTS
VOICES = {
    "N": ("ru-RU-DmitryNeural", "+10%", "-6Hz", None),
    "M": ("ru-RU-SvetlanaNeural", "-4%", "-12Hz", "aecho=0.8:0.6:90|180:0.3|0.15"),
    "F": ("ru-RU-DmitryNeural", "-24%", "-28Hz", "lowpass=f=1700,aecho=0.8:0.75:70|150:0.4|0.25,volume=1.4"),
    "R": ("ru-RU-DmitryNeural", "-10%", "-4Hz", "highpass=f=450,lowpass=f=2600,acrusher=bits=8:mix=0.25,volume=1.6"),
}
TRIM = ("silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
        "silenceremove=start_periods=1:start_threshold=-45dB,areverse")


async def _tts(text, voice, path):
    import edge_tts
    v, rate, pitch, _ = VOICES[voice]
    await edge_tts.Communicate(text, v, rate=rate, pitch=pitch).save(path)


def tts(text, voice, idx):
    os.makedirs(BUILD, exist_ok=True)
    mp3 = os.path.join(BUILD, f"vo_{idx:02d}_{voice}.mp3")
    if not os.path.exists(mp3) or os.path.getsize(mp3) == 0:
        asyncio.run(_tts(text, voice, mp3))
    af = TRIM + ("," + VOICES[voice][3] if VOICES[voice][3] else "")
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", mp3, "-af", af, "-ar", str(SR), "-ac", "1", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).astype(np.float64)


# ---------------------------------------------------------------- scenes
class Scene:
    def __init__(self, name, segs, render, pad_start=0.35, pad_end=0.7, min_dur=0, tag=None, sfx=None,
                 intensity=0.5, fade_in=0.12, fade_out=0.1):
        self.name, self.segs, self.render_fn = name, segs, render
        self.pad_start, self.pad_end, self.min_dur = pad_start, pad_end, min_dur
        self.tag, self.sfx, self.intensity, self.fade_in, self.fade_out = tag, sfx, intensity, fade_in, fade_out


def seg(voice, text, caps=()):
    return {"voice": voice, "text": text, "caps": list(caps)}


def cap(text, y, size=82, at=0.0, weight="Black"):
    return {"text": text, "y": y, "size": size, "at": at, "weight": weight}


_bases = {}


def B(name, fn, *a):
    if name not in _bases:
        _bases[name] = fn(*a)
    return _bases[name]


def r_hook(t, d, sc):
    f = kb(B("fire", bg_city_fire), t, d, 1.0, 1.06, 0.45, 0.55)
    draw_heli(f, t)
    if (t % 0.9) < 0.45 and t > 0.6:
        impact_lines(f, 820, 1100, 40)
    return f


def r_calm(t, d, sc):
    return kb(B("calm", bg_city_calm), t, d, 1.0, 1.08, 0.5, 0.6)


def flicker(t, seed=0):
    r = random.Random(int(t * 15) + seed)
    return 1.0 if r.random() > 0.45 else r.uniform(0.25, 0.6)


def r_room(t, d, sc):
    f = kb(B("room_lit", bg_room, True), t, d, 1.0, 1.06, 0.4, 0.6)
    if t > sc.seg_t[1][0]:
        f = darken(f, flicker(t))
    return f


def r_blackout(t, d, sc):
    if t < 0.25:
        return kb(B("room_lit", bg_room, True), 0, 1, 1.06, 1.06, 0.4, 0.6)
    if t < 0.33:
        return WHITE.copy()
    sh = max(0, 22 * (1 - (t - 0.33) / 1.2))
    f = kb(B("room_dark", bg_room, False), t, d, 1.04, 1.1, 0.45, 0.6, shake=sh)
    if t < 0.6:
        f = darken(f, (t - 0.33) / 0.27)
    on = int(t * 4) % 2
    p = kb_map(B("room_dark", bg_room, False), t, d, (700, 1080), 1.04, 1.1, 0.45, 0.6)
    q = kb_map(B("room_dark", bg_room, False), t, d, (880, 1095), 1.04, 1.1, 0.45, 0.6)
    glow_dots(f, [p], 14, (255, 40, 40) if on else (40, 90, 255))
    glow_dots(f, [q], 10, (40, 90, 255) if on else (255, 40, 40))
    return f


def r_nosignal(t, d, sc):
    f = kb(B("ph_nosig", bg_phone, scr_nosignal), t, d, 1.0, 1.06, 0.5, 0.55)
    if (t % 0.9) < 0.5 and t < 1.8:
        impact_lines(f, 880, 420, 45)
    return f


def r_alert(t, d, sc):
    if t < 0.7:
        return kb(B("ph_lock", bg_phone, scr_lock("22:56")), t, d, 1.0, 1.08, 0.5, 0.5)
    a = kb(B("ph_alert", bg_phone, scr_alert, 600, 1150, 3), t, d, 1.0, 1.08, 0.5, 0.5)
    g = kb(B("ph_alert_g", bg_phone, scr_alert, 600, 1150, 3, (210, 30, 30)), t, d, 1.0, 1.08, 0.5, 0.5)
    f = mix(a, g, 0.5 + 0.5 * math.sin(t * 6))
    if t < 0.82:
        f = mix(f, WHITE, 0.4)
    return f


def r_clock(t, d, sc):
    f = kb(B("clk2257", bg_clock, "22:57"), t, d, 1.0, 1.05)
    if (t % 1) > 0.5:
        ImageDraw.Draw(f).rectangle([526, 760, 556, 920], fill=(20, 22, 28))
    return f


def r_street(t, d, sc):
    base, pts = B("street_one", bg_street, "one")
    args = (1.0, 1.35, 0.52, 0.5)
    f = kb(base, t, d, *args)
    if t > sc.seg_t[1][0] + 1.0:
        x, y = kb_map(base, t, d, pts[0], *args)
        a = min(1, (t - sc.seg_t[1][0] - 1.0) / 0.6)
        glow_dots(f, [(x - 9, y), (x + 9, y)], 4, (220, 225, 240), a)
    return f


def r_msg1(t, d, sc):
    name = "ph_chat0" if t < sc.seg_t[1][0] else "ph_chat1"
    msgs = [] if name == "ph_chat0" else [("ОНИ УЖЕ ЗНАЮТ,\nЧТО ТЫ ДОМА.", (48, 51, 62), (240, 240, 244))]
    f = kb(B(name, bg_phone, scr_chat(msgs), 600, 1150, -4), t, d, 1.0, 1.08, 0.5, 0.45)
    if t < 0.9 and (t % 0.45) < 0.3:
        impact_lines(f, 900, 330, 45)
    return f


def r_clock23(t, d, sc):
    if t < 1.0:
        f = kb(B("clk2259", bg_clock, "22:59"), t, d, 1.0, 1.04)
        return f
    f = kb(B("clk2300", bg_clock, "23:00", RED), t, d, 1.0, 1.04)
    if t < 1.35:
        f = glitch(f, 30 * (1.35 - t) / 0.35 + 4, int(t * 30))
    return f


def r_crowd(t, d, sc):
    off = sc.seg_t[1][0] + 0.6
    args = (1.0, 1.1, 0.5, 0.55)
    if t < 0.12:
        return WHITE.copy()
    lit, pts = B("street_lit", bg_street, "lit")
    dark, _ = B("street_dark", bg_street, "dark")
    if t < off:
        f = kb(lit, t, d, *args)
        return mix(WHITE, f, (t - 0.12) / 0.3) if t < 0.42 else f
    if t < off + 0.5:
        f = kb(lit if random.Random(int(t * 20)).random() > 0.5 else dark, t, d, *args)
    else:
        f = kb(dark, t, d, *args)
    pul = 0.75 + 0.25 * math.sin(t * 5)
    eyes = []
    for p in pts:
        x, y = kb_map(dark, t, d, p, *args)
        eyes += [(x - p[1] * 0 - 6, y), (x + 6, y)]
    glow_dots(f, eyes[::2], 4, (190, 205, 235), pul * 0.8)
    return f


def r_knock(t, d, sc):
    base = B("door", bg_door)
    s0 = sc.seg_t[0][1]
    knocks = [s0 + k for k in (0.3, 0.62, 0.94, 2.0, 2.32, 2.64)]
    sh = 0
    for k in knocks:
        if 0 <= t - k < 0.15:
            sh = 14 * (1 - (t - k) / 0.15)
    f = kb(base, t, d, 1.0, 1.08, 0.5, 0.45, shake=sh, seed=3)
    for k in knocks:
        if 0 <= t - k < 0.25:
            impact_lines(f, 850, 820, 55, 1 - (t - k) / 0.25)
    return f


def r_voice(t, d, sc):
    return kb(B("door_p", bg_door, True), t, d, 1.04, 1.14, 0.55, 0.4)


def r_memorial(t, d, sc):
    a = kb(B("mem1", bg_memorial, True), t, d, 1.0, 1.12, 0.45, 0.45)
    b = kb(B("mem0", bg_memorial, False), t, d, 1.0, 1.12, 0.45, 0.45)
    fl = 0.75 + 0.25 * math.sin(t * 9) * math.sin(t * 3.7)
    f = mix(b, a, fl)
    x, y = kb_map(B("mem1", bg_memorial, True), t, d, (853, 1172), 1.0, 1.12, 0.45, 0.45)
    glow_dots(f, [(x, y + 2 * math.sin(t * 11))], 9, (255, 170, 70))
    return f


def r_peep(t, d, sc):
    on = flicker(t, 5) > 0.9 or t < 0.8
    return kb(B("peep1" if on else "peep0", bg_peephole, on), t, d, 1.0, 1.15, 0.5, 0.48)


def r_msg2(t, d, sc):
    m1 = ("ОНИ УЖЕ ЗНАЮТ,\nЧТО ТЫ ДОМА.", (48, 51, 62), (240, 240, 244))
    m2 = ("НЕ ОТКРЫВАЙ.\nМЫ ПРЕДУПРЕЖДАЛИ\nТЕБЯ.", (120, 22, 24), (255, 235, 235))
    name = "ph_chat1b" if t < sc.seg_t[1][0] else "ph_chat2"
    msgs = [m1] if name == "ph_chat1b" else [m1, m2]
    f = kb(B(name, bg_phone, scr_chat(msgs), 600, 1150, 5), t, d, 1.0, 1.12, 0.5, 0.5)
    if t > d - 0.7:
        f = glitch(f, 10 + 40 * (t - d + 0.7), int(t * 30))
    return f


def r_end(t, d, sc):
    f = kb(B("radio", bg_radio), t, d, 1.0, 1.04, 0.5, 0.3)
    on = random.Random(int(t * 12)).random() > 0.3
    glow_dots(f, [(675, 560)], 22 if on else 12, (255, 50, 40), 1 if on else 0.5)
    if t < 0.5:
        f = glitch(f, 30, int(t * 30))
    return f


# ---- per-scene sound design. A.add(sound, time_in_scene, gain)
def fx_hook(A, sc):
    A.add(s_boom(), 0, 0.8)
    A.add(s_siren(sc.dur + 1), 0, 0.07)
    A.add(s_rotor(sc.dur), 0, 0.12)
    A.add(s_buzz(1.8), 0.6, 0.25)


def fx_room(A, sc):
    A.add(s_hum(sc.dur - sc.seg_t[1][0], True), sc.seg_t[1][0], 0.12)


def fx_blackout(A, sc):
    A.add(s_bang(), 0.25, 1.0)
    A.add(s_siren(sc.dur + 4), 0.8, 0.12)
    A.add(s_siren(sc.dur + 3), 1.6, 0.06)
    A.add(filt(noise(1.2), lo=1500) * np.exp(-tt(1.2) / 0.25), 1.9, 0.25)


def fx_nosig(A, sc):
    A.add(s_buzz(1.8), 0.1, 0.3)
    A.add(s_static(0.4), sc.dur - 0.3, 0.12)


def fx_alert(A, sc):
    A.add(s_alert(1.6), 0.7, 0.18)
    A.add(s_boom(), 0.7, 0.5)


def fx_clock(A, sc):
    for k in np.arange(0.0, sc.dur, 1.0):
        A.add(s_tick(), k, 0.5)


def fx_street(A, sc):
    A.add(s_heart(sc.dur, 62), 0.5, 0.5)
    A.add(s_whine(sc.dur), sc.seg_t[1][0], 0.015)


def fx_msg1(A, sc):
    A.add(s_buzz(0.9), 0.1, 0.3)
    A.add(s_blip(), sc.seg_t[1][0] - 0.15, 0.25)
    A.add(s_heart(sc.dur, 72), 0, 0.45)


def fx_clock23(A, sc):
    for k in (0.0, 0.5):
        A.add(s_tick(), k, 0.6)
    A.add(s_glitch(0.35), 1.0, 0.25)
    A.add(s_boom(), 1.0, 0.7)


def fx_crowd(A, sc):
    A.add(s_bang()[: int(0.8 * SR)], 0, 0.5)
    A.add(s_hum(sc.seg_t[1][0] + 0.6), 0, 0.12)
    A.add(s_whine(sc.dur), 0.3, 0.03)
    A.add(s_heart(sc.dur, 92), 0.3, 0.6)
    A.add(s_hum(0.5, True), sc.seg_t[1][0] + 0.6, 0.2)


def fx_knock(A, sc):
    s0 = sc.seg_t[0][1]
    for k in (0.3, 0.62, 0.94, 2.0, 2.32, 2.64):
        A.add(s_knock(), s0 + k, 0.9)


def fx_voice(A, sc):
    A.add(s_riser(sc.dur), 0, 0.12)


def fx_memorial(A, sc):
    s1, e1 = sc.seg_t[1]
    A.add(s_stinger(), s1 + (e1 - s1) * 0.5, 0.7)


def fx_peep(A, sc):
    A.add(s_heart(sc.dur, 56), 0, 0.5)
    A.add(s_hum(sc.dur, True), 0, 0.06)


def fx_msg2(A, sc):
    A.add(s_buzz(0.9), 0.1, 0.3)
    A.add(s_blip(), sc.seg_t[1][0] - 0.15, 0.25)
    A.add(s_riser(1.5), sc.dur - 1.5, 0.25)
    A.add(s_glitch(0.6), sc.dur - 0.6, 0.3)


def fx_end(A, sc):
    A.add(s_boom(), 0, 0.9)
    A.add(s_static(sc.dur), 0, 0.09)


SCENES = [
    Scene("hook", [seg("N", "Люди, которые выходят на улицу после двадцати трёх ноль-ноль... бесследно исчезают.",
                       [cap("ЛЮДИ, КОТОРЫЕ ВЫХОДЯТ\nНА УЛИЦУ ПОСЛЕ 23:00,", 690, 66),
                        cap("БЕССЛЕДНО\n*ИСЧЕЗАЮТ*…", 1560, 128, at=0.55)])],
          r_hook, pad_start=0.25, pad_end=0.8, tag="СЕРИЯ 1 · «23:00»", sfx=fx_hook, intensity=0.6, fade_in=0),
    Scene("calm", [seg("N", "Одиннадцатое ноября. Две тысячи двадцать шестой год."),
                   seg("N", "Никто даже не подозревал, что этот вечер станет последним обычным вечером в истории человечества.",
                       [cap("НИКТО ДАЖЕ\nНЕ ПОДОЗРЕВАЛ,\nЧТО ЭТОТ ВЕЧЕР\nСТАНЕТ *ПОСЛЕДНИМ*…", 800, 80)])],
          r_calm, tag="11 НОЯБРЯ 2026 ГОДА.", intensity=0.2),
    Scene("room", [seg("N", "Александр сидел дома и листал телефон.", [cap("АЛЕКСАНДР, 24 ГОДА", 250, 58)]),
                   seg("N", "Внезапно свет начал мигать.", [cap("ВНЕЗАПНО СВЕТ\nНАЧИНАЕТ *МИГАТЬ*…", 250, 80)])],
          r_room, sfx=fx_room, intensity=0.3, pad_end=0.5),
    Scene("blackout", [seg("N", "Весь город погружается в темноту.",
                           [cap("*БАХ!*", 330, 170), cap("ВЕСЬ ГОРОД\nПОГРУЖАЕТСЯ\nВ *ТЕМНОТУ*.", 1540, 86, at=0.1)]),
                       seg("N", "Крики. Аварии. Сирены.",
                           [cap("НА УЛИЦЕ —\n*КРИКИ* И СИРЕНЫ…", 1540, 86)])],
          r_blackout, pad_start=0.8, sfx=fx_blackout, intensity=0.6, fade_in=0),
    Scene("nosignal", [seg("N", "Он берёт телефон. Сети нет.", [cap("ТЕЛЕФОНЫ\nПЕРЕСТАЮТ *РАБОТАТЬ*.", 280, 80)])],
          r_nosignal, sfx=fx_nosig, intensity=0.55),
    Scene("alert", [seg("N", "И вдруг на экране появляется неизвестное уведомление.",
                        [cap("НЕИЗВЕСТНОЕ\n*УВЕДОМЛЕНИЕ*", 230, 72)]),
                    seg("M", "Не выходите на улицу после двадцати трёх ноль-ноль.")],
          r_alert, sfx=fx_alert, intensity=0.65),
    Scene("clock", [seg("N", "Он смотрит на часы. Двадцать два пятьдесят семь.",
                        [cap("ДО 23:00 ОСТАЛОСЬ\n*ТРИ МИНУТЫ*…", 1250, 80, at=0.45)])],
          r_clock, sfx=fx_clock, intensity=0.6),
    Scene("street", [seg("N", "На другой стороне дороги появляется человек.",
                         [cap("НА ДРУГОЙ СТОРОНЕ\nДОРОГИ *КТО-ТО СТОИТ*…", 1470, 76)]),
                     seg("N", "Он стоит неподвижно... и смотрит прямо в его окно.",
                         [cap("И СМОТРИТ\n*ПРЯМО В ЕГО ОКНО*.", 1470, 80)])],
          r_street, sfx=fx_street, intensity=0.7, pad_end=1.0),
    Scene("msg1", [seg("N", "Снова вибрация. Неизвестный номер."),
                   seg("M", "Они уже знают, что ты дома.")],
          r_msg1, sfx=fx_msg1, intensity=0.75),
    Scene("clock23", [seg("N", "Двадцать три ноль-ноль.")], r_clock23, pad_start=1.1, pad_end=0.4, sfx=fx_clock23,
          intensity=0.8),
    Scene("crowd", [seg("N", "Все фонари на секунду загораются... и он видит их.",
                        [cap("ФОНАРИ\n*ЗАГОРАЮТСЯ*…", 230, 80)]),
                    seg("N", "Десятки людей посреди дороги. И все они смотрят на его окно.",
                        [cap("ДЕСЯТКИ ЛЮДЕЙ.\nВСЕ СМОТРЯТ *НА НЕГО*.", 230, 80)])],
          r_crowd, pad_start=0.6, pad_end=1.2, sfx=fx_crowd, intensity=1.0, fade_in=0),
    Scene("knock", [seg("N", "А потом в дверь начинают стучать.",
                        [cap("А ПОТОМ НАЧИНАЮТ\n*СТУЧАТЬ В ДВЕРЬ*…", 230, 76)])],
          r_knock, pad_end=3.3, sfx=fx_knock, intensity=0.55),
    Scene("voice", [seg("F", "Александр... открой. Это я.", [cap("«АЛЕКСАНДР…\nОТКРОЙ. ЭТО Я…»", 280, 80, weight="ExtraBold")])],
          r_voice, pad_start=0.5, pad_end=0.9, sfx=fx_voice, intensity=0.7),
    Scene("memorial", [seg("N", "Он узнаёт этот голос. Это голос его отца.", [cap("ЭТО ГОЛОС\nЕГО *ОТЦА*.", 300, 86)]),
                       seg("N", "Но его отец умер пять лет назад.", [cap("НО ОТЕЦ УМЕР\n*ПЯТЬ ЛЕТ НАЗАД*.", 300, 86)])],
          r_memorial, sfx=fx_memorial, intensity=0.85, pad_end=1.2),
    Scene("peep", [seg("N", "Он смотрит в глазок..."),
                   seg("N", "За дверью никого нет.", [cap("ЗА ДВЕРЬЮ\n*НИКОГО НЕТ*.", 1560, 92)])],
          r_peep, sfx=fx_peep, intensity=0.35, pad_end=1.0),
    Scene("msg2", [seg("N", "Новое сообщение."),
                   seg("M", "Не открывай. Мы предупреждали тебя.")],
          r_msg2, sfx=fx_msg2, intensity=0.9, pad_end=1.3, fade_out=0),
    Scene("end", [seg("R", "...если кто-нибудь меня слышит... это не ваши близкие. Не открывайте двери.",
                      [cap("*ПРОДОЛЖЕНИЕ\nСЛЕДУЕТ…*", 1060, 112, at=0.0),
                       cap("КТО СТУЧАЛ В ДВЕРЬ?", 1330, 58, at=0.5, weight="ExtraBold"),
                       cap("СЕРИЯ 2 — ПОДПИШИСЬ, ЧТОБЫ НЕ ПРОПУСТИТЬ", 1440, 34, at=0.75, weight="Bold")])],
          r_end, pad_start=0.9, pad_end=2.2, sfx=fx_end, intensity=0.5, fade_in=0, fade_out=0.6),
]


# ---------------------------------------------------------------- timeline
def build_timeline():
    vo, idx, t = [], 0, 0.0
    for sc in SCENES:
        sc.start, sc.seg_t, cur = t, [], sc.pad_start
        for s in sc.segs:
            a = tts(s["text"], s["voice"], idx)
            idx += 1
            dur = len(a) / SR
            sc.seg_t.append((cur, cur + dur))
            vo.append((t + cur, a, s["voice"]))
            cur += dur + 0.22
        sc.dur = max(sc.min_dur, sc.seg_t[-1][1] + sc.pad_end)
        sc.caps = []
        for i, s in enumerate(sc.segs):
            s0, s1 = sc.seg_t[i]
            end = sc.seg_t[i + 1][0] if i + 1 < len(sc.segs) else sc.dur
            for c in s["caps"]:
                sc.caps.append((s0 + (s1 - s0) * c["at"] - (0.1 if c["at"] == 0 else 0), end, c))
        t += sc.dur
    return vo, t


class Mixer:
    def __init__(self, total):
        self.a = np.zeros(int((total + 4) * SR))
        self.off = 0.0

    def add(self, x, t, gain=1.0):
        i = int((self.off + t) * SR)
        if i < 0 or i >= len(self.a):
            return
        n = min(len(x), len(self.a) - i)
        self.a[i:i + n] += x[:n] * gain


def build_audio(vo, total, path):
    A = Mixer(total)
    inten = np.zeros(len(A.a))
    for sc in SCENES:
        i0, i1 = int(sc.start * SR), int((sc.start + sc.dur) * SR)
        inten[i0:i1] = sc.intensity
        A.off = sc.start
        if sc.sfx:
            sc.sfx(A, sc)
    A.off = 0
    inten = filt(inten, hi=0.6)
    music = s_drone(len(A.a) / SR, np.clip(inten, 0, 1))[:len(A.a)]
    voice = np.zeros(len(A.a))
    for t0, a, v in vo:
        i = int(t0 * SR)
        voice[i:i + len(a)] += a / (np.max(np.abs(a)) + 1e-9) * (0.9 if v != "R" else 0.75)
    for t0, a, v in vo:
        if v == "R":
            A.add(s_static(len(a) / SR), t0, 0.05)
    duck = filt(np.abs(voice), hi=3)
    duck = 1 - 0.45 * np.clip(duck / (np.max(duck) * 0.3 + 1e-9), 0, 1)
    out = voice + (A.a + music * 0.22) * duck
    end = int(total * SR)
    out = out[:end]
    fade = int(0.8 * SR)
    out[-fade:] *= np.linspace(1, 0, fade)
    out /= np.max(np.abs(out)) + 1e-9
    raw = os.path.join(BUILD, "mix_raw.wav")
    pcm = (out * 0.95 * 32767).astype(np.int16)
    import wave
    with wave.open(raw, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                    "-ar", str(SR), "-ac", "2", path], check=True)


# ---------------------------------------------------------------- post
_vig = None
_grain = []


def post(frame, idx):
    global _vig
    if _vig is None:
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        v = np.clip(1.15 - 0.55 * r ** 1.8, 0.25, 1)
        _vig = Image.fromarray((np.stack([v] * 3, -1) * 255).astype(np.uint8))
        g = np.random.default_rng(1)
        for _ in range(8):
            n = g.integers(0, 22, (H, W), dtype=np.uint8)
            _grain.append(Image.fromarray(np.stack([n] * 3, -1)))
    frame = ImageChops.multiply(frame, _vig)
    return ImageChops.add(frame, _grain[idx % 8], 1.0, -11)


_tags = {}


def compose(t_global, idx):
    sc = next((s for s in SCENES if s.start <= t_global < s.start + s.dur), SCENES[-1])
    t = t_global - sc.start
    f = sc.render_fn(t, sc.dur, sc)
    if f.mode != "RGB":
        f = f.convert("RGB")
    for t0, t1, c in sc.caps:
        if t0 <= t < t1:
            im = caption_img(c["text"], c["size"], c["weight"])
            p = (t - t0) / 0.16
            paste_center(f, im, W / 2, c["y"], alpha=min(1, p), scale=1 + 0.18 * (1 - ease(p)))
    if sc.tag:
        n = len(sc.tag) if sc.name != "calm" else min(len(sc.tag), int(t / 0.07))
        if n > 0:
            key = sc.tag[:n]
            if key not in _tags:
                _tags[key] = tag_img(key, 44 if sc.name == "calm" else 34)
            im = _tags[key]
            if sc.name == "calm":
                f.paste(im, (int(W / 2 - tag_img(sc.tag, 44).size[0] / 2), 230), im)
            else:
                f.paste(im, (60, 80), im)
    if sc.fade_in and t < sc.fade_in:
        f = darken(f, t / sc.fade_in)
    if sc.fade_out and t > sc.dur - sc.fade_out:
        f = darken(f, (sc.dur - t) / sc.fade_out)
    return post(f, idx)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    vo, total = build_timeline()
    print(f"Длительность: {total:.1f} с, сцен: {len(SCENES)}")
    for s in SCENES:
        print(f"  {s.name:10s} {s.start:6.2f} +{s.dur:5.2f}")
    if "--preview" in sys.argv:
        thumbs = []
        for s in SCENES:
            for fr in (0.35, 0.85):
                thumbs.append(compose(s.start + s.dur * fr, 0).resize((270, 480)))
        cols = 8
        sheet = Image.new("RGB", (270 * cols, 480 * math.ceil(len(thumbs) / cols)))
        for i, th in enumerate(thumbs):
            sheet.paste(th, ((i % cols) * 270, (i // cols) * 480))
        sheet.save(os.path.join(BUILD, "preview.jpg"), quality=85)
        print("preview:", os.path.join(BUILD, "preview.jpg"))
        return
    audio = os.path.join(BUILD, "audio.wav")
    build_audio(vo, total, audio)
    out = os.path.join(OUTDIR, "seriya1_23-00.mp4")
    n = int(total * FPS)
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-i", audio, "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                           "-shortest", out], stdin=subprocess.PIPE)
    for i in range(n):
        ff.stdin.write(compose(i / FPS, i).tobytes())
        if i % 150 == 0:
            print(f"кадр {i}/{n}", flush=True)
    ff.stdin.close()
    ff.wait()
    hook = SCENES[0]
    compose(hook.start + hook.dur * 0.9, 0).save(os.path.join(OUTDIR, "cover.png"))
    print("готово:", out)


if __name__ == "__main__":
    main()
