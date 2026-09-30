"""바닥, 압력계, 표지판 텍스처를 코드로 그려서 PNG로 저장 (Pillow 사용)."""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import layout as L
from .config import D, W

FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgunbd.ttf", "C:/Windows/Fonts/malgun.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]


def _korean_font(size):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return None


def floor_texture(path, px_per_m=32, seed=0):
    rng = np.random.default_rng(seed)
    w, h = int(W * px_per_m), int(D * px_per_m)
    base = np.full((h, w, 3), (128, 134, 139), dtype=np.float32)
    base += rng.normal(0, 6, (h, w, 1))
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img, "RGBA")

    def P(x, y):  # 월드 (x, y) -> 이미지 픽셀 (북쪽이 위)
        return ((x + W / 2) * px_per_m, (D / 2 - y) * px_per_m)

    for _ in range(40):
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(20, 90)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(35, 35, 35, 8))
    for x in range(0, int(W) + 1, 4):
        d.line([P(x - W / 2, D / 2), P(x - W / 2, -D / 2)], fill=(25, 25, 25, 50), width=2)
    for y in range(0, int(D) + 1, 4):
        d.line([P(-W / 2, y - D / 2), P(W / 2, y - D / 2)], fill=(25, 25, 25, 50), width=2)

    lw = int(0.12 * px_per_m)
    for key, (x0, y0, x1, y1) in L.AISLES.items():
        a, b = P(x0, y1), P(x1, y0)
        d.rectangle([a, b], fill=(58, 112, 86, 107))
    for key, (x0, y0, x1, y1) in L.AISLES.items():
        a, b = P(x0, y1), P(x1, y0)
        if key in ("A", "B"):
            d.rectangle([a[0], a[1], b[0], a[1] + lw], fill=(242, 194, 0, 255))
            d.rectangle([a[0], b[1] - lw, b[0], b[1]], fill=(242, 194, 0, 255))
        else:
            d.rectangle([a[0], a[1], a[0] + lw, b[1]], fill=(242, 194, 0, 255))
            d.rectangle([b[0] - lw, a[1], b[0], b[1]], fill=(242, 194, 0, 255))

    font = _korean_font(30)
    text = "보행자 통로" if font else "WALKWAY"
    font = font or ImageFont.load_default()
    for x, y in [(-13, 9), (3, 9), (13, -7), (-4, -7)]:
        cx, cy = P(x, y)
        d.text((cx, cy), text, fill=(255, 255, 255, 130), font=font, anchor="mm")

    # 로봇팔 주변 빗금 안전 구역
    a, b = P(6.9, 4.7), P(10.6, 1.75)
    hatch = Image.new("RGBA", (int(b[0] - a[0]), int(b[1] - a[1])), (29, 33, 38, 255))
    hd = ImageDraw.Draw(hatch)
    hw, hh = hatch.size
    for k in range(-hh, hw + hh, 26):
        hd.line([(k, hh), (k + hh, 0)], fill=(242, 194, 0, 255), width=9)
    img.paste(hatch, (int(a[0]), int(a[1])))

    # 출하 대기 구역 점선
    a, b = P(0.8, -10.4), P(16.2, -13.6)
    for x in np.arange(a[0], b[0], 30):
        d.line([(x, a[1]), (min(x + 18, b[0]), a[1])], fill=(255, 255, 255, 140), width=4)
        d.line([(x, b[1]), (min(x + 18, b[0]), b[1])], fill=(255, 255, 255, 140), width=4)
    for y in np.arange(a[1], b[1], 30):
        d.line([(a[0], y), (a[0], min(y + 18, b[1]))], fill=(255, 255, 255, 140), width=4)
        d.line([(b[0], y), (b[0], min(y + 18, b[1]))], fill=(255, 255, 255, 140), width=4)
    img.save(path)
    return path


def gauge_texture(path, ok, seed=0, size=256):
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (size, size), (20, 20, 20))
    d = ImageDraw.Draw(img)
    c, R = size / 2, size / 2 - 4
    d.ellipse([c - R, c - R, c + R, c + R], fill=(244, 244, 240), outline=(34, 34, 34), width=10)
    r2 = R * 0.72
    box = [c - r2, c - r2, c + r2, c + r2]
    # PIL 각도: 3시 방향 0도, 시계방향 +
    d.arc(box, 135, 200, fill=(224, 49, 49), width=28)
    d.arc(box, 200, 340, fill=(47, 158, 68), width=28)
    d.arc(box, 340, 405, fill=(224, 49, 49), width=28)
    ang = math.radians(rng.uniform(245, 295) if ok else rng.uniform(142, 175))
    d.line([(c, c), (c + math.cos(ang) * R * 0.8, c + math.sin(ang) * R * 0.8)], fill=(17, 17, 17), width=10)
    d.ellipse([c - 14, c - 14, c + 14, c + 14], fill=(17, 17, 17))
    img.save(path)
    return path


def sign_texture(path):
    img = Image.new("RGB", (512, 256), (212, 42, 42))
    d = ImageDraw.Draw(img)
    font = _korean_font(120)
    text = "소화기" if font else "FIRE EXT."
    font = font or ImageFont.load_default()
    d.text((256, 132), text, fill=(255, 255, 255), font=font, anchor="mm")
    img.save(path)
    return path


def make_all(out_dir, seed=0):
    """모든 텍스처를 만들고 {키: 절대경로} 반환."""
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    tex = {
        "floor": floor_texture(os.path.join(out_dir, "floor.png"), seed=seed),
        "sign": sign_texture(os.path.join(out_dir, "sign.png")),
    }
    for v in range(3):
        tex[f"gauge_ok_{v}"] = gauge_texture(os.path.join(out_dir, f"gauge_ok_{v}.png"), True, seed=10 + v)
        tex[f"gauge_low_{v}"] = gauge_texture(os.path.join(out_dir, f"gauge_low_{v}.png"), False, seed=20 + v)
    return {k: v.replace("\\", "/") for k, v in tex.items()}   # USD 경로는 / 로 통일
