"""공장 평면도 그림 만들기 (일반 파이썬). 배치를 고친 뒤 눈으로 확인할 때 쓴다.

    python scripts/plot_layout.py                    # docs/layout.png
    python scripts/plot_layout.py --seed 5 --out outputs/plan_5.png
"""
import argparse
import os
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from factory_safety import layout as L  # noqa: E402
from factory_safety.config import D, W  # noqa: E402
from factory_safety.patrol import ClosedPath  # noqa: E402
from factory_safety.scenario import sample_scenario  # noqa: E402
from factory_safety.textures import _korean_font  # noqa: E402

COL = {"puddle": (77, 171, 247), "tool": (255, 167, 38), "stack": (255, 77, 79),
       "ext_ok": (61, 220, 132), "ext_low": (255, 212, 59)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--out", default=os.path.join(ROOT, "docs", "layout.png"))
    p.add_argument("--scale", type=int, default=26, help="1 m 당 픽셀")
    a = p.parse_args()

    s, pad, top, bottom = a.scale, 30, 70, 60
    img = Image.new("RGB", (int(W * s) + 2 * pad, int(D * s) + top + bottom), (27, 32, 37))
    d = ImageDraw.Draw(img, "RGBA")
    X = lambda x: pad + (x + W / 2) * s  # noqa: E731
    Y = lambda y: top + (D / 2 - y) * s  # noqa: E731
    font = _korean_font(22) or ImageFont.load_default()
    small = _korean_font(15) or ImageFont.load_default()

    d.rectangle([X(-W / 2), Y(D / 2), X(W / 2), Y(-D / 2)], fill=(43, 50, 57), outline=(91, 100, 109), width=3)
    for x0, y0, x1, y1 in L.AISLES.values():
        d.rectangle([X(x0), Y(y1), X(x1), Y(y0)], fill=(47, 74, 63))
    for m in L.build_static(0):
        if m.name in ("Shell", "Ceiling", "Signs"):
            continue
        for part in m.parts:
            c = m.transform(part.local_corners())
            mn, mx = c.min(axis=0), c.max(axis=0)
            if mn[2] > 2.5 or (mx[0] - mn[0]) * (mx[1] - mn[1]) < 0.02:
                continue
            col = tuple(int(255 * min(1, v ** (1 / 2.2))) for v in part.color)
            d.rectangle([X(mn[0]), Y(mx[1]), X(mx[0]), Y(mn[1])], fill=col + (150,))

    path = ClosedPath()
    pts = [(X(q[0]), Y(q[1])) for q in path.pts[::6]]
    for i in range(len(pts)):
        if i % 2 == 0:
            d.line([pts[i], pts[(i + 1) % len(pts)]], fill=(77, 171, 247, 200), width=3)

    sc = sample_scenario(a.seed)
    for h in sc.hazards:
        key = h.type if h.type != "ext" else ("ext_ok" if h.ok else "ext_low")
        x, y = X(h.center[0]), Y(h.center[1])
        r = 7 if h.type == "ext" else 10
        d.ellipse([x - r, y - r, x + r, y + r], fill=COL[key], outline=(18, 22, 26), width=2)
    for m in sc.extras:
        mn, mx = m.aabb()
        d.rectangle([X(mn[0]), Y(mx[1]), X(mx[0]), Y(mn[1])], outline=(200, 170, 120), width=2)

    d.text((pad, 22), f"공장 평면도  (위험 요소 배치 시드 {a.seed})", fill=(236, 240, 243), font=font)
    lx, ly = pad, top + D * s + 20
    items = [("웅덩이", "puddle"), ("공구", "tool"), ("불안정 적재물", "stack"), ("소화기 정상", "ext_ok"), ("소화기 압력 부족", "ext_low")]
    for label, key in items:
        d.ellipse([lx, ly, lx + 14, ly + 14], fill=COL[key])
        d.text((lx + 20, ly - 2), label, fill=(210, 216, 222), font=small)
        lx += 40 + int(d.textlength(label, font=small))
    d.line([(lx, ly + 7), (lx + 30, ly + 7)], fill=(77, 171, 247), width=3)
    d.text((lx + 38, ly - 2), "고정 순찰 경로", fill=(210, 216, 222), font=small)
    for name, (x, y) in {"북측 랙": (0, 14.5), "A 통로": (-8, 9), "가공 구역": (-2, 0.2), "B 통로": (-8, -7),
                         "작업대, 출하 대기": (-2, -13.8)}.items():
        d.text((X(x), Y(y)), name, fill=(236, 240, 243, 190), font=small, anchor="mm")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    img.save(a.out)
    print(f"[완료] {a.out}")


if __name__ == "__main__":
    main()
