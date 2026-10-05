"""창고 평면도: 랙, 순찰 경로, 위험/안전 물체와 위험 영역 (한 시나리오) (일반 파이썬, matplotlib).

    python scripts/plot_layout.py               # docs/layout.png
    python scripts/plot_layout.py --seed 3
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from factory_safety import warehouse as W  # noqa: E402
from factory_safety.scenario import sample_scenario  # noqa: E402

RACKS = [(-9.8, -3.8, 1.6, 16.2), (-0.8, -3.8, 1.6, 16.2), (8.2, -3.8, 1.6, 16.2)]   # x, y, 폭, 길이
MARK = {"spill": "o", "tool": "s", "stack": "^", "ext": "D"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(ROOT, "docs", "layout.png"))
    a = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.add_patch(Rectangle((-10.4, -12.0), 20.2, 30.0, fill=False, lw=2, color="#444"))
    for x, y, w, h in RACKS:
        ax.add_patch(Rectangle((x, y), w, h, color="#4a78c2", alpha=0.75))
    for x, y, _ in W.TABLES:
        ax.add_patch(Rectangle((x - 1.24, y - 0.39), 2.47, 0.78, color="#8b6a45"))
        ax.text(x, y - 0.95, "작업대", ha="center", fontsize=8)
    path = W.PatrolPath()
    ax.plot(path.pts[:, 0], path.pts[:, 1], "--", color="#1f9bd8", lw=1.8, label=f"작업자 순찰 경로 ({path.length:.0f} m)")
    sc = sample_scenario(a.seed)
    from matplotlib.patches import Circle
    for z in sc.zones:
        # 위험 영역: 라바콘 링 또는 DANGER 표지 (반경은 배치 자리 크기)
        ax.add_patch(Circle((z.x, z.y), z.radius, color="#e03131", alpha=0.12, zorder=2))
        for cx, cy in z.params.get("cones", []):
            ax.scatter([cx], [cy], marker="^", s=28, color="#fd7e14", edgecolor="k", lw=0.5, zorder=4)
        if z.params.get("sign"):
            ax.scatter([z.x], [z.y], marker="X", s=70, color="#ffd43b", edgecolor="k", zorder=4)
        ax.text(z.x, z.y - z.radius - 0.45, f"{z.id} 위험 영역", ha="center", fontsize=6.5, color="#c92a2a")
    for o in sc.objects:
        c = "#e03131" if o.hazard else "#2f9e44"
        ax.scatter([o.x], [o.y], marker=MARK[o.kind], s=90, color=c, edgecolor="k", zorder=5)
        ax.text(o.x + 0.35, o.y + 0.25, o.label.split(" (")[0], fontsize=6.5, color=c)
    for k, v in {"o": "바닥 유출", "s": "공구", "^": "적재", "D": "소화기"}.items():
        ax.scatter([], [], marker=k, color="#888", edgecolor="k", label=v)
    ax.scatter([], [], marker="o", color="#e03131", label="위험")
    ax.scatter([], [], marker="o", color="#2f9e44", label="안전")
    ax.scatter([], [], marker="^", color="#fd7e14", edgecolor="k", label="라바콘")
    ax.scatter([], [], marker="X", color="#ffd43b", edgecolor="k", label="DANGER 표지")
    ax.set_xlim(-11, 11)
    ax.set_ylim(-12.8, 18.8)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=4, fontsize=8)
    ax.set_title(f"창고 평면도 (위험 요소 배치 시드 {a.seed}: 위험 {len(sc.hazards)}개, 안전 {len(sc.objects) - len(sc.hazards)}개)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    plt.savefig(a.out, dpi=110)
    print(f"[완료] {a.out}")


if __name__ == "__main__":
    main()
