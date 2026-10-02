"""YOLO 가중치 여러 개를 같은 검증 데이터로 비교 (일반 파이썬, GPU).

    python scripts/compare_yolo.py
    python scripts/compare_yolo.py --weights a.pt b.pt --data outputs/dataset/data.yaml
"""
import argparse
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(description="YOLO 가중치 비교")
    p.add_argument("--weights", nargs="+", default=[os.path.join(ROOT, "outputs", "yolo", "warehouse", "weights", "best.pt")])
    p.add_argument("--data", default=os.path.join(ROOT, "outputs", "dataset", "data.yaml"))
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--device", default="0")
    a = p.parse_args()

    from ultralytics import YOLO
    rows = []
    for w in a.weights:
        r = YOLO(w).val(data=a.data, imgsz=a.imgsz, batch=16, device=a.device, verbose=False, plots=False,
                        project=os.path.join(ROOT, "outputs", "yolo", "_compare"), name="val", exist_ok=True)
        names = r.names
        rows.append((w, r.box.map50, r.box.map, [r.box.class_result(i)[2] for i in range(len(names))]))
    tags = [os.path.basename(os.path.dirname(os.path.dirname(w))) for w, *_ in rows]
    lines = [f"검증 데이터: {a.data}", "", "| 클래스 (mAP50) | " + " | ".join(tags) + " |", "|---|" + ":-:|" * len(rows)]
    for i in range(len(names)):
        lines.append(f"| {names[i]} | " + " | ".join(f"{r[3][i]:.3f}" for r in rows) + " |")
    lines.append("| **전체 mAP50** | " + " | ".join(f"**{r[1]:.3f}**" for r in rows) + " |")
    lines.append("| 전체 mAP50-95 | " + " | ".join(f"{r[2]:.3f}" for r in rows) + " |")
    out = os.path.join(ROOT, "outputs", "eval", "yolo_compare.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[완료] {out}")


if __name__ == "__main__":
    main()
