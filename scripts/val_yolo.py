"""학습한 YOLO 를 검증 데이터로 채점해서 클래스별 mAP50 을 metrics.json 으로 (일반 파이썬, GPU). 보고서·발표자료가 이 숫자를 씀.

    python scripts/val_yolo.py --weights outputs/yolo/warehouse_v3/weights/best.pt --data outputs/data_v3.yaml
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(description="YOLO 검증 점수 저장")
    p.add_argument("--weights", default=os.path.join(ROOT, "outputs", "yolo", "warehouse_v3", "weights", "best.pt"))
    p.add_argument("--data", default=os.path.join(ROOT, "outputs", "data_v3.yaml"))
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--device", default="0")
    a = p.parse_args()
    from ultralytics import YOLO
    r = YOLO(a.weights).val(data=a.data, imgsz=a.imgsz, batch=16, device=a.device, verbose=False, plots=True,
                            project=os.path.join(ROOT, "outputs", "yolo", "_val"), name=os.path.basename(os.path.dirname(os.path.dirname(a.weights))),
                            exist_ok=True)
    out = {"map50": round(float(r.box.map50), 3), "map5095": round(float(r.box.map), 3),
           "precision": round(float(r.box.mp), 3), "recall": round(float(r.box.mr), 3),
           "per_class": {r.names[int(c)]: round(float(r.box.class_result(k)[2]), 3) for k, c in enumerate(r.box.ap_class_index)}}
    path = os.path.join(os.path.dirname(os.path.dirname(a.weights)), "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"[저장] {path}")


if __name__ == "__main__":
    main()
