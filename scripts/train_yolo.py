"""합성 데이터로 YOLO 학습 (일반 파이썬, GPU 권장).

    pip install ultralytics
    python scripts/train_yolo.py --data outputs/dataset/data.yaml
    python scripts/train_yolo.py --data outputs/dataset/data.yaml --model yolo11s.pt --epochs 150
"""
import argparse
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(description="YOLO 학습")
    p.add_argument("--data", default=os.path.join(ROOT, "outputs", "dataset", "data.yaml"))
    p.add_argument("--model", default="yolo11n.pt", help="시작 가중치 (n < s < m 순으로 크고 정확)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="예: 0 (GPU), cpu")
    p.add_argument("--name", default="factory_hazard")
    a = p.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.model)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                project=os.path.join(ROOT, "outputs", "yolo"), name=a.name, exist_ok=True,
                # 합성 데이터라 색, 밝기 변형을 조금 더 세게
                hsv_h=0.02, hsv_s=0.6, hsv_v=0.5, degrees=3.0, mosaic=1.0)
    best = os.path.join(ROOT, "outputs", "yolo", a.name, "weights", "best.pt")
    print(f"\n[완료] 가중치: {best}")
    print(f"       Isaac Sim에서 확인: python.sh scripts/run_patrol.py --detector yolo --weights {best}")


if __name__ == "__main__":
    main()
