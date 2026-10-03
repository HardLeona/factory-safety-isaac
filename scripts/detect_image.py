"""사진 한 장을 학습한 YOLO 로 판정 (일반 파이썬). 박스를 그린 그림과 결과 표를 저장한다.

    python scripts/detect_image.py 사진.jpg                       # outputs/detect/사진_yolo.jpg
    python scripts/detect_image.py 사진.jpg --conf 0.15 --device cpu --weights outputs/yolo/warehouse_v2/weights/best.pt

합성 데이터로만 학습한 모델이라 실제 사진에서는 확신도가 낮거나 비슷한 클래스로 잡힐 수 있다 (실사 검증용).
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--weights", default=os.path.join(ROOT, "outputs", "yolo", "warehouse_v3", "weights", "best.pt"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=os.path.join(ROOT, "outputs", "detect"))
    a = p.parse_args()
    from PIL import Image
    from ultralytics import YOLO
    from factory_safety import overlay
    from factory_safety.config import CLASS_KO, HAZARD
    model = YOLO(a.weights)
    img = Image.open(a.image).convert("RGB")
    r = model(img, imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)[0]
    dets = []
    for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
        name = model.names[int(c)]
        dets.append((name, float(s), [round(v, 1) for v in b]))
    dets.sort(key=lambda d: -d[1])
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.image))[0]
    size = max(14, img.width // 50)
    overlay.draw(img, [(n, s, b) for n, s, b in dets], size=size, width=max(2, img.width // 400)).save(
        os.path.join(a.out, f"{stem}_yolo.jpg"), quality=92)
    rows = [{"class": n, "ko": CLASS_KO.get(n, n), "hazard": HAZARD.get(n), "conf": round(s, 3), "box": b} for n, s, b in dets]
    json.dump({"image": a.image, "weights": a.weights, "size": [img.width, img.height], "detections": rows},
              open(os.path.join(a.out, f"{stem}_yolo.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[YOLO] 가중치 {a.weights}")
    print(f"[YOLO] {len(dets)}개 (확신도 {a.conf} 이상)")
    for row in rows:
        print(f"  {row['class']:15s} {row['ko']:22s} {row['conf'] * 100:5.1f}%  박스 {row['box']}")
    print(f"[저장] {os.path.join(a.out, stem + '_yolo.jpg')}")


if __name__ == "__main__":
    main()
