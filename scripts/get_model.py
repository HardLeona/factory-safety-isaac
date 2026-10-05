"""학습된 YOLO 가중치 내려받기 (일반 파이썬). Hugging Face (IBDPLab/factory-safety-isaac-yolo) 에서
outputs/yolo/<이름>/weights/best.pt 로 받는다. get_assets.py 와 같은 흐름(한 번 받으면 그대로 재사용).

    python scripts/get_model.py                 # warehouse_v3 + pinch_v1 둘 다
    python scripts/get_model.py --only pinch_v1  # 끼임점 모델만
"""
import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPO_ID = "IBDPLab/factory-safety-isaac-yolo"
MODELS = {
    "warehouse_v3": "warehouse_v3/best.pt",   # YOLO26s, 19클래스 (위험/안전 상태, 공구 8종, 운반 카트 등)
    "pinch_v1": "pinch_v1/best.pt",           # 끼임점(pinch_point) 1클래스 경량 모델
}


def fetch(name, filename):
    out = os.path.join(ROOT, "outputs", "yolo", name, "weights", "best.pt")
    if os.path.exists(out):
        print(f"  {name}: 이미 있음 ({out})")
        return
    from huggingface_hub import hf_hub_download
    os.makedirs(os.path.dirname(out), exist_ok=True)
    path = hf_hub_download(repo_id=REPO_ID, filename=filename)
    shutil.copyfile(path, out)
    print(f"  {name}: 받음 -> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", choices=list(MODELS), default=None, help="하나만 받기 (기본: 둘 다)")
    args = p.parse_args()
    names = [args.only] if args.only else list(MODELS)
    for name in names:
        fetch(name, MODELS[name])
    print(f"[완료] https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    sys.exit(main())
