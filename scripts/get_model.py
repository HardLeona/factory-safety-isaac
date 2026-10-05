"""학습된 YOLO 가중치 내려받기 (일반 파이썬). Hugging Face (IBDPLab/factory-safety-isaac-yolo) 에서
outputs/yolo/<이름>/weights/best.pt 로 받는다. get_assets.py 와 같은 흐름(한 번 받으면 그대로 재사용).
run_patrol.py 는 가중치가 기본 자리에 없으면 이 로직(factory_safety/model_weights.py)으로 알아서 받는다.

    python scripts/get_model.py                 # warehouse_v3 + pinch_v1 둘 다
    python scripts/get_model.py --only pinch_v1  # 끼임점 모델만
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from factory_safety.model_weights import MODELS, REPO_ID, download  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", choices=list(MODELS), default=None, help="하나만 받기 (기본: 둘 다)")
    args = p.parse_args()
    for name in ([args.only] if args.only else list(MODELS)):
        _, local = MODELS[name]
        if os.path.exists(local):
            print(f"  {name}: 이미 있음 ({local})")
            continue
        path = download(name)
        print(f"  {name}: 받음 -> {path}")
    print(f"[완료] https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    sys.exit(main())
