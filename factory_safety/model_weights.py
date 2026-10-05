"""학습된 YOLO 가중치 자동 다운로드 (Hugging Face IBDPLab/factory-safety-isaac-yolo).

scripts/get_model.py (수동 CLI) 와 scripts/run_patrol.py (가중치가 없을 때 자동) 가 같이 쓴다.
huggingface_hub 가 없는 환경(Isaac Sim 전용 파이썬 등)에서는 조용히 건너뛴다 — 호출부가
그 경우 기존 안내 메시지로 처리한다.
"""
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ID = "IBDPLab/factory-safety-isaac-yolo"

# name -> (Hugging Face 안의 파일 경로, 이 저장소에서 기대하는 로컬 경로)
MODELS = {
    "warehouse_v3": ("warehouse_v3/best.pt", os.path.join(ROOT, "outputs", "yolo", "warehouse_v3", "weights", "best.pt")),
    "pinch_v1": ("pinch_v1/best.pt", os.path.join(ROOT, "outputs", "yolo", "pinch_v1", "weights", "best.pt")),
}


def download(name):
    """Hugging Face 에서 받아 MODELS[name] 의 로컬 경로에 둔다 (이미 있으면 그대로)."""
    hf_file, local = MODELS[name]
    if os.path.exists(local):
        return local
    from huggingface_hub import hf_hub_download
    os.makedirs(os.path.dirname(local), exist_ok=True)
    got = hf_hub_download(repo_id=REPO_ID, filename=hf_file)
    shutil.copyfile(got, local)
    return local


def ensure_weights(path):
    """path 가 기본 가중치 자리(warehouse_v3/pinch_v1)와 같은데 없으면 Hugging Face 에서 받아 온다.
    huggingface_hub 가 없거나 받기에 실패하면 (인터넷 없음 등) 조용히 원래 path 를 돌려주고,
    호출부가 여전히 없는 걸 보고 알아서 안내하게 한다. 모르는 경로(사용자가 --weights 로 직접 지정)는 손대지 않는다."""
    if os.path.exists(path):
        return path
    for name, (_, local) in MODELS.items():
        if os.path.abspath(path) == os.path.abspath(local):
            try:
                print(f"[다운로드] {name} YOLO 가중치가 없어 Hugging Face ({REPO_ID}) 에서 받습니다...")
                got = download(name)
                print(f"[다운로드] 완료 -> {got}")
                return got
            except ImportError:
                print("[다운로드] huggingface_hub 가 이 환경에 없어 자동으로 못 받았어요 "
                      "(일반 파이썬에서 pip install -r requirements.txt 또는 python scripts/get_model.py 로 받아 두세요).")
            except Exception as e:
                print(f"[다운로드] 실패: {e}")
            return path
    return path
