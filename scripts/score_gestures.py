"""손동작 시험 다시 채점 (Isaac 없이, .venv-assistant): test_gestures.py --raw 로 저장한 화면을 지금 손 인식으로 다시 본다.

    .venv-assistant\\Scripts\\python scripts/score_gestures.py --raw outputs/eval/gesture_raw

손 인식 기준을 바꾼 뒤 같은 화면으로 비교할 때 쓴다. 결과 형식은 test_gestures.py 와 같음 (outputs/eval/gesture_test.json).
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from assistant_worker import hand  # noqa: E402
from factory_safety.hand_count import GestureFilter  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=os.path.join(ROOT, "outputs", "eval", "gesture_raw"))
    p.add_argument("--save", default=os.path.join(ROOT, "outputs", "eval", "gesture_frames"))
    p.add_argument("--out", default=os.path.join(ROOT, "outputs", "eval", "gesture_test.json"))
    p.add_argument("--dt", type=float, default=0.1, help="화면 사이 시간 (test_gestures 의 --every 3 = 0.1초)")
    a = p.parse_args()
    seqs = defaultdict(list)
    for f in glob.glob(os.path.join(a.raw, "*.jpg")):
        m = re.match(r"spot(\d+)_g(\d)_(\d+)_a(\d+)\.jpg", os.path.basename(f))
        if m:
            seqs[(int(m[1]), int(m[2]))].append((int(m[3]), int(m[4]) / 100, f))
    os.makedirs(a.save, exist_ok=True)
    trials = []
    for (spot, c), seq in sorted(seqs.items()):
        flt, fired, full, moving = GestureFilter(), [], [], []
        for k, alpha, f in sorted(seq):
            r = hand(f)
            cnt = r["count"]
            (full if alpha >= 0.999 else moving).append(cnt)
            x = flt.update(k * a.dt, cnt, r.get("pts"))
            if x:
                fired.append(x)
            out = os.path.join(a.save, f"spot{spot}_g{c}.jpg")
            # 명령이 확정된 화면 (확정이 없으면 다 올린 세 번째 화면)
            if x or (alpha >= 0.999 and len(full) == 3 and not fired):
                from PIL import Image
                from factory_safety import overlay
                overlay.draw_hand(Image.open(f).convert("RGB"), r.get("pts"), cnt).save(out, quality=85)
        trials.append({"spot": spot, "shown": c, "fired": fired, "ok": fired == [c], "wrong": any(x != c for x in fired),
                       "full": full, "moving": moving})
        print(f"[G] 지점 {spot} 손가락 {c} → 명령 {fired or '없음'} | 다 올림 {full}", flush=True)
    res = {"spots": len({t["spot"] for t in trials}), "trials": len(trials), "ok": sum(t["ok"] for t in trials),
           "wrong": sum(t["wrong"] for t in trials), "missed": sum(not t["fired"] for t in trials),
           "frames": sum(len(t["full"]) for t in trials), "frames_ok": sum(k == t["shown"] for t in trials for k in t["full"]),
           "mid_frames": sum(len(t["moving"]) for t in trials),
           "mid_false": sum(1 for t in trials for k in t["moving"] if k not in (0, t["shown"])), "rows": trials}
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[완료] 명령 {res['ok']}/{res['trials']} 맞음, 다른 명령 {res['wrong']}, 안 나옴 {res['missed']} | "
          f"다 올린 화면 {res['frames_ok']}/{res['frames']} | 움직이는 중 다른 수 {res['mid_false']}/{res['mid_frames']} → {a.out}")


if __name__ == "__main__":
    main()
