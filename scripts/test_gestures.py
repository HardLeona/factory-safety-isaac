"""손동작 인식 시험 (Isaac Sim): 순찰 경로 여러 지점에서 조명을 바꿔 가며 손가락 1~5 를 보이고, 순찰 때와 똑같이
손을 올리고 → 들고 있고 → 내리는 동안의 바디캠 화면을 명령 확정 필터에 넣어 명령이 맞게 나오는지 본다.

    <isaac>/python.sh scripts/test_gestures.py                  # 지점 8곳 x 손가락 1~5 = 40번
    <isaac>/python.sh scripts/test_gestures.py --spots 12 --raw outputs/eval/gesture_raw    # 원본 화면 저장 (기준 조정용)

결과: outputs/eval/gesture_test.json
    trials / ok      명령이 보인 수대로 한 번 나온 경우
    wrong            다른 명령이 나온 경우 (가장 나쁨)
    frames_ok        손을 다 올린 화면 중 수를 맞힌 비율
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

parser = argparse.ArgumentParser(description="손동작 인식 시험")
parser.add_argument("--spots", type=int, default=8, help="경로 위 시험 지점 수")
parser.add_argument("--seed", type=int, default=7, help="위험 요소 배치와 조명")
parser.add_argument("--every", type=int, default=3, help="몇 프레임마다 손 인식 (순찰의 --yolo-every)")
parser.add_argument("--save", default=os.path.join(ROOT, "outputs", "eval", "gesture_frames"), help="손을 다 올린 화면 (손 관절 표시) 저장 폴더")
parser.add_argument("--raw", default=None, help="모든 시험 화면 원본을 저장할 폴더")
parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "eval", "gesture_test.json"))
args, _ = parser.parse_known_args()

from factory_safety.isaac_utils import make_app  # noqa: E402

app = make_app(headless=True)

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from PIL import Image  # noqa: E402

from factory_safety import overlay  # noqa: E402
from factory_safety.assistant_client import AssistantClient  # noqa: E402
from factory_safety.config import IMG_H, IMG_W  # noqa: E402
from factory_safety.hand_count import GestureFilter  # noqa: E402
from factory_safety.isaac_utils import attach, get_annotator, isaac_labeler, new_stage, rgb_array, timeline_setter  # noqa: E402
from factory_safety.scenario import sample_scenario  # noqa: E402
from factory_safety.scene import WarehouseScene  # noqa: E402
from factory_safety.walker import PathWalker  # noqa: E402

stage = new_stage()
scene = WarehouseScene(stage, labeler=isaac_labeler, set_time=timeline_setter()).build()
scene.set_scenario(sample_scenario(args.seed))
rp = rep.create.render_product(scene.cam_path, (IMG_W, IMG_H))
a_rgb = get_annotator("rgb")
attach(a_rgb, rp)
client = AssistantClient()
if not client.ok:
    sys.exit(f"[오류] 손 인식 도우미를 못 띄움: {client.error}")
rng = np.random.default_rng(args.seed)
walker = PathWalker(seed=args.seed)
dt = 1 / 30
for d in (args.save, args.raw):
    if d:
        os.makedirs(d, exist_ok=True)


def shoot():
    x, y, yaw, at = walker.base_pose
    scene.set_worker(x, y, yaw, at)
    scene.set_camera(walker.camera())
    rep.orchestrator.step(delta_time=0.0, rt_subframes=3)
    return rgb_array(a_rgb.get_data())


trials = []
for i in range(args.spots):
    walker.reset(s0=walker.path.length * (i + 0.5) / args.spots)
    walker.t = float(rng.uniform(0, 10))
    scene.randomize_lighting(rng)
    for c in range(1, 6):
        flt = GestureFilter()
        walker.start_gesture(c, hold=1.2)
        seq, fired, n = [], [], 0
        while walker.gesture:
            if n % args.every == 0:
                img = shoot()
                if img is not None:
                    alpha = walker.gesture_alpha
                    cnt, pts = client.hand(img)
                    seq.append((round(alpha, 2), cnt))
                    f = flt.update(walker.t, cnt, pts)
                    if f:
                        fired.append(f)
                    tag = f"spot{i}_g{c}"
                    if args.raw:
                        Image.fromarray(img).save(os.path.join(args.raw, f"{tag}_{len(seq):02d}_a{int(alpha * 100):03d}.jpg"), quality=92)
                    # 명령이 확정된 화면 (확정이 없으면 다 올린 첫 화면)
                    if args.save and (f or (alpha >= 0.999 and not fired and not os.path.exists(os.path.join(args.save, f"{tag}.jpg")))):
                        overlay.draw_hand(Image.fromarray(img), pts, cnt).save(os.path.join(args.save, f"{tag}.jpg"), quality=85)
            walker.step(dt)
            n += 1
        full = [k for a, k in seq if a >= 0.999]
        moving = [k for a, k in seq if a < 0.999]
        trials.append({"spot": i, "shown": c, "fired": fired, "ok": fired == [c], "wrong": any(f != c for f in fired),
                       "full": full, "moving": moving})
        print(f"[G] 지점 {i} ({walker.s:.0f} m) 손가락 {c} → 명령 {fired or '없음'} | 다 올림 {full} | 움직이는 중 {moving}", flush=True)
client.close()
res = {"spots": args.spots, "trials": len(trials), "ok": sum(t["ok"] for t in trials), "wrong": sum(t["wrong"] for t in trials),
       "missed": sum(not t["fired"] for t in trials),
       "frames": sum(len(t["full"]) for t in trials), "frames_ok": sum(k == t["shown"] for t in trials for k in t["full"]),
       "mid_frames": sum(len(t["moving"]) for t in trials), "mid_false": sum(1 for t in trials for k in t["moving"] if k not in (0, t["shown"])),
       "rows": trials}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"[완료] 명령 {res['ok']}/{res['trials']} 맞음, 다른 명령 {res['wrong']}, 안 나옴 {res['missed']} | 다 올린 화면 {res['frames_ok']}/{res['frames']} | "
      f"움직이는 중 다른 수 {res['mid_false']}/{res['mid_frames']} → {args.out}")
app.close()
