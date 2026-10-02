"""창고 장면을 만들어 Isaac Sim 에 띄우고 USD 파일과 정답표로 저장한다.

실행 (Isaac Sim 파이썬):
    <isaac>/python.sh scripts/build_scene.py
    <isaac>/python.sh scripts/build_scene.py --seed 3 --headless
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

parser = argparse.ArgumentParser(description="창고 장면 생성")
parser.add_argument("--seed", type=int, default=0, help="위험 요소 배치 시드")
parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "usd", "warehouse.usd"))
parser.add_argument("--headless", action="store_true", help="창 없이 저장만 하고 종료")
parser.add_argument("--duration", type=float, default=0.0, help="창을 이 시간(초)만 보여주고 종료, 0이면 닫을 때까지")
args, _ = parser.parse_known_args()

from factory_safety.isaac_utils import make_app  # noqa: E402

app = make_app(headless=args.headless)

from factory_safety.isaac_utils import isaac_labeler, new_stage, set_viewport_top_view, timeline_setter  # noqa: E402
from factory_safety.scenario import sample_scenario  # noqa: E402
from factory_safety.scene import WarehouseScene  # noqa: E402

stage = new_stage()
scene = WarehouseScene(stage, labeler=isaac_labeler, set_time=timeline_setter()).build()
scenario = sample_scenario(args.seed)
scene.set_scenario(scenario)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
scene.save(args.out)
key = os.path.splitext(args.out)[0] + f"_answer_key_seed{args.seed}.json"
scenario.save_answer_key(key)
print(f"[완료] 장면 저장: {args.out}")
print(f"       정답표: {key}")
for o in scenario.objects:
    print(f"       {o.id} {'위험' if o.hazard else '안전'}  {o.label}  ({o.zone})")

if not args.headless:
    set_viewport_top_view(stage)
    print("[안내] 창을 닫으면 종료돼요.")
    t0 = time.time()
    while app.is_running():
        app.update()
        if args.duration and time.time() - t0 > args.duration:
            break
app.close()
