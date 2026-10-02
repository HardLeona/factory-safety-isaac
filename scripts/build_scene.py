"""공장 장면을 만들어 Isaac Sim에 띄우고 USD 파일로 저장한다.

실행 (Isaac Sim 파이썬):
    <isaac>/python.sh scripts/build_scene.py
    <isaac>/python.sh scripts/build_scene.py --seed 3 --headless
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

parser = argparse.ArgumentParser(description="공장 장면 생성")
parser.add_argument("--seed", type=int, default=0, help="위험 요소 배치 시드")
parser.add_argument("--layout-seed", type=int, default=0, help="랙 적재물 배치 시드")
parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "usd", "factory.usd"))
parser.add_argument("--headless", action="store_true", help="창 없이 저장만 하고 종료")
parser.add_argument("--duration", type=float, default=0.0, help="창을 이 시간(초)만 보여주고 종료, 0이면 닫을 때까지")
args, _ = parser.parse_known_args()

from factory_safety.isaac_utils import make_app  # noqa: E402

app = make_app(headless=args.headless)

from factory_safety.isaac_utils import isaac_labeler, new_stage, set_viewport_top_view  # noqa: E402
from factory_safety.scenario import sample_scenario  # noqa: E402
from factory_safety.usd_scene import FactoryStage  # noqa: E402

stage = new_stage()
scene = FactoryStage(stage, os.path.join(ROOT, "outputs", "textures"), static_seed=args.layout_seed,
                     labeler=isaac_labeler).build()
scenario = sample_scenario(args.seed)
scene.set_scenario(scenario)
scene.set_robot(-19.0, 9.0, 0.0)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
scene.save(args.out)
print(f"[완료] 장면 저장: {args.out}")
print(f"       위험 요소 {len(scenario.targets)}개, 소화기 {sum(1 for h in scenario.hazards if h.type == 'ext')}개")
for h in scenario.hazards:
    if h.is_hazard:
        print(f"       - {h.label} ({h.zone})")

if not args.headless:
    import time
    set_viewport_top_view(stage)
    print("[안내] 창을 닫으면 종료돼요.")
    t0 = time.time()
    while app.is_running():
        app.update()
        if args.duration and time.time() - t0 > args.duration:
            break
app.close()
