"""run_patrol.py --dump-dets 로 저장한 검출 기록을 Isaac 없이 다시 판단하고 채점한다 (일반 파이썬).

판단 규칙(factory_safety/yolo_judge.py)을 고친 뒤 몇 초 만에 효과를 확인할 때 쓴다.
카메라 경로는 기록 그대로라서, 판단이 바뀌어 로봇 고개가 달리 돌아가는 효과는 반영되지 않는다.

    python scripts/replay_dets.py outputs/eval/dets_seed0.json
    python scripts/replay_dets.py outputs/eval/dets_seed0.json --verbose
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from factory_safety import layout as L  # noqa: E402
from factory_safety.geometry import CameraPose  # noqa: E402
from factory_safety.scenario import sample_scenario  # noqa: E402
from factory_safety.yolo_judge import YoloJudge, judge_line, judge_summary  # noqa: E402


def replay(path, verbose=False):
    rec = json.load(open(path, encoding="utf-8"))
    sc = sample_scenario(rec["seed"])
    judge = YoloJudge(*L.map_boxes_3d(L.build_static(rec.get("layout_seed", 0))), zoom=rec["zoom"])
    t = 0.0
    for fr in rec["frames"]:
        t = fr["t"]
        x, y, z, yaw, pitch, roll, vfov = fr["cam"]
        cam = CameraPose(pos=np.array([x, y, z]), yaw=yaw, pitch=pitch, roll=roll, vfov=vfov)
        events = judge.update(t, cam, [(n, c, b) for n, c, b in fr["dets"]])
        if fr.get("zoom") and fr["zoom"][1]:
            events += judge.add_zoom_reading(t, cam, fr["zoom"][0], fr["zoom"][1])
        if verbose:
            for tr in events:
                print(judge_line(tr, t, color=False))
    return sc, judge, t


def main():
    p = argparse.ArgumentParser(description="검출 기록 다시 판단")
    p.add_argument("paths", nargs="+")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    for path in a.paths:
        sc, judge, t = replay(path, a.verbose)
        print(f"== {path}")
        print(judge_summary(judge, t, sc.hazards))


if __name__ == "__main__":
    main()
