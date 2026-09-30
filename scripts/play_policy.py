"""학습한 강화학습 정책을 Isaac Sim 화면에서 재생한다.

실행 (Isaac Sim 파이썬):
    <isaac>/python.sh scripts/play_policy.py                        # outputs/rl/ppo_patrol.npz (numpy만 필요)
    <isaac>/python.sh scripts/play_policy.py --baseline             # 비교용: 고정 경로 순찰
    <isaac>/python.sh scripts/play_policy.py --model xxx.zip        # SB3 .zip 직접 (Isaac 파이썬에 SB3 필요)
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

parser = argparse.ArgumentParser(description="강화학습 정책 재생")
parser.add_argument("--model", default=os.path.join(ROOT, "outputs", "rl", "ppo_patrol.npz"))
parser.add_argument("--baseline", action="store_true", help="학습 정책 대신 고정 경로 순찰로 재생")
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--view", choices=["top", "pov"], default="top")
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--headless", action="store_true")
args, _ = parser.parse_known_args()

from factory_safety.isaac_utils import make_app  # noqa: E402

app = make_app(headless=args.headless)

import numpy as np  # noqa: E402

from factory_safety.geometry import CameraPose  # noqa: E402
from factory_safety.isaac_utils import DebugBoxes, isaac_labeler, new_stage, set_viewport_camera, set_viewport_top_view  # noqa: E402
from factory_safety.report import box_color, clock, event_line, summary  # noqa: E402
from factory_safety.rl_env import FactoryPatrolEnv, PathFollower  # noqa: E402
from factory_safety.usd_scene import FactoryStage  # noqa: E402

stage = new_stage()
scene = FactoryStage(stage, os.path.join(ROOT, "outputs", "textures"), labeler=isaac_labeler).build()
env = FactoryPatrolEnv()
if args.baseline:
    policy = PathFollower(env, track=True)
    act = lambda obs: policy(obs)  # noqa: E731
elif args.model.endswith(".zip"):
    from stable_baselines3 import PPO
    model = PPO.load(args.model, device="cpu")
    act = lambda obs: model.predict(obs, deterministic=True)[0]  # noqa: E731
else:
    from factory_safety.policy_numpy import NumpyPolicy
    if not os.path.exists(args.model):
        sys.exit(f"[오류] 정책 파일이 없어요: {args.model}\n       먼저 python scripts/train_rl.py 로 학습하거나 --baseline 으로 실행하세요.")
    act = NumpyPolicy(args.model)

if args.view == "pov":
    set_viewport_camera(scene.cam_path)
else:
    set_viewport_top_view(stage)
boxes = DebugBoxes()


def lerp_pose(a, b, k):
    dyaw = (b.yaw - a.yaw + np.pi) % (2 * np.pi) - np.pi
    return CameraPose(pos=a.pos + (b.pos - a.pos) * k, yaw=a.yaw + dyaw * k, pitch=b.pitch, roll=b.roll, vfov=b.vfov)


try:
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        if args.baseline:
            policy.reset()
        scene.set_scenario(env.scenario)
        print(f"\n[에피소드 {ep + 1}] 위험 요소 {info['total']}개, 소화기 {info['ext_total']}개")
        prev = cur = env.camera()
        acc, last, done = 0.0, time.time(), False
        step_dt = env.dt / max(args.speed, 1e-3)
        while app.is_running() and not done:
            now = time.time()
            acc += min(0.1, now - last)
            last = now
            while acc >= step_dt and not done:
                acc -= step_dt
                obs, r, term, trunc, info = env.step(act(obs))
                done = term or trunc
                prev, cur = cur, env.camera()
                for i in info["new"]:
                    print(event_line(env.detector.hazards[i], env.steps * env.dt, env.detector.conf[i]))
            pose = lerp_pose(prev, cur, min(1.0, acc / step_dt))
            scene.set_camera(pose)
            head = (pose.yaw - env.heading + np.pi) % (2 * np.pi) - np.pi
            scene.set_robot(float(pose.pos[0]), float(pose.pos[1]), env.heading, head, visible=args.view == "top")
            scene.animate(env.steps * env.dt)
            det = env.detector
            boxes.show([(h.aabb_min, h.aabb_max, box_color(h, det.detected[i], det.conf[i]))
                        for i, h in enumerate(det.hazards) if det.detected[i] or (det.in_view[i] and det.conf[i] > det.SHOW_CONF)])
            app.update()
        print(summary(env.detector, env.steps * env.dt))
        print(f"  충돌 {env.collisions}회, 본 구역 {info['coverage'] * 100:.0f}%, 소요 {clock(env.steps * env.dt)}")
        if not app.is_running():
            break
except KeyboardInterrupt:
    pass
app.close()
