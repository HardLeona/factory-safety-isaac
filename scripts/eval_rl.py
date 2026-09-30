"""고정 경로 순찰과 강화학습 정책 비교 평가 (일반 파이썬).

    python scripts/eval_rl.py                            # 기준 정책만
    python scripts/eval_rl.py --model outputs/rl/ppo_patrol.zip --episodes 30
    python scripts/eval_rl.py --model outputs/rl/ppo_patrol.npz       # numpy 정책 (결과 같아야 정상)
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from factory_safety.rl_env import FactoryPatrolEnv, PathFollower  # noqa: E402


def run(env, act, reset_fn, episodes, seed):
    rows = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        reset_fn()
        ret, done = 0.0, False
        while not done:
            obs, r, term, trunc, info = env.step(act(obs))
            ret += r
            done = term or trunc
        rows.append(dict(success=term, time=env.steps * env.dt, found=info["found"] / max(1, info["total"]),
                         ext=info["ext_checked"] / max(1, info["ext_total"]), coll=env.collisions, ret=ret))
    return rows


def report(name, rows):
    s = np.mean([r["success"] for r in rows]) * 100
    t = np.mean([r["time"] for r in rows if r["success"]]) if any(r["success"] for r in rows) else float("nan")
    print(f"{name:<22} 성공 {s:5.1f}%   완료 시간 {t:6.1f}s   위험 발견 {np.mean([r['found'] for r in rows]) * 100:5.1f}%   "
          f"소화기 {np.mean([r['ext'] for r in rows]) * 100:5.1f}%   충돌 {np.mean([r['coll'] for r in rows]):5.1f}   보상 {np.mean([r['ret'] for r in rows]):6.2f}")


def main():
    p = argparse.ArgumentParser(description="순찰 정책 비교")
    p.add_argument("--model", default=None)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=1000)
    a = p.parse_args()

    env = FactoryPatrolEnv()
    print(f"에피소드 {a.episodes}개, 같은 시나리오로 비교 (최대 {env.max_steps * env.dt:.0f}초)\n")
    for track in (False, True):
        pol = PathFollower(env, track=track)
        report("고정 경로 + 추적" if track else "고정 경로 (훑기만)", run(env, pol, pol.reset, a.episodes, a.seed))
    if a.model and a.model.endswith(".npz"):
        from factory_safety.policy_numpy import NumpyPolicy
        report("강화학습 정책 (npz)", run(env, NumpyPolicy(a.model), lambda: None, a.episodes, a.seed))
    elif a.model:
        from stable_baselines3 import PPO
        model = PPO.load(a.model, device="cpu")
        report("강화학습 정책", run(env, lambda o: model.predict(o, deterministic=True)[0], lambda: None, a.episodes, a.seed))


if __name__ == "__main__":
    main()
