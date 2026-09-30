"""강화학습 (PPO) 으로 순찰 정책 학습. Isaac Sim 없이 일반 파이썬에서 돈다.

    pip install -r requirements.txt
    python scripts/train_rl.py                     # 기본 300만 스텝, 환경 8개 병렬
    python scripts/train_rl.py --steps 1000000 --envs 4
    python scripts/train_rl.py --export outputs/rl/checkpoints/ppo_patrol_1000000_steps.zip   # 체크포인트를 .npz로
    tensorboard --logdir outputs/rl/tb             # 학습 곡선 보기
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def make_env(max_steps):
    def _f():
        from factory_safety.rl_env import FactoryPatrolEnv
        return FactoryPatrolEnv(max_steps=max_steps)
    return _f


def main():
    p = argparse.ArgumentParser(description="순찰 정책 강화학습 (PPO)")
    p.add_argument("--steps", type=int, default=3_000_000)
    p.add_argument("--envs", type=int, default=8, help="병렬 환경 수 (CPU 코어 수 정도)")
    p.add_argument("--max-steps", type=int, default=1500, help="에피소드 길이 (0.1초 단위, 1500 = 150초)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(ROOT, "outputs", "rl"))
    p.add_argument("--resume", default=None, help="이어서 학습할 모델 .zip")
    p.add_argument("--export", default=None, help="학습 없이 이 .zip 을 Isaac 재생용 .npz 로 변환만")
    a = p.parse_args()

    from stable_baselines3 import PPO
    from factory_safety.policy_numpy import export_sb3_policy

    if a.export:
        out = os.path.splitext(a.export)[0] + ".npz"
        export_sb3_policy(PPO.load(a.export, device="cpu"), out)
        print(f"[완료] {out}")
        return

    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    os.makedirs(a.out, exist_ok=True)
    venv = make_vec_env(make_env(a.max_steps), n_envs=a.envs, seed=a.seed,
                        vec_env_cls=SubprocVecEnv if a.envs > 1 else None)
    if a.resume:
        model = PPO.load(a.resume, env=venv, device="cpu")
    else:
        model = PPO("MlpPolicy", venv, n_steps=1024, batch_size=1024, n_epochs=10, learning_rate=3e-4,
                    gamma=0.995, gae_lambda=0.95, clip_range=0.2, ent_coef=0.005,
                    policy_kwargs=dict(net_arch=[256, 256]), tensorboard_log=os.path.join(a.out, "tb"),
                    seed=a.seed, verbose=1, device="cpu")
    ckpt = CheckpointCallback(save_freq=max(1, 250_000 // a.envs), save_path=os.path.join(a.out, "checkpoints"),
                              name_prefix="ppo_patrol")
    model.learn(total_timesteps=a.steps, callback=ckpt, reset_num_timesteps=a.resume is None)
    path = os.path.join(a.out, "ppo_patrol.zip")
    model.save(path)
    npz = export_sb3_policy(model, os.path.join(a.out, "ppo_patrol.npz"))
    venv.close()
    print(f"\n[완료] 정책 저장: {path}")
    print(f"       Isaac 재생용 (numpy): {npz}")
    print("       비교 평가: python scripts/eval_rl.py")
    print("       화면 재생: <isaac>/python.sh scripts/play_policy.py")


if __name__ == "__main__":
    main()
