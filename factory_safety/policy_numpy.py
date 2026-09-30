"""학습한 PPO 정책을 numpy만으로 실행.

Isaac Sim 파이썬에 torch / stable-baselines3 를 따로 설치하지 않아도 되게,
train_rl.py 가 학습 끝에 정책 신경망 가중치를 .npz 로 내보낸다.
"""
import re

import numpy as np


def export_sb3_policy(model, path):
    """stable-baselines3 PPO(MlpPolicy) -> .npz. 결정적 행동(평균값)만 쓴다."""
    sd = {k: v.detach().cpu().numpy() for k, v in model.policy.state_dict().items()}
    idx = sorted({int(m.group(1)) for k in sd for m in [re.match(r"mlp_extractor\.policy_net\.(\d+)\.weight", k)] if m})
    arrays = {}
    for n, i in enumerate(idx):
        arrays[f"pi_w{n}"] = sd[f"mlp_extractor.policy_net.{i}.weight"]
        arrays[f"pi_b{n}"] = sd[f"mlp_extractor.policy_net.{i}.bias"]
    arrays["act_w"] = sd["action_net.weight"]
    arrays["act_b"] = sd["action_net.bias"]
    act_fn = getattr(model.policy, "activation_fn", None)
    arrays["activation"] = np.array("relu" if act_fn is not None and "ReLU" in act_fn.__name__ else "tanh")
    np.savez(path, **arrays)
    return path


class NumpyPolicy:
    def __init__(self, path):
        z = np.load(path)
        self.layers = []
        i = 0
        while f"pi_w{i}" in z:
            self.layers.append((z[f"pi_w{i}"], z[f"pi_b{i}"]))
            i += 1
        self.out_w, self.out_b = z["act_w"], z["act_b"]
        self.relu = str(z["activation"]) == "relu"

    def __call__(self, obs):
        x = np.asarray(obs, dtype=np.float64)
        for w, b in self.layers:
            x = x @ w.T + b
            x = np.maximum(x, 0) if self.relu else np.tanh(x)
        return np.clip(x @ self.out_w.T + self.out_b, -1.0, 1.0).astype(np.float32)
