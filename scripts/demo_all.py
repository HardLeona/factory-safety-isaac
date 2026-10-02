"""전체 시연: Isaac Sim 창을 띄워 단계별로 차례로 보여준다 (일반 파이썬으로 실행, 단계마다 창이 저절로 닫힘).

    python scripts/demo_all.py
    python scripts/demo_all.py --steps patrol top        # 일부만
    python scripts/demo_all.py --seed 3                  # 다른 위험 요소 배치

단계
  scene   장면 전체 모습 (위에서)
  patrol  로봇 시점 순찰: YOLO 화면 판단 + 압력계 줌 판독
  top     관제 화면: 위에서 로봇이 도는 모습
  bodycam 작업자 바디캠 시점 (흔들림)
  policy  강화학습 정책 재생 (위에서)

각 단계 로그: outputs/logs/demo_<단계>.txt
"""
import argparse
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "outputs", "yolo", "factory_hazard_v2", "weights", "best.pt")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def steps(seed):
    yolo = ["--detector", "yolo", "--weights", WEIGHTS, "--seed", str(seed)]
    return {
        "scene": ("장면 전체 모습", ["build_scene.py", "--seed", str(seed), "--duration", "20"]),
        "patrol": ("로봇 시점 순찰 (YOLO + 압력계 줌 판독)", ["run_patrol.py", *yolo, "--duration", "80"]),
        "top": ("관제 화면 (2배속)", ["run_patrol.py", *yolo, "--view", "top", "--speed", "2", "--duration", "80"]),
        "bodycam": ("작업자 바디캠", ["run_patrol.py", *yolo, "--camera", "bodycam", "--duration", "60"]),
        "policy": ("강화학습 정책 재생", ["play_policy.py", "--episodes", "1", "--seed", str(seed), "--view", "top"]),
    }


def main():
    p = argparse.ArgumentParser(description="전체 시연")
    p.add_argument("--steps", nargs="+", default=["scene", "patrol", "top", "bodycam", "policy"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--isaac", default=os.environ.get("ISAACSIM_PYTHON") or os.path.join(ROOT, ".venv-isaac", "Scripts", "python.exe"))
    a = p.parse_args()
    table = steps(a.seed)
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    os.makedirs(os.path.join(ROOT, "outputs", "logs"), exist_ok=True)
    for k, name in enumerate(a.steps, 1):
        title, cmd = table[name]
        log = os.path.join(ROOT, "outputs", "logs", f"demo_{name}.txt")
        print(f"\n=== [{k}/{len(a.steps)}] {title} ===", flush=True)
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as f:
            proc = subprocess.Popen([a.isaac, os.path.join(ROOT, "scripts", cmd[0]), *cmd[1:]], cwd=ROOT, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                    errors="replace")
            for line in proc.stdout:
                f.write(line)
                s = ANSI.sub("", line).rstrip()
                # Isaac 내부 로그는 빼고 순찰 결과만 화면에
                if re.match(r"^\[\d\d:\d\d\]|^순찰 시간|^\s+\[채점\]|^\s+못 찾음|^\[시나리오\]|^\[완료\]|^\[에피소드|^\s+충돌|^\[경고\]", s):
                    print("  " + s, flush=True)
            proc.wait()
        print(f"  (끝, {time.time() - t0:.0f}초, 로그 {log})", flush=True)
    print("\n[완료] 전체 시연이 끝났어요.")


if __name__ == "__main__":
    sys.exit(main())
