"""전체 시연: Isaac Sim 창을 띄워 단계별로 차례로 보여준다 (일반 파이썬으로 실행, 단계마다 창이 저절로 닫힘).

    python scripts/demo_all.py
    python scripts/demo_all.py --steps bodycam ptz        # 일부만
    python scripts/demo_all.py --seed 3                  # 다른 위험 요소 배치

단계
  scene    창고 전체 모습 (위에서, 위험/안전 물체 배치)
  bodycam  작업자 바디캠 시점 순찰: 에이전트가 YOLO 위험/안전 판정, CCTV 접근 경고, CCTV 확대 재확인,
           작업자 손동작 (손가락 1~5) 명령에 중국어·영어·일본어 음성 안내
  top      관제 화면: 위에서 작업자가 걷는 모습
  ptz      CCTV 확대(PTZ) 화면: 에이전트가 고른 CCTV 가 애매한 물체와 점검 지점을 확대해 다시 판정
끝나면 조치 지시서(outputs/agent/dashboard_seed<시드>.html)를 브라우저로 연다.

각 단계 로그: outputs/logs/demo_<단계>.txt
"""
import argparse
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "outputs", "yolo", "warehouse_v3", "weights", "best.pt")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def steps(seed):
    run = ["run_patrol.py", "--weights", WEIGHTS, "--seed", str(seed)]
    return {
        "scene": ("창고 전체 모습", ["build_scene.py", "--seed", str(seed), "--duration", "20"]),
        "bodycam": ("작업자 바디캠 순찰 (판정 + 위험 영역 + 음성 경고 + 손동작 명령)", [*run, "--gestures", "demo", "--laps", "1.25"]),
        "top": ("관제 화면 (2배속)", [*run, "--view", "top", "--speed", "2"]),
        "ptz": ("CCTV 확대 재확인 화면", [*run, "--view", "ptz"]),
        "cctv": ("CCTV 화면 (서쪽 통로)", [*run, "--view", "cctv_west"]),
    }


def main():
    p = argparse.ArgumentParser(description="전체 시연")
    p.add_argument("--steps", nargs="+", default=["scene", "bodycam", "top", "ptz"])
    p.add_argument("--seed", type=int, default=5)
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
                if re.match(r"^\[\d\d:\d\d\]|^조치 지시서|^\s+\d+\. \[|^에이전트 채점|^\s+(위험을|안전을|없는|현장|재확인|바디캠만)|^CCTV 접근|^\[시나리오\]|^\[완료\]|^\[저장\]|^\[손동작\]", s):
                    print("  " + s, flush=True)
            proc.wait()
        print(f"  (끝, {time.time() - t0:.0f}초, 로그 {log})", flush=True)
    dash = os.path.join(ROOT, "outputs", "agent", f"dashboard_seed{a.seed}.html")
    if os.path.exists(dash) and hasattr(os, "startfile"):
        os.startfile(dash)
    print("\n[완료] 전체 시연이 끝났어요." + (f" 조치 지시서: {dash}" if os.path.exists(dash) else ""))


if __name__ == "__main__":
    sys.exit(main())
