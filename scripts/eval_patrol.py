"""Isaac Sim 순찰을 여러 시나리오로 돌려서 채점 결과를 표로 모은다 (일반 파이썬으로 실행).

    python scripts/eval_patrol.py --seeds 0 1 2 3 4
    python scripts/eval_patrol.py --seeds 0 1 2 --modes yolo_zoom yolo

모드
  yolo_zoom  YOLO 화면 판단 + 압력계 줌 판독 (기본)
  yolo       YOLO 화면 판단만 (압력계 정상/부족도 YOLO 클래스로)
  sim        가상 검출기 (정답 위치 기반, 비교 기준)

각 실행은 --sim-dt 로 시간 간격을 고정해서 같은 시드면 같은 결과가 나온다.
결과: outputs/eval/patrol_<모드>_seed<시드>.txt (로그), outputs/eval/patrol_results.md (표),
      outputs/eval/dets_<모드>_seed<시드>.json (검출 기록, scripts/replay_dets.py 로 다시 판단)
"""
import argparse
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAT_JUDGE = re.compile(r"\[채점\] 위험 요소 (\d+)/(\d+)\s+\|\s+소화기 점검 (\d+)/(\d+)\s+\|\s+오탐 (\d+)건")
PAT_SIM = re.compile(r"순찰 시간 \S+\s+\|\s+위험 요소 (\d+)/(\d+)\s+\|\s+소화기 점검 (\d+)/(\d+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def run_one(isaac, mode, seed, args, out_dir):
    cmd = [isaac, os.path.join(ROOT, "scripts", "run_patrol.py"), "--headless", "--seed", str(seed),
           "--duration", str(args.duration), "--sim-dt", str(args.sim_dt)]
    if mode.startswith("yolo"):
        cmd += ["--detector", "yolo", "--weights", args.weights,
                "--dump-dets", os.path.join(out_dir, f"dets_{mode}_seed{seed}.json")]
        if mode == "yolo":
            cmd += ["--no-zoom"]
    log = os.path.join(out_dir, f"patrol_{mode}_seed{seed}.txt")
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES", PYTHONIOENCODING="utf-8")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    text = ANSI.sub("", open(log, encoding="utf-8", errors="replace").read())
    m = PAT_JUDGE.search(text)
    if m:
        found, total, ext, ext_total, fa = map(int, m.groups())
    else:
        m = PAT_SIM.search(text)
        if not m:
            print(f"  [실패] {mode} seed {seed}: 결과 줄을 못 찾음 ({log})")
            return None
        found, total, ext, ext_total = map(int, m.groups())
        fa = 0
    row = dict(mode=mode, seed=seed, found=found, total=total, ext=ext, ext_total=ext_total, fa=fa,
               wall=time.time() - t0)
    print(f"  {mode:10s} seed {seed}: 위험 {found}/{total}  소화기 {ext}/{ext_total}  오탐 {fa}  ({row['wall'] / 60:.1f}분)")
    return row


def main():
    p = argparse.ArgumentParser(description="Isaac 순찰 여러 시나리오 평가")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--modes", nargs="+", default=["yolo_zoom"], choices=["yolo_zoom", "yolo", "sim"])
    v2 = os.path.join(ROOT, "outputs", "yolo", "factory_hazard_v2", "weights", "best.pt")
    v1 = os.path.join(ROOT, "outputs", "yolo", "factory_hazard", "weights", "best.pt")
    p.add_argument("--weights", default=v2 if os.path.exists(v2) else v1)
    p.add_argument("--duration", type=float, default=150.0)
    p.add_argument("--sim-dt", type=float, default=1 / 30)
    p.add_argument("--isaac", default=os.environ.get("ISAACSIM_PYTHON", os.path.join(ROOT, ".venv-isaac", "Scripts", "python.exe")))
    a = p.parse_args()
    out_dir = os.path.join(ROOT, "outputs", "eval")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for mode in a.modes:
        for seed in a.seeds:
            r = run_one(a.isaac, mode, seed, a, out_dir)
            if r:
                rows.append(r)
    if not rows:
        sys.exit("[오류] 결과가 없어요.")
    lines = [f"# Isaac Sim 순찰 평가 (시나리오 {len(a.seeds)}개, {a.duration:.0f}초, 시드 {a.seeds})", "",
             "| 방식 | 위험 요소 발견 | 소화기 점검 | 오탐 (건/시나리오) |", "|---|:-:|:-:|:-:|"]
    for mode in a.modes:
        rs = [r for r in rows if r["mode"] == mode]
        if not rs:
            continue
        f, t = sum(r["found"] for r in rs), sum(r["total"] for r in rs)
        e, et = sum(r["ext"] for r in rs), sum(r["ext_total"] for r in rs)
        fa = sum(r["fa"] for r in rs) / len(rs)
        lines.append(f"| {mode} | {f}/{t} ({100 * f / t:.0f}%) | {e}/{et} ({100 * e / et:.0f}%) | {fa:.1f} |")
    lines += ["", "| 방식 | 시드 | 위험 | 소화기 | 오탐 |", "|---|:-:|:-:|:-:|:-:|"]
    for r in rows:
        lines.append(f"| {r['mode']} | {r['seed']} | {r['found']}/{r['total']} | {r['ext']}/{r['ext_total']} | {r['fa']} |")
    path = os.path.join(out_dir, "patrol_results.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[:4 + len(a.modes)]))
    print(f"\n[완료] {path}")


if __name__ == "__main__":
    main()
