"""Isaac Sim 순찰을 여러 시나리오로 돌려서 정답표 채점 결과를 표로 모은다 (일반 파이썬으로 실행).

    python scripts/eval_patrol.py --seeds 0 1 2 3 4

각 실행은 --sim-dt 로 시간 간격을 고정해서 같은 시드면 같은 결과가 나온다.
결과: outputs/eval/inspection_seed<시드>.json (물체별 판정), outputs/eval/answer_key_seed<시드>.json (정답표),
      outputs/eval/patrol_results.md (표)
"""
import argparse
import json
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(description="Isaac 순찰 여러 시나리오 평가")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--weights", default=os.path.join(ROOT, "outputs", "yolo", "warehouse", "weights", "best.pt"))
    p.add_argument("--laps", type=float, default=1.0)
    p.add_argument("--sim-dt", type=float, default=1 / 30)
    p.add_argument("--isaac", default=os.environ.get("ISAACSIM_PYTHON") or os.path.join(ROOT, ".venv-isaac", "Scripts", "python.exe"))
    a = p.parse_args()
    out_dir = os.path.join(ROOT, "outputs", "eval")
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES", PYTHONIOENCODING="utf-8")
    results = []
    for seed in a.seeds:
        res_path = os.path.join(out_dir, f"inspection_seed{seed}.json")
        log = os.path.join(out_dir, f"patrol_seed{seed}.txt")
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as f:
            subprocess.run([a.isaac, os.path.join(ROOT, "scripts", "run_patrol.py"), "--headless", "--seed", str(seed),
                            "--laps", str(a.laps), "--sim-dt", str(a.sim_dt), "--weights", a.weights, "--result", res_path],
                           stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        if not os.path.exists(res_path):
            print(f"  [실패] seed {seed}: 결과가 없어요 ({log})")
            continue
        r = json.load(open(res_path, encoding="utf-8"))
        results.append(r)
        b, c = r["bodycam"], r["cctv"] or {}
        print(f"  seed {seed}: 위험 {b['hazard_found']}/{b['hazard_total']}  안전 {b['safe_ok']}/{b['safe_total']}  "
              f"안전->위험 오판 {b['safe_as_hazard']}  CCTV 접근 사건 {c.get('events_detected')}/{c.get('events_visible')} 경고"
              f" (사각지대 {c.get('events_blind')}, 오경보 {c.get('false_alert_episodes')})  ({(time.time() - t0) / 60:.1f}분)", flush=True)
    if not results:
        return

    def tot(k, part="bodycam"):
        return sum((r[part] or {}).get(k, 0) or 0 for r in results)

    def med(k):
        v = [r["cctv"][k] for r in results if r["cctv"] and r["cctv"].get(k) is not None]
        return f"{sorted(v)[len(v) // 2]:.2f} m" if v else "-"

    hz, sf = tot("hazard_total"), tot("safe_total")
    lines = [f"# Isaac Sim 창고 순찰 평가 (시나리오 {len(results)}개, 한 바퀴, 시드 {[r['seed'] for r in results]})", "",
             "## 바디캠 YOLO 판정 (정답표와 비교, 경로에서 보인 물체만)", "",
             "| 항목 | 결과 |", "|---|:-:|",
             f"| 위험 물체를 위험으로 판정 | {tot('hazard_found')}/{hz} ({100 * tot('hazard_found') / max(1, hz):.0f}%) |",
             f"| 위험 물체를 안전으로 오판 | {tot('hazard_as_safe')} |",
             f"| 위험 물체 놓침 | {tot('hazard_missed')} |",
             f"| 안전 물체를 안전으로 판정 | {tot('safe_ok')}/{sf} ({100 * tot('safe_ok') / max(1, sf):.0f}%) |",
             f"| 안전 물체를 위험으로 오판 | {tot('safe_as_hazard')} |",
             f"| 상태까지 정확 | {tot('exact')}/{tot('seen')} |",
             f"| 정답 없는 곳의 위험 박스 (프레임 단위) | {tot('false_hazard_boxes')} / {tot('frames')}프레임 |", "",
             "## CCTV 작업자-위험물 접근 경고 (2 m, CCTV 3대를 묶어서 사건 단위)", "",
             "| 항목 | 결과 |", "|---|:-:|",
             f"| 작업자가 위험물에 다가간 사건 | {tot('events', 'cctv')} |",
             f"| CCTV 에 보인 사건 중 경고 | {tot('events_detected', 'cctv')}/{tot('events_visible', 'cctv')} |",
             f"| 어느 CCTV 에도 안 보인 사건 (사각지대) | {tot('events_blind', 'cctv')} |",
             f"| 오경보 (실제 3 m 넘는데 경고) | {tot('false_alert_episodes', 'cctv')}번 |",
             f"| 작업자 위치 오차 (중앙값) | {med('worker_err_median')} |",
             f"| 거리 오차 (중앙값) | {med('dist_err_median')} |", "",
             "## 시나리오별", "", "| 시드 | 위험 판정 | 안전 판정 | 안전->위험 오판 | CCTV 접근 사건 경고 | 오경보 |", "|:-:|:-:|:-:|:-:|:-:|:-:|"]
    for r in results:
        b, c = r["bodycam"], r["cctv"] or {}
        lines.append(f"| {r['seed']} | {b['hazard_found']}/{b['hazard_total']} | {b['safe_ok']}/{b['safe_total']} | {b['safe_as_hazard']} | "
                     f"{c.get('events_detected')}/{c.get('events_visible')} | {c.get('false_alert_episodes')} |")
    path = os.path.join(out_dir, "patrol_results.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[완료] {path}")


if __name__ == "__main__":
    main()
