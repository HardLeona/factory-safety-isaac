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


def checkpoint_checks(r):
    """카메라로 못 봐서 현장 확인으로 넘긴 점검 지점 수와, 그 자리에 실제 물체가 있었던 수 (정답표와 비교)."""
    cps = [c for c in r["agent"]["report"]["checkpoints"] if c["status"] == "현장 확인 필요" and not c["finding"]]
    key = os.path.join(ROOT, "outputs", "eval", f"answer_key_seed{r['seed']}.json")
    objs = json.load(open(key, encoding="utf-8"))["objects"] if os.path.exists(key) else []
    real = 0
    for c in cps:
        kind = "ext_" if c["name"].startswith("소화기") else "tool_stored"
        real += any(o["class"].startswith(kind) and ((o["x"] - c["x"]) ** 2 + (o["y"] - c["y"]) ** 2) ** 0.5 < 2.0 for o in objs)
    return len(cps), real


def main():
    p = argparse.ArgumentParser(description="Isaac 순찰 여러 시나리오 평가")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--weights", default=os.path.join(ROOT, "outputs", "yolo", "warehouse", "weights", "best.pt"))
    p.add_argument("--laps", type=float, default=1.0)
    p.add_argument("--sim-dt", type=float, default=1 / 30)
    p.add_argument("--report-only", action="store_true", help="순찰은 다시 안 돌리고 저장된 결과로 표만 다시 만들기")
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
        if a.report_only:
            if os.path.exists(res_path):
                results.append(json.load(open(res_path, encoding="utf-8")))
            continue
        if os.path.exists(res_path):
            os.remove(res_path)
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
        if r.get("agent"):
            e = r["agent"]["evaluation"]
            print(f"           에이전트: 위험 {e['after']['hazard_found']}/{e['after']['hazard_total']} (바디캠만 {e['before']['hazard_found']})  "
                  f"안전 {e['after']['safe_ok']}/{e['after']['safe_total']} (바디캠만 {e['before']['safe_ok']})  "
                  f"거꾸로 {e['after']['hazard_as_safe'] + e['after']['safe_as_hazard']}  없는 위험 {e['after']['false_reports']}  "
                  f"현장 확인 {e['after']['need_check']}  재확인 {e['recheck']['recheck_run']}", flush=True)
    if not results:
        return

    def tot(k, part="bodycam"):
        return sum((r[part] or {}).get(k, 0) or 0 for r in results)

    def med(k):
        v = [r["cctv"][k] for r in results if r["cctv"] and r["cctv"].get(k) is not None]
        return f"{sorted(v)[len(v) // 2]:.2f} m" if v else "-"

    def ag(part, k):
        return sum(r["agent"]["evaluation"][part][k] for r in results if r.get("agent"))

    def rc(k):
        return sum(r["agent"]["evaluation"]["recheck"][k] for r in results if r.get("agent"))

    cps = [checkpoint_checks(r) for r in results if r.get("agent")]
    cpc = (sum(c[0] for c in cps), sum(c[1] for c in cps))
    hz, sf = tot("hazard_total"), tot("safe_total")
    ah, asf = ag("after", "hazard_total"), ag("after", "safe_total")
    lines = [f"# Isaac Sim 창고 순찰 평가 (시나리오 {len(results)}개, 한 바퀴, 시드 {[r['seed'] for r in results]})", "",
             f"## 에이전트 최종 판정 (창고 전체 물체 {ah + asf}개, 정답표와 비교)", "",
             "바디캠만: 바디캠이 확정한 물체만 바디캠 판정으로. 에이전트: 재확인(CCTV 확대)과 점검표까지 거친 최종 위험물 대장.", "",
             "| 항목 | 바디캠만 | 에이전트 |", "|---|:-:|:-:|",
             f"| 위험 물체를 위험으로 | {ag('before', 'hazard_found')}/{ah} ({100 * ag('before', 'hazard_found') / max(1, ah):.0f}%) | "
             f"{ag('after', 'hazard_found')}/{ah} ({100 * ag('after', 'hazard_found') / max(1, ah):.0f}%) |",
             f"| 안전 물체를 안전으로 | {ag('before', 'safe_ok')}/{asf} ({100 * ag('before', 'safe_ok') / max(1, asf):.0f}%) | "
             f"{ag('after', 'safe_ok')}/{asf} ({100 * ag('after', 'safe_ok') / max(1, asf):.0f}%) |",
             f"| 위험을 안전으로 오판 | {ag('before', 'hazard_as_safe')} | {ag('after', 'hazard_as_safe')} |",
             f"| 안전을 위험으로 오판 | {ag('before', 'safe_as_hazard')} | {ag('after', 'safe_as_hazard')} |",
             f"| 상태까지 정확 | {ag('before', 'exact')}/{ah + asf} | {ag('after', 'exact')}/{ah + asf} |",
             f"| 없는 위험 보고 | {ag('before', 'false_reports')} | {ag('after', 'false_reports')} |",
             f"| 현장 확인 요청 (그중 실제 물체) | - | {ag('after', 'need_check') + cpc[0]} ({ag('after', 'need_check_real') + cpc[1]}) |", "",
             f"재확인 {rc('recheck_run')}건 실행 (요청 {rc('recheck_requested')}, 기다리는 동안 바디캠이 확정해서 취소 {rc('recheck_canceled')}): "
             f"찾음 {rc('recheck_found')}, 판정 고침 {rc('recheck_changed')} (맞게 고침 {rc('changed_correct')}), "
             f"다른 CCTV 로 재시도 {rc('recheck_retry')}, 현장 확인 {rc('recheck_escalated')}, 오검출로 뺌 {rc('recheck_rejected')}", "",
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
             "## 시나리오별 (대표 테스트 케이스)", "",
             "| 시드 | 물체 (위험) | 에이전트 위험 판정 | 에이전트 안전 판정 | 거꾸로 판정 | 없는 위험 보고 | 현장 확인 | 재확인 (판정 고침) | 바디캠 YOLO 위험/안전 | CCTV 접근 경고 | 오경보 |",
             "|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|"]
    for r in results:
        b, c = r["bodycam"], r["cctv"] or {}
        e = r.get("agent", {}).get("evaluation")
        if e:
            a_, rr = e["after"], e["recheck"]
            ag_cols = (f"{a_['hazard_total'] + a_['safe_total']} ({a_['hazard_total']}) | {a_['hazard_found']}/{a_['hazard_total']} | "
                       f"{a_['safe_ok']}/{a_['safe_total']} | {a_['hazard_as_safe'] + a_['safe_as_hazard']} | {a_['false_reports']} | "
                       f"{a_['need_check'] + checkpoint_checks(r)[0]} | {rr['recheck_run']} ({rr['recheck_changed']})")
        else:
            ag_cols = "- | - | - | - | - | - | -"
        lines.append(f"| {r['seed']} | {ag_cols} | {b['hazard_found']}/{b['hazard_total']}, {b['safe_ok']}/{b['safe_total']} | "
                     f"{c.get('events_detected')}/{c.get('events_visible')} | {c.get('false_alert_episodes')} |")
    path = os.path.join(out_dir, "patrol_results.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[완료] {path}")


if __name__ == "__main__":
    main()
