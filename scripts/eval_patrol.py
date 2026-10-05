"""Isaac Sim 순찰을 여러 시나리오로 돌려서 정답표 채점 결과를 표로 모은다 (일반 파이썬으로 실행).

    python scripts/eval_patrol.py --seeds 0 1 2 3 4
    python scripts/eval_patrol.py --seeds 0 1 2 --agent both      # 규칙 vs LangGraph 재확인 비교 (LLM 호출로 느려서 시드 적게 권장)

각 실행은 --sim-dt 로 시간 간격을 고정해서 같은 시드면 같은 결과가 나온다.
결과: outputs/eval/inspection_seed<시드>.json (물체별 판정), outputs/eval/answer_key_seed<시드>.json (정답표),
      outputs/eval/patrol_results.md (표, --agent both 면 patrol_results_compare.md 도)
"""
import argparse
import json
import os
import subprocess
import time
from types import SimpleNamespace

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


def extra_lines(results):
    """위험 영역, 음성 경고, 공구 이름 표 (새 기능, 예전 결과에는 없음)."""
    ex = [r["agent"]["evaluation"] for r in results if r.get("agent") and "zones" in r["agent"]["evaluation"]]
    if not ex:
        return []

    def z(k):
        return sum(e["zones"][k] for e in ex)

    def v(k):
        return sum(e["voice"][k] for e in ex)

    def t(k):
        return sum(e["tools"][k] for e in ex)
    return ["## 위험 영역, 음성 경고, 공구 이름", "",
            "| 항목 | 결과 |", "|---|:-:|",
            f"| 위험 영역 (라바콘 링, DANGER 표지) 알아봄 | {z('found')}/{z('gt')} |",
            f"| 라바콘이 놓인 조치된 유출을 영역으로 | {z('marked_spill')} |",
            f"| 엉뚱한 곳에 만든 표시 영역 | {z('false')} |",
            f"| 에이전트가 스스로 판단한 위험 영역 (실제 위험 주변인 것) | {z('agent')} ({z('agent_real')}) |",
            f"| 작업자가 닿기 직전 (위험물 1 m, 영역 0.5 m) 사건에 음성 경고 | {v('warned')}/{v('events')} |",
            f"| 음성 경고 중 실제 사건에 맞은 것 | {v('useful')}/{v('voices')} |",
            f"| 공구 이름 맞힘 (정답 공구 종류 중) | {t('named')}/{t('total')} |", ""]


def collect_results(seeds, isaac, weights, laps, sim_dt, report_only, out_dir, agent_mode="langgraph", recheck_llm="qwen2.5:7b",
                    suffix=""):
    """시드마다 run_patrol.py 를 (headless, 고정 sim-dt 로) 돌려 결과를 모은다.
    suffix 를 주면 inspection_seed<시드><suffix>.json 에 따로 저장 (--agent both 로 같은 시드를 두 번 돌릴 때 안 겹치게)."""
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES", PYTHONIOENCODING="utf-8")
    results = []
    for seed in seeds:
        res_path = os.path.join(out_dir, f"inspection_seed{seed}{suffix}.json")
        log = os.path.join(out_dir, f"patrol_seed{seed}{suffix}.txt")
        t0 = time.time()
        if report_only:
            if os.path.exists(res_path):
                results.append(json.load(open(res_path, encoding="utf-8")))
            continue
        if os.path.exists(res_path):
            os.remove(res_path)
        with open(log, "w", encoding="utf-8") as f:
            cmd = [isaac, os.path.join(ROOT, "scripts", "run_patrol.py"), "--headless", "--seed", str(seed),
                  "--laps", str(laps), "--sim-dt", str(sim_dt), "--weights", weights, "--result", res_path,
                  "--agent", agent_mode, "--report-llm", "off"]   # 채점 수치엔 영향 없는 문구 보강이라 평가 때는 꺼서 빠르게
            if agent_mode == "langgraph":
                cmd += ["--recheck-llm", recheck_llm]
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        if not os.path.exists(res_path):
            print(f"  [실패] seed {seed} ({agent_mode}): 결과가 없어요 ({log})")
            continue
        r = json.load(open(res_path, encoding="utf-8"))
        results.append(r)
        b, c = r["bodycam"], r["cctv"] or {}
        print(f"  seed {seed} [{agent_mode}]: 위험 {b['hazard_found']}/{b['hazard_total']}  안전 {b['safe_ok']}/{b['safe_total']}  "
              f"안전->위험 오판 {b['safe_as_hazard']}  CCTV 접근 사건 {c.get('events_detected')}/{c.get('events_visible')} 경고"
              f" (사각지대 {c.get('events_blind')}, 오경보 {c.get('false_alert_episodes')})  ({(time.time() - t0) / 60:.1f}분)", flush=True)
        if r.get("agent"):
            e = r["agent"]["evaluation"]
            print(f"           에이전트: 위험 {e['after']['hazard_found']}/{e['after']['hazard_total']} (바디캠만 {e['before']['hazard_found']})  "
                  f"안전 {e['after']['safe_ok']}/{e['after']['safe_total']} (바디캠만 {e['before']['safe_ok']})  "
                  f"거꾸로 {e['after']['hazard_as_safe'] + e['after']['safe_as_hazard']}  없는 위험 {e['after']['false_reports']}  "
                  f"현장 확인 {e['after']['need_check']}  미확정 {e['after'].get('undetermined', 0)}  재확인 {e['recheck']['recheck_run']}",
                  flush=True)
    return results


def summarize(results):
    """결과 목록 -> 집계 함수들을 담은 SimpleNamespace (여러 모드를 같은 방식으로 요약할 때 재사용)."""
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
    return SimpleNamespace(results=results, tot=tot, med=med, ag=ag, rc=rc, cpc=cpc,
                           hz=tot("hazard_total"), sf=tot("safe_total"),
                           ah=ag("after", "hazard_total"), asf=ag("after", "safe_total"))


def format_report(s):
    """기존(단일 모드) 전체 표 (patrol_results.md)."""
    results, tot, med, ag, rc, cpc, hz, sf, ah, asf = (s.results, s.tot, s.med, s.ag, s.rc, s.cpc, s.hz, s.sf, s.ah, s.asf)
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
             f"| 현장 확인 요청 (그중 실제 물체) | - | {ag('after', 'need_check') + cpc[0]} ({ag('after', 'need_check_real') + cpc[1]}) |",
             f"| 미확정 (주의, 그중 실제 물체) | - | {ag('after', 'undetermined')} ({ag('after', 'undetermined_real')}) |", "",
             f"재확인 {rc('recheck_run')}건 실행 (요청 {rc('recheck_requested')}, 기다리는 동안 바디캠이 확정해서 취소 {rc('recheck_canceled')}): "
             f"찾음 {rc('recheck_found')}, 판정 고침 {rc('recheck_changed')} (맞게 고침 {rc('changed_correct')}), "
             f"다른 CCTV 로 재시도 {rc('recheck_retry')}, 재확인 소진 후 위험으로 둠 {rc('recheck_defaulted_hazard')}, "
             f"미확정 {rc('recheck_undetermined')}, 현장 확인(점검표) {rc('recheck_escalated')}, 오검출로 뺌 {rc('recheck_rejected')}", "",
             *extra_lines(results),
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
    return lines


def format_compare(seeds, s_rule, s_lg):
    """바디캠만(규칙 실행 기준) / 규칙 에이전트 / LangGraph 에이전트 3단 비교표 (patrol_results_compare.md)."""
    ar = {"hazard_total": s_rule.ah, "hazard_found": s_rule.ag("after", "hazard_found"),
         "safe_total": s_rule.asf, "safe_ok": s_rule.ag("after", "safe_ok"),
         "hazard_as_safe": s_rule.ag("after", "hazard_as_safe"), "safe_as_hazard": s_rule.ag("after", "safe_as_hazard"),
         "false_reports": s_rule.ag("after", "false_reports"), "need_check": s_rule.ag("after", "need_check") + s_rule.cpc[0],
         "undetermined": s_rule.ag("after", "undetermined")}
    al = {"hazard_total": s_lg.ah, "hazard_found": s_lg.ag("after", "hazard_found"),
         "safe_total": s_lg.asf, "safe_ok": s_lg.ag("after", "safe_ok"),
         "hazard_as_safe": s_lg.ag("after", "hazard_as_safe"), "safe_as_hazard": s_lg.ag("after", "safe_as_hazard"),
         "false_reports": s_lg.ag("after", "false_reports"), "need_check": s_lg.ag("after", "need_check") + s_lg.cpc[0],
         "undetermined": s_lg.ag("after", "undetermined")}
    bh, bs = s_rule.ag("before", "hazard_found"), s_rule.ag("before", "safe_ok")
    lines = [f"# 규칙 vs LangGraph 재확인 비교 (시드 {seeds}, --recheck-llm 만 다름)", "",
             "바디캠만: 두 모드 공통 (재확인 전 단계라 recheck_llm 과 무관, rule 실행 기준). "
             "규칙: --agent rule (PTZ_SURE 임계값). LangGraph: --agent langgraph (로컬 LLM 이 재확인 확신 판단).", "",
             "| 항목 | 바디캠만 | 규칙 에이전트 | LangGraph 에이전트 |", "|---|:-:|:-:|:-:|",
             f"| 위험을 위험으로 | {bh}/{ar['hazard_total']} | {ar['hazard_found']}/{ar['hazard_total']} "
             f"({100 * ar['hazard_found'] / max(1, ar['hazard_total']):.0f}%) | {al['hazard_found']}/{al['hazard_total']} "
             f"({100 * al['hazard_found'] / max(1, al['hazard_total']):.0f}%) |",
             f"| 안전을 안전으로 | {bs}/{ar['safe_total']} | {ar['safe_ok']}/{ar['safe_total']} "
             f"({100 * ar['safe_ok'] / max(1, ar['safe_total']):.0f}%) | {al['safe_ok']}/{al['safe_total']} "
             f"({100 * al['safe_ok'] / max(1, al['safe_total']):.0f}%) |",
             f"| 거꾸로 판정 (위험↔안전) | - | {ar['hazard_as_safe'] + ar['safe_as_hazard']} | {al['hazard_as_safe'] + al['safe_as_hazard']} |",
             f"| 없는 위험 보고 | - | {ar['false_reports']} | {al['false_reports']} |",
             f"| 현장 확인 요청 | - | {ar['need_check']} | {al['need_check']} |",
             f"| 미확정 (주의) | - | {ar['undetermined']} | {al['undetermined']} |", "",
             f"규칙: 재확인 {s_rule.rc('recheck_run')}건, 위험으로 둠 {s_rule.rc('recheck_defaulted_hazard')}, 미확정 {s_rule.rc('recheck_undetermined')}",
             f"LangGraph: 재확인 {s_lg.rc('recheck_run')}건, 위험으로 둠 {s_lg.rc('recheck_defaulted_hazard')}, 미확정 {s_lg.rc('recheck_undetermined')}",
             ""]
    return lines


def main():
    p = argparse.ArgumentParser(description="Isaac 순찰 여러 시나리오 평가")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--weights", default=os.path.join(ROOT, "outputs", "yolo", "warehouse_v3", "weights", "best.pt"))
    p.add_argument("--laps", type=float, default=1.0)
    p.add_argument("--sim-dt", type=float, default=1 / 30)
    p.add_argument("--report-only", action="store_true", help="순찰은 다시 안 돌리고 저장된 결과로 표만 다시 만들기")
    p.add_argument("--isaac", default=os.environ.get("ISAACSIM_PYTHON") or os.path.join(ROOT, ".venv-isaac", "Scripts", "python.exe"))
    p.add_argument("--agent", choices=["rule", "langgraph", "both"], default="langgraph",
                   help="재확인 판단: langgraph (기본, 로컬 LLM) | rule (PTZ_SURE 임계값만) | both (둘 다 돌려 비교표, 시드 적게 권장)")
    p.add_argument("--recheck-llm", default="qwen2.5:7b", help="--agent langgraph/both 일 때 쓸 로컬 LLM")
    a = p.parse_args()
    out_dir = os.path.join(ROOT, "outputs", "eval")
    os.makedirs(out_dir, exist_ok=True)

    if a.agent != "both":
        results = collect_results(a.seeds, a.isaac, a.weights, a.laps, a.sim_dt, a.report_only, out_dir,
                                  agent_mode=a.agent, recheck_llm=a.recheck_llm)
        if not results:
            return
        lines = format_report(summarize(results))
        path = os.path.join(out_dir, "patrol_results.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))
        print(f"\n[완료] {path}")
        return

    print("== 규칙 (--agent rule) ==")
    results_rule = collect_results(a.seeds, a.isaac, a.weights, a.laps, a.sim_dt, a.report_only, out_dir,
                                   agent_mode="rule", suffix="_rule")
    print("== LangGraph (--agent langgraph) ==")
    results_lg = collect_results(a.seeds, a.isaac, a.weights, a.laps, a.sim_dt, a.report_only, out_dir,
                                 agent_mode="langgraph", recheck_llm=a.recheck_llm, suffix="_langgraph")
    if not results_rule or not results_lg:
        print("[실패] 두 모드 중 하나라도 결과가 없어서 비교표를 못 만들어요.")
        return
    s_rule, s_lg = summarize(results_rule), summarize(results_lg)
    lines = format_compare(a.seeds, s_rule, s_lg)     # 바디캠만은 규칙 실행 기준 (recheck_llm 과 무관한 단계)
    path = os.path.join(out_dir, "patrol_results_compare.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[완료] {path}")
    # 개별 표도 참고용으로 저장
    with open(os.path.join(out_dir, "patrol_results_rule.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(format_report(s_rule)) + "\n")
    with open(os.path.join(out_dir, "patrol_results_langgraph.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(format_report(s_lg)) + "\n")


if __name__ == "__main__":
    main()
