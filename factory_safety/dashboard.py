"""순찰이 끝나면 만드는 조치 지시서 화면 (HTML 한 장, 인터넷 없이 열림).

    write_dashboard(path, report, evaluation=None, seed=None)

평면도(SVG)에 위험물 위치와 우선순위 번호, 아래에 조치 목록, 점검표, 에이전트 기록을 넣는다.
evaluation 을 주면 정답표 비교 표도 붙인다 (평가용).
"""
import html

from . import warehouse as W
from .report import clock

X0, X1, Y0, Y1 = -11.0, 11.0, -12.6, 18.6
S = 17.0          # 1 m 당 픽셀
COLOR = {"위험": "#d93a3a", "안전": "#2e9d5b", "확인 필요": "#e09a1a", "주의": "#8a5fd1"}


def _p(x, y):
    return (x - X0) * S, (Y1 - y) * S


def _map_svg(rep):
    w, h = (X1 - X0) * S, (Y1 - Y0) * S
    out = [f'<svg viewBox="0 0 {w:.0f} {h:.0f}" xmlns="http://www.w3.org/2000/svg" class="map">']
    x, y = _p(-10.3, 18.0)
    out.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{20.6 * S:.0f}" height="{30.2 * S:.0f}" fill="#f4f1ea" stroke="#777" stroke-width="2"/>')
    for (a, b) in W.RACKS:
        x, y = _p(a[0], b[1])
        out.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{(b[0] - a[0]) * S:.0f}" height="{(b[1] - a[1]) * S:.0f}" '
                   f'fill="#b9b4a8" rx="2"/>')
    for (a, b) in W.PILLARS:
        x, y = _p(a[0], b[1])
        out.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{(b[0] - a[0]) * S:.0f}" height="{(b[1] - a[1]) * S:.0f}" fill="#555"/>')
    for (tx, ty, _) in W.TABLES:
        x, y = _p(tx - 1.24, ty + 0.39)
        out.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{2.47 * S:.0f}" height="{0.78 * S:.0f}" fill="#a07850"/>')
    for zz in rep.get("zones", []):
        pts = " ".join("{:.0f},{:.0f}".format(*_p(a, b)) for a, b in zz["poly"])
        col = "#d93a3a" if zz["source"] != "agent" else "#e07a1a"
        out.append(f'<polygon points="{pts}" fill="{col}" fill-opacity="0.16" stroke="{col}" stroke-width="2" stroke-dasharray="5 3"/>')
        cx, cy = _p(zz["x"], zz["y"])
        out.append(f'<text x="{cx:.0f}" y="{cy - zz["radius"] * S - 4:.0f}" class="zl" fill="{col}">{zz["id"]}</text>')
    path = W.PatrolPath()
    pts = " ".join("{:.0f},{:.0f}".format(*_p(px, py)) for px, py in path.pts[::4])
    out.append(f'<polygon points="{pts}" fill="none" stroke="#3a78c9" stroke-width="2" stroke-dasharray="6 5" opacity="0.7"/>')
    for c in rep["checkpoints"]:
        x, y = _p(c["x"], c["y"])
        done = c["status"] == "바디캠 확인"
        out.append(f'<rect x="{x - 5:.0f}" y="{y - 5:.0f}" width="10" height="10" fill="{"#fff" if done else "#ffe9b8"}" '
                   f'stroke="#555"/><text x="{x:.0f}" y="{y + 3.5:.0f}" class="ck">{"✓" if done else "?"}</text>')
    for r in sorted(rep["findings"], key=lambda r: -r["rank"]):
        x, y = _p(r["x"], r["y"])
        col = COLOR[r["state"]]
        rad = 11 if r["state"] != "안전" else 7
        out.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{rad}" fill="{col}" stroke="#fff" stroke-width="2"/>')
        if r["state"] != "안전":
            out.append(f'<text x="{x:.0f}" y="{y + 4:.0f}" class="num">{r["rank"]}</text>')
    out.append("</svg>")
    return "".join(out)


def _rows(rep):
    out = []
    for r in rep["findings"]:
        ev = [f"바디캠 {r['n_body']}프레임"]
        if r["near_miss"]:
            ev.append(f"<b>접근 경고 {r['near_miss']}회</b>")
        if r["changed_by_recheck"]:
            ev.append("재관측 재판단으로 판정 고침")
        action = html.escape(r["action"])
        if r.get("action_source"):
            action += f'<br><span class="muted">근거: {html.escape(r["action_source"])}</span>'
        out.append(f'<tr class="{"safe" if r["state"] == "안전" else ""}"><td>{r["rank"]}</td>'
                   f'<td><span class="pri p{r["priority"]}">{r["priority"]}</span></td>'
                   f'<td><span class="st" style="background:{COLOR[r["state"]]}">{r["state"]}</span> {html.escape(r["label"])}</td>'
                   f'<td class="zone">{html.escape(r["zone"])}<br><small>({r["x"]:+.1f}, {r["y"]:+.1f})</small></td>'
                   f'<td>{action}</td><td><small>{" · ".join(ev)}<br>확신도 {r["confidence"] * 100:.0f}%</small></td></tr>')
    return "".join(out)


def _eval_html(ev):
    """before: BodycamInspector 프레임 단위 YOLO 원시 판정. after: 에이전트 최종 대장 (물체 단위).
    집계 단위가 달라 "없는 위험 판정" 은 프레임 단위 박스 수(before) 와 물체 단위 보고 건수(after) 를 나란히만 보여준다."""
    b, a, r = ev["before"], ev["after"], ev["rejudge"]
    rows = [("위험을 위험으로", f"{b['hazard_found']}/{b['hazard_total']}", f"{a['hazard_found']}/{a['hazard_total']}"),
            ("안전을 안전으로", f"{b['safe_ok']}/{b['safe_total']}", f"{a['safe_ok']}/{a['safe_total']}"),
            ("위험을 안전으로 오판", b["hazard_as_safe"], a["hazard_as_safe"]),
            ("안전을 위험으로 오판", b["safe_as_hazard"], a["safe_as_hazard"]),
            ("없는 위험 판정 (박스 수 / 보고 건수)", f"{b['false_hazard_boxes']}개", f"{a['false_reports']}건"),
            ("현장 확인 요청", "-", f"{a['need_check']} (실제 물체 {a['need_check_real']})"),
            ("주의 (안전·위험 못 가름)", "-", f"{a['caution']} (실제 물체 {a['caution_real']})")]
    if "zones" in ev:
        z, v, t = ev["zones"], ev["voice"], ev["tools"]
        rows += [("위험 영역 (라바콘·표지) 알아봄", "-", f"{z['found']}/{z['gt']}"),
                 ("에이전트가 판단한 위험 영역 (실제 위험 주변)", "-", f"{z['agent']} ({z['agent_real']})"),
                 ("닿기 직전 사건에 음성 경고", "-", f"{v['warned']}/{v['events']}"),
                 ("공구 이름 맞힘", "-", f"{t['named']}/{t['total']}")]
    if "gestures" in ev:
        g = ev["gestures"]
        rows += [("손동작 명령 맞게 인식 (잘못 실행)", "-", f"{g['recognized']}/{g['shown']} ({g['extra']})")]
    body = "".join(f"<tr><td>{k}</td><td>{v1}</td><td>{v2}</td></tr>" for k, v1, v2 in rows)
    return (f'<h2>정답표 비교 (평가용)</h2>'
            f'<p class="muted">바디캠 (프레임 단위 YOLO 원시 판정, 에이전트 판단 전) vs 에이전트 (2단계 분류 + 재관측 재판단 + 위험 영역·점검표 반영)</p>'
            f'<table class="ev"><tr><th></th><th>바디캠 프레임 원시판정</th><th>에이전트</th></tr>{body}</table>'
            f'<p class="muted">애매한 판정 {r["ambiguous_hazard"] + r["ambiguous_caution"]}건 '
            f'(즉시 위험 확정 {r["ambiguous_hazard"]}, 주의 분류 {r["ambiguous_caution"]}) · '
            f'주의 재관측 재판단 {r["caution_rejudged"]}건 중 확정 전환 {r["caution_rejudged_confirmed"]} (맞게 고침 {r["changed_correct"]})</p>')


def llm_note(e):
    """LLM 에이전트가 정한 근거와 부른 도구."""
    m = e.get("llm") or {}
    if not m:
        return ""
    if not m.get("llm"):
        return '<br><span class="muted">판단: 규칙 (LLM 못 씀)</span>'
    return (f'<br><span class="llm">AI 판단 ({html.escape(str(m.get("model")))}, {m.get("sec", 0):.1f}초): {html.escape(m.get("reason_ko") or "")}'
            f'<br>{html.escape(" → ".join(m.get("trace") or []))}</span>')


def write_dashboard(path, rep, evaluation=None, seed=None, title="창고 안전 순찰 조치 지시서"):
    s = rep["summary"]
    tl = "".join(f'<li><span class="t">{clock(e["t"])}</span><span class="k k{e["kind"].replace(" ", "")}">{e["kind"]}</span>'
                 f'{html.escape(e["text"])}</li>' for e in rep["timeline"] if e["kind"] != "계획" or "점검표" not in e["text"])
    srcname = {"cone": "라바콘 표시", "sign": "DANGER 표지", "agent": "에이전트 판단"}
    zrows = "".join(f'<tr><td>{z["id"]}</td><td><b>{srcname[z["source"]]}</b>: {html.escape(z["reason"])}</td>'
                    f'<td>{html.escape(z["zone"])} <small>({z["x"]:+.1f}, {z["y"]:+.1f})</small></td><td>반지름 {z["radius"]:.1f} m</td></tr>'
                    for z in rep.get("zones", [])) or '<tr><td colspan="4">없음</td></tr>'
    vrows = "".join(
        f'<tr><td>{clock(v["t"])}</td><td><span class="st" style="background:{"#d93a3a" if v["level"] == "hazard" else "#8a5fd1"}">'
        f'{"위험" if v["level"] == "hazard" else "주의"}</span></td><td>{html.escape(v["what"])}</td></tr>'
        for v in rep.get("voice", [])) or '<tr><td colspan="3">없음</td></tr>'
    cps = "".join(f'<tr><td>{c["id"]}</td><td>{html.escape(c["name"])}</td><td>{c["status"]}</td><td>{c["finding"] or "-"}</td></tr>'
                  for c in rep["checkpoints"])
    asst = rep.get("assistant", [])
    arows = "".join(
        f'<tr class="a{e["count"]}"><td>{clock(e["t"])}</td><td><b>{e["count"]}</b></td><td class="zone">{html.escape(e["cmd"])}</td>'
        f'<td class="zone">{html.escape(e["lang_name"])}</td><td>{html.escape(e["text"])}<br><span class="muted">{html.escape(e["text_ko"])}</span>'
        f'{llm_note(e)}</td></tr>'
        for e in asst)
    alerts = "".join(
        f'<div class="alert {"sos" if e["count"] == 5 else "mgr"}"><b>{"SOS 신고" if e["count"] == 5 else "관리자 호출"}</b> '
        f'{clock(e["t"])} · {html.escape(e["zone"])} ({e["worker"][0]:+.1f}, {e["worker"][1]:+.1f}) · 작업자 언어 {html.escape(e["lang_name"])}'
        f'{"<br>" + html.escape(e["manager_ko"]) if e.get("manager_ko") else ""}</div>'
        for e in asst if e["count"] in (4, 5))
    doc = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{{margin:0;font-family:"Malgun Gothic","Apple SD Gothic Neo","Microsoft YaHei","Yu Gothic",sans-serif;background:#f6f7f9;color:#1d2330}}
.alert{{margin:10px 24px 0;padding:10px 14px;border-radius:8px;font-size:14px}} .alert.sos{{background:#d93a3a;color:#fff}}
.alert.mgr{{background:#f6e3a6;color:#4a3a00}} .k손동작{{color:#0a7cc4}} .k관리자호출{{color:#c27c00}} .kSOS{{color:#d93a3a}}
tr.a5 td{{background:#fdeaea}} tr.a4 td{{background:#fdf6e0}} .llm{{color:#6b3fc7;font-size:12px}} .kLLM판단{{color:#6b3fc7}}
header{{background:#1f2b3d;color:#fff;padding:14px 24px}} header h1{{margin:0;font-size:22px}} header p{{margin:4px 0 0;opacity:.8}}
main{{display:grid;grid-template-columns:400px 1fr;gap:18px;padding:18px 24px}}
.card{{background:#fff;border-radius:10px;padding:14px 16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.kpi{{display:flex;gap:10px;margin-bottom:12px}} .kpi div{{flex:1;background:#fff;border-radius:10px;padding:10px 12px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.kpi b{{display:block;font-size:26px}} .kpi span{{font-size:12px;color:#667}}
.map{{width:100%;height:auto}} .lbl{{font-size:10px;fill:#333}} .num{{font-size:11px;fill:#fff;font-weight:700;text-anchor:middle}}
.ck{{font-size:9px;text-anchor:middle;fill:#333}} .zl{{font-size:12px;font-weight:700;text-anchor:middle}}
table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{border-bottom:1px solid #e4e6ea;padding:6px 6px;text-align:left;vertical-align:top}}
tr.safe td{{color:#667}} .st{{color:#fff;border-radius:4px;padding:1px 6px;font-size:12px}}
.pri{{border-radius:4px;padding:1px 6px;font-size:12px;background:#eee;white-space:nowrap}} .st{{white-space:nowrap}}
td.zone{{white-space:nowrap}} .p긴급{{background:#d93a3a;color:#fff}} .p높음{{background:#f0a35a}} .p보통{{background:#f6e3a6}}
h2{{font-size:16px;margin:4px 0 10px}} ul.tl{{list-style:none;padding:0;margin:0;font-size:12.5px;max-height:520px;overflow:auto}}
ul.tl li{{padding:3px 0;border-bottom:1px dashed #e4e6ea}} .t{{color:#889;margin-right:6px;font-family:monospace}}
.k{{display:inline-block;min-width:62px;margin-right:6px;font-weight:700;color:#3a78c9}} .k음성경고,.k현장확인,.k위험영역{{color:#d93a3a}} .k판정수정{{color:#9b4dca}}
.k주의,.k주의안내,.k접근기록{{color:#e08a00}} .muted{{color:#667;font-size:12px}} .legend span{{margin-right:10px;font-size:12px}}
.ev td,.ev th{{text-align:center}} .ev td:first-child{{text-align:left}}
@media (max-width:900px){{main{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>{title}</h1><p>작업자 바디캠 + YOLO 판정, 에이전트 재판단 결과{f" · 시나리오 {seed}" if seed is not None else ""}</p></header>
{alerts}
<main><section>
<div class="kpi"><div><b style="color:#d93a3a">{s['hazards']}</b><span>위험 (긴급 {s['urgent']})</span></div>
<div><b style="color:#e09a1a">{s['need_check']}</b><span>현장 확인</span></div>
<div><b style="color:#8a5fd1">{s['caution']}</b><span>주의 (미확정)</span></div>
<div><b style="color:#2e9d5b">{s['safe']}</b><span>안전 확인</span></div>
<div><b>{s['checkpoints_done']}/{s['checkpoints']}</b><span>점검표</span></div>
<div><b style="color:#d93a3a">{s.get('zones', 0)}</b><span>위험 영역</span></div>
<div><b style="color:#9b4dca">{s.get('voice', 0)}</b><span>음성 경고</span></div></div>
<div class="card">{_map_svg(rep)}
<p class="legend"><span style="color:#d93a3a">● 위험 (번호=우선순위)</span><span style="color:#e09a1a">● 확인 필요</span>
<span style="color:#2e9d5b">● 안전</span><span>□ 점검 지점</span><span style="color:#3a78c9">- - 순찰 경로</span></p></div>
<div class="card" style="margin-top:14px"><h2>위험 영역</h2><table><tr><th></th><th>근거</th><th>위치</th><th>크기</th></tr>{zrows}</table></div>
<div class="card" style="margin-top:14px"><h2>음성 경고 (위험: "멈추세요! 위험 요소가 식별되었습니다" · 주의: "발밑을 확인하세요")</h2><table><tr><th>시각</th><th>톤</th><th>작업자가 다가간 것</th></tr>{vrows}</table></div>
<div class="card" style="margin-top:14px"><h2>점검표</h2><table><tr><th></th><th>지점</th><th>결과</th><th>대장</th></tr>{cps}</table></div>
</section><section>
<div class="card"><h2>조치 목록 (우선순위 순)</h2><table><tr><th>#</th><th>우선</th><th>판정</th><th>위치</th><th>조치</th><th>근거</th></tr>{_rows(rep)}</table></div>
{f'<div class="card" style="margin-top:14px"><h2>작업자 손동작 요청 (손가락 1~5 → 작업자 언어 안내)</h2><table><tr><th>시각</th><th>손가락</th><th>명령</th><th>언어</th><th>안내 (원문 / 한국어)</th></tr>{arows}</table></div>' if asst else ""}
{f'<div class="card" style="margin-top:14px">{_eval_html(evaluation)}</div>' if evaluation else ""}
<div class="card" style="margin-top:14px"><h2>에이전트 기록</h2><ul class="tl">{tl}</ul></div>
</section></main></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path
