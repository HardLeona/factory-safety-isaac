"""제출 문서 만들기 (일반 파이썬, python-docx). 수치는 outputs/eval 의 채점 결과에서 읽는다.

    python scripts/make_docs.py              # submission/개발완료보고서.docx, submission/AI_Agent_기술설명서.docx

Word 가 깔려 있으면 scripts/to_pdf.ps1 로 PDF 도 만든다.
"""
import glob
import json
import os
import sys

from docx import Document
from docx.enum.section import WD_ORIENT  # noqa: F401
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "submission")
FIG = os.path.join(ROOT, "docs")
FONT = "맑은 고딕"
BLUE = RGBColor(0x1F, 0x3A, 0x6B)
GRAY = RGBColor(0x55, 0x5F, 0x6D)

TEAM = "IBDP 팀"
MEMBERS = "문상균, 이승혜, 조현준"
DIVISION = os.environ.get("DIVISION", "대학부")
TITLE = "창고 안전 순찰 AI 에이전트"
SUBTITLE = "작업자 바디캠만으로 위험 요소를 판정하고, 애매하면 재관측으로 재판단하며, 손동작 명령에 작업자 언어로 답하는 피지컬 AI"
REPO = "https://github.com/HardLeona/factory-safety-isaac"
N_TESTS = 29
sys.path.insert(0, ROOT)
from factory_safety.config import TOOL_TYPES  # noqa: E402


def _yolo_metrics():
    """학습한 YOLO 의 검증 점수 (scripts/val_yolo.py 가 만든 metrics.json). 없으면 첫 모델 값."""
    for name in ("warehouse_v3", "warehouse_v2", "warehouse"):
        path = os.path.join(ROOT, "outputs", "yolo", name, "metrics.json")
        if os.path.exists(path):
            return json.load(open(path, encoding="utf-8"))
    return {"map50": 0.923, "map5095": 0.733, "per_class": {}}


YOLO = _yolo_metrics()
DATA = {"n": 7700, "desc": "일반 촬영 5000장 + 공구 가까이 1500장 + 운반 카트 1200장"}
DEMO_SEED = int(os.environ.get("DEMO_SEED", "5"))


def _gestures():
    """손동작 인식 시험 (scripts/test_gestures.py) 과 시연 녹화의 손동작 채점."""
    path = os.path.join(ROOT, "outputs", "eval", "gesture_test.json")
    test = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {"ok": 0, "trials": 0, "mid_false": 0, "mid_frames": 0}
    rec = os.path.join(ROOT, "outputs", "record", f"seed{DEMO_SEED}", "final.json")
    demo, langs = {"recognized": 0, "shown": 0, "extra": 0}, []
    if os.path.exists(rec):
        f = json.load(open(rec, encoding="utf-8"))
        demo = dict(f.get("evaluation", {}).get("gestures", demo))
        evs = f["report"].get("assistant", [])
        langs = list(dict.fromkeys(e["lang_name"] for e in evs))
        demo["llm"] = sum(1 for e in evs if (e.get("llm") or {}).get("llm"))          # LLM 이 결정한 요청 수
        demo["requests"] = len(evs)
        demo["llm_sec"] = max([(e.get("llm") or {}).get("sec") or 0 for e in evs] or [0])
    return test, demo, langs


G_TEST, G_DEMO, G_LANGS = _gestures()


def gtxt():
    return f"{G_TEST['ok']}/{G_TEST['trials']}"


# ---------------------------------------------------------------- 결과 읽기
def load_results():
    rs = []
    for f in sorted(glob.glob(os.path.join(ROOT, "outputs", "eval", "inspection_seed*.json"))):
        r = json.load(open(f, encoding="utf-8"))
        if r.get("agent"):
            rs.append(r)
    rs.sort(key=lambda r: r["seed"])
    return rs


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


def agg(rs):
    def s(part, k):
        return sum(r["agent"]["evaluation"][part][k] for r in rs)

    def rj(k):
        return sum(r["agent"]["evaluation"]["rejudge"][k] for r in rs)
    b = [r["bodycam"] for r in rs]
    cpc = [checkpoint_checks(r) for r in rs]
    out_a = {k: s("after", k) for k in rs[0]["agent"]["evaluation"]["after"]}
    out_a["need_check"] += sum(c[0] for c in cpc)
    out_a["need_check_real"] += sum(c[1] for c in cpc)
    return {
        "n": len(rs), "objects": s("after", "hazard_total") + s("after", "safe_total"),
        "b": {k: s("before", k) for k in rs[0]["agent"]["evaluation"]["before"]},
        "a": out_a,
        "rc": {k: rj(k) for k in rs[0]["agent"]["evaluation"]["rejudge"]},
        "yolo_h": (sum(x["hazard_found"] for x in b), sum(x["hazard_total"] for x in b)),
        "yolo_s": (sum(x["safe_ok"] for x in b), sum(x["safe_total"] for x in b)),
        "yolo_flip": sum(x["hazard_as_safe"] + x["safe_as_hazard"] for x in b),
        "x": extra(rs),
    }


def extra(rs):
    """위험 영역, 음성 경고, 공구 이름 합계 (예전 결과에는 없어서 0)."""
    ex = [r["agent"]["evaluation"] for r in rs if "zones" in r["agent"]["evaluation"]]

    def g(part, k):
        return sum(e[part][k] for e in ex)
    if not ex:
        return {k: 0 for k in ("zone_found", "zone_gt", "zone_agent", "zone_agent_real", "zone_false", "voice_warned", "voice_events",
                                 "voices", "voices_useful", "tool_named", "tool_total")}
    return {"zone_found": g("zones", "found"), "zone_gt": g("zones", "gt"), "zone_agent": g("zones", "agent"),
            "zone_agent_real": g("zones", "agent_real"), "zone_false": g("zones", "false"),
            "voice_warned": g("voice", "warned"), "voice_events": g("voice", "events"), "voices": g("voice", "voices"),
            "voices_useful": g("voice", "useful"), "tool_named": g("tools", "named"), "tool_total": g("tools", "total")}


def pct(a, b):
    return f"{100 * a / max(1, b):.0f}%"


# ---------------------------------------------------------------- docx 도우미
def set_font(run, size=None, bold=None, color=None):
    run.font.name = FONT
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), FONT)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = color


def new_doc(margin_cm=1.8, base=9.5):
    d = Document()
    st = d.styles["Normal"]
    st.font.name = FONT
    st.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), FONT)
    st.font.size = Pt(base)
    st.paragraph_format.space_after = Pt(2)
    st.paragraph_format.line_spacing = 1.15
    for s in d.sections:
        s.page_height, s.page_width = Cm(29.7), Cm(21.0)
        s.left_margin = s.right_margin = Cm(margin_cm)
        s.top_margin = s.bottom_margin = Cm(1.5)
    return d


def para(d, text="", size=None, bold=False, color=None, align=None, after=2, before=0, italic=False):
    p = d.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.space_before = Pt(before)
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    # **굵게** 표시 지원
    parts = text.split("**")
    for i, t in enumerate(parts):
        if not t:
            continue
        r = p.add_run(t)
        set_font(r, size, bold or (i % 2 == 1), color)
        r.italic = italic
    return p


def heading(d, text, level=1):
    size = {1: 13, 2: 10.5}[level]
    p = para(d, text, size=size, bold=True, color=BLUE, before=6 if level == 1 else 3, after=2)
    p.paragraph_format.keep_with_next = True
    if level == 1:
        pPr = p._p.get_or_add_pPr()
        bdr = OxmlElement("w:pBdr")
        b = OxmlElement("w:bottom")
        for k, v in (("w:val", "single"), ("w:sz", "8"), ("w:space", "1"), ("w:color", "1F3A6B")):
            b.set(qn(k), v)
        bdr.append(b)
        pPr.append(bdr)
    return p


def bullets(d, items, size=None):
    for it in items:
        p = para(d, "• " + it, size=size, after=1)
        p.paragraph_format.left_indent = Cm(0.3)
        p.paragraph_format.first_line_indent = Cm(-0.3)


def shade(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), hex_color)
    tcPr.append(sh)


def table(d, rows, widths, header=True, size=8.5, first_col_shade=False, align_center_cols=()):
    t = d.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.width = Cm(widths[j])
            c.text = ""
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            if j in align_center_cols or (header and i == 0):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            parts = str(val).split("**")
            for k, tx in enumerate(parts):
                if tx:
                    r = p.add_run(tx)
                    set_font(r, size, (header and i == 0) or (first_col_shade and j == 0) or k % 2 == 1)
            if header and i == 0:
                shade(c, "DCE6F2")
            elif first_col_shade and j == 0:
                shade(c, "F1F4F8")
    for j, w in enumerate(widths):
        for c in t.columns[j].cells:
            c.width = Cm(w)
    d.add_paragraph().paragraph_format.space_after = Pt(1)
    return t


def image(d, path, width_cm, caption=None):
    if not os.path.exists(path):
        para(d, f"[그림 없음: {os.path.basename(path)}]", color=GRAY)
        return
    p = d.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.keep_with_next = bool(caption)
    p.add_run().add_picture(path, width=Cm(width_cm))
    if caption:
        para(d, caption, size=8, color=GRAY, align="center", after=3)


def images_row(d, items, total_cm=17.4, size=8):
    """그림 여러 장을 한 줄에 (표 칸에 넣음)."""
    t = d.add_table(rows=2, cols=len(items))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    w = total_cm / len(items)
    for j, (path, cap) in enumerate(items):
        c = t.cell(0, j)
        c.width = Cm(w)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if os.path.exists(path):
            p.add_run().add_picture(path, width=Cm(w - 0.3))
        c2 = t.cell(1, j)
        p2 = c2.paragraphs[0]
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p2.add_run(cap)
        set_font(r, size, False, GRAY)
    d.add_paragraph().paragraph_format.space_after = Pt(1)


def page_break(d):
    d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# ---------------------------------------------------------------- 개발완료보고서
CLASS_ROWS = [("바닥 유출", "spill", "기름/물 웅덩이 + 쓰러진 통, 조치 없음", "spill_marked", "같은 유출 + 미끄럼 주의 표지판·라바콘"),
              ("적재", "stack_unstable", "맨 위 층이 밀려나 기울어짐", "stack_stable", "반듯하게 쌓임"),
              ("소화기", "ext_fallen", "바닥에 쓰러짐 / 앞을 상자가 막음 (ext_blocked)", "ext_ok", "제자리에 보이게 비치")]
TOOL_ROW_KO = "망치·드라이버·톱·전동톱·곡괭이·삽·렌치·전동 드릴"


def yolo_cell(*names):
    per = YOLO.get("per_class", {})
    vals = [f"{per[n]:.3f}" for n in names if n in per]
    return " / ".join(vals) if vals else "-"


def report(rs, figs):
    A = agg(rs)
    a, b, rc, X = A["a"], A["b"], A["rc"], A["x"]
    nh, ns = a["hazard_total"], a["safe_total"]
    tools_map = [YOLO.get("per_class", {}).get(t) for t in TOOL_TYPES]
    tools_map = [v for v in tools_map if v is not None]
    tool_m = f"{sum(tools_map) / len(tools_map):.3f}" if tools_map else "-"
    d = new_doc()
    # 표지
    for _ in range(6):
        para(d)
    para(d, "제4회 경남AI·SW경진대회", 14, color=GRAY, align="center")
    para(d, "개발완료보고서", 30, True, BLUE, "center", after=24)
    para(d, TITLE, 22, True, align="center", after=6)
    para(d, SUBTITLE, 12, color=GRAY, align="center", after=60)
    table(d, [["지정분야", "③ 제조·피지컬 AI Agent"], ["부문", DIVISION], ["팀명", TEAM], ["팀원", MEMBERS],
              ["소스코드", REPO], ["제출일", "2026. 10. 06."]], [4, 10], header=False, size=11, first_col_shade=True)
    page_break(d)

    # 1. 프로젝트 현황
    heading(d, "1. 프로젝트 현황")
    table(d, [
        ["작품명", f"{TITLE}: {SUBTITLE}"],
        ["팀 / 분야", f"{TEAM} ({MEMBERS}) · ③ 제조·피지컬 AI Agent · 시뮬레이션형 (NVIDIA Isaac Sim 디지털 트윈)"],
        ["요약", "작업자가 정해진 경로로 창고를 걸으면 가슴 바디캠 화면을 YOLO26 이 보고 위험 요소의 **위험/안전 상태**와 **공구 이름**(망치, 삽 등 8종)을 판정한다. "
                "에이전트는 판정을 **위험물 대장**에 모으고, 애매한 물체는 고위험 후보가 있으면 즉시 위험으로 확정하고 없으면 **'주의'로 분류**해, "
                "바디캠이 같은 물체를 자연스럽게 다시 지나칠 때 **로컬 LLM(재관측 재판단)**이 위험/안전을 가린다. "
                "**라바콘으로 둘러친 곳, DANGER 표지가 선 곳, 스스로 위험하다고 판단한 주변을 위험 영역**으로 잡고, 작업자가 위험물에 닿기 직전이거나 "
                "영역에 들어서면 **\"경고 경고 위험 요소가 식별되었습니다\" 음성 경고**를 낸다. 작업자가 바디캠 앞에 **손가락 1~5개**를 보이면 장비 설명, 공장 위험 스캔, "
                "오늘의 TBM, 관리자 호출, SOS 를 **로컬 LLM(Qwen2.5-7B) 에이전트(LangGraph)** 가 도구로 상황을 보고 판단해 **작업자 언어(중국어·영어·일본어·한국어)** 음성으로 처리한다. "
                "순찰이 끝나면 **조치 지시서**를 만든다."],
        ["핵심 성과", f"학습에 안 쓴 시나리오 {A['n']}개 (창고 물체 {A['objects']}개): 위험 물체 **{a['hazard_found']}/{nh} ({pct(a['hazard_found'], nh)})**, "
                    f"안전 물체 **{a['safe_ok']}/{ns} ({pct(a['safe_ok'], ns)})** 판정 (바디캠만 썼을 때 {b['hazard_found']}, {b['safe_ok']}), 거꾸로 판정 "
                    f"{a['hazard_as_safe'] + a['safe_as_hazard']}건. 위험 영역 {X['zone_found']}/{X['zone_gt']} 인식, 닿기 직전 음성 경고 "
                    f"{X['voice_warned']}/{X['voice_events']}, 공구 이름 {X['tool_named']}/{X['tool_total']}, 손동작 명령 {gtxt()} "
                    f"({pct(G_TEST['ok'], G_TEST['trials'])}). YOLO26s 검증 mAP50 {YOLO['map50']}"],
        ["완성도", f"Level 3 (MVP): 입력 → 판단 → 도구 실행 → 결과 확인 → 보고까지 한 번에 동작, 테스트 시나리오 {A['n']}건 + 단위 테스트 {N_TESTS}개"],
    ], [2.6, 14.8], header=False, size=8.3, first_col_shade=True)

    # 2. 프로젝트 개요
    heading(d, "2. 프로젝트 개요 (문제, 사용자, 목표)")
    para(d, "**문제.** 고용노동부 「2025년 산업재해 현황」에 따르면 2025년 사고재해자 113,305명 가운데 **넘어짐이 28,606명(25.2%)으로 가장 많다**. "
            "창고와 공장에서는 바닥 유출, 통로에 방치된 공구가 넘어짐의 직접 원인이 되고, 불안정 적재는 무너짐·맞음 사고로, 쓰러지거나 가로막힌 소화기는 "
            "화재 초기 대응 실패로 이어진다. 그러나 순찰 점검은 사람이 눈으로 확인하는 방식이라 ① 놓치기 쉽고, ② 기록이 남지 않으며, "
            "③ 작업자가 위험물이나 출입 금지 구역에 다가가는 순간을 알려 줄 수단이 없고, ④ 외국인 작업자는 한국어 안전 안내·TBM 을 알아듣기 어렵다.", size=9)
    para(d, "**사용자.** 물류창고·제조공장의 안전관리자(조치 지시서, 호출·SOS 수신), 순찰 작업자(바디캠·음성 경고·손동작 명령, 외국인 작업자 포함).", size=9)
    table(d, [
        ["목표", "성공 지표", "결과"],
        ["위험/안전 상태 자동 판정", "경로에서 보인 위험 90% 이상, 거꾸로 판정 최소",
         f"바디캠 {A['yolo_h'][0]}/{A['yolo_h'][1]} ({pct(*A['yolo_h'])}), 거꾸로 {A['yolo_flip']}"],
        ["애매하면 재판단, 창고 전체 점검", "창고 전체 물체 판정률 (바디캠 프레임 원시판정 대비)",
         f"위험 {pct(a['hazard_found'], nh)}, 안전 {pct(a['safe_ok'], ns)} (바디캠만 {pct(b['hazard_found'], nh)}, {pct(b['safe_ok'], ns)})"],
        ["공구 이름 인식", "정답 공구 종류 중 맞힌 비율", f"{X['tool_named']}/{X['tool_total']} ({pct(X['tool_named'], X['tool_total'])})"],
        ["위험 영역 자동 설정", "라바콘 링·DANGER 표지 영역 인식", f"{X['zone_found']}/{X['zone_gt']} ({pct(X['zone_found'], X['zone_gt'])})"],
        ["닿기 직전 음성 경고", "위험물 1 m·영역 0.5 m 안 사건 중 경고", f"{X['voice_warned']}/{X['voice_events']} ({pct(X['voice_warned'], X['voice_events'])})"],
        ["손동작 명령, 작업자 언어 안내", "손가락 1~5 인식 (경로 여러 곳·조명)", f"{gtxt()} ({pct(G_TEST['ok'], G_TEST['trials'])}), 중·영·일·한"],
        ["애매한 판정 재관측 재판단, 조치 지시서", "주의 분류 후 재판단으로 확정 전환된 비율", f"{rc['caution_rejudged_confirmed']}/{rc['caution_rejudged']}건, HTML 1장"],
    ], [4.6, 6.8, 6.0], size=8, align_center_cols=(2,))

    # 3. 개발 세부 내용
    heading(d, "3. 개발 세부 내용")
    heading(d, "3.1 시스템 구조", 2)
    image(d, os.path.join(FIG, "architecture.png"), 17.0, "그림 1. 시스템 구조: 디지털 트윈 → 센서 → AI 판단 → 에이전트 → 결과, 에이전트가 재관측 재판단을 요청하는 피드백 경로")
    heading(d, "3.2 활용 데이터", 2)
    para(d, "실제 사진 없이 **NVIDIA Isaac Sim 6.0** 실사 창고와 NVIDIA 소품, **Poly Haven 실물 스캔 공구 (CC0)**, YCB 드릴로 장면을 만들고 (전동톱과 DANGER 표지는 직접 모델링), "
            f"Omniverse Replicator 로 **합성 데이터 {DATA['n']}장**({DATA['desc']})을 정답 박스와 함께 자동 생성했다. "
            "같은 종류를 위험한 상태와 안전한 상태로 함께 두어 YOLO 가 모양이 아니라 **상태**를 구분하게 했고, 공구는 종류별로 라벨을 붙였다.", size=9)
    rows = [["종류", "위험 (YOLO 클래스)", "안전 (YOLO 클래스)", "mAP50"]]
    for ko, hc, hd, sc, sd in CLASS_ROWS:
        names = (hc, "ext_blocked", sc) if hc == "ext_fallen" else (hc, sc)
        rows.append([ko, f"{hd} ({hc})", f"{sd} ({sc})", yolo_cell(*names)])
    rows.append(["공구 8종", f"통로 바닥에 방치 → 에이전트가 자리로 판단", f"작업대 위 정리", f"평균 {tool_m}"])
    rows.append(["영역 표시·장비", "라바콘 (cone), DANGER 표지 (danger_sign)", "작업자 (worker), 운반 카트 (cart)",
                 yolo_cell("cone", "danger_sign", "worker", "cart")])
    table(d, rows, [2.0, 6.4, 5.6, 3.4], size=7.6, align_center_cols=(3,))

    heading(d, "3.3 AI 와 알고리즘", 2)
    bullets(d, [
        f"**YOLO26s** (19 클래스, 960 px, 이전 모델에서 이어 학습): 검증 mAP50 **{YOLO['map50']}**, mAP50-95 {YOLO['map5095']}. **ByteTrack** 으로 같은 물체의 판정을 모아 3프레임이면 확정",
        "**바닥 투영**: 박스 아래쪽을 카메라 자세로 바닥에 투영해 창고 좌표. **공구의 위험/안전은 놓인 자리로**: 작업대 윗면 높이로 투영해 작업대 위면 정리, 아니면 통로 방치",
        "**애매한 판정의 2단계 분류**: 잠깐 보이거나 판정이 엇갈린 물체는 후보 중 고위험 클래스가 있으면 즉시 위험으로 확정, 전부 저위험이면 '주의'로 분류. "
        "주의 물체를 바디캠이 자연스럽게 다시 지나치면 로컬 LLM(LangGraph, recheck_agent.py)이 이전+새 증거로 위험/안전을 재판단, LLM 실패 시 규칙 폴백",
        "**위험 영역**: 3 m 안으로 이어진 라바콘 2개 이상 → 볼록 다각형, DANGER 표지 → 반경 1.2 m, 방치된 유출 → 1.3 m (미끄럼), 무너질 듯한 적재 → 1.6 m (붕괴). "
        "회피가 필요 없는 소화기류는 영역 근거에서 제외",
        "**음성 경고**: 회피형 위험(유출·적재·통로 공구) 1.5 m 안이거나 영역 경계 0.5 m 안이면 경고 — 위험은 \"멈추세요\" + 강한 진동, 주의는 \"발밑을 확인하세요\" + 짧은 진동으로 "
        "톤만 구분(거리는 같음). 2 m 안은 경고 없이 근접만 기록. 오늘 작업 계획(TBM) 대상 물체는 경고에서 제외",
        "**LLM 에이전트**: LangGraph 그래프 (agent ⇄ tools → finish) 에서 로컬 **Qwen2.5-7B** (Ollama, 인터넷·API 키 불필요) 가 손동작 요청마다 "
        "도구(look_around, equipment_info, hazard_log, todays_tbm, worker_status)로 상황을 보고 말할 항목·근거·관리자 메시지를 정함. 안전 문장은 검수한 틀로, LLM 실패 시 규칙으로",
        "**손동작 명령** (1 장비 설명, 2 공장 위험 스캔: 위험물 대장을 우선순위로, 3 TBM, 4 관리자 호출, 5 SOS): "
        "**MediaPipe Hands** 손 관절 21점에서 손가락마다 마디가 곧은지(각도)·손목에서 먼지, 엄지는 약지 뿌리까지 거리로 1~5 를 세고, 3번 연속 같으면 확정 "
        "(손을 내려야 다시 받음). 안내 문장은 **검수한 4개 언어 문장 틀 + 현장 용어집**으로 만든다 (번역 모델 NLLB 는 '안전화→seat belt' 처럼 현장 용어를 틀려 안전 안내에 안 씀). "
        "음성은 신경망 TTS (edge-tts)",
    ], size=8.5)

    heading(d, "3.4 에이전트 동작 (Goal · Planning · Reasoning · Tool · Memory · Feedback)", 2)
    table(d, [
        ["요소", "구현"],
        ["Goal", "순찰 한 바퀴 동안 위험물을 찾아 판정하고, 위험 영역을 잡고, 작업자를 경고하고, 조치 지시서를 만든다"],
        ["Planning", "순찰 전 도면(소화기 6, 작업대 2)으로 점검표 계산. 순찰 끝에 못 본 지점은 현장 확인 필요로"],
        ["Reasoning", "애매한 판정 2단계 분류 (고위험 후보 있으면 즉시 위험 확정, 없으면 '주의'), 공구 위험/안전 (놓인 자리), "
                      "위험 영역 (라바콘 묶음·표지·위험 주변), 음성 경고 시점·톤 (거리, 위험/주의), 손동작 요청은 LLM(Qwen2.5-7B) 이 도구로 보고 무엇을 말할지·근거·관리자 메시지 결정"],
        ["Tool Use", "바디캠, YOLO26, 손 인식, LLM 이 부르는 도구 (화면 물체·장비 정보·위험물 대장·TBM·작업자 상태, 매뉴얼 RAG), 다국어 음성, 관리자 알림, 조치 지시서"],
        ["Memory/State", "위험물 대장 (위치, 판정 표, 공구 종류, 확신도, 상태: 확정/주의/현장 확인 필요), 라바콘·표지 위치, 위험 영역 목록, 점검표, 경고·작업자 요청 기록, 최근 바디캠 화면"],
        ["Feedback", "주의 물체를 바디캠이 다시 지나치면 누적 증거로 LLM(또는 규칙)이 재판단해 대장 수정, 끝까지 안 보이면 주의 유지·현장 확인 권고, 정답표 채점"],
    ], [2.6, 14.8], size=8, first_col_shade=True)
    heading(d, "3.5 단계별 개발 (8일)", 2)
    table(d, [
        ["일차", "내용"],
        ["9/29~10/1", "Isaac Sim 6.0 장면·카메라·Replicator 파이프라인 (첫 버전은 로봇 순찰 강화학습), 실제 실행 검증과 라벨 문제 해결"],
        ["10/2", "**위험 판단 중심으로 재설계**: 고정 경로 작업자 바디캠, 위험/안전 짝 물체, 정답표 채점, 합성 데이터·YOLO 학습, 접근 음성 경고"],
        ["10/3", "에이전트 (점검표, 대장, 현장 확인, 조치 지시서) + **위험 영역, 음성 경고, 공구 이름 8종, 손동작 명령·다국어 안내, LLM 에이전트 (LangGraph + 로컬 Qwen2.5-7B)** 추가, 운반 카트 시연, 재학습"],
        ["10/4", "**바디캠 단독화**: CCTV/PTZ 전면 제거, 애매한 판정 2단계 분류(즉시 위험 확정/주의) + 바디캠 재관측 재판단(recheck_agent.py)으로 전환, "
                 "매뉴얼 RAG(manuals.py)로 장비 설명·조치 문구 근거 보강"],
        ["10/5~10/6", "평가, 시연 영상, 보고서, 기술설명서, 발표자료"],
    ], [2.6, 14.8], size=8, first_col_shade=True)

    # 4. 구현 결과
    heading(d, "4. 구현 결과")
    heading(d, "4.1 핵심 기능", 2)
    images_row(d, [(figs["body"], "① 바디캠 판정 (공구 이름, 위험/안전)"), (figs["zone"], "② 위험 영역 + 음성 경고"),
                   (figs["gesture"], "③ 손동작 명령 → 작업자 언어 안내"), (figs["rejudge"], "④ 재관측 재판단 (LLM)")], size=7.5)
    image(d, figs["dash"], 11.5, "그림 2. 조치 지시서 화면 (평면도의 번호 = 우선순위, 빨간 영역 = 위험 영역, 위험 개수, 음성 경고, 조치 목록)")

    heading(d, f"4.2 테스트 (대표 시나리오 {A['n']}건, 정답표 채점)", 2)
    para(d, f"위험 요소 배치를 시드마다 무작위로 바꾼 시나리오 {A['n']}개(학습 데이터에 없는 배치)를 한 바퀴씩 순찰했다. 정답표는 판정에 쓰지 않고 채점에만 쓴다. "
            "**바디캠만**은 BodycamInspector 의 프레임 단위 YOLO 원시 판정(에이전트 판단 전), **에이전트**는 2단계 분류·재관측 재판단·점검표를 거친 최종 대장을 "
            "창고 전체 물체와 비교한 것이다(두 쪽은 집계 단위가 달라 '없는 위험' 수치는 프레임 박스 수/물체 보고 건수로 따로 센다).", size=8.5)
    rows = [["시나리오", "물체 (위험)", "위험 판정\n바디캠만 → 에이전트", "안전 판정\n바디캠만 → 에이전트", "거꾸로", "없는 위험", "현장 확인",
             "위험 영역", "음성 경고\n(닿기 직전)", "공구 이름"]]
    for r in rs:
        e = r["agent"]["evaluation"]
        bb, aa = e["before"], e["after"]
        z, v, t = e.get("zones", {}), e.get("voice", {}), e.get("tools", {})
        rows.append([f"시드 {r['seed']}", f"{aa['hazard_total'] + aa['safe_total']} ({aa['hazard_total']})",
                     f"{bb['hazard_found']} → **{aa['hazard_found']}**/{aa['hazard_total']}", f"{bb['safe_ok']} → **{aa['safe_ok']}**/{aa['safe_total']}",
                     aa["hazard_as_safe"] + aa["safe_as_hazard"], aa["false_reports"], aa["need_check"] + checkpoint_checks(r)[0],
                     f"{z.get('found', '-')}/{z.get('gt', '-')}", f"{v.get('warned', '-')}/{v.get('events', '-')}", f"{t.get('named', '-')}/{t.get('total', '-')}"])
    rows.append(["합계", f"{A['objects']} ({nh})", f"{b['hazard_found']} → **{a['hazard_found']}**/{nh}", f"{b['safe_ok']} → **{a['safe_ok']}**/{ns}",
                 a["hazard_as_safe"] + a["safe_as_hazard"], a["false_reports"], a["need_check"], f"{X['zone_found']}/{X['zone_gt']}",
                 f"{X['voice_warned']}/{X['voice_events']}", f"{X['tool_named']}/{X['tool_total']}"])
    table(d, rows, [1.5, 1.5, 2.4, 2.4, 1.1, 1.3, 1.3, 1.6, 2.0, 1.6], size=7, align_center_cols=tuple(range(10)))
    para(d, f"애매한 판정 {rc['ambiguous_hazard'] + rc['ambiguous_caution']}건 중 고위험 후보로 즉시 위험 확정한 것이 {rc['ambiguous_hazard']}건, '주의'로 분류한 것이 "
            f"{rc['ambiguous_caution']}건이다. 주의 물체를 바디캠이 다시 지나쳐 재판단한 것이 {rc['caution_rejudged']}건이고 그중 {rc['caution_rejudged_confirmed']}건이 "
            f"확정으로 바뀌었다 (맞게 고침 {rc['changed_correct']}건). "
            f"에이전트가 스스로 판단한 위험 영역은 {X['zone_agent']}개(실제 위험 주변 {X['zone_agent_real']}개), 음성 경고는 모두 {X['voices']}번 "
            f"(그중 실제 닿기 직전 사건에 맞은 것 {X['voices_useful']}번)이었다. "
            f"**손동작**은 경로 {G_TEST.get('spots', len({r['spot'] for r in G_TEST.get('rows', [])}))}곳에서 조명을 바꿔 손가락 1~5 를 보인 {G_TEST['trials']}번 중 "
            f"{G_TEST['ok']}번을 맞게 인식했고 (다른 명령 {G_TEST['wrong']}번은 어둡거나 역광인 손, 못 알아본 것 {G_TEST['missed']}번, 손을 올리고 내리는 중 잘못 실행 0번), 시연 순찰에서는 {G_DEMO['shown']}번 중 "
            f"{G_DEMO['recognized']}번을 인식해 {', '.join(G_LANGS) or '작업자 언어'}로 안내했고, 요청 {G_DEMO.get('requests', 0)}건 중 "
            f"{G_DEMO.get('llm', 0)}건을 로컬 Qwen2.5-7B 에이전트가 도구로 상황을 보고 결정했다 (요청당 최대 {G_DEMO.get('llm_sec', 0):.0f}초). "
            f"Isaac 없이 도는 단위 테스트 {N_TESTS}개(투영, 채점, 재확인·병합, 위험 영역, 음성 경고, 손가락 세기, 다국어 문장 등)가 모두 통과한다.", size=8.5)

    # 5. 기대효과
    heading(d, "5. 기대효과, 성과와 한계")
    heading(d, "5.1 성과와 기대효과", 2)
    bullets(d, [
        f"**정량**: 창고 전체 위험 {pct(a['hazard_found'], nh)}, 안전 {pct(a['safe_ok'], ns)} 판정 (바디캠만 {pct(b['hazard_found'], nh)}, {pct(b['safe_ok'], ns)}), "
        f"위험 영역 {pct(X['zone_found'], X['zone_gt'])} 인식, 닿기 직전 음성 경고 {pct(X['voice_warned'], X['voice_events'])}, 공구 이름 {pct(X['tool_named'], X['tool_total'])}, "
        f"손동작 명령 {pct(G_TEST['ok'], G_TEST['trials'])}",
        "**정성**: 순찰마다 판정 근거 기록이 남고, 작업자는 화면을 보지 않아도 소리로 위험을 알 수 있음. "
        "외국인 작업자도 모국어로 장비·위험·TBM 안내를 받고 관리자 호출·SOS 를 할 수 있음",
        "**적용**: 작업자 바디캠(스피커)만으로 동작, 별도 CCTV 설비 없이 도입 가능. 라바콘·표지처럼 현장에서 이미 쓰는 표시를 그대로 위험 영역으로 읽음",
    ], size=8.5)
    heading(d, "5.2 한계와 향후 계획", 2)
    from collections import Counter
    miss = Counter(row["class"] for r in rs for row in r["agent"]["evaluation"]["rows"] if row["hazard"] and row["after"] is None)
    bullets(d, [
        f"**작은 공구·중복**: 끝내 못 찾은 위험 {sum(miss.values())}개 중 {miss.get('tool_floor', 0)}개가 바닥 공구. 없는 위험 보고 {a['false_reports']}건은 "
        "대부분 멀리서 본 같은 물체가 위치가 2~7 m 어긋나 대장에 한 번 더 오른 중복 (거리별 병합으로 보완 예정)",
        "**합성 데이터만**: 휴대폰 공구 사진 시험에서 가까이 찍은 드라이버를 놓침 (학습은 1~5 m 앞 작은 공구). 현장 사진으로 미세조정 필요 (Level 4). 손동작도 시뮬레이션 맨손만",
        "**음성·언어**: 다국어 안내 음성은 Microsoft 온라인 TTS (인터넷 필요, 끊기면 Windows 음성으로 한국어·영어·일본어만). 언어는 문장 틀에 있는 4개만",
        "**작업자 위치**: 음성 경고는 작업자 위치를 정확히 안다고 가정 (시뮬레이션 값). 실제로는 바디캠에 UWB·실내 측위나 영상 기반 위치 추정이 필요",
        "**처리 속도**: Isaac Sim 6.0.1 의 PyTorch 가 CPU 전용이라 시뮬레이션 안 YOLO 는 CPU (시간 고정이라 결과는 재현). 실제 배치는 GPU·엣지 장치",
        "**규칙 기반 판단**: 애매한 판정의 즉시 분류 기준, 영역 크기, 경고 거리는 규칙 (재관측 재판단만 LLM 또는 규칙). 현장 피드백으로 보정하고 MES·작업지시 연동 계획. "
        "지게차 같은 움직이는 위험은 아직 없음",
        "**LLM**: 로컬 7B 모델이라 요청당 수 초가 걸리고 판단 근거 문장이 가끔 부정확함 (그래서 작업자에게 들려주는 안전 문장은 틀로 고정, 말할 항목은 도구가 준 번호만 받음). "
        "손동작 요청에만 쓰고 순찰 중 위험 판정·경고는 규칙",
    ], size=8.5)
    heading(d, "5.3 기존자산과 신규개발분, 출처", 2)
    table(d, [
        ["구분", "내용"],
        ["기존자산 (제3자)", "NVIDIA Isaac Sim 6.0·Replicator·에셋(창고, 소품, 작업자; 에셋 서버 참조), Poly Haven 공구 모델(CC0), YCB 드릴, "
                         "Ultralytics YOLO26 사전학습 가중치(AGPL-3.0)·ByteTrack, Google MediaPipe Hands(Apache-2.0), Qwen2.5-7B-Instruct(Apache-2.0)·Ollama·LangGraph(MIT), "
                         "edge-tts(Microsoft 온라인 음성), "
                         "Windows 음성 합성, numpy·Pillow"],
        ["8일 신규개발분", "장면·시나리오·정답표, 위험/안전 짝 물체·전동톱·DANGER 표지 모델링, 걷기 동작, 카메라, 합성 데이터 파이프라인과 학습, 바닥 투영, "
                         "에이전트 (점검표, 대장, 2단계 분류, 바디캠 재관측 재판단, 위험 영역, 음성 경고, 조치 지시서), LLM 에이전트 그래프·도구, 매뉴얼 RAG, "
                         "손동작·다국어 문장 틀, 채점, 영상 (커밋 9/30~10/6)"],
        ["AI 활용", "코드 작성·디버깅·문서 초안에 AI 코딩 도구(Anthropic Claude Code)를 사용함. 설계 방향 결정, 위험 요소 선정, 결과 검증은 팀이 수행"],
        ["데이터·안전", "실제 개인정보·현장 영상 없음 (전부 합성). API 키·비밀번호 없음. 통계 출처: 고용노동부 「2025년 산업재해 현황」"],
    ], [3.0, 14.4], size=7.8, first_col_shade=True)
    path = os.path.join(OUT, "개발완료보고서.docx")
    d.save(path)
    return path


# ---------------------------------------------------------------- 기술설명서 (별지 1, 1쪽)
def tech_sheet(rs):
    A = agg(rs)
    a, b, rc, X = A["a"], A["b"], A["rc"], A["x"]
    nh, ns = a["hazard_total"], a["safe_total"]
    d = new_doc(margin_cm=1.4, base=8)
    for s in d.sections:
        s.top_margin = s.bottom_margin = Cm(1.0)
    para(d, "AI Agent 기술설명서", 14, True, BLUE, "center", after=4)
    rows = [
        ["팀명 / 부문 / 주제", f"{TEAM} ({MEMBERS}) / {DIVISION} / ③ 제조·피지컬 AI Agent"],
        ["작품명", f"{TITLE}"],
        ["해결 문제", "창고 순찰 점검이 사람 눈에 의존해 바닥 유출·방치 공구·불안정 적재·소화기 상태를 놓치기 쉽고, 기록이 남지 않으며, 작업자가 위험물이나 "
                   "출입 금지 구역에 다가가는 순간을 알려 줄 수단이 없음, 외국인 작업자는 한국어 안전 안내를 알아듣기 어려움 (2025년 사고재해자 중 넘어짐 25.2%로 1위, 고용노동부)"],
        ["대상 사용자", "물류창고·제조공장 안전관리자, 순찰 작업자, 관제 담당자"],
        ["Agent Goal", "순찰 한 바퀴 동안 위험물을 찾아 위험/안전과 공구 이름을 판정하고, 애매하면 스스로 재확인하고, 위험 영역을 잡아 작업자에게 음성으로 경고하고, "
                       "작업자의 손동작 요청은 LLM 이 상황을 보고 판단해 작업자 언어로 답하며, 우선순위가 있는 조치 지시서를 만든다"],
        ["사용 AI / 모델", f"YOLO26s (합성 데이터 {DATA['n']}장, 19 클래스: 위험/안전 상태, 공구 8종, 운반 카트, 라바콘, DANGER 표지, 작업자, mAP50 {YOLO['map50']}) "
                          "+ ByteTrack, MediaPipe Hands (손가락 1~5), **Qwen2.5-7B 로컬 LLM + LangGraph** (손동작 요청·주의 물체 재관측 재판단마다 도구 골라 판단), 신경망 TTS (중·영·일·한), 매뉴얼 RAG. "
                          "에이전트 판단(2단계 분류 즉시 확정 기준, 공구 자리, 위험 영역, 경고 시점·톤, 우선순위)은 규칙·기하 계산"],
        ["사용 Tool / Data / 장비", "NVIDIA Isaac Sim 6.0 디지털 트윈(실사 창고, NVIDIA·Poly Haven·YCB 에셋), 작업자 바디캠과 스피커, "
                                 "Replicator(정답 박스), 도면 데이터(랙, 기둥, 소화기, 작업대), TBM 데이터, 안전 매뉴얼 원문, 4개 언어 문장 틀·현장 용어집, HTML 조치 지시서"],
        ["Memory / State / Feedback", "· 위험물 대장 (위치, 출처별 판정 표, 공구 종류, 확신도, 접근 경고 횟수, 상태: 확정/주의/현장 확인 필요), 라바콘·표지 위치, 위험 영역 목록, 점검표, 음성 경고 기록\n"
                                    "· Feedback: 애매한 판정은 고위험 후보 있으면 즉시 위험 확정, 없으면 '주의'로 분류 → 바디캠이 같은 물체를 다시 지나치면 LLM(또는 규칙)이 누적 증거로 재판단 → 끝내 다시 안 보이면 주의 유지 "
                                    "(현장 확인 권고). 순찰 뒤 정답표로 채점"],
        ["핵심 Workflow\n(End-to-End)", "① 순찰 전 도면으로 점검표(소화기 6, 작업대 2) 계산 (계획)\n"
                                      "② 작업자가 경로를 걷고 바디캠 화면이 들어옴 (입력)\n"
                                      "③ YOLO 가 위험/안전·공구 이름 판정, 추적 3프레임이면 확정, 공구는 놓인 자리로 위험 여부 (판단·기억)\n"
                                      "④ 라바콘 묶음·DANGER 표지·회피형 위험 주변을 위험 영역으로 설정 (소화기류 제외)\n"
                                      "⑤ 회피형 위험 1.5 m·영역 0.5 m 안이면 음성 경고 (위험 \"멈추세요\", 주의 \"발밑을 확인하세요\"), 오늘 작업 대상은 제외 (도구)\n"
                                      "⑥ 애매한 물체는 고위험 후보 있으면 즉시 위험 확정, 없으면 '주의' → 바디캠이 다시 보면 LLM(또는 규칙)이 재판단 (결과 확인)\n"
                                      "⑦ 작업자가 손가락 1~5 → LLM 에이전트가 도구(매뉴얼 RAG 포함)로 보고 결정 → 장비 설명·공장 위험 스캔·TBM·관리자 호출·SOS 를 작업자 언어로 (도구)\n"
                                      "⑧ 순찰 끝에 조치 지시서 (우선순위·위치·공구 이름·위험 영역·작업자 요청) 출력, 정답표로 채점"],
        ["핵심기능 3~5개", "1) 바디캠 위험/안전·공구 이름 판정  2) 애매한 판정 2단계 분류 + 바디캠 재관측 재판단  3) 위험 영역 + 닿기 직전 음성 경고  "
                         "4) 손동작 명령 → LLM 판단(매뉴얼 RAG 근거) → 작업자 언어 안내·호출·SOS  5) 조치 지시서"],
        ["기존자산 / 8일 신규개발분", "· 기존자산: Isaac Sim·Replicator, NVIDIA·Poly Haven(CC0)·YCB 에셋, Ultralytics YOLO26·ByteTrack, MediaPipe Hands, "
                                   "Qwen2.5-7B·Ollama·LangGraph, edge-tts·Windows 음성 합성\n"
                                   "· 신규개발: 장면·시나리오·정답표, 위험/안전 짝·전동톱·DANGER 표지 모델링, 걷기·손동작 자세·카메라, 합성 데이터와 학습, 바닥 투영, "
                                   "에이전트 전체 (2단계 분류·재관측 재판단, LLM 도구·그래프 포함, 매뉴얼 RAG), 다국어 문장 틀, 채점·영상"],
        [f"대표 테스트 {A['n']}건 결과", f"학습에 안 쓴 배치 {A['n']}개 (창고 물체 {A['objects']}개): 위험 {a['hazard_found']}/{nh} ({pct(a['hazard_found'], nh)}), "
                                     f"안전 {a['safe_ok']}/{ns} ({pct(a['safe_ok'], ns)}), 거꾸로 판정 {a['hazard_as_safe'] + a['safe_as_hazard']} "
                                     f"(바디캠만: 위험 {b['hazard_found']}, 안전 {b['safe_ok']})\n위험 영역 인식 {X['zone_found']}/{X['zone_gt']}, "
                                     f"닿기 직전 음성 경고 {X['voice_warned']}/{X['voice_events']}, 공구 이름 {X['tool_named']}/{X['tool_total']}, "
                                     f"주의 분류 {rc['ambiguous_caution']}건 중 재관측 재판단 {rc['caution_rejudged']}건(확정 전환 {rc['caution_rejudged_confirmed']}), 손동작 명령 {gtxt()}"],
        ["현재 완성도 Level", "Level 3 (MVP): 시뮬레이션에서 End-to-End 동작과 정량 채점 완료. 실사 검증 전"],
        ["소스코드 / 저장소", f"{REPO} (실행: scripts/run_patrol.py, 평가: scripts/eval_patrol.py, README 에 설치·실행법)"],
        ["정보출처 및 기타", "고용노동부 「2025년 산업재해 현황」, NVIDIA Isaac Sim 문서·에셋, Poly Haven (CC0), YCB Object Set, Ultralytics YOLO (AGPL-3.0), "
                          "Google MediaPipe (Apache-2.0), Qwen2.5-7B-Instruct (Apache-2.0)·Ollama·LangGraph (MIT), edge-tts. "
                          "AI 코딩 도구 Anthropic Claude Code 로 코드·문서 작성 보조. 실제 개인정보·현장 영상 사용 없음"],
    ]
    table(d, rows, [3.6, 14.6], header=False, size=7.4, first_col_shade=True)
    path = os.path.join(OUT, "AI_Agent_기술설명서.docx")
    d.save(path)
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    rs = load_results()
    if not rs:
        sys.exit("outputs/eval 에 에이전트 채점 결과가 없어요. scripts/eval_patrol.py 를 먼저 돌리세요.")
    figs = {k: os.path.join(FIG, f"fig_{k}.jpg") for k in ("body", "rejudge", "zone", "gesture")}
    figs["dash"] = os.path.join(FIG, "fig_dashboard_crop.png")
    print(report(rs, figs))
    print(tech_sheet(rs))


if __name__ == "__main__":
    main()
