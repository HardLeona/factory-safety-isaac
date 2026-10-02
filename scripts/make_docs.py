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
SUBTITLE = "작업자 바디캠과 CCTV로 위험 요소를 판정·재확인하고 조치 지시서를 만드는 피지컬 AI"
REPO = "https://github.com/HardLeona/factory-safety-isaac"
N_TESTS = 19
YOLO = {"map50": 0.923, "map5095": 0.733, "per": [("spill", 0.884), ("spill_marked", 0.962), ("tool_floor", 0.865),
        ("tool_stored", 0.906), ("stack_unstable", 0.959), ("stack_stable", 0.954), ("ext_fallen", 0.912),
        ("ext_blocked", 0.898), ("ext_ok", 0.926), ("worker", 0.965)]}


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

    def rc(k):
        return sum(r["agent"]["evaluation"]["recheck"][k] for r in rs)
    c = [r["cctv"] for r in rs if r.get("cctv")]
    b = [r["bodycam"] for r in rs]
    med = sorted(x["dist_err_median"] for x in c if x.get("dist_err_median") is not None)
    wmed = sorted(x["worker_err_median"] for x in c if x.get("worker_err_median") is not None)
    cpc = [checkpoint_checks(r) for r in rs]
    out_a = {k: s("after", k) for k in rs[0]["agent"]["evaluation"]["after"]}
    out_a["need_check"] += sum(c[0] for c in cpc)
    out_a["need_check_real"] += sum(c[1] for c in cpc)
    return {
        "n": len(rs), "objects": s("after", "hazard_total") + s("after", "safe_total"),
        "b": {k: s("before", k) for k in rs[0]["agent"]["evaluation"]["before"]},
        "a": out_a,
        "rc": {k: rc(k) for k in rs[0]["agent"]["evaluation"]["recheck"]},
        "ev": (sum(x["events_detected"] for x in c), sum(x["events_visible"] for x in c), sum(x["events"] for x in c),
               sum(x["false_alert_episodes"] for x in c)),
        "dist": med[len(med) // 2] if med else None, "werr": wmed[len(wmed) // 2] if wmed else None,
        "yolo_h": (sum(x["hazard_found"] for x in b), sum(x["hazard_total"] for x in b)),
        "yolo_s": (sum(x["safe_ok"] for x in b), sum(x["safe_total"] for x in b)),
        "yolo_flip": sum(x["hazard_as_safe"] + x["safe_as_hazard"] for x in b),
    }


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
def report(rs, figs):
    A = agg(rs)
    a, b, rc = A["a"], A["b"], A["rc"]
    nh, ns = a["hazard_total"], a["safe_total"]
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
        ["요약", "작업자가 정해진 경로로 창고를 걸으면 가슴 바디캠 화면을 YOLO26 이 보고 위험 요소 4종의 **위험/안전 상태**를 판정한다. "
                "에이전트는 판정을 **위험물 대장**에 모으고, 잠깐 보이고 지나쳤거나 판정이 엇갈린 물체와 순찰 중 못 본 점검 지점을 골라 "
                "**볼 수 있는 CCTV 를 선택해 확대(PTZ)로 다시 판정**한다. 못 찾으면 다른 CCTV 로 다시 하고, 그래도 안 되면 사람에게 현장 확인을 넘긴다. "
                "CCTV 3대는 작업자와 위험물 사이 거리를 재서 2 m 안이면 경고하고, 순찰이 끝나면 우선순위와 조치 방법을 담은 **조치 지시서**를 만든다."],
        ["핵심 성과", f"바디캠 YOLO 판정: 경로에서 보인 위험 물체 **{A['yolo_h'][0]}/{A['yolo_h'][1]}**, 안전 물체 **{A['yolo_s'][0]}/{A['yolo_s'][1]}**. "
                    f"에이전트 최종 대장, 학습에 안 쓴 시나리오 {A['n']}개 (창고 물체 {A['objects']}개): 위험 물체 **{a['hazard_found']}/{nh} ({pct(a['hazard_found'], nh)})**, "
                    f"안전 물체 **{a['safe_ok']}/{ns} ({pct(a['safe_ok'], ns)})** 판정 (바디캠만 썼을 때 {b['hazard_found']}/{nh}, {b['safe_ok']}/{ns}). "
                    f"위험↔안전 거꾸로 판정 {a['hazard_as_safe'] + a['safe_as_hazard']}건. CCTV 접근 경고 {A['ev'][0]}/{A['ev'][1]}건, 거리 오차 중앙값 {A['dist']:.2f} m"],
        ["완성도", f"Level 3 (MVP): 입력 → 판단 → 도구 실행 → 결과 확인 → 보고까지 한 번에 동작, 테스트 시나리오 {A['n']}건 + 단위 테스트 {N_TESTS}개"],
    ], [2.6, 14.8], header=False, size=8.5, first_col_shade=True)

    # 2. 프로젝트 개요
    heading(d, "2. 프로젝트 개요 (문제, 사용자, 목표)")
    para(d, "**문제.** 고용노동부 「2025년 산업재해 현황」에 따르면 2025년 재해자는 147,130명이고, 사고재해자 113,305명 가운데 "
            "**넘어짐이 28,606명(25.2%)으로 가장 많다**. 창고와 공장에서는 바닥 유출, 통로에 방치된 공구·자재가 넘어짐의 직접 원인이 되고, "
            "불안정 적재는 무너짐·맞음 사고로, 쓰러지거나 가로막힌 소화기는 화재 초기 대응 실패로 이어진다. "
            "그러나 순찰 점검은 사람이 눈으로 확인하는 방식이라 ① 놓치기 쉽고, ② 무엇을 어디서 봤는지 기록이 남지 않으며, "
            "③ 작업자가 위험물에 다가가는 순간(아차사고)을 알 수 없다.")
    para(d, "**사용자.** 물류창고·제조공장의 안전관리자(조치 지시서를 받아 조치), 순찰 작업자(바디캠 착용, 현장 확인 요청 수행), 관제 담당자(CCTV 접근 경고 확인).")
    table(d, [
        ["목표", "성공 지표", "결과"],
        ["위험 요소 4종의 위험/안전 상태 자동 판정", "경로에서 보인 위험 물체 90% 이상, 위험↔안전 거꾸로 판정 최소",
         f"바디캠 {A['yolo_h'][0]}/{A['yolo_h'][1]} ({pct(*A['yolo_h'])}), 거꾸로 {A['yolo_flip']}건"],
        ["경로에서 안 보이는 곳까지 창고 전체 점검", "창고 전체 물체 기준 판정률 (바디캠만 대비)",
         f"위험 {pct(a['hazard_found'], nh)}, 안전 {pct(a['safe_ok'], ns)} (바디캠만 {pct(b['hazard_found'], nh)}, {pct(b['safe_ok'], ns)})"],
        ["애매한 판정은 스스로 재확인", "재확인으로 바디캠만 썼을 때보다 판정 물체 증가",
         f"위험 +{a['hazard_found'] - b['hazard_found']}, 안전 +{a['safe_ok'] - b['safe_ok']}"],
        ["작업자-위험물 접근 경고", "CCTV 에 보인 접근 사건 경고율, 거리 오차", f"{A['ev'][0]}/{A['ev'][1]}건, {A['dist']:.2f} m"],
        ["조치 지시서 자동 생성", "순찰 1회마다 우선순위·위치·조치 포함 보고", "HTML 1장 (평면도·점검표·기록)"],
    ], [5.6, 6.8, 5.0], size=8.5, align_center_cols=(2,))

    # 3. 개발 세부 내용
    heading(d, "3. 개발 세부 내용")
    heading(d, "3.1 시스템 구조", 2)
    image(d, os.path.join(FIG, "architecture.png"), 17.2, "그림 1. 시스템 구조: 디지털 트윈 → 센서 → AI 판단 → 에이전트 → 결과, 에이전트가 CCTV 확대를 다시 명령하는 피드백 경로")
    heading(d, "3.2 활용 데이터", 2)
    para(d, "실제 현장 사진 없이 **NVIDIA Isaac Sim 6.0** 의 실사 창고(warehouse_multiple_shelves)와 NVIDIA 소품(소화기, 표지판, 라바콘, 통, 상자, 팔레트, 작업대, 작업자), "
            "YCB 실물 스캔 공구(드릴, 클램프, 가위, 나무토막)로 장면을 만들고, Omniverse Replicator 로 **합성 데이터 4000장**(학습 3429, 검증 571)을 정답 박스와 함께 자동 생성했다. "
            "촬영 시점은 바디캠 55%, CCTV 30%, 자유 시점 15%이고, 조명과 흔들림·노이즈를 무작위로 바꿨다. 같은 종류의 물체를 위험한 상태와 안전한 상태로 함께 두어 "
            "YOLO 가 모양만이 아니라 **상태**를 구분하게 했다.")
    table(d, [
        ["종류", "위험 (YOLO 클래스)", "안전 (YOLO 클래스)", "mAP50"],
        ["바닥 유출", "기름/물 웅덩이 + 쓰러진 통, 조치 없음 (spill)", "같은 유출 + 미끄럼 주의 표지판·라바콘 (spill_marked)", "0.884 / 0.962"],
        ["공구·자재", "통로 바닥에 흩어진 공구 (tool_floor)", "작업대 위 정리 (tool_stored)", "0.865 / 0.906"],
        ["적재", "맨 위 층이 밀려나 기울어짐 (stack_unstable)", "반듯하게 쌓임 (stack_stable)", "0.959 / 0.954"],
        ["소화기", "바닥에 쓰러짐 (ext_fallen), 앞을 상자가 막음 (ext_blocked)", "제자리에 보이게 비치 (ext_ok)", "0.912, 0.898 / 0.926"],
        ["작업자", "CCTV 거리 측정용 (worker)", "", "0.965"],
    ], [2.2, 6.0, 6.0, 3.2], size=8, align_center_cols=(3,))

    heading(d, "3.3 AI 와 알고리즘", 2)
    bullets(d, [
        f"**YOLO26s** (Ultralytics, NMS 없는 출력, 작은 물체 라벨 할당 개선): 960 px, 80 epoch, RTX 4060 Ti 에서 약 2.2시간 학습. 검증 mAP50 **{YOLO['map50']}**, mAP50-95 {YOLO['map5095']}. "
        "혼동 행렬에서 위험/안전 짝끼리 헷갈린 비율은 1~4%",
        "**ByteTrack 추적**으로 같은 물체의 여러 프레임 판정을 신뢰도 합으로 투표. 3프레임 이상 같은 물체로 추적되면 확정",
        "**바닥 투영**: 박스 아래쪽(유출은 가운데)을 카메라 자세로 바닥 평면에 투영해 창고 좌표를 구함. 소화기는 도면의 거치 위치, 작업대 공구는 작업대 윗면 높이에 맞춤. "
        "가까이서 본 위치일수록 가중치를 크게 주고, 벽 밖 추정은 버리고 랙 안 추정은 랙 면으로 밀어냄. 같은 물체로 묶는 거리는 2 m (적재물 3 m)",
        "**CCTV 선택**: 재확인할 점과 각 CCTV 사이 선분이 랙(높이 6 m, CCTV 5 m 보다 높음)에 막히는지, 거리가 32 m 안인지 계산해 가까운 순으로 고름. "
        "PTZ 방향은 목표점을 향하게, 화각은 화면 세로가 4 m → 2.6 m 가 되게 두 장 촬영",
        "**접근 경고**: CCTV 화면의 작업자·위험물 박스를 바닥에 투영해 거리 2 m 안이면 경고(같은 물체는 5초에 한 번). 20 m 보다 먼 추정과 작업자 발밑 그림자 오인은 뺌",
    ])

    heading(d, "3.4 에이전트 동작 (Goal · Planning · Reasoning · Tool · Memory · Feedback)", 2)
    table(d, [
        ["요소", "구현"],
        ["Goal", "순찰 한 바퀴 동안 위험물을 찾아 위험/안전을 판정하고, 작업자 접근을 경고하고, 조치 지시서를 만든다"],
        ["Planning", "순찰 전 도면(소화기 6곳, 작업대 2곳)으로 점검표를 만들고 각 지점을 볼 수 있는 CCTV 를 미리 계산. 순찰이 끝나면 못 본 지점을 CCTV 확대 일정으로 바꿈"],
        ["Reasoning", "재확인이 필요한 경우를 고름: 바디캠에 1~2프레임만 보이고 지나감, 판정 표 1등이 70% 미만, CCTV 경고로 처음 알게 된 물체. "
                      "2초 기다렸다가 그사이 바디캠이 확정하면 재확인 취소"],
        ["Tool Use", "바디캠·CCTV 3대 화면, YOLO26, CCTV PTZ 방향·화각 명령, 거리 계산, 조치 지시서(HTML) 작성"],
        ["Memory/State", "위험물 대장 (위치, 판정 표, 확신도, 본 횟수, 출처, 접근 경고 횟수, 상태: 확정/재확인 대기/재확인 완료/현장 확인 필요/기각/병합), 점검표 상태"],
        ["Feedback", "확대 판정을 대장에 합쳐 판정 수정, 확대 화면(높이 5 m)에서 구한 위치로 대장 위치를 고치고 같은 물체면 병합. "
                     "못 찾거나 확신이 50% 미만이면 다음 CCTV 로 재시도, 끝까지 안 되면 현장 확인 요청. 바디캠이 나중에 확정하면 현장 확인 취소. 순찰 뒤 정답표로 채점"],
    ], [2.6, 14.8], size=8, first_col_shade=True)

    heading(d, "3.5 단계별 개발 (8일)", 2)
    table(d, [
        ["일차", "내용"],
        ["9/29~9/30", "Isaac Sim 6.0 설치, 장면·카메라·Replicator 라벨 파이프라인. 첫 버전은 로봇이 순찰 경로를 강화학습으로 배우는 구조였음"],
        ["10/1", "Isaac 실제 실행 검증, Replicator 라벨 누락·이전 프레임 문제 해결"],
        ["10/2", "**위험 판단 중심으로 전면 재설계**: 순찰 경로는 고정하고 작업자 바디캠으로 촬영, 위험/안전 짝 물체를 실사 에셋으로 모델링, 정답표 채점. "
                 "합성 데이터 4000장 생성, YOLO26s 학습, CCTV 접근 경고, 시나리오 평가"],
        ["10/3", "에이전트 층 추가: 점검표 계획, 위험물 대장, CCTV 선택·PTZ 확대 재확인, 재시도·현장 확인, 조치 지시서. 재평가"],
        ["10/4~10/6", "시연 영상, 보고서, 기술설명서, 발표자료, 저장소 정리"],
    ], [2.6, 14.8], size=8, first_col_shade=True)

    # 4. 구현 결과
    heading(d, "4. 구현 결과")
    heading(d, "4.1 핵심 기능", 2)
    images_row(d, [(figs["body"], "① 바디캠 YOLO 위험/안전 판정"), (figs["ptz"], "② CCTV 선택·확대 재확인"),
                   (figs["cctv"], "③ CCTV 작업자-위험물 접근 경고")])
    para(d, "① 바디캠: 한 화면에서 위험(유출, 바닥 공구, 가로막힌 소화기)과 안전(반듯한 적재, 표지된 유출)을 함께 판정. "
            "② 재확인: 에이전트가 고른 CCTV 가 PTZ 로 확대해 다시 판정 (화면 가운데가 목표점). ③ 접근 경고: CCTV 화면을 바닥 좌표로 바꿔 작업자-위험물 거리 계산.", size=8.5)
    image(d, figs["dash"], 13.0, "그림 2. 조치 지시서 화면 (평면도의 번호 = 우선순위, 위험 개수, 현장 확인, 점검표, 조치 목록)")

    heading(d, f"4.2 테스트 (대표 시나리오 {A['n']}건, 정답표 채점)", 2)
    para(d, f"위험 요소 배치를 시드마다 무작위로 바꾼 시나리오 {A['n']}개(학습 데이터에 없는 배치)를 한 바퀴씩 순찰했다. 정답표(물체별 클래스, 위험 여부, 위치)는 장면을 만들 때 저장하고 "
            "판정에는 쓰지 않으며 채점에만 쓴다. **바디캠만**은 바디캠이 확정한 물체를 바디캠 판정 그대로, **에이전트**는 재확인과 점검표를 거친 최종 대장을 창고 전체 물체와 비교한 것이다.")
    rows = [["시나리오", "물체 (위험)", "위험 판정\n바디캠만 → 에이전트", "안전 판정\n바디캠만 → 에이전트", "거꾸로 판정", "없는 위험 보고", "현장 확인", "재확인", "CCTV 접근 경고"]]
    for r in rs:
        e = r["agent"]["evaluation"]
        bb, aa, rr = e["before"], e["after"], e["recheck"]
        c = r.get("cctv") or {}
        rows.append([f"시드 {r['seed']}", f"{aa['hazard_total'] + aa['safe_total']} ({aa['hazard_total']})",
                     f"{bb['hazard_found']} → **{aa['hazard_found']}**/{aa['hazard_total']}", f"{bb['safe_ok']} → **{aa['safe_ok']}**/{aa['safe_total']}",
                     aa["hazard_as_safe"] + aa["safe_as_hazard"], aa["false_reports"], aa["need_check"] + checkpoint_checks(r)[0], rr["recheck_run"],
                     f"{c.get('events_detected', '-')}/{c.get('events_visible', '-')}"])
    rows.append(["합계", f"{A['objects']} ({nh})", f"{b['hazard_found']} → **{a['hazard_found']}**/{nh}", f"{b['safe_ok']} → **{a['safe_ok']}**/{ns}",
                 a["hazard_as_safe"] + a["safe_as_hazard"], a["false_reports"], a["need_check"], rc["recheck_run"], f"{A['ev'][0]}/{A['ev'][1]}"])
    table(d, rows, [1.7, 1.7, 2.7, 2.7, 1.5, 1.6, 1.5, 1.4, 2.0], size=7.5, align_center_cols=tuple(range(9)))
    para(d, f"재확인 {rc['recheck_run']}건 중 {rc['recheck_found']}건에서 물체를 다시 찾았고, 판정을 고친 것이 {rc['recheck_changed']}건(정답과 맞게 고침 {rc['changed_correct']}건), "
            f"다른 CCTV 로 다시 시도한 것이 {rc['recheck_retry']}번이었다. 바디캠 프레임 단위 YOLO 판정은 경로에서 보인 위험 물체 {A['yolo_h'][0]}/{A['yolo_h'][1]}, "
            f"안전 물체 {A['yolo_s'][0]}/{A['yolo_s'][1]}였다. 이 밖에 Isaac 없이 도는 단위 테스트 {N_TESTS}개(배치, 경로, 투영, 채점, 에이전트 재확인·재시도·병합 등)가 모두 통과한다.", size=8.5)

    # 5. 기대효과
    heading(d, "5. 기대효과, 성과와 한계")
    heading(d, "5.1 성과와 기대효과", 2)
    bullets(d, [
        f"**정량**: 창고 전체 물체 기준 위험 {pct(a['hazard_found'], nh)}, 안전 {pct(a['safe_ok'], ns)} 판정 (바디캠만 {pct(b['hazard_found'], nh)}, {pct(b['safe_ok'], ns)}). "
        f"재확인과 점검표가 바디캠 경로에서 멀거나 잠깐 보인 물체(작업대, 랙 끝 소화기)를 채움. CCTV 접근 경고 {A['ev'][0]}/{A['ev'][1]}건, 거리 오차 중앙값 {A['dist']:.2f} m",
        "**정성**: 순찰 1회마다 무엇을 어디서 어떤 근거로 판정했는지 기록이 남고, 조치 지시서가 바로 나옴. 카메라로 확정 못 한 것은 숨기지 않고 '현장 확인'으로 사람에게 넘김",
        "**적용**: 바디캠과 기존 CCTV(PTZ) 만으로 동작하는 구조라 로봇 없이 적용 가능. 디지털 트윈에서 위험 상태 데이터를 만들 수 있어 사고 장면을 실제로 연출할 필요가 없음",
    ])
    heading(d, "5.2 한계와 향후 계획", 2)
    from collections import Counter
    miss = Counter(row["class"] for r in rs for row in r["agent"]["evaluation"]["rows"] if row["hazard"] and row["after"] is None)
    n_miss = sum(miss.values())
    bullets(d, [
        f"**작은 공구를 놓침**: 에이전트가 끝내 못 찾은 위험 물체 {n_miss}개 중 {miss.get('tool_floor', 0)}개가 통로 바닥의 작은 공구 "
        f"(YOLO 에서 가장 약한 클래스, mAP50 0.865). 바디캠에 한 번도 검출되지 않으면 재확인 대상에도 오르지 않음. 공구 데이터 보강과 고해상도 바디캠이 필요",
        "**합성 데이터만 사용**: 실제 사진으로 검증하지 않았다. 현장 적용 전 실사 사진 수백 장으로 미세조정과 검증이 필요 (Level 4 Pilot 단계)",
        "**처리 속도**: Isaac Sim 6.0.1 의 PyTorch 가 CPU 전용이라 시뮬레이션 안 YOLO 는 CPU 로 돌림 (시뮬레이션 시간 고정으로 결과는 재현 가능). 실제 배치는 GPU·엣지 장치에서 실시간 가능",
        f"**CCTV 원거리**: 접근 경고를 놓친 {A['ev'][1] - A['ev'][0]}건은 모두 북쪽 끝 유출 자리로, 그곳을 보는 CCTV 가 약 29 m 떨어져 작업자가 너무 작게 찍힘 "
        "(거리 계산은 20 m 안만). 북쪽에 CCTV 1대를 더 두는 배치 최적화가 필요",
        "**규칙 기반 판단**: 재확인 조건, 우선순위 점수는 규칙. 현장 피드백(조치 완료, 오탐 표시)으로 임계값을 보정하는 학습과 MES·작업지시 시스템 연동을 계획",
        "**정적인 위험만**: 지게차 같은 움직이는 위험, 넘어짐 순간 같은 사건은 아직 다루지 않음",
    ])
    heading(d, "5.3 기존자산과 신규개발분, 출처", 2)
    table(d, [
        ["구분", "내용"],
        ["기존자산 (제3자)", "NVIDIA Isaac Sim 6.0·Omniverse Replicator, NVIDIA Isaac 에셋(창고, 소품, 작업자; 저장소에 포함하지 않고 NVIDIA 에셋 서버에서 참조), "
                         "YCB 물체 스캔, Ultralytics YOLO26 사전학습 가중치(AGPL-3.0), ByteTrack(Ultralytics 내장), numpy, Pillow, matplotlib"],
        ["8일 신규개발분", "장면·시나리오·정답표 생성, 위험/안전 짝 물체 모델링, 작업자 걷기 동작, 바디캠·CCTV·PTZ 카메라, 합성 데이터 파이프라인, YOLO 학습 설정, "
                         "바닥 투영·거리 계산, 에이전트(점검표 계획, 대장, 재확인·재시도·병합, 조치 지시서), 채점·평가, 시연 영상 합성 (저장소 커밋 9/30~10/6)"],
        ["AI 활용", "코드 작성·디버깅·문서 초안에 AI 코딩 도구(Anthropic Claude Code)를 사용함. 설계 방향 결정, 위험 요소 선정, 결과 검증은 팀이 수행"],
        ["데이터·안전", "실제 개인정보·현장 영상 없음 (전부 합성). API 키·비밀번호 없음. 통계 출처: 고용노동부 「2025년 산업재해 현황」"],
    ], [3.0, 14.4], size=8, first_col_shade=True)
    path = os.path.join(OUT, "개발완료보고서.docx")
    d.save(path)
    return path


# ---------------------------------------------------------------- 기술설명서 (별지 1, 1쪽)
def tech_sheet(rs):
    A = agg(rs)
    a, b, rc = A["a"], A["b"], A["rc"]
    nh, ns = a["hazard_total"], a["safe_total"]
    d = new_doc(margin_cm=1.4, base=8)
    for s in d.sections:
        s.top_margin = s.bottom_margin = Cm(1.1)
    para(d, "AI Agent 기술설명서", 14, True, BLUE, "center", after=4)
    tests = []
    for r in rs:
        e = r["agent"]["evaluation"]["after"]
        tests.append(f"S{r['seed']} 위험 {e['hazard_found']}/{e['hazard_total']}·안전 {e['safe_ok']}/{e['safe_total']}")
    rows = [
        ["팀명 / 부문 / 주제", f"{TEAM} ({MEMBERS}) / {DIVISION} / ③ 제조·피지컬 AI Agent"],
        ["작품명", f"{TITLE}"],
        ["해결 문제", "창고 순찰 점검이 사람 눈에 의존해 바닥 유출·방치 공구·불안정 적재·소화기 상태를 놓치기 쉽고, 기록이 남지 않으며, 작업자가 위험물에 다가가는 순간을 알 수 없음 "
                   "(2025년 사고재해자 중 넘어짐 25.2%로 1위, 고용노동부)"],
        ["대상 사용자", "물류창고·제조공장 안전관리자, 순찰 작업자, 관제 담당자"],
        ["Agent Goal", "순찰 한 바퀴 동안 위험물을 찾아 위험/안전을 판정하고, 애매하면 스스로 재확인하고, 작업자 접근을 경고하고, 우선순위가 있는 조치 지시서를 만든다"],
        ["사용 AI / 모델", f"YOLO26s (합성 데이터 4000장, 10 클래스, mAP50 {YOLO['map50']}) + ByteTrack 추적. 에이전트 판단(재확인 조건, CCTV 선택, 판정 결합, 우선순위)은 규칙·기하 계산"],
        ["사용 Tool / Data / 장비", "NVIDIA Isaac Sim 6.0 디지털 트윈(실사 창고, NVIDIA·YCB 에셋), 작업자 바디캠, CCTV 3대, CCTV PTZ(방향·화각 명령), Replicator(정답 박스), "
                                 "도면 데이터(랙, 소화기, 작업대, CCTV 위치), HTML 조치 지시서"],
        ["Memory / State / Feedback", "· 위험물 대장: 위치, 출처별 판정 표, 확신도, 본 횟수, 접근 경고 횟수, 상태(확정/재확인 대기/재확인 완료/현장 확인 필요/기각/병합). 점검표 8곳 상태\n"
                                    "· Feedback: 확대 판정을 대장에 합쳐 판정 수정, 확대 화면 위치로 대장 위치 보정·중복 병합 → 못 찾거나 확신 50% 미만이면 다음 CCTV 로 재시도 "
                                    "→ 끝까지 안 되면 현장 확인 요청 (나중에 바디캠이 확정하면 취소). 순찰 뒤 정답표로 채점"],
        ["핵심 Workflow\n(End-to-End)", "① 순찰 전 도면으로 점검표 8곳과 지점별 볼 수 있는 CCTV 계산 (계획)\n"
                                      "② 작업자가 경로를 걷고 바디캠·CCTV 화면이 들어옴 (입력)\n"
                                      "③ YOLO 가 위험/안전 판정, 추적으로 3프레임 모이면 확정하고 바닥 위치를 구해 대장에 기록 (판단·기억)\n"
                                      "④ 잠깐 보인 물체, 엇갈린 판정, 못 본 점검 지점을 재확인 목록에 올리고, 랙 가림·거리로 CCTV 를 골라 PTZ 확대 촬영 (계획·도구)\n"
                                      "⑤ 확대 화면 YOLO 로 다시 판정해 대장 수정, 실패 시 다른 CCTV → 현장 확인 요청 (결과 확인·재시도)\n"
                                      "⑥ CCTV 가 작업자-위험물 거리 2 m 안이면 경고하고 그 위험물 우선순위를 올림\n"
                                      "⑦ 순찰 끝에 우선순위·위치·조치가 담긴 조치 지시서 출력, 정답표로 채점"],
        ["핵심기능 3~5개", "1) 바디캠 YOLO 위험/안전 판정  2) CCTV 선택·확대 재확인(재시도·현장 확인)  3) CCTV 작업자-위험물 접근 경고  4) 위험물 대장·점검표  5) 조치 지시서"],
        ["기존자산 / 8일 신규개발분", "· 기존자산: Isaac Sim·Replicator, NVIDIA·YCB 에셋, Ultralytics YOLO26 사전학습 가중치·ByteTrack, numpy·Pillow\n"
                                   "· 신규개발: 장면·시나리오·정답표, 위험/안전 짝 모델링, 걷기 동작·카메라, 합성 데이터 파이프라인과 학습, 바닥 투영·거리, 에이전트 전체, 채점·영상 합성"],
        [f"대표 테스트 {A['n']}건 결과", f"학습에 안 쓴 배치 {A['n']}개, 창고 전체 물체 기준\n" + "  ".join(tests) +
                              f"\n합계 위험 {a['hazard_found']}/{nh} ({pct(a['hazard_found'], nh)}), 안전 {a['safe_ok']}/{ns} ({pct(a['safe_ok'], ns)}), "
                              f"바디캠만 대비 위험 +{a['hazard_found'] - b['hazard_found']}, 안전 +{a['safe_ok'] - b['safe_ok']}. 재확인 {rc['recheck_run']}건. "
                              f"CCTV 접근 경고 {A['ev'][0]}/{A['ev'][1]}건, 거리 오차 중앙값 {A['dist']:.2f} m"],
        ["현재 완성도 Level", "Level 3 (MVP): 시뮬레이션에서 End-to-End 동작과 정량 채점 완료. 실사 검증 전"],
        ["소스코드 / 저장소", f"{REPO} (실행: scripts/run_patrol.py, 평가: scripts/eval_patrol.py, README 에 설치·실행법)"],
        ["정보출처 및 기타", "고용노동부 「2025년 산업재해 현황」, NVIDIA Isaac Sim 문서·에셋, YCB Object Set, Ultralytics YOLO (AGPL-3.0). "
                          "AI 코딩 도구 Anthropic Claude Code 로 코드·문서 작성 보조. 실제 개인정보·현장 영상 사용 없음"],
    ]
    table(d, rows, [3.6, 14.6], header=False, size=7.6, first_col_shade=True)
    path = os.path.join(OUT, "AI_Agent_기술설명서.docx")
    d.save(path)
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    rs = load_results()
    if not rs:
        sys.exit("outputs/eval 에 에이전트 채점 결과가 없어요. scripts/eval_patrol.py 를 먼저 돌리세요.")
    figs = {k: os.path.join(FIG, f"fig_{k}.jpg") for k in ("body", "ptz", "cctv")}
    figs["dash"] = os.path.join(FIG, "fig_dashboard_crop.png")
    print(report(rs, figs))
    print(tech_sheet(rs))


if __name__ == "__main__":
    main()
