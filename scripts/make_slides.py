"""발표자료 10장 (python-pptx). IBDP 양식이 있으면 그 위에 만든다. 수치는 outputs/eval 채점 결과에서.

    python scripts/make_slides.py --template <팀 PPT 양식.pptx>      # 양식 없이 돌리면 빈 16:9 슬라이드에 만듦
"""
import argparse
import os
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from make_docs import MEMBERS, REPO, SUBTITLE, TEAM, TITLE, YOLO, agg, load_results, pct  # noqa: E402

FIG = os.path.join(ROOT, "docs")
BLUE = RGBColor(0x28, 0x4E, 0x92)
INK = RGBColor(0x1E, 0x24, 0x30)
GRAY = RGBColor(0x5A, 0x63, 0x70)
RED = RGBColor(0xD9, 0x3A, 0x3A)
GREEN = RGBColor(0x2E, 0x9D, 0x5B)
ORANGE = RGBColor(0xE0, 0x8A, 0x00)
LIGHT = RGBColor(0xEE, 0xF2, 0xF8)
FONT = "맑은 고딕"


def drop_slides(prs):
    lst = prs.slides._sldIdLst
    for sld in list(lst):
        prs.part.drop_rel(sld.rId)
        lst.remove(sld)


def tb(slide, x, y, w, h, lines, size=16, color=INK, bold=False, align=None, anchor=None, fill=None, line=None):
    """lines: 문자열 또는 (문자열, 크기, 색, 굵게) 목록. '**' 로 굵게."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)) if fill is None else \
        slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is not None:
        box.fill.solid()
        box.fill.fore_color.rgb = fill
        if line is None:
            box.line.fill.background()
        else:
            box.line.color.rgb = line
        box.adjustments[0] = 0.08
    tf = box.text_frame
    tf.word_wrap = True
    for side in ("margin_left", "margin_right"):
        setattr(tf, side, Inches(0.12))
    tf.margin_top = tf.margin_bottom = Inches(0.06)
    if anchor or fill is not None:
        tf.vertical_anchor = anchor or MSO_ANCHOR.TOP
    if isinstance(lines, (str, tuple)):
        lines = [lines]
    for i, ln in enumerate(lines):
        if isinstance(ln, tuple):
            text, sz, col, bd = (list(ln) + [None, None, None])[:4]
        else:
            text, sz, col, bd = ln, None, None, None
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if align:
            p.alignment = align
        p.space_after = Pt(4)
        for k, part in enumerate(text.split("**")):
            if not part:
                continue
            r = p.add_run()
            r.text = part
            r.font.name = FONT
            r.font.size = Pt(sz or size)
            r.font.bold = bool(bd if bd is not None else bold) or k % 2 == 1
            r.font.color.rgb = col or color
    return box


def pic(slide, path, x, y, w=None, h=None):
    if not os.path.exists(path):
        return tb(slide, x, y, w or 3, 0.4, f"[그림 없음: {os.path.basename(path)}]", 10, GRAY)
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(path, Inches(x), Inches(y), **kw)


def table(slide, rows, x, y, w, col_w, size=12, header_fill=BLUE, row_h=0.36, bold_last=False):
    shp = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    t = shp.table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.03)
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if j > 0 or i == 0 else PP_ALIGN.LEFT
            for k, part in enumerate(str(val).split("**")):
                if not part:
                    continue
                r = p.add_run()
                r.text = part
                r.font.name = FONT
                r.font.size = Pt(size)
                r.font.bold = i == 0 or k % 2 == 1 or (bold_last and i == len(rows) - 1)
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if i == 0 else INK
            c.fill.solid()
            c.fill.fore_color.rgb = header_fill if i == 0 else (LIGHT if i % 2 == 0 else RGBColor(0xFF, 0xFF, 0xFF))
    return shp


def content_slide(prs, title, layout_idx):
    s = prs.slides.add_slide(prs.slide_layouts[layout_idx])
    for ph in list(s.placeholders):
        idx = ph.placeholder_format.idx
        if idx == 0:
            ph.text = title
            for p in ph.text_frame.paragraphs:
                for r in p.runs:
                    r.font.name = FONT
        elif idx not in (12, 10, 11):
            ph._element.getparent().remove(ph._element)
    return s


def build(template, out):
    rs = load_results()
    A = agg(rs)
    a, b, rc = A["a"], A["b"], A["rc"]
    nh, ns = a["hazard_total"], a["safe_total"]
    if template and os.path.exists(template):
        prs = Presentation(template)
        drop_slides(prs)
        L_TITLE, L_BODY = 0, 1
    else:
        prs = Presentation()
        prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
        L_TITLE, L_BODY = 0, 5

    # 1. 표지
    s = prs.slides.add_slide(prs.slide_layouts[L_TITLE])
    for ph in list(s.placeholders):
        idx = ph.placeholder_format.idx
        if idx == 0:
            ph.text = TITLE
        elif idx == 1:
            ph.text = f"{SUBTITLE}\n\n{TEAM} · {MEMBERS}\n2026-10-06"
            for k, p in enumerate(ph.text_frame.paragraphs):
                for r in p.runs:
                    r.font.size = Pt(16 if k == 0 else 18)
                    r.font.name = FONT
    tb(s, 2.5, 1.15, 8.3, 0.45, "제4회 경남AI·SW경진대회 · ③ 제조·피지컬 AI Agent", 16, GRAY, align=PP_ALIGN.CENTER)

    # 2. 문제
    s = content_slide(prs, "1. 문제: 창고 순찰 점검은 사람 눈에 의존한다", L_BODY)
    tb(s, 0.6, 1.4, 4.2, 2.3, [("25.2%", 66, RED, True), ("2025년 사고재해자 113,305명 중 넘어짐 28,606명, 재해 유형 1위", 14, GRAY),
                                ("(고용노동부 「2025년 산업재해 현황」)", 11, GRAY)], align=PP_ALIGN.CENTER)
    tb(s, 0.6, 4.1, 4.2, 2.6, [("창고의 위험 요소", 18, BLUE, True), "• 바닥 유출 → 미끄러짐", "• 통로에 방치된 공구·자재 → 걸려 넘어짐",
                                "• 불안정 적재 → 무너짐·맞음", "• 쓰러지거나 가로막힌 소화기 → 화재 초기 대응 실패"], 15)
    tb(s, 5.3, 1.4, 7.5, 0.5, ("지금의 순찰 점검", 20, BLUE, True))
    for k, (h, body) in enumerate([("놓치기 쉽다", "작업자가 걸으며 눈으로만 확인. 잠깐 보고 지나친 것, 경로에서 먼 것은 빠짐"),
                                   ("기록이 남지 않는다", "무엇을 어디서 어떤 근거로 위험하다고 봤는지 남지 않아 조치 지시와 추적이 어려움"),
                                   ("다가가는 순간을 모른다", "작업자가 위험물 바로 옆을 지나가는 아차사고를 관리자가 알 수 없음")]):
        tb(s, 5.3, 2.0 + k * 1.55, 7.5, 1.35, [(h, 18, INK, True), (body, 14, GRAY)], fill=LIGHT, anchor=MSO_ANCHOR.MIDDLE)

    # 3. 해결 방안
    s = content_slide(prs, "2. 해결: 순찰하며 판정하고, 애매하면 스스로 재확인하는 AI 에이전트", L_BODY)
    tb(s, 0.6, 1.3, 12.1, 0.9, [("작업자 바디캠과 CCTV 로 위험/안전을 판정하고, 애매한 것은 **볼 수 있는 CCTV 를 골라 확대해 다시 확인**한 뒤, "
                                 "작업자 접근을 경고하고 **조치 지시서**를 만든다", 17, INK)])
    pic(s, os.path.join(FIG, "workflow.png"), 0.6, 2.2, w=12.1)
    feats = [("① 바디캠 판정", "YOLO26 이 10개 클래스로\n위험/안전 상태 판정", BLUE), ("② 확대 재확인", "CCTV 선택 → PTZ 확대\n재시도 → 현장 확인", ORANGE),
             ("③ 접근 경고", "CCTV 3대로 작업자-위험물\n거리 2 m 안이면 경고", RED), ("④ 위험물 대장", "위치·상태·확신도 기억\n점검표 8곳", RGBColor(0x7A, 0x5A, 0xC8)),
             ("⑤ 조치 지시서", "우선순위·위치·조치\n평면도와 기록", GREEN)]
    for k, (h, body, col) in enumerate(feats):
        tb(s, 0.6 + k * 2.47, 4.5, 2.3, 1.9, [(h, 17, col, True), (body, 13, INK)], fill=LIGHT, align=PP_ALIGN.CENTER)
    tb(s, 0.6, 6.5, 12.1, 0.4, "로봇 없이 바디캠과 기존 CCTV(PTZ) 만으로 동작 · 위험 데이터는 디지털 트윈에서 합성", 13, GRAY, align=PP_ALIGN.CENTER)

    # 4. 구조
    s = content_slide(prs, "3. 에이전트 구조 (Goal · Plan · Reasoning · Tool · Memory · Feedback)", L_BODY)
    pic(s, os.path.join(FIG, "architecture.png"), 0.5, 1.25, w=12.3)
    table(s, [["Goal", "Planning", "Reasoning", "Tool Use", "Memory", "Feedback"],
              ["위험물 판정, 접근 경고, 조치 지시서", "도면으로 점검표 8곳, 지점별 CCTV 계산", "재확인 필요 판단 (잠깐 보임, 엇갈림)",
               "바디캠·CCTV·PTZ 명령·YOLO", "위험물 대장 (위치·판정 표·상태)", "판정 수정, 다른 CCTV 재시도, 현장 확인"]],
          0.5, 5.75, 12.3, [2.05] * 6, size=11, row_h=0.5)

    # 5. 디지털 트윈과 데이터
    s = content_slide(prs, "4. 디지털 트윈과 학습 데이터", L_BODY)
    pic(s, os.path.join(FIG, "fig_cctv_raw.jpg"), 0.5, 1.3, w=6.0)
    tb(s, 0.5, 4.75, 6.0, 1.9, [("NVIDIA Isaac Sim 6.0 실사 창고", 16, BLUE, True), "• NVIDIA 소품(소화기, 표지판, 라바콘, 상자, 팔레트, 작업대, 작업자)",
                                 "• YCB 실물 스캔 공구 (드릴, 클램프, 가위, 나무토막)", "• Replicator 로 정답 박스 자동 생성, 정답표로 채점"], 13)
    table(s, [["종류", "위험", "안전"], ["바닥 유출", "조치 없음", "표지판·라바콘"], ["공구·자재", "통로 바닥", "작업대 위"],
              ["적재", "기울어짐", "반듯함"], ["소화기", "쓰러짐 / 가로막힘", "제자리"]], 6.9, 1.3, 5.9, [1.7, 2.2, 2.0], size=13)
    tb(s, 6.9, 3.4, 5.9, 3.2, [("합성 데이터 4000장 → YOLO26s", 16, BLUE, True),
                                "• 바디캠 55%, CCTV 30%, 자유 시점 15% · 조명·흔들림 무작위",
                                "• 같은 종류를 위험/안전 상태로 함께 배치 → 모양이 아니라 상태를 학습",
                                f"• 검증 mAP50 **{YOLO['map50']}** (mAP50-95 {YOLO['map5095']}), 위험/안전 짝 혼동 1~4%",
                                "• 960 px, 80 epoch, RTX 4060 Ti 에서 약 2.2시간"], 13)

    # 6. 핵심 구현 1
    s = content_slide(prs, "5. 핵심 구현 ①  바디캠 판정과 CCTV 접근 경고", L_BODY)
    pic(s, os.path.join(FIG, "fig_body.jpg"), 0.5, 1.3, w=6.1)
    pic(s, os.path.join(FIG, "fig_cctv.jpg"), 6.75, 1.3, w=6.1)
    tb(s, 0.5, 4.85, 6.1, 1.9, [("바디캠 YOLO 판정", 16, BLUE, True), "• 가슴 높이 1.38 m, 걸음마다 흔들리는 시점",
                                 "• ByteTrack 추적으로 같은 물체 판정을 모아 3프레임이면 확정", "• 박스를 바닥에 투영해 위치 → 위험물 대장"], 13)
    tb(s, 6.75, 4.85, 6.1, 1.9, [("CCTV 접근 경고", 16, RED, True), "• 작업자·위험물 박스를 바닥 좌표로 바꿔 거리 계산",
                                  f"• 2 m 안이면 경고, 그 위험물 우선순위 상향 (거리 오차 중앙값 {A['dist']:.2f} m)",
                                  "• 20 m 넘는 추정, 발밑 그림자 오인은 제외"], 13)

    # 7. 핵심 구현 2
    s = content_slide(prs, "6. 핵심 구현 ②  CCTV 선택·확대 재확인과 조치 지시서", L_BODY)
    pic(s, os.path.join(FIG, "fig_ptz.jpg"), 0.5, 1.3, w=5.6)
    tb(s, 0.5, 4.5, 5.6, 2.3, [("언제 재확인하나", 15, ORANGE, True), "• 바디캠에 1~2프레임만 보이고 지나감",
                                "• 판정 표 1등 70% 미만 (엇갈림) · CCTV 경고로 처음 본 물체", "• 순찰이 끝나도 못 본 점검 지점",
                                ("어떻게", 15, ORANGE, True), "• 랙·적재물 가림, 거리로 CCTV 선택 → PTZ 확대 2장",
                                "• 못 찾거나 확신 50% 미만 → 다른 CCTV → 현장 확인", "• 확대 화면 위치로 대장 위치 보정, 중복 병합"], 12)
    pic(s, os.path.join(FIG, "fig_dashboard_crop.png"), 6.4, 1.3, w=6.45)
    tb(s, 6.4, 6.35, 6.45, 0.45, "조치 지시서: 평면도(번호 = 우선순위), 조치 목록, 점검표, 기록", 12, GRAY, align=PP_ALIGN.CENTER)

    # 8. 시연
    s = content_slide(prs, "7. 시연: 입력 → 판단 → 도구 실행 → 결과", L_BODY)
    pic(s, os.path.join(FIG, "fig_video.png"), 0.5, 1.25, w=8.6)
    tb(s, 9.3, 1.25, 3.55, 5.3, [("시연 영상 (1분 44초)", 17, BLUE, True), ("시나리오 1, 실시간 1배속", 13, GRAY),
                                  "① 순찰 전 점검 계획", "② 바디캠 위험/안전 판정", "③ 애매한 물체 → CCTV 선택 → 확대 재확인",
                                  "④ 작업자 접근 경고", "⑤ 순찰 끝 점검표 확대 확인", "⑥ 조치 지시서", "⑦ 정답표 채점 결과",
                                  ("", 8), ("실행", 15, BLUE, True), ("run_patrol.py --view bodycam|top|ptz", 11, GRAY)], 14)

    # 9. 성과
    s = content_slide(prs, f"8. 성과: 학습에 안 쓴 시나리오 {A['n']}개, 정답표 채점", L_BODY)
    table(s, [["창고 전체 물체 기준", "바디캠만", "에이전트"],
              [f"위험 물체를 위험으로 ({nh}개)", f"{b['hazard_found']} ({pct(b['hazard_found'], nh)})", f"**{a['hazard_found']} ({pct(a['hazard_found'], nh)})**"],
              [f"안전 물체를 안전으로 ({ns}개)", f"{b['safe_ok']} ({pct(b['safe_ok'], ns)})", f"**{a['safe_ok']} ({pct(a['safe_ok'], ns)})**"],
              ["위험↔안전 거꾸로 판정", b["hazard_as_safe"] + b["safe_as_hazard"], a["hazard_as_safe"] + a["safe_as_hazard"]],
              ["없는 위험 보고", b["false_reports"], a["false_reports"]],
              ["현장 확인 요청 (실제 물체)", "-", f"{a['need_check']} ({a['need_check_real']})"]],
          0.5, 1.35, 7.4, [3.6, 1.9, 1.9], size=14, row_h=0.5)
    tb(s, 0.5, 4.6, 7.4, 2.1, [("재확인과 점검표가 바디캠이 놓친 물체를 채움", 15, BLUE, True),
                                f"• 재확인 {rc['recheck_run']}건 실행: 찾음 {rc['recheck_found']}, 판정 고침 {rc['recheck_changed']}, 다른 CCTV 재시도 {rc['recheck_retry']}",
                                "• 작업대 위 공구, 랙 끝 소화기처럼 경로에서 먼 물체를 CCTV 확대로 확인",
                                "• 확정 못 한 것은 숨기지 않고 '현장 확인' 으로 사람에게"], 13)
    rows = [["시드", "위험", "안전", "거꾸로", "재확인"]]
    for r in rs:
        e = r["agent"]["evaluation"]
        rows.append([r["seed"], f"{e['after']['hazard_found']}/{e['after']['hazard_total']}", f"{e['after']['safe_ok']}/{e['after']['safe_total']}",
                     e["after"]["hazard_as_safe"] + e["after"]["safe_as_hazard"], e["recheck"]["recheck_run"]])
    table(s, rows, 8.3, 1.35, 4.5, [0.8, 1.0, 1.0, 0.85, 0.85], size=10 if len(rows) > 6 else 13, row_h=0.25 if len(rows) > 6 else 0.42)
    tb(s, 8.3, 4.6, 4.5, 2.1, [("CCTV 접근 경고", 15, RED, True), (f"{A['ev'][0]}/{A['ev'][1]} 건", 30, INK, True),
                                 (f"CCTV 에 보인 접근 사건 · 오경보 {A['ev'][3]}번 · 거리 오차 {A['dist']:.2f} m", 12, GRAY)], fill=LIGHT)

    # 10. 한계·발전·BM
    s = content_slide(prs, "9. 한계, 발전 계획, 비즈니스 모델", L_BODY)
    tb(s, 0.5, 1.3, 4.0, 5.4, [("한계 (현장 적용 전 보완점)", 16, RED, True), "• 합성 데이터만 사용, 실사 미검증",
                                "• Isaac 6.0.1 torch 가 CPU 전용이라 시뮬 안 YOLO 는 CPU (실배치는 GPU·엣지)", "• 놓친 접근 경고 4건은 모두 CCTV 에서 약 29 m 먼 북쪽 끝 (CCTV 추가 필요)", "• 통로 바닥 작은 공구를 자주 놓침 (못 찾은 위험 14개 중 12개)",
                                "• 재확인 조건·우선순위는 규칙", "• 움직이는 위험(지게차)은 아직 없음"], 13, fill=LIGHT)
    tb(s, 4.7, 1.3, 4.0, 5.4, [("발전 계획", 16, BLUE, True), ("1단계 실사 검증", 14, INK, True), "실습실·창고 사진 수백 장으로 미세조정",
                                ("2단계 파일럿 (Level 4)", 14, INK, True), "실제 CCTV 보정, 바디캠 1대 + PTZ 1대 현장 시험",
                                ("3단계 연동", 14, INK, True), "MES·작업지시 연동, 조치 완료 피드백으로 기준 보정, 지게차 등 동적 위험"], 13, fill=LIGHT)
    tb(s, 8.9, 1.3, 3.95, 5.4, [("비즈니스 모델", 16, GREEN, True), ("대상", 14, INK, True), "중소 물류창고·제조공장 (사고재해 40%가 5~49인 사업장)",
                                 ("제공", 14, INK, True), "기존 CCTV + 바디캠에 붙이는 소프트웨어 구독 (사업장·카메라 수 기준)",
                                 ("차별점", 14, INK, True), "디지털 트윈으로 그 현장의 위험 상태 데이터를 만들어 맞춤 학습",
                                 ("기대효과", 14, INK, True), "점검 누락·기록 공백 감소, 아차사고 관리"], 13, fill=LIGHT)
    tb(s, 0.5, 6.75, 12.3, 0.3, f"소스코드 {REPO}", 11, GRAY, align=PP_ALIGN.CENTER)
    prs.save(out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--template", default=os.environ.get("SLIDE_TEMPLATE"), help="제목/제목+내용 레이아웃이 있는 pptx 양식")
    p.add_argument("--out", default=os.path.join(ROOT, "submission", "발표자료.pptx"))
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    print(build(a.template, a.out))


if __name__ == "__main__":
    main()
