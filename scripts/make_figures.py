"""보고서·발표용 그림 (일반 파이썬, Pillow).

    python scripts/make_figures.py          # docs/architecture.png, docs/workflow.png
"""
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = "C:/Windows/Fonts/malgun.ttf"
FONT_B = "C:/Windows/Fonts/malgunbd.ttf"
INK = (33, 41, 56)
MUTED = (100, 110, 128)
COL = {"twin": (64, 120, 200), "sensor": (40, 150, 180), "ai": (120, 90, 200), "agent": (220, 130, 30), "out": (50, 150, 90)}


def f(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def box(d, xy, title, lines, col, title_size=34, size=25):
    x0, y0, x1, y1 = xy
    d.rounded_rectangle(xy, 22, fill=(255, 255, 255), outline=col, width=5)
    d.rounded_rectangle([x0, y0, x1, y0 + 66], 22, fill=col)
    d.rectangle([x0, y0 + 40, x1, y0 + 66], fill=col)
    d.text(((x0 + x1) / 2, y0 + 33), title, font=f(title_size, True), fill=(255, 255, 255), anchor="mm")
    y = y0 + 92
    for ln in lines:
        bold = ln.startswith("*")
        d.text((x0 + 24, y), ln.lstrip("*"), font=f(size, bold), fill=INK if bold else MUTED if ln.startswith("  ") else INK)
        y += int(size * 1.55)


def arrow(d, x0, y, x1, col=(120, 128, 140), w=6, label=None, up=False):
    d.line([(x0, y), (x1 - 18, y)], fill=col, width=w)
    s = 1 if x1 > x0 else -1
    d.polygon([(x1, y), (x1 - 22 * s, y - 14), (x1 - 22 * s, y + 14)], fill=col)
    if label:
        d.text(((x0 + x1) / 2, y + (-34 if up else 12)), label, font=f(21, True), fill=col, anchor="ma")


def architecture(path):
    W_, H_ = 2600, 940
    im = Image.new("RGB", (W_, H_), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.text((W_ / 2, 40), "창고 안전 순찰 AI 에이전트 구조", font=f(48, True), fill=INK, anchor="ma")
    bw, gap, top, bh = 440, 72, 130, 600
    xs = [60 + i * (bw + gap) for i in range(5)]
    box(d, (xs[0], top, xs[0] + bw, top + bh), "디지털 트윈", [
        "*NVIDIA Isaac Sim 6.0 실사 창고", "*위험 4종 + 안전 짝",
        "  유출 / 표지된 유출", "  통로 공구 / 작업대 공구 (8종)", "  불안정 / 반듯한 적재", "  쓰러짐·가로막힘 / 정상 소화기",
        "*위험 영역", "  라바콘 링, DANGER 표지", "*걷는 작업자, 운반 카트", "  손가락 1~5 손동작", "*정답표 (채점 전용)"], COL["twin"])
    box(d, (xs[1], top, xs[1] + bw, top + bh), "센서 (입력·도구)", [
        "*작업자 가슴 바디캠", "  높이 1.38 m, 걸음 흔들림", "", "*CCTV 3대 (고정)", "  천장 4.6~5 m",
        "", "*CCTV PTZ 확대", "  에이전트가 방향·화각 지정", "", "*스피커 (경고·다국어 안내)", "*Replicator 렌더"], COL["sensor"])
    box(d, (xs[2], top, xs[2] + bw, top + bh), "AI 판단", [
        "*YOLO26s (19 클래스)", "  위험/안전 상태, 공구 이름 8종", "  운반 카트, 라바콘, DANGER 표지", "*ByteTrack 추적",
        "  같은 물체 판정 모으기", "*바닥 투영", "  화면 박스 → 창고 좌표", "  공구: 바닥 / 작업대 판단",
        "*MediaPipe 손 관절 21점", "  → 손가락 1~5"], COL["ai"])
    box(d, (xs[3], top, xs[3] + bw, top + bh), "에이전트", [
        "*계획: 점검표 8곳", "*기억: 위험물 대장, 위험 영역", "*판단: 재확인 필요 여부", "  잠깐 보임 / 판정 엇갈림",
        "*판단: 위험 영역 설정", "  라바콘·표지 / 스스로 판단", "*도구: CCTV 고르기, 음성 경고", "  닿기 직전 1 m, 영역 0.5 m",
        "*피드백: 판정 수정, 재시도", "  현장 확인 요청", "*손동작 명령: 장비·스캔·TBM", "  관리자 호출·SOS"], COL["agent"])
    box(d, (xs[4], top, xs[4] + bw, top + bh), "결과", [
        "*음성 경고", "  \"경고 경고 위험 요소가", "   식별되었습니다\"", "*실시간 알림", "  판정, 접근 경고, 위험 영역",
        "*작업자 언어 음성 안내", "  중·영·일·한 (검수한 문장 틀)", "*조치 지시서 (HTML)", "  우선순위·위치·공구 이름",
        "  위험 영역, 호출·SOS 알림", "*정답표 채점"], COL["out"])
    ym = top + bh / 2
    for i in range(4):
        arrow(d, xs[i] + bw + 6, ym, xs[i + 1] - 6)
    # 에이전트 -> 센서 피드백 (확대 명령)
    yb = top + bh + 70
    c = COL["agent"]
    d.line([(xs[3] + bw / 2, top + bh), (xs[3] + bw / 2, yb)], fill=c, width=6)
    d.line([(xs[3] + bw / 2, yb), (xs[1] + bw / 2, yb)], fill=c, width=6)
    d.line([(xs[1] + bw / 2, yb), (xs[1] + bw / 2, top + bh + 22)], fill=c, width=6)
    d.polygon([(xs[1] + bw / 2, top + bh + 4), (xs[1] + bw / 2 - 14, top + bh + 28), (xs[1] + bw / 2 + 14, top + bh + 28)], fill=c)
    d.text(((xs[1] + xs[3]) / 2 + bw / 2, yb + 16), "재확인: CCTV 선택 → PTZ 방향·화각 명령 → 확대 화면 YOLO 재판정 → 대장 수정 (못 찾으면 다음 CCTV, 그래도 안 되면 현장 확인)",
           font=f(25, True), fill=c, anchor="ma")
    im.save(path)
    return path


def workflow(path):
    """End-to-End 흐름 (가로 한 줄)."""
    steps = [("① 입력", "바디캠·CCTV 화면\n작업자 손동작"), ("② 판단", "위험/안전·공구 이름\n위험 영역 설정"), ("③ 계획", "애매한 물체,\n못 본 점검 지점"),
             ("④ 도구 실행", "CCTV 확대 재판정\n음성 경고·손동작 응답"), ("⑤ 결과 확인", "판정 수정·재시도\n현장 확인 요청"), ("⑥ 보고", "조치 지시서\n호출·SOS 알림")]
    W_, H_ = 2600, 420
    im = Image.new("RGB", (W_, H_), (255, 255, 255))
    d = ImageDraw.Draw(im)
    bw, gap = 370, 54
    x = 40
    cols = [COL["sensor"], COL["ai"], COL["agent"], COL["agent"], COL["agent"], COL["out"]]
    for i, (h, body) in enumerate(steps):
        d.rounded_rectangle([x, 60, x + bw, 360], 20, fill=(255, 255, 255), outline=cols[i], width=5)
        d.text((x + bw / 2, 110), h, font=f(36, True), fill=cols[i], anchor="mm")
        for j, ln in enumerate(body.split("\n")):
            d.text((x + bw / 2, 190 + j * 50), ln, font=f(28), fill=INK, anchor="mm")
        if i < len(steps) - 1:
            arrow(d, x + bw + 4, 210, x + bw + gap - 4)
        x += bw + gap
    im.save(path)
    return path


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    print(architecture(os.path.join(ROOT, "docs", "architecture.png")))
    print(workflow(os.path.join(ROOT, "docs", "workflow.png")))
