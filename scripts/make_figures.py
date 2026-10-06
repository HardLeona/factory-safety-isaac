"""보고서·발표용 그림 (일반 파이썬, Pillow).

    python scripts/make_figures.py          # docs/architecture.png, docs/workflow.png
"""
import math
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = "C:/Windows/Fonts/malgun.ttf"
FONT_B = "C:/Windows/Fonts/malgunbd.ttf"
INK = (33, 41, 56)
MUTED = (100, 110, 128)
COL = {"twin": (64, 120, 200), "sensor": (40, 150, 180), "ai": (120, 90, 200), "agent": (220, 130, 30), "out": (50, 150, 90)}


SS = 2       # 글자를 몇 배로 그려 줄일지


def f(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


class Draw(ImageDraw.ImageDraw):
    """글자만 SS 배로 그려 줄여 붙이는 ImageDraw. 맑은 고딕을 작은 크기로 그대로 그리면 받침 ㅌ 이 깨짐 (같은, 발밑)."""

    def __init__(self, im):
        super().__init__(im)
        self.base = im

    def text(self, xy, text, fill=None, font=None, anchor=None, **kw):
        big = ImageFont.truetype(font.path, font.size * SS)
        l, t, r, b = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), text, font=big, anchor=anchor)
        pad = 4 * SS
        mask = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 0)
        ImageDraw.Draw(mask).text((pad - l, pad - t), text, font=big, fill=255, anchor=anchor)
        mask = mask.resize((math.ceil(mask.width / SS), math.ceil(mask.height / SS)), Image.LANCZOS)
        pos = (round(xy[0] + (l - pad) / SS), round(xy[1] + (t - pad) / SS))
        self.base.paste(Image.new("RGB", mask.size, fill), pos, mask)


def box(d, xy, title, lines, col, title_size=34, size=25):
    x0, y0, x1, y1 = xy
    d.rounded_rectangle(xy, 22, fill=(255, 255, 255), outline=col, width=5)
    d.rounded_rectangle([x0, y0, x1, y0 + 66], 22, fill=col)
    d.rectangle([x0, y0 + 40, x1, y0 + 66], fill=col)
    d.text(((x0 + x1) / 2, y0 + 33), title, font=f(title_size, True), fill=(255, 255, 255), anchor="mm")
    y = y0 + 92
    for ln in lines:
        bold = ln.startswith("*")
        fits(d, ln.lstrip("*"), f(size, bold), x1 - x0 - 40, title)
        d.text((x0 + 24, y), ln.lstrip("*"), font=f(size, bold), fill=INK if bold else MUTED if ln.startswith("  ") else INK)
        y += int(size * 1.55)
    if y - int(size * 0.55) > y1 - 10:
        print(f"[경고] '{title}' 칸: 줄이 많아 상자 아래로 넘침")


def fits(d, text, font, width, where):
    """글이 상자 폭을 넘으면 알려 줌 (그림을 다시 만들 때 넘침을 바로 알게)."""
    if d.textlength(text, font=font) > width:
        print(f"[경고] '{where}' 칸: '{text.strip()}' 이 상자 폭을 넘음 ({d.textlength(text, font=font):.0f} > {width} px)")


def arrow(d, x0, y, x1, col=(120, 128, 140), w=6, label=None, up=False):
    d.line([(x0, y), (x1 - 18, y)], fill=col, width=w)
    s = 1 if x1 > x0 else -1
    d.polygon([(x1, y), (x1 - 22 * s, y - 14), (x1 - 22 * s, y + 14)], fill=col)
    if label:
        d.text(((x0 + x1) / 2, y + (-34 if up else 12)), label, font=f(21, True), fill=col, anchor="ma")


def architecture(path):
    W_, H_ = 2600, 980
    im = Image.new("RGB", (W_, H_), (255, 255, 255))
    d = Draw(im)
    d.text((W_ / 2, 40), "바디캠 안전관리 에이전트 구조", font=f(48, True), fill=INK, anchor="ma")
    bw, gap, top, bh = 440, 72, 130, 640
    xs = [60 + i * (bw + gap) for i in range(5)]
    box(d, (xs[0], top, xs[0] + bw, top + bh), "디지털 트윈", [
        "*NVIDIA Isaac Sim 6.0 실사 창고", "*위험 4종 + 안전 짝",
        "  유출 / 표지된 유출", "  통로 공구 / 작업대 공구 (8종)", "  불안정 / 반듯한 적재", "  쓰러짐·가로막힘 / 정상 소화기",
        "*위험 영역", "  라바콘 링, DANGER 표지", "*걷는 작업자, 운반 카트", "  손가락 1~3 손동작", "*정답표 (채점 전용)"], COL["twin"])
    box(d, (xs[1], top, xs[1] + bw, top + bh), "센서 (입력·도구)", [
        "*작업자 가슴 바디캠 (하나뿐)", "  높이 1.38 m, 걸음 흔들림", "", "*스피커 (경고·다국어 안내)",
        "", "*Replicator 렌더"], COL["sensor"])
    box(d, (xs[2], top, xs[2] + bw, top + bh), "AI 판단", [
        "*YOLO26s (19 클래스)", "  위험/안전 상태, 공구 이름 8종", "  운반 카트, 라바콘, DANGER 표지", "*ByteTrack 추적",
        "  같은 물체 판정 모으기", "*바닥 투영", "  화면 박스 → 창고 좌표", "  공구: 바닥 / 작업대 판단",
        "*MediaPipe 손 관절 21점", "  → 손가락 1~3"], COL["ai"])
    box(d, (xs[3], top, xs[3] + bw, top + bh), "에이전트", [
        "*계획: 점검표 8곳", "*기억: 위험물 대장, 위험 영역", "*판단: 고위험 후보면 즉시 확정", "  아니면 '주의'로 분류",
        "*판단: 위험 영역 설정", "  라바콘·표지 / 스스로 판단", "*도구: 음성 경고 (위험/주의 톤)", "  회피형 위험 1.5 m, 영역 0.5 m",
        "*피드백: 주의 물체 재관측 재판단", "  끝까지 안 보이면 현장 확인", "*LLM 에이전트 (LangGraph)", "  Qwen2.5-7B 로컬", "  손동작 요청·재판단에", "  도구 골라 판단"], COL["agent"])
    box(d, (xs[4], top, xs[4] + bw, top + bh), "결과", [
        "*음성 경고 (위험)", "  \"멈추세요! 위험 요소가", "   식별되었습니다\"", "*음성 안내 (주의)", "  \"발밑을 확인하세요\"",
        "*실시간 알림", "  판정, 위험 영역, 재판단", "*작업자 언어 음성 안내", "  중·영·일·한 (검수한 문장 틀)", "*조치 지시서 (HTML)",
        "  우선순위·위치·공구 이름", "  위험 영역, 호출·SOS 알림", "*정답표 채점"], COL["out"])
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
    d.text(((xs[1] + xs[3]) / 2 + bw / 2, yb + 16), "재판단: '주의' 물체를 바디캠이 다시 지나침 → LangGraph + 로컬 LLM 재판단 (실패 시 규칙) → 대장 수정 (끝까지 안 보이면 현장 확인 권고)",
           font=f(25, True), fill=c, anchor="ma")
    im.save(path)
    return path


def workflow(path):
    """End-to-End 흐름 (가로 한 줄)."""
    steps = [("① 입력", "바디캠 화면\n작업자 손동작"), ("② 판단", "위험/안전·공구 이름\n위험 영역 설정"), ("③ 주의 분류", "안전·위험 미확정\n재관측 대기"),
             ("④ 도구 실행", "재관측 재판단 (LLM)\n음성 경고\n손동작: LLM 도구 호출"), ("⑤ 결과 확인", "판정 확정 / 주의 유지\n현장 확인 요청"), ("⑥ 보고", "조치 지시서\n호출·SOS 알림")]
    W_, H_ = 2600, 420
    im = Image.new("RGB", (W_, H_), (255, 255, 255))
    d = Draw(im)
    bw, gap = 370, 54
    x = 40
    cols = [COL["sensor"], COL["ai"], COL["agent"], COL["agent"], COL["agent"], COL["out"]]
    for i, (h, body) in enumerate(steps):
        d.rounded_rectangle([x, 60, x + bw, 360], 20, fill=(255, 255, 255), outline=cols[i], width=5)
        d.text((x + bw / 2, 110), h, font=f(36, True), fill=cols[i], anchor="mm")
        for j, ln in enumerate(body.split("\n")):
            fits(d, ln, f(28), bw - 30, h)
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
