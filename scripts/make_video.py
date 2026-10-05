"""시연 영상 만들기 (일반 파이썬). run_patrol.py --record 로 저장한 화면과 에이전트 기록을 1920x1080 영상으로 합친다.

    python scripts/make_video.py --record outputs/record/seed1 --out outputs/video/demo.mp4

화면 구성: 왼쪽 바디캠(입력, YOLO 판정) | 오른쪽 평면도(작업자·위험물·위험 영역) | 위험물 대장 | 에이전트 기록(판정→주의→재판단→경고→결과)
앞뒤로 제목, 문제와 구조, 점검 계획, 조치 지시서 화면, 평가 결과를 붙인다.
음성 경고가 난 시각에는 위험("멈추세요! 위험 요소가 식별되었습니다") 또는 주의("발밑을 확인하세요") 음성을 소리 트랙에 넣는다.
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from factory_safety import warehouse as W  # noqa: E402
from factory_safety.config import CLASS_KO, HAZARD  # noqa: E402
from factory_safety.report import clock  # noqa: E402

FPS = 30
def _yolo_map50():
    """학습한 YOLO 검증 mAP50 (scripts/val_yolo.py 가 만든 metrics.json, 새 모델부터)."""
    for name in ("warehouse_v3", "warehouse_v2", "warehouse"):
        path = os.path.join(ROOT, "outputs", "yolo", name, "metrics.json")
        if os.path.exists(path):
            return f"{json.load(open(path, encoding='utf-8'))['map50']:.3f}"
    return "-"


YOLO_MAP50 = os.environ.get("YOLO_MAP50") or _yolo_map50()
WIDTH, HEIGHT = 1920, 1080
FONT = "C:/Windows/Fonts/malgun.ttf"
FONT_B = "C:/Windows/Fonts/malgunbd.ttf"
BG = (18, 24, 34)
PANEL = (28, 36, 50)
FG = (232, 236, 242)
MUTED = (150, 160, 178)
RED, GREEN, AMBER, BLUE, ORANGE, PURPLE = (232, 72, 72), (60, 184, 110), (240, 170, 40), (80, 150, 240), (255, 140, 40), (170, 110, 230)
KIND_COLOR = {"계획": BLUE, "판정": (110, 200, 255), "판정 수정": PURPLE, "주의": AMBER, "현장 확인": RED,
              "점검표": GREEN, "위험 영역": (255, 110, 110), "음성 경고": (255, 80, 200), "주의 안내": (255, 170, 120),
              "접근 기록": MUTED, "손동작": (120, 220, 255), "관리자 호출": AMBER, "SOS": RED, "LLM 판단": (190, 150, 255)}
BODY = (0, 70, 1280, 790)       # 화면 배치: 바디캠 1280x720 (왼쪽), 오른쪽 평면도·대장, 아래 에이전트 기록
# 영상에서 잠깐 멈춰 보여줄 행동 (CCTV 확대 같은 연출이 없는 바디캠 단독 모드라 "현장 확인"/"주의"는 안 멈춤)
HOLD_KINDS = ("판정 수정", "위험 영역", "음성 경고", "주의 안내", "LLM 판단")
HOLD_S = 1.5
# 자막으로 보여줄 행동 (앞에 있을수록 우선)
CAPTION_KINDS = ["음성 경고", "주의 안내", "위험 영역", "판정 수정", "현장 확인", "LLM 판단", "주의", "점검표", "판정"]
_fonts = {}


LANG_FONT = {"zh": ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc"), "ja": ("C:/Windows/Fonts/YuGothM.ttc", "C:/Windows/Fonts/YuGothB.ttc")}
_lang_fonts = {}


def lang_font(lang, size, bold=False):
    """중국어·일본어 자막 글꼴 (맑은 고딕에 없는 글자가 있어서)."""
    if lang not in LANG_FONT:
        return font(size, bold)
    key = (lang, size, bold)
    if key not in _lang_fonts:
        path = LANG_FONT[lang][1 if bold else 0]
        _lang_fonts[key] = ImageFont.truetype(path, size) if os.path.exists(path) else font(size, bold)
    return _lang_fonts[key]


def font(size, bold=False):
    k = (size, bold)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(FONT_B if bold else FONT, size)
    return _fonts[k]


def text(d, xy, s, size=20, fill=FG, bold=False, anchor="la"):
    d.text(xy, s, font=font(size, bold), fill=fill, anchor=anchor)


def fit(d, s, size, width, bold=False):
    """width 픽셀에 맞게 자름."""
    f = font(size, bold)
    if d.textlength(s, font=f) <= width:
        return s
    while s and d.textlength(s + "…", font=f) > width:
        s = s[:-1]
    return s + "…"


def wrap(d, s, size, width, max_lines):
    """width 픽셀 줄로 나눔 (넘치면 마지막 줄 말줄임)."""
    f, out, cur = font(size), [], ""
    for ch in s:
        if d.textlength(cur + ch, font=f) > width:
            out.append(cur)
            cur = ch
        else:
            cur += ch
    out.append(cur)
    if len(out) > max_lines:
        out = out[:max_lines]
        out[-1] = fit(d, out[-1] + "…", size, width)
    return out


def load(path, size):
    if path and os.path.exists(path):
        return Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
    return None


# ---------------------------------------------------------------- 평면도
MAP_S = 14.2
MX0, MY1 = -10.8, 18.4


def mp(x, y, ox, oy):
    return ox + (x - MX0) * MAP_S, oy + (MY1 - y) * MAP_S


def draw_map(d, ox, oy, st, alerts, path_pts):
    w, h = 21.6 * MAP_S, 31.0 * MAP_S
    d.rectangle([ox, oy, ox + w, oy + h], fill=(236, 232, 222))
    x0, y0 = mp(-10.3, 18.0, ox, oy)
    x1, y1 = mp(10.3, -12.2, ox, oy)
    d.rectangle([x0, y0, x1, y1], outline=(90, 90, 90), width=2)
    for a, b in W.RACKS:
        p0, p1 = mp(a[0], b[1], ox, oy), mp(b[0], a[1], ox, oy)
        d.rectangle([p0, p1], fill=(176, 170, 158))
    for a, b in W.PILLARS:
        d.rectangle([mp(a[0], b[1], ox, oy), mp(b[0], a[1], ox, oy)], fill=(80, 80, 80))
    for tx, ty, _ in W.TABLES:
        d.rectangle([mp(tx - 1.24, ty + 0.39, ox, oy), mp(tx + 1.24, ty - 0.39, ox, oy)], fill=(160, 120, 80))
    pts = [mp(px, py, ox, oy) for px, py in path_pts]
    for i in range(0, len(pts) - 1, 2):
        d.line([pts[i], pts[i + 1]], fill=(80, 130, 210), width=2)
    for zz in st.get("zones", []):
        poly = [mp(a, b, ox, oy) for a, b in zz["poly"]]
        col = (220, 50, 50) if zz["source"] != "agent" else (230, 120, 30)
        d.polygon(poly, fill=col + (55,), outline=col + (255,))
        cx = sum(q[0] for q in poly) / len(poly)
        cy = min(q[1] for q in poly)
        text(d, (cx, cy - 15), zz["id"], 13, col, True, anchor="ma")
    wx, wy, wyaw = st["worker"]
    for fx, fy in alerts:
        d.line([mp(wx, wy, ox, oy), mp(fx, fy, ox, oy)], fill=RED, width=4)
    for f in st["findings"]:
        if f["status"] == "기각" or not f["cls"]:
            continue
        x, y = mp(f["x"], f["y"], ox, oy)
        col = AMBER if f["status"] in ("주의", "현장 확인 필요") else RED if HAZARD[f["cls"]] else GREEN
        r = 8 if col != GREEN else 6
        d.ellipse([x - r, y - r, x + r, y + r], fill=col, outline=(255, 255, 255), width=2)
        text(d, (x + 9, y - 9), f["id"], 13, fill=(30, 30, 30), bold=True)
    bx, by = st["bodycam"][:2]
    byaw, bvfov = st["bodycam"][2], st["bodycam"][3]
    hf = math.radians(bvfov * 16 / 9 / 2)
    c = mp(bx, by, ox, oy)
    poly = [c] + [mp(bx + 7 * math.cos(byaw + s), by + 7 * math.sin(byaw + s), ox, oy) for s in (-hf, hf)]
    d.polygon(poly, fill=(80, 150, 240, 60), outline=BLUE)
    p = mp(wx, wy, ox, oy)
    d.ellipse([p[0] - 8, p[1] - 8, p[0] + 8, p[1] + 8], fill=BLUE, outline=(255, 255, 255), width=2)


# ---------------------------------------------------------------- 화면 한 장
def header(d, t, stage, live=True):
    d.rectangle([0, 0, WIDTH, 66], fill=(12, 16, 24))
    text(d, (22, 12), "창고 안전 순찰 AI 에이전트", 30, bold=True)
    text(d, (400, 22), f"Isaac Sim 시뮬레이션 · {'실시간 1배속' if live else ''} · {clock(t)}", 18, fill=MUTED)
    steps = ["① 입력: 바디캠·손동작", "② 판단: YOLO·위험 영역", "③ 도구: LLM 에이전트·음성", "④ 결과: 대장·조치 지시서"]
    x = 900
    for i, s in enumerate(steps):
        f = font(18, True)
        wdt = d.textlength(s, font=f) + 24
        on = i == stage
        d.rounded_rectangle([x, 16, x + wdt, 50], 8, fill=ORANGE if on else (40, 50, 66))
        d.text((x + 12, 22), s, font=f, fill=(20, 20, 20) if on else MUTED)
        x += wdt + 10


def panel_label(d, x, y, s, color=FG):
    f = font(18, True)
    wdt = d.textlength(s, font=f) + 16
    d.rectangle([x, y, x + wdt, y + 28], fill=(0, 0, 0))
    d.text((x + 8, y + 3), s, font=f, fill=color)


def compose(rec, st, tl, path_pts, cache):
    """바디캠 1280x720 (왼쪽), 오른쪽 평면도·위험물 대장, 아래 에이전트 기록."""
    k, t = st["k"], st["t"]
    im = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(im, "RGBA")
    recent = [e for e in tl[:st["n_timeline"]] if e["kind"] != "계획" or e["t"] > 0]
    window = [e for e in recent if t - e["t"] <= 2.0]
    if any(e["kind"] in ("LLM 판단", "손동작", "관리자 호출", "SOS", "음성 경고", "주의 안내", "위험 영역", "주의") for e in window) \
            or (st.get("hand") or {}).get("count"):
        stage = 2
    elif any(e["kind"] in ("판정", "판정 수정") for e in window):
        stage = 1
    else:
        stage = 0
    header(d, t, stage)
    bx0, by0, bx1, by1 = BODY
    body_path = os.path.join(rec, f"body_{k:05d}.jpg")
    if os.path.exists(body_path):
        cache["body"] = body_path
    if cache.get("body"):
        im.paste(load(cache["body"], (bx1 - bx0, by1 - by0)), (bx0, by0))
    panel_label(d, bx0 + 8, by0 + 8, "작업자 바디캠 + YOLO 판정")
    if st.get("story"):
        panel_label(d, bx0 + 8, by0 + 42, f"작업자: {st['story']}", (120, 220, 255))
    key = [e for e in recent if t - e["t"] <= 2.5 and e["kind"] in CAPTION_KINDS]
    if key:
        e = max(key, key=lambda e: (CAPTION_KINDS.index(e["kind"]) * -1, e["t"]))
        lines = wrap(d, e["text"], 24, bx1 - bx0 - 30, 2)
        top_y = by1 - 50 - 34 * len(lines)
        d.rectangle([bx0, top_y, bx1 - 1, by1 - 1], fill=(0, 0, 0, 175))
        text(d, (bx0 + 14, top_y + 8), e["kind"], 26, KIND_COLOR.get(e["kind"], FG), True)
        for i, ln in enumerate(lines):
            text(d, (bx0 + 14, top_y + 46 + 34 * i), ln, 24, FG)
    # 음성 경고 중인 물체 쪽으로 평면도에 빨간 선 (최근 2초 안의 음성 경고/주의 안내)
    alerting = [e for e in window if e["kind"] in ("음성 경고", "주의 안내")]
    alerts = []
    for e in alerting:
        fid = e["text"].split("↔ ")[-1].split()[0]
        f = next((f for f in st["findings"] if f["id"] == fid), None)
        if f:
            alerts.append((f["x"], f["y"]))
    # 오른쪽: 평면도 + 위험물 대장
    rx = bx1 + 10
    d.rectangle([bx1, 66, WIDTH, HEIGHT], fill=PANEL)
    text(d, (rx + 6, 76), "평면도 (작업자·위험물·위험 영역)", 20, bold=True)
    mimg = Image.new("RGB", (int(21.6 * MAP_S), int(31.0 * MAP_S)), (236, 232, 222))
    draw_map(ImageDraw.Draw(mimg, "RGBA"), 0, 0, st, alerts, path_pts)
    mw = 360
    mimg = mimg.resize((mw, int(mimg.height * mw / mimg.width)), Image.LANCZOS)
    im.paste(mimg, (rx + (WIDTH - rx - mw) // 2, 106))
    ly = 106 + mimg.height + 12
    text(d, (rx + 6, ly), "위험물 대장 (기억)", 20, bold=True)
    y = ly + 32
    rows = [f for f in st["findings"] if f["status"] != "기각" and f["cls"]]
    for f in rows[-((HEIGHT - y) // 28):]:
        col, s_ = (RED, "위험") if HAZARD[f["cls"]] else (GREEN, "안전")
        if f["status"] == "주의":
            col, s_ = AMBER, "주의"
        elif f["status"] == "현장 확인 필요":
            col, s_ = AMBER, "확인필요"
        d.rounded_rectangle([rx + 6, y, rx + 58, y + 22], 5, fill=col)
        text(d, (rx + 32, y + 11), s_, 14, (255, 255, 255), True, anchor="mm")
        text(d, (rx + 66, y + 1), fit(d, f"{f['id']} {CLASS_KO[f['cls']]}", 16, WIDTH - rx - 80), 16)
        y += 28
    # 아래: 에이전트 기록
    gy = by1 + 6
    d.rectangle([0, by1, bx1, HEIGHT], fill=PANEL)
    text(d, (14, gy), "에이전트 기록 (판정 → 주의 → 재관측 재판단 → 경고 → 결과)", 20, bold=True)
    y = gy + 32
    for e in recent[-((HEIGHT - y) // 30):]:
        col = KIND_COLOR.get(e["kind"], FG)
        if t - e["t"] < 1.5:
            d.rectangle([8, y - 2, bx1 - 8, y + 26], fill=(60, 50, 30))
        text(d, (14, y), clock(e["t"]), 16, MUTED)
        text(d, (70, y), e["kind"], 16, col, True)
        text(d, (175, y), fit(d, e["text"], 16, bx1 - 195), 16)
        y += 30
    return im


# ---------------------------------------------------------------- 앞뒤 카드
def card(lines, title=None, sub=None):
    im = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(im)
    y = 120
    if title:
        text(d, (WIDTH // 2, y), title, 54, bold=True, anchor="mt")
        y += 90
    if sub:
        text(d, (WIDTH // 2, y), sub, 26, MUTED, anchor="mt")
        y += 70
    for ln in lines:
        if isinstance(ln, tuple):
            s, size, col, bold = ln
        else:
            s, size, col, bold = ln, 28, FG, False
        text(d, (WIDTH // 2, y), s, size, col, bold, anchor="mt")
        y += int(size * 1.7)
    return im


def workflow_card():
    im = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(im)
    text(d, (WIDTH // 2, 70), "문제와 에이전트 구조", 46, bold=True, anchor="mt")
    text(d, (WIDTH // 2, 150), "창고 순찰 점검은 사람이 눈으로 확인해서 놓치기 쉽고, 무엇을 어디서 봤는지 기록이 남지 않습니다", 26, MUTED, anchor="mt")
    boxes = [("입력", "작업자 가슴 바디캠\n(착용자 시점)", BLUE), ("판단", "YOLO26 위험/안전 판정\n공구 이름 8종\n라바콘·DANGER 표지", (110, 200, 255)),
             ("즉시 반응", "고위험 후보는 즉시 확정\n저위험만 있으면 '주의' 분류\n위험 영역 자동 설정", AMBER),
             ("에이전트 재판단", "주의 물체를 다시 지나치면\nLangGraph + 로컬 LLM 재판단\n위험/안전 어느 쪽인지 확정", PURPLE),
             ("결과", "조치 지시서, 1.5m 음성 경고\n다국어 손동작 안내", GREEN)]
    bw, gap = 300, 50
    x = (WIDTH - (bw * 5 + gap * 4)) // 2
    for i, (h, body, col) in enumerate(boxes):
        d.rounded_rectangle([x, 300, x + bw, 620], 18, fill=PANEL, outline=col, width=4)
        text(d, (x + bw // 2, 330), h, 32, col, True, anchor="mt")
        for j, ln in enumerate(body.split("\n")):
            text(d, (x + bw // 2, 410 + j * 46), ln, 24, FG, anchor="mt")
        if i < 4:
            d.polygon([(x + bw + 10, 450), (x + bw + gap - 10, 460), (x + bw + 10, 470)], fill=MUTED)
        x += bw + gap
    lines = ["위험 요소: 바닥 유출 · 방치된 공구(망치·드라이버·톱·전동톱·곡괭이·삽·렌치·드릴) · 불안정 적재 · 소화기 · 라바콘/DANGER 표지 영역",
             "정답표를 미리 만들어 두고 에이전트 판정을 채점 (판정에는 정답표를 쓰지 않음)",
             "작업자 손동작 (손가락 1~5): 장비 설명 · 공장 위험 스캔 · 오늘의 TBM · 관리자 호출 · SOS → 작업자 언어 음성 (중·영·일·한)",
             "NVIDIA Isaac Sim 6.0 실사 창고 · 합성 데이터 7700장으로 학습 · 실제 사진 없음"]
    for j, ln in enumerate(lines):
        text(d, (WIDTH // 2, 720 + j * 56), ln, 26, FG if j == 0 else MUTED, anchor="mt")
    return im


def plan_card(tl):
    plan = [e["text"] for e in tl if e["kind"] == "계획" and e["t"] == 0]
    lines = [("에이전트가 순찰 전에 세운 점검 계획", 40, FG, True), ("", 12, FG, False)]
    for s in plan:
        lines.append((s, 23, FG if s.startswith("점검표") else AMBER, False))
    return card(lines)


def screenshot(html_path, out_png):
    edge = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
    if not os.path.exists(edge) or not os.path.exists(html_path):
        return None
    subprocess.run([edge, "--headless", "--disable-gpu", "--hide-scrollbars", f"--screenshot={out_png}",
                    "--window-size=1920,1500", "file:///" + os.path.abspath(html_path).replace("\\", "/")],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    return out_png if os.path.exists(out_png) else None


def dashboard_frames(png):
    """조치 지시서 화면을 위에서 아래로 천천히 훑음."""
    src = Image.open(png).convert("RGB")
    sw, sh = src.size
    scale = WIDTH / sw
    src = src.resize((WIDTH, int(sh * scale)), Image.BILINEAR)
    view_h = HEIGHT - 70
    span = max(0, src.size[1] - view_h)
    n = FPS * 12
    for i in range(n):
        off = int(span * min(1.0, max(0.0, (i - FPS * 3) / (n - FPS * 5))))
        im = Image.new("RGB", (WIDTH, HEIGHT), BG)
        im.paste(src.crop((0, off, WIDTH, off + view_h)), (0, 70))
        d = ImageDraw.Draw(im)
        header(d, 0, 3, live=False)
        yield im


def results_card(eval_dir, gestures=None, langs=None):
    files = sorted(glob.glob(os.path.join(eval_dir, "inspection_seed*.json")))
    rs = [json.load(open(f, encoding="utf-8")) for f in files]
    rs = [r for r in rs if r.get("agent")]
    if not rs:
        return None
    tot = lambda part, k: sum(r["agent"]["evaluation"][part][k] for r in rs)  # noqa: E731
    rc = sum(r["agent"]["evaluation"]["rejudge"]["caution_rejudged"] for r in rs)
    ch = sum(r["agent"]["evaluation"]["rejudge"]["caution_rejudged_confirmed"] for r in rs)
    im = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(im)
    text(d, (WIDTH // 2, 70), f"평가: 학습에 안 쓴 배치 {len(rs)}개를 정답표와 비교", 44, bold=True, anchor="mt")
    text(d, (WIDTH // 2, 140), "창고 전체 물체 기준 (경로에서 안 보이는 물체 포함)", 24, MUTED, anchor="mt")
    rows = [("", "바디캠 프레임 원시판정", "에이전트 (재판단 + 점검표 포함)"),
            ("위험 물체를 위험으로", f"{tot('before', 'hazard_found')}/{tot('before', 'hazard_total')}",
             f"{tot('after', 'hazard_found')}/{tot('after', 'hazard_total')}"),
            ("안전 물체를 안전으로", f"{tot('before', 'safe_ok')}/{tot('before', 'safe_total')}",
             f"{tot('after', 'safe_ok')}/{tot('after', 'safe_total')}"),
            ("위험↔안전 거꾸로 판정", f"{tot('before', 'hazard_as_safe') + tot('before', 'safe_as_hazard')}",
             f"{tot('after', 'hazard_as_safe') + tot('after', 'safe_as_hazard')}"),
            ("없는 위험 판정 (박스 수 / 보고 건수)", f"{tot('before', 'false_hazard_boxes')}개", f"{tot('after', 'false_reports')}건")]
    y = 230
    for i, (a, b, cc) in enumerate(rows):
        bold = i == 0
        d.rectangle([360, y - 8, 1560, y + 50], fill=PANEL if i % 2 else BG)
        text(d, (400, y), a, 30, MUTED if bold else FG, bold)
        text(d, (1050, y), b, 30, MUTED if bold else FG, bold, anchor="ma")
        text(d, (1380, y), cc, 30, ORANGE if bold else FG, True, anchor="ma")
        y += 66
    y += 30
    text(d, (WIDTH // 2, y), f"주의 분류 {tot('after', 'caution')}건 (실제 물체 {tot('after', 'caution_real')}건) · "
                             f"주의 재관측 재판단 {rc}건 중 확정 전환 {ch}건", 28, AMBER, anchor="mt")
    ex = [r["agent"]["evaluation"] for r in rs if "zones" in r["agent"]["evaluation"]]
    if ex:
        zs = lambda k: sum(e["zones"][k] for e in ex)   # noqa: E731
        vs = lambda k: sum(e["voice"][k] for e in ex)   # noqa: E731
        ts = lambda k: sum(e["tools"][k] for e in ex)   # noqa: E731
        text(d, (WIDTH // 2, y + 60), f"위험 영역 (라바콘·DANGER 표지) 알아봄 {zs('found')}/{zs('gt')} · 에이전트 판단 영역 {zs('agent')}개 · "
                                      f"1.5m 음성 경고 {vs('warned')}/{vs('events')} · 공구 이름 {ts('named')}/{ts('total')}", 26, (255, 120, 200),
             anchor="mt")
        y += 60
    if gestures:
        text(d, (WIDTH // 2, y + 60), f"시연 손동작 명령 {gestures['recognized']}/{gestures['shown']} 인식 (잘못 실행 {gestures['extra']}) · "
                                      f"안내 언어 {', '.join(langs or [])}", 26, (120, 220, 255), anchor="mt")
        y += 60
    text(d, (WIDTH // 2, y + 60), f"YOLO26s 검증 mAP50 {YOLO_MAP50} · 합성 데이터만으로 학습, 실제 현장 적용 전 실사 검증 필요", 24, MUTED, anchor="mt")
    return im


def main():
    p = argparse.ArgumentParser(description="시연 영상 합치기")
    p.add_argument("--record", default=os.path.join(ROOT, "outputs", "record", "seed5"))
    p.add_argument("--out", default=os.path.join(ROOT, "outputs", "video", "demo.mp4"))
    p.add_argument("--eval-dir", default=os.path.join(ROOT, "outputs", "eval"))
    p.add_argument("--team", default="IBDP 팀 (문상균, 이승혜, 조현준)")
    p.add_argument("--preview", type=int, default=None, help="이 단계 번호의 화면 한 장만 PNG 로 저장")
    a = p.parse_args()
    rec = a.record
    states = [json.loads(ln) for ln in open(os.path.join(rec, "state.jsonl"), encoding="utf-8") if ln.strip()]
    final = json.load(open(os.path.join(rec, "final.json"), encoding="utf-8"))
    tl = final["report"]["timeline"]
    path_pts = W.PatrolPath().pts[::3]
    if a.preview is not None:
        st = next(s for s in states if s["k"] >= a.preview)
        out = os.path.splitext(a.out)[0] + f"_preview_{a.preview}.png"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        compose(rec, st, tl, path_pts, {}).save(out)
        print(out)
        return
    import imageio_ffmpeg
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    writer = imageio_ffmpeg.write_frames(a.out, (WIDTH, HEIGHT), fps=FPS, codec="libx264", quality=None,
                                         bitrate="6M", pix_fmt_out="yuv420p", macro_block_size=8)
    writer.send(None)
    n = [0]

    def emit(im, seconds=None, frames=None):
        buf = np.asarray(im.convert("RGB"), dtype=np.uint8).tobytes()
        for _ in range(frames if frames is not None else int(round(seconds * FPS))):
            writer.send(buf)
            n[0] += 1

    seed = final["seed"]
    emit(card([(a.team, 30, FG, True), ("제4회 경남AI·SW경진대회 · 제조·피지컬 AI Agent", 26, MUTED, False), ("", 20, FG, False),
               ("작업자 바디캠으로 창고를 순찰하며 위험/안전을 판정하고, 위험 영역을 잡아 닿기 전에 경고하는 AI 에이전트", 28, FG, False),
               ("애매하면 '주의'로 분류하고, 같은 물체를 바디캠이 다시 지나칠 때 로컬 LLM (Qwen2.5-7B, LangGraph) 이 재판단", 28, FG, False),
               ("작업자가 손가락 1~5 를 보이면 장비 설명·공장 위험 스캔·TBM 안내·관리자 호출·SOS 를 작업자 언어로 처리", 26, (120, 220, 255), False),
               ("", 20, FG, False), (f"NVIDIA Isaac Sim 6.0 디지털 트윈 · YOLO26 · 시나리오 {seed}", 24, MUTED, False)],
              title="창고 안전 순찰 AI 에이전트"), 5)
    emit(workflow_card(), 8)
    emit(plan_card(tl), 6)
    cache = {}
    walk = [s for s in states]
    prev_n = 0
    sys.path.insert(0, ROOT)
    from factory_safety.config import VOICE_TEXT_CAUTION
    from factory_safety.voice import VOICE_CAUTION_WAV, VOICE_WAV, duration, ensure_voice
    ensure_voice()
    ensure_voice(VOICE_CAUTION_WAV, VOICE_TEXT_CAUTION)
    vdur = {"hazard": duration(VOICE_WAV), "caution": duration(VOICE_CAUTION_WAV)}
    voice_marks, voice_end, voice_level = [], -1.0, "hazard"
    assist = final["report"].get("assistant", [])
    clips, speaking = [], None          # 손동작 안내 음성 (영상 시각, wav), 지금 나오는 안내 (사건, 시작 시각)
    n_assist = 0
    for i, st in enumerate(walk):
        nxt = walk[i + 1]["t"] if i + 1 < len(walk) else st["t"] + 0.4
        dt = max(0.0, nxt - st["t"])
        im = compose(rec, st, tl, path_pts, cache)
        new_ev = assist[n_assist:st.get("n_assist", 0)]
        n_assist = max(n_assist, st.get("n_assist", 0))
        for ev in new_ev:
            speaking = (ev, n[0] / FPS)
            if ev.get("wav") and os.path.exists(ev["wav"]):
                clips.append((n[0] / FPS, ev["wav"]))
        if speaking and n[0] / FPS > speaking[1] + speaking[0].get("dur", 0) + 0.5:
            speaking = None
        if speaking:
            assist_caption(im, speaking[0], (n[0] / FPS - speaking[1]) / max(speaking[0].get("dur", 1.0), 1.0))
        if st.get("voice"):
            for v in st["voice"]:
                voice_marks.append((n[0] / FPS, v["level"]))
            voice_level = st["voice"][-1]["level"]
            voice_end = n[0] / FPS + vdur[voice_level]
        if n[0] / FPS < voice_end:
            voice_banner(im, voice_level)
        emit(im, frames=max(1, int(round(dt * FPS))))
        # 주요 장면 (판정 수정, 위험 영역, 음성 경고, LLM 판단) 은 1.5초 멈춰서 읽을 수 있게
        new = tl[prev_n:st["n_timeline"]]
        prev_n = st["n_timeline"]
        key = [e for e in new if e["kind"] in HOLD_KINDS and "갱신" not in e["text"]]
        bx1 = BODY[2]
        if new_ev:
            # 손동작 인식 순간: 손 관절과 명령이 보이게 잠깐 멈춤
            d = ImageDraw.Draw(im)
            d.rounded_rectangle([bx1 - 260, 78, bx1 - 8, 112], 8, fill=(0, 140, 220))
            text(d, (bx1 - 134, 95), "손동작 명령 인식 · 잠깐 멈춤", 18, (255, 255, 255), True, anchor="mm")
            emit(im, 1.2)
        elif key and dt < 0.3:
            d = ImageDraw.Draw(im)
            d.rounded_rectangle([bx1 - 230, 78, bx1 - 8, 112], 8, fill=ORANGE)
            text(d, (bx1 - 119, 95), "주요 장면 · 잠깐 멈춤", 18, (20, 20, 20), True, anchor="mm")
            emit(im, HOLD_S)
    html_path = os.path.join(ROOT, "outputs", "agent", f"dashboard_{dash_tag(final, states)}.html")
    png = screenshot(html_path, os.path.join(tempfile.gettempdir(), f"dash_{seed}.png"))
    if png:
        for im in dashboard_frames(png):
            emit(im, frames=1)
    res = results_card(a.eval_dir, final.get("evaluation", {}).get("gestures"), list(dict.fromkeys(e["lang_name"] for e in assist)))
    if res:
        emit(res, 10)
    writer.close()
    clips += [(m, VOICE_WAV if lvl == "hazard" else VOICE_CAUTION_WAV) for m, lvl in voice_marks]
    if clips:
        add_audio(a.out, clips, n[0] / FPS)
    nh = sum(1 for _, lvl in voice_marks if lvl == "hazard")
    print(f"[완료] {a.out}  ({n[0] / FPS:.0f}초, 음성 경고 {nh}번 · 주의 안내 {len(voice_marks) - nh}번, 손동작 안내 {len(assist)}번)")


def dash_tag(final, states):
    """조치 지시서 파일 이름: 시연 이야기 녹화는 story_seed<시드>, 그 밖은 seed<시드> (run_patrol.py 와 같게)."""
    story = final.get("story", any(s.get("story") for s in states))
    return f"story_seed{final['seed']}" if story else f"seed{final['seed']}"


def voice_banner(im, level="hazard"):
    """음성 경고가 나오는 동안 바디캠 화면 위쪽에 띠: 위험(빨강, 강한 경고) / 주의(주황, 약한 안내)."""
    d = ImageDraw.Draw(im, "RGBA")
    bx0, by0, bx1, _ = BODY
    if level == "hazard":
        col, msg = (200, 20, 30, 225), "음성 경고 (강함)  \"멈추세요! 위험 요소가 식별되었습니다\""
    else:
        col, msg = (190, 120, 20, 225), "주의 안내 (약함)  \"발밑을 확인하세요\""
    d.rectangle([bx0, by0 + 42, bx1 - 1, by0 + 106], fill=col)
    text(d, ((bx0 + bx1) // 2, by0 + 74), msg, 28, (255, 255, 255), True, anchor="mm")


def _sentences(s):
    """자막을 문장 단위로 (음성 진행에 맞춰 차례로 보여 줌)."""
    import re
    parts = [p.strip() for p in re.split(r"(?<=[.!?。!?])\s*", s) if p.strip()]
    return parts or [s]


def assist_caption(im, ev, progress):
    """손동작 안내 자막: 명령, 작업자 언어 문장 (음성 진행에 맞춰), 한국어 번역."""
    d = ImageDraw.Draw(im, "RGBA")
    lang = ev["lang"]

    def current(s):
        sents = _sentences(s)
        total = sum(len(x) for x in sents)
        acc = 0
        for x in sents:
            acc += len(x)
            if acc / total >= min(max(progress, 0.0), 1.0) - 1e-6:
                return x
        return sents[-1]

    bx0, _, bx1, by1 = BODY
    f_lang, f_ko = lang_font(lang, 30, True), font(24)
    head = f"손가락 {ev['count']} → {ev['cmd']}  ·  {ev['lang_name']} 안내"
    llm = ev.get("llm") or {}
    ai = (f"AI 판단 ({llm.get('model')}): " + " → ".join(llm.get("trace") or [])) if llm.get("llm") else ""
    cur = current(ev["text"])
    cur_ko = current(ev["text_ko"])

    def lines_of(s, f, width, n):
        out, line = [], ""
        for ch in s:
            if d.textlength(line + ch, font=f) > width:
                out.append(line)
                line = ch
            else:
                line += ch
        out.append(line)
        return out[:n]
    width = bx1 - bx0 - 40
    l1 = lines_of(cur, f_lang, width, 2)
    l2 = lines_of(cur_ko, f_ko, width, 2)
    lh1, lh2 = 40, 34
    h = 44 + (28 if ai else 0) + lh1 * len(l1) + lh2 * len(l2) + 12
    top = min(by1 - h, by1 - 140)          # 아래 에이전트 자막을 다 덮게
    d.rectangle([bx0, top, bx1 - 1, by1 - 1], fill=(0, 34, 70, 250))
    d.rectangle([bx0, top, bx1 - 1, top + 4], fill=(0, 160, 240))
    text(d, (bx0 + 14, top + 10), head, 24, (120, 220, 255), True)
    y = top + 44
    if ai:
        text(d, (bx0 + 14, y), fit(d, ai, 17, width), 17, (190, 150, 255))
        y += 28
    for ln in l1:
        d.text((bx0 + 14, y), ln, font=f_lang, fill=(255, 255, 255))
        y += lh1
    for ln in l2:
        d.text((bx0 + 14, y), ln, font=f_ko, fill=(190, 200, 215))
        y += lh2


def add_audio(video, clips, total_s):
    """무음 트랙에 음성 (경고, 손동작 안내) 을 시각마다 얹고 영상과 합친다. clips: [(시각 초, wav)]."""
    import wave
    import imageio_ffmpeg
    rate, cache = 22050, {}
    track = np.zeros(int(total_s * rate) + rate, np.int32)
    for m, wav in clips:
        if wav not in cache:
            with wave.open(wav, "rb") as w:
                data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.int32)
                if w.getframerate() != rate:
                    idx = np.arange(0, len(data), w.getframerate() / rate).astype(int)
                    data = data[idx[idx < len(data)]]
            cache[wav] = data
        voice = cache[wav]
        i = int(m * rate)
        j = min(len(track), i + len(voice))
        if j > i:
            track[i:j] += voice[:j - i]
    track = np.clip(track, -32768, 32767).astype(np.int16)
    tmp_wav = video + ".audio.wav"
    with wave.open(tmp_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(track.tobytes())
    tmp_mp4 = video + ".tmp.mp4"
    os.replace(video, tmp_mp4)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-i", tmp_mp4, "-i", tmp_wav, "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "128k", "-shortest", video], check=True)
    os.remove(tmp_mp4)
    os.remove(tmp_wav)


if __name__ == "__main__":
    main()
