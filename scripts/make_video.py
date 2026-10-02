"""시연 영상 만들기 (일반 파이썬). run_patrol.py --record 로 저장한 화면과 에이전트 기록을 1920x1080 영상으로 합친다.

    python scripts/make_video.py --record outputs/record/seed1 --out outputs/video/demo.mp4

화면 구성: 위쪽 바디캠(입력, YOLO 판정) | CCTV 3대 + 확대(PTZ) 화면(도구)
          아래쪽 평면도(작업자 위치, 위험물 대장) | 위험물 대장 | 에이전트 기록(계획, 판정, 재확인, 경고)
앞뒤로 제목, 문제와 구조, 점검 계획, 조치 지시서 화면, 평가 결과를 붙인다. 소리는 없음.
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
WIDTH, HEIGHT = 1920, 1080
FONT = "C:/Windows/Fonts/malgun.ttf"
FONT_B = "C:/Windows/Fonts/malgunbd.ttf"
BG = (18, 24, 34)
PANEL = (28, 36, 50)
FG = (232, 236, 242)
MUTED = (150, 160, 178)
RED, GREEN, AMBER, BLUE, ORANGE, PURPLE = (232, 72, 72), (60, 184, 110), (240, 170, 40), (80, 150, 240), (255, 140, 40), (170, 110, 230)
KIND_COLOR = {"계획": BLUE, "판정": (110, 200, 255), "판정 수정": PURPLE, "재확인 계획": AMBER, "CCTV 선택": AMBER,
              "재확인": AMBER, "재시도": AMBER, "재확인 취소": MUTED, "현장 확인": RED, "접근 경고": RED, "점검표": GREEN}
# 영상에서 잠깐 멈춰 보여줄 행동
HOLD_KINDS = ("판정 수정", "재확인", "현장 확인", "재확인 취소", "위치 보정")
HOLD_S = 1.5
# 자막으로 보여줄 행동 (앞에 있을수록 우선)
CAPTION_KINDS = ["판정 수정", "접근 경고", "현장 확인", "재확인", "재시도", "CCTV 선택", "재확인 계획", "재확인 취소", "점검표", "판정"]
_fonts = {}


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
    for n, cx, cy, cz, yaw, pitch, vfov in W.CCTVS:
        c = mp(cx, cy, ox, oy)
        a = math.radians(yaw)
        fan = [c] + [mp(cx + 9 * math.cos(a + s), cy + 9 * math.sin(a + s), ox, oy) for s in np.linspace(-0.55, 0.55, 7)]
        d.polygon(fan, fill=(60, 60, 60, 28))
        d.rectangle([c[0] - 6, c[1] - 6, c[0] + 6, c[1] + 6], fill=(40, 40, 40))
        text(d, (c[0], c[1] + (10 if cy < 0 else -24)), n.split("_")[1], 12, (40, 40, 40), True, anchor="ma")
    ptz = st.get("ptz")
    if ptz:
        cam = next(c for c in W.CCTVS if c[0] == ptz["cam"])
        d.line([mp(cam[1], cam[2], ox, oy), mp(ptz["target"][0], ptz["target"][1], ox, oy)], fill=ORANGE, width=3)
        tx, ty = mp(ptz["target"][0], ptz["target"][1], ox, oy)
        d.ellipse([tx - 13, ty - 13, tx + 13, ty + 13], outline=ORANGE, width=3)
    wx, wy, wyaw = st["worker"]
    for fx, fy in alerts:
        d.line([mp(wx, wy, ox, oy), mp(fx, fy, ox, oy)], fill=RED, width=4)
    for f in st["findings"]:
        if f["status"] == "기각" or not f["cls"]:
            continue
        x, y = mp(f["x"], f["y"], ox, oy)
        if f["status"] in ("재확인 대기", "현장 확인 필요"):
            col = AMBER
        else:
            col = RED if HAZARD[f["cls"]] else GREEN
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
def header(d, t, stage, now_text, live=True):
    d.rectangle([0, 0, WIDTH, 66], fill=(12, 16, 24))
    text(d, (22, 12), "창고 안전 순찰 AI 에이전트", 30, bold=True)
    text(d, (400, 22), f"Isaac Sim 시뮬레이션 · {'실시간 1배속' if live else ''} · {clock(t)}", 18, fill=MUTED)
    steps = ["① 입력: 바디캠·CCTV", "② 판단: YOLO 위험/안전", "③ 도구: CCTV 확대·경고", "④ 결과: 대장·조치 지시서"]
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


def compose(rec, st, tl, last_ptz, path_pts, cache):
    k, t = st["k"], st["t"]
    im = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(im, "RGBA")
    recent = [e for e in tl[:st["n_timeline"]] if e["kind"] != "계획" or e["t"] > 0]
    window = [e for e in recent if t - e["t"] <= 2.0]
    alerting = [e for e in window if e["kind"] == "접근 경고"]
    if st.get("ptz") or alerting or any(e["kind"] in ("재확인", "CCTV 선택", "재시도", "재확인 계획", "현장 확인") for e in window):
        stage = 2
    elif any(e["kind"] in ("판정", "판정 수정") for e in window):
        stage = 1
    else:
        stage = 0
    now = recent[-1]["text"] if recent else ""
    header(d, t, stage, now)
    # 바디캠 (순찰이 끝난 뒤 확대 재확인 중이면 확대 화면을 크게)
    body_path = os.path.join(rec, f"body_{k:05d}.jpg")
    if os.path.exists(body_path):
        im.paste(load(body_path, (960, 540)), (0, 70))
        panel_label(d, 8, 78, "작업자 바디캠 + YOLO 판정")
        cache["body"] = body_path
    elif st.get("ptz") and os.path.exists(os.path.join(rec, f"ptz_{k:05d}.jpg")):
        im.paste(load(os.path.join(rec, f"ptz_{k:05d}.jpg"), (960, 540)), (0, 70))
        d.rectangle([0, 70, 959, 609], outline=ORANGE, width=6)
        what = st["ptz"]["finding"] or st["ptz"]["checkpoint"]
        panel_label(d, 8, 78, f"순찰 끝: 남은 점검 지점 확인 · CCTV {st['ptz']['cam'].split('_')[1]} 확대 → {what}", ORANGE)
    elif cache.get("body"):
        im.paste(load(cache["body"], (960, 540)), (0, 70))
        panel_label(d, 8, 78, "작업자 바디캠 + YOLO 판정")
    # 바디캠 아래 자막: 최근 2.5초 안의 중요한 에이전트 행동을 크게
    key = [e for e in recent if t - e["t"] <= 2.5 and e["kind"] in CAPTION_KINDS]
    if key:
        e = max(key, key=lambda e: (CAPTION_KINDS.index(e["kind"]) * -1, e["t"]))
        lines = wrap(d, e["text"], 22, 930, 2)
        top_y = 609 - 46 - 32 * len(lines)
        d.rectangle([0, top_y, 959, 609], fill=(0, 0, 0, 175))
        text(d, (14, top_y + 8), e["kind"], 24, KIND_COLOR.get(e["kind"], FG), True)
        for i, ln in enumerate(lines):
            text(d, (14, top_y + 44 + 32 * i), ln, 22, FG)
    # CCTV 2x2
    cells = [("cctv_west", (962, 70)), ("cctv_east", (1442, 70)), ("cctv_south", (962, 342))]
    alert_cams = {e["text"].split(":")[0] for e in alerting}
    for n, (x, y) in cells:
        cp = os.path.join(rec, f"{n}_{k:05d}.jpg")
        if os.path.exists(cp):
            cache[n] = cp
        c = load(cache.get(n), (478, 270))
        if c:
            im.paste(c, (x, y))
        if n in alert_cams:
            d.rectangle([x, y, x + 477, y + 269], outline=RED, width=5)
        panel_label(d, x + 6, y + 6, f"CCTV {n.split('_')[1]}", RED if n in alert_cams else FG)
    px, py = 1442, 342
    if st.get("ptz"):
        last_ptz[0] = os.path.join(rec, f"ptz_{k:05d}.jpg")
        last_ptz[1] = st["ptz"]
    pimg = load(last_ptz[0], (478, 270))
    if pimg:
        im.paste(pimg, (px, py))
    else:
        d.rectangle([px, py, px + 477, py + 269], fill=PANEL)
        text(d, (px + 239, py + 135), "확대 재확인 대기", 22, MUTED, anchor="mm")
    if st.get("ptz"):
        d.rectangle([px, py, px + 477, py + 269], outline=ORANGE, width=5)
        what = st["ptz"]["finding"] or st["ptz"]["checkpoint"]
        panel_label(d, px + 6, py + 6, f"CCTV {st['ptz']['cam'].split('_')[1]} 확대 → {what}", ORANGE)
    else:
        panel_label(d, px + 6, py + 6, "CCTV 확대(PTZ) 최근 화면", MUTED)
    # 아래: 평면도 | 대장 | 기록
    top = 618
    d.rectangle([0, top, WIDTH, HEIGHT], fill=PANEL)
    alerts = []
    for e in alerting:
        fid = e["text"].split("대장 ")[-1].split()[0]
        f = next((f for f in st["findings"] if f["id"] == fid), None)
        if f:
            alerts.append((f["x"], f["y"]))
    mimg = Image.new("RGB", (int(21.6 * MAP_S), int(31.0 * MAP_S)), (236, 232, 222))
    draw_map(ImageDraw.Draw(mimg, "RGBA"), 0, 0, st, alerts, path_pts)
    im.paste(mimg, (8, top + 8))
    # 위험물 대장
    lx = 340
    text(d, (lx, top + 10), "위험물 대장 (기억)", 22, bold=True)
    rows = [f for f in st["findings"] if f["status"] != "기각" and f["cls"]]
    y = top + 44
    for f in rows[-14:]:
        if f["status"] in ("재확인 대기", "현장 확인 필요"):
            col, s = AMBER, "확인중" if f["status"] == "재확인 대기" else "현장확인"
        else:
            col, s = (RED, "위험") if HAZARD[f["cls"]] else (GREEN, "안전")
        d.rounded_rectangle([lx, y, lx + 64, y + 24], 5, fill=col)
        text(d, (lx + 32, y + 12), s, 15, (255, 255, 255), True, anchor="mm")
        line = f"{f['id']} {CLASS_KO[f['cls']]}" + (" · 재확인" if f["status"] == "재확인 완료" else "") + \
               (f" · 접근 {f['near_miss']}" if f["near_miss"] else "")
        text(d, (lx + 72, y + 1), fit(d, line, 17, 430), 17)
        y += 30
    # 에이전트 기록
    gx = 870
    text(d, (gx, top + 10), "에이전트 기록 (계획 → 판정 → 재확인 → 결과)", 22, bold=True)
    y = top + 46
    for e in recent[-12:]:
        col = KIND_COLOR.get(e["kind"], FG)
        age = t - e["t"]
        if age < 1.5:
            d.rectangle([gx - 6, y - 2, WIDTH - 10, y + 28], fill=(60, 50, 30))
        text(d, (gx, y), clock(e["t"]), 17, MUTED)
        text(d, (gx + 58, y), e["kind"], 17, col, True)
        text(d, (gx + 168, y), fit(d, e["text"], 17, WIDTH - gx - 190), 17)
        y += 34
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
    boxes = [("입력", "작업자 가슴 바디캠\nCCTV 3대", BLUE), ("판단", "YOLO26 위험/안전 판정\n(10개 클래스)", (110, 200, 255)),
             ("계획·도구", "애매하면 CCTV 선택\nPTZ 확대로 재확인\n작업자 접근 경고", AMBER),
             ("기억·평가", "위험물 대장\n판정 수정, 재시도\n현장 확인 요청", PURPLE), ("결과", "조치 지시서\n우선순위·위치·조치", GREEN)]
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
    lines = ["위험 요소 4종: 바닥 유출 · 방치된 공구 · 불안정 적재 · 소화기 상태 (같은 종류의 안전한 상태와 함께 배치)",
             "정답표를 미리 만들어 두고 에이전트 판정을 채점 (판정에는 정답표를 쓰지 않음)",
             "NVIDIA Isaac Sim 6.0 실사 창고 · 합성 데이터 4000장으로 학습 · 실제 사진 없음"]
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
        header(d, 0, 3, "", live=False)
        yield im


def results_card(eval_dir):
    files = sorted(glob.glob(os.path.join(eval_dir, "inspection_seed*.json")))
    rs = [json.load(open(f, encoding="utf-8")) for f in files]
    rs = [r for r in rs if r.get("agent")]
    if not rs:
        return None
    tot = lambda part, k: sum(r["agent"]["evaluation"][part][k] for r in rs)  # noqa: E731
    c = [r["cctv"] for r in rs if r.get("cctv")]
    ev = sum(x["events_detected"] for x in c), sum(x["events_visible"] for x in c), sum(x["false_alert_episodes"] for x in c)
    rc = sum(r["agent"]["evaluation"]["recheck"]["recheck_run"] for r in rs)
    ch = sum(r["agent"]["evaluation"]["recheck"]["recheck_changed"] for r in rs)
    im = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(im)
    text(d, (WIDTH // 2, 70), f"평가: 학습에 안 쓴 배치 {len(rs)}개를 정답표와 비교", 44, bold=True, anchor="mt")
    text(d, (WIDTH // 2, 140), "창고 전체 물체 기준 (경로에서 안 보이는 물체 포함)", 24, MUTED, anchor="mt")
    rows = [("", "바디캠만", "에이전트 (재확인 + 점검표)"),
            ("위험 물체를 위험으로", f"{tot('before', 'hazard_found')}/{tot('before', 'hazard_total')}",
             f"{tot('after', 'hazard_found')}/{tot('after', 'hazard_total')}"),
            ("안전 물체를 안전으로", f"{tot('before', 'safe_ok')}/{tot('before', 'safe_total')}",
             f"{tot('after', 'safe_ok')}/{tot('after', 'safe_total')}"),
            ("위험↔안전 거꾸로 판정", f"{tot('before', 'hazard_as_safe') + tot('before', 'safe_as_hazard')}",
             f"{tot('after', 'hazard_as_safe') + tot('after', 'safe_as_hazard')}"),
            ("없는 위험을 보고", f"{tot('before', 'false_reports')}", f"{tot('after', 'false_reports')}")]
    y = 230
    for i, (a, b, cc) in enumerate(rows):
        bold = i == 0
        d.rectangle([360, y - 8, 1560, y + 50], fill=PANEL if i % 2 else BG)
        text(d, (400, y), a, 30, MUTED if bold else FG, bold)
        text(d, (1050, y), b, 30, MUTED if bold else FG, bold, anchor="ma")
        text(d, (1380, y), cc, 30, ORANGE if bold else FG, True, anchor="ma")
        y += 66
    y += 30
    text(d, (WIDTH // 2, y), f"CCTV 확대 재확인 {rc}건 (판정 고침 {ch}건) · CCTV 접근 경고 {ev[0]}/{ev[1]}건 · 오경보 {ev[2]}번", 28,
         AMBER, anchor="mt")
    text(d, (WIDTH // 2, y + 60), "YOLO26s 검증 mAP50 0.923 · 합성 데이터만으로 학습, 실제 현장 적용 전 실사 검증 필요", 24, MUTED, anchor="mt")
    return im


def main():
    p = argparse.ArgumentParser(description="시연 영상 합치기")
    p.add_argument("--record", default=os.path.join(ROOT, "outputs", "record", "seed1"))
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
        compose(rec, st, tl, [None, None], path_pts, {}).save(out)
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
               ("작업자 바디캠과 CCTV 로 창고를 순찰하며 위험/안전을 판정하고,", 28, FG, False),
               ("애매한 것은 CCTV 를 골라 확대해 다시 확인한 뒤 조치 지시서를 만드는 AI 에이전트", 28, FG, False),
               ("", 20, FG, False), (f"NVIDIA Isaac Sim 6.0 디지털 트윈 · YOLO26 · 시나리오 {seed}", 24, MUTED, False)],
              title="창고 안전 순찰 AI 에이전트"), 5)
    emit(workflow_card(), 8)
    emit(plan_card(tl), 6)
    last_ptz = [None, None]
    cache = {}
    walk = [s for s in states]
    prev_n, alerted = 0, False
    for i, st in enumerate(walk):
        nxt = walk[i + 1]["t"] if i + 1 < len(walk) else st["t"] + 0.4
        dt = max(0.0, nxt - st["t"])
        if st.get("ptz") and i > 0 and st["t"] == walk[i - 1]["t"]:
            dt = 0.6
        im = compose(rec, st, tl, last_ptz, path_pts, cache)
        emit(im, frames=max(1, int(round(max(dt, 0.6 if st.get("ptz") and dt < 0.05 else dt) * FPS))))
        # 주요 장면 (판정 수정, 재확인 결과, 현장 확인, 첫 접근 경고) 은 1.5초 멈춰서 읽을 수 있게
        new = tl[prev_n:st["n_timeline"]]
        prev_n = st["n_timeline"]
        key = [e for e in new if e["kind"] in HOLD_KINDS or (e["kind"] == "접근 경고" and not alerted)]
        if key and dt < 0.3:
            alerted |= any(e["kind"] == "접근 경고" for e in key)
            d = ImageDraw.Draw(im)
            d.rounded_rectangle([730, 78, 952, 112], 8, fill=ORANGE)
            text(d, (841, 95), "주요 장면 · 잠깐 멈춤", 18, (20, 20, 20), True, anchor="mm")
            emit(im, HOLD_S)
    html_path = os.path.join(ROOT, "outputs", "agent", f"dashboard_seed{seed}.html")
    png = screenshot(html_path, os.path.join(tempfile.gettempdir(), f"dash_{seed}.png"))
    if png:
        for im in dashboard_frames(png):
            emit(im, frames=1)
    res = results_card(a.eval_dir)
    if res:
        emit(res, 10)
    writer.close()
    print(f"[완료] {a.out}  ({n[0] / FPS:.0f}초)")


if __name__ == "__main__":
    main()
