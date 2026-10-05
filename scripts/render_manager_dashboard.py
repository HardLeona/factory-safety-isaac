"""BodyGuard v2 시연 영상용: 관리자 알림 피드 화면을 프레임 시퀀스로 그린다 (Isaac Sim 불필요, 순수 합성).

스토리보드 ④ "남기기" 단계: 에이전트가 자동으로 보낸 알림(끼임 경보, 지게차 접근, 주의 분류)이
관리자 화면에 시각·구역·사건으로 쌓이고, 관리자가 "확인" 을 누르면 처리 완료 상태로 바뀐다.

    python scripts/render_manager_dashboard.py --out outputs/record/v2_manager

각 scripts/record_scenario.py 결과(outputs/record/v2_*/final.json)의 manager_notifications 를 실제로
읽어서 쓴다 (가짜 데이터 아님). 알림이 하나씩 피드에 나타나고, 가상 커서가 "확인" 버튼을 눌러 상태가
바뀌는 과정을 프레임으로 저장한다 (body_XXXXX.jpg, make_video2.py 가 이어 붙임).
"""
import argparse
import glob
import json
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

parser = argparse.ArgumentParser(description="관리자 알림 피드 화면 녹화 (상호작용 포함)")
parser.add_argument("--record-dir", default=os.path.join(ROOT, "outputs", "record"), help="v2_* 녹화 결과가 있는 폴더")
parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "record", "v2_manager"))
parser.add_argument("--fps", type=int, default=30)
args = parser.parse_args()

WIDTH, HEIGHT = 1920, 1080
FPS = args.fps
BG = (16, 20, 28)
PANEL = (26, 32, 44)
PANEL_HOVER = (32, 40, 54)
FG = (232, 236, 242)
MUTED = (146, 156, 174)
CRITICAL = (232, 72, 72)
CAUTION = (190, 150, 60)
DONE = (60, 184, 110)
BLUE = (80, 150, 240)
FONT = "C:/Windows/Fonts/malgun.ttf"
FONT_B = "C:/Windows/Fonts/malgunbd.ttf"

_fonts = {}


def font(size, bold=False):
    k = (size, bold)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(FONT_B if bold else FONT, size)
    return _fonts[k]


def text(d, xy, s, size=20, fill=FG, bold=False, anchor="la"):
    d.text(xy, s, font=font(size, bold), fill=fill, anchor=anchor)


def clock(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


LEVEL_STYLE = {
    "critical": (CRITICAL, "긴급"),
    "caution": (CAUTION, "주의"),
}


def load_notifications():
    """outputs/record/v2_*/final.json 에서 실제 manager_notifications 를 모아, 시연 순서(끼임→지게차→주의)로 정렬."""
    order = {"v2_pinch": 0, "v2_forklift": 1, "v2_spill": 2, "v2_sign": 3}
    items = []
    for path in sorted(glob.glob(os.path.join(args.record_dir, "v2_*", "final.json"))):
        name = os.path.basename(os.path.dirname(path))
        if name not in order:
            continue
        d = json.load(open(path, encoding="utf-8"))
        for n in d["report"].get("manager_notifications", []):
            items.append({**n, "clip_t": round(float(n["t"]), 1), "_order": order[name]})
    items.sort(key=lambda n: n["_order"])
    return items


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def card_rect(idx, n_total, reveal=1.0):
    """idx 번째 카드의 (x0, y0, x1, y1). reveal<1 이면 위에서 슬라이드 인 중."""
    x0, w = 420, 1080
    y0 = 200 + idx * 150
    y0 -= (1.0 - reveal) * 60
    return x0, y0, x0 + w, y0 + 120


def draw_header(d, t):
    d.rectangle([0, 0, WIDTH, 84], fill=(10, 14, 20))
    text(d, (36, 16), "BodyGuard 관리자 화면", 32, bold=True)
    text(d, (36, 54), "에이전트가 자동으로 보낸 알림 — 작업자가 직접 부르지 않아도 쌓인다", 18, fill=MUTED)
    text(d, (WIDTH - 36, 30), clock(t), 26, fill=MUTED, anchor="ra")


def draw_card(im, d, n, idx, n_total, reveal, ack_u, hover=False):
    """카드 한 장을 im(RGB) 에 그린다. reveal<1 이면 위에서 슬라이드 인 중(위치만 바뀜, 불투명)."""
    if reveal < 0.03:
        return None
    x0, y0, x1, y1 = card_rect(idx, n_total, reveal)
    color, label = LEVEL_STYLE[n["level"]]
    acked = ack_u >= 1.0
    bg = (24, 30, 26) if acked else (PANEL_HOVER if hover else PANEL)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([x0, y0, x1, y1], 14, fill=bg)
    bar_col = DONE if acked else color
    d.rounded_rectangle([x0, y0, x0 + 8, y1], 4, fill=bar_col)
    pad = 28
    badge_w = 74
    badge_col = DONE if acked else bar_col
    d.rounded_rectangle([x0 + pad, y0 + 18, x0 + pad + badge_w, y0 + 44], 6, fill=badge_col)
    text(d, (x0 + pad + badge_w / 2, y0 + 31), "완료" if acked else label, 15, fill=(18, 18, 18), bold=True, anchor="mm")
    text(d, (x0 + pad + badge_w + 16, y0 + 18), f"{clock(n['clip_t'])} · {n['zone']}", 17, fill=MUTED, anchor="la")
    summary = n["summary"]
    text(d, (x0 + pad, y0 + 54), summary if len(summary) <= 58 else summary[:57] + "…", 19, fill=FG if not acked else MUTED)
    # 확인 버튼
    bw, bh = 108, 40
    bx0, by0 = x1 - pad - bw, y0 + (y1 - y0) / 2 - bh / 2
    if acked:
        d.rounded_rectangle([bx0, by0, bx0 + bw, by0 + bh], 8, outline=DONE, width=2)
        cx, cy = bx0 + 22, by0 + bh / 2
        d.line([(cx - 7, cy), (cx - 2, cy + 6), (cx + 9, cy - 8)], fill=DONE, width=3, joint="curve")
        text(d, (bx0 + bw / 2 + 12, by0 + bh / 2), "확인됨", 16, fill=DONE, bold=True, anchor="mm")
    else:
        press = ease(min(1.0, ack_u * 3)) if 0 < ack_u < 1.0 else 0.0
        fill_col = tuple(int(c1 + (c2 - c1) * press) for c1, c2 in zip(PANEL, BLUE))
        d.rounded_rectangle([bx0, by0, bx0 + bw, by0 + bh], 8, fill=fill_col, outline=BLUE, width=2)
        text(d, (bx0 + bw / 2, by0 + bh / 2), "확인", 17, fill=FG, bold=True, anchor="mm")
    return bx0 + bw / 2, by0 + bh / 2


def draw_cursor(d, pos):
    x, y = pos
    pts = [(x, y), (x, y + 18), (x + 5, y + 13), (x + 9, y + 20), (x + 12, y + 18), (x + 8, y + 11), (x + 15, y + 11)]
    d.polygon(pts, fill=(255, 255, 255), outline=(10, 10, 10))


def draw_summary(d, items, acked_n):
    y = 150
    text(d, (36, 200 - 40), "알림 피드", 22, bold=True)
    total, done = len(items), acked_n
    text(d, (WIDTH - 36, 200 - 40), f"처리 {done}/{total}", 20, fill=MUTED, anchor="ra")


def main():
    os.makedirs(args.out, exist_ok=True)
    items = load_notifications()
    if not items:
        print("[경고] manager_notifications 가 없습니다. scripts/record_scenario.py 를 먼저 돌리세요.")
        return
    print(f"[관리자 화면] 알림 {len(items)}건: " + ", ".join(f"{n['id']}({n['level']})" for n in items))

    frames = []

    def emit(im, n=1):
        for _ in range(n):
            frames.append(im.copy())

    t_vid = 0.0
    im = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(im)
    draw_header(d, t_vid)
    draw_summary(d, items, 0)
    emit(im, int(0.6 * FPS))

    REVEAL_S, HOLD_S, CLICK_S, SETTLE_S = 0.5, 0.9, 0.45, 0.7
    acked = [False] * len(items)
    for i, n in enumerate(items):
        # 카드 슬라이드 인
        steps = int(REVEAL_S * FPS)
        for k in range(steps + 1):
            im = Image.new("RGB", (WIDTH, HEIGHT), BG)
            d = ImageDraw.Draw(im)
            draw_header(d, t_vid)
            draw_summary(d, items, sum(acked))
            for j in range(i + 1):
                reveal = 1.0 if j < i else k / steps
                draw_card(im, None, items[j], j, len(items), reveal, 1.0 if acked[j] else 0.0)
            emit(im)
        # 잠깐 멈춰서 알림을 보여줌
        im = Image.new("RGB", (WIDTH, HEIGHT), BG)
        d = ImageDraw.Draw(im)
        draw_header(d, t_vid)
        draw_summary(d, items, sum(acked))
        btn_pos = None
        for j in range(i + 1):
            r = draw_card(im, None, items[j], j, len(items), 1.0, 1.0 if acked[j] else 0.0)
            if j == i:
                btn_pos = r
        emit(im, int(HOLD_S * FPS))
        # 커서가 확인 버튼으로 이동해 클릭
        cur_from = (WIDTH - 200, HEIGHT - 120)
        steps = int(CLICK_S * FPS)
        for k in range(steps + 1):
            u = ease(k / steps)
            cx = cur_from[0] + (btn_pos[0] - cur_from[0]) * u
            cy = cur_from[1] + (btn_pos[1] - cur_from[1]) * u
            im = Image.new("RGB", (WIDTH, HEIGHT), BG)
            d = ImageDraw.Draw(im)
            draw_header(d, t_vid)
            draw_summary(d, items, sum(acked))
            for j in range(i + 1):
                hover = j == i and k >= steps - 2
                draw_card(im, None, items[j], j, len(items), 1.0, 1.0 if acked[j] else (0.5 if (j == i and k >= steps - 2) else 0.0), hover=hover)
            d = ImageDraw.Draw(im)
            draw_cursor(d, (cx, cy))
            emit(im)
        acked[i] = True
        # 처리 완료 상태로 전환, 잠깐 멈춤
        im = Image.new("RGB", (WIDTH, HEIGHT), BG)
        d = ImageDraw.Draw(im)
        draw_header(d, t_vid)
        draw_summary(d, items, sum(acked))
        for j in range(i + 1):
            draw_card(im, None, items[j], j, len(items), 1.0, 1.0 if acked[j] else 0.0)
        d = ImageDraw.Draw(im)
        draw_cursor(d, btn_pos)
        emit(im, int(SETTLE_S * FPS))

    # 마지막: 전부 처리된 화면 유지
    im = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(im)
    draw_header(d, t_vid)
    draw_summary(d, items, sum(acked))
    for j in range(len(items)):
        draw_card(im, None, items[j], j, len(items), 1.0, 1.0)
    emit(im, int(1.2 * FPS))

    for k, f in enumerate(frames):
        f.convert("RGB").save(os.path.join(args.out, f"body_{k:05d}.jpg"), quality=90)
    print(f"[완료] 프레임 {len(frames)}장 ({len(frames) / FPS:.1f}초) -> {args.out}")


if __name__ == "__main__":
    main()
