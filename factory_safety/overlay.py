"""YOLO 결과를 한국어 이름으로 그리기 (영상, 보고서용). 공구는 종류 + 놓인 자리 (바닥이면 빨강, 작업대면 초록)."""
import os

from PIL import Image, ImageDraw, ImageFont

from .config import CLASS_KO, HAZARD, TOOL_KO, TOOL_TYPES

FONT = "C:/Windows/Fonts/malgunbd.ttf"
RED, GREEN, ORANGE, YELLOW, WHITE = (235, 60, 60), (60, 200, 110), (255, 150, 30), (255, 215, 0), (240, 240, 240)
SHORT = {"spill": "방치된 유출", "spill_marked": "조치된 유출", "stack_unstable": "무너질 듯한 적재", "stack_stable": "반듯한 적재",
         "ext_fallen": "쓰러진 소화기", "ext_blocked": "가로막힌 소화기", "ext_ok": "정상 소화기", "worker": "작업자",
         "cone": "라바콘", "danger_sign": "DANGER 표지"}
_font = {}


def font(size):
    if size not in _font:
        try:
            _font[size] = ImageFont.truetype(FONT if os.path.exists(FONT) else "arial.ttf", size)
        except OSError:
            _font[size] = ImageFont.load_default()
    return _font[size]


def label_of(name, tool_state=None):
    """(글자, 색)."""
    if name in TOOL_TYPES:
        where = {"tool_floor": "바닥 · 위험", "tool_stored": "작업대 · 정리"}.get(tool_state, "")
        return f"{TOOL_KO[name]}" + (f" ({where})" if where else ""), RED if tool_state == "tool_floor" else GREEN if tool_state else ORANGE
    if name in ("cone", "danger_sign"):
        return SHORT[name], YELLOW
    if name == "worker":
        return SHORT[name], WHITE
    if name == "cart":
        return CLASS_KO[name], (90, 170, 255)
    return SHORT.get(name, CLASS_KO.get(name, name)), RED if HAZARD.get(name) else GREEN


def draw(img, dets, tool_state=None, size=17, width=3):
    """img: numpy (H, W, 3). dets: [(이름, 신뢰도, xyxy, ...)]. tool_state(xyxy) -> 'tool_floor'|'tool_stored'."""
    im = Image.fromarray(img) if not isinstance(img, Image.Image) else img
    d = ImageDraw.Draw(im)
    f = font(size)
    for name, conf, b, *_ in dets:
        st = tool_state(b) if (tool_state and name in TOOL_TYPES) else None
        text, col = label_of(name, st)
        x0, y0, x1, y1 = [float(v) for v in b]
        d.rectangle([x0, y0, x1, y1], outline=col, width=width)
        s = f"{text} {conf * 100:.0f}%"
        tw = d.textlength(s, font=f)
        ty = y0 - size - 6 if y0 > size + 6 else y1 + 2
        d.rectangle([x0, ty, x0 + tw + 8, ty + size + 5], fill=col)
        d.text((x0 + 4, ty + 1), s, font=f, fill=(15, 15, 15))
    return im


BADGE_COLOR = {"LLM 판단": (90, 170, 255), "판정": (235, 60, 60), "주의": (138, 95, 209)}


def draw_status_badge(im, text, kind="판정"):
    """바디캠 화면 아래쪽에 처리 경로/상태 전환 문구를 찍는다 (시연 영상에서 바로 보이도록, --record 전용)."""
    d = ImageDraw.Draw(im)
    f = font(16)
    col = BADGE_COLOR.get(kind, (90, 90, 90))
    w, h = im.size
    tw = d.textlength(text, font=f)
    x0, y0 = 8, h - 32
    d.rectangle([x0, y0, min(x0 + tw + 14, w - 8), y0 + 26], fill=col)
    d.text((x0 + 7, y0 + 4), text, font=f, fill=(255, 255, 255))
    return im


HAND_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def draw_hand(im, pts, count, label=None, scale=1.0):
    """손 관절 21점과 인식한 손가락 수를 그린다 (PIL 이미지에 바로)."""
    from PIL import ImageDraw
    if not pts:
        return im
    d = ImageDraw.Draw(im)
    p = [(x * scale, y * scale) for x, y in pts]
    for a, b in HAND_LINKS:
        d.line([p[a], p[b]], fill=(80, 220, 255), width=3)
    for x, y in p:
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 255, 255), outline=(0, 150, 220))
    x0, y0 = min(x for x, _ in p), min(y for _, y in p)
    text = f"손가락 {count}" + (f" · {label}" if label else "") if count else "손 인식"
    f = font(22)
    tw = d.textlength(text, font=f)
    d.rectangle([x0, y0 - 34, x0 + tw + 14, y0 - 4], fill=(0, 110, 190))
    d.text((x0 + 7, y0 - 32), text, fill=(255, 255, 255), font=f)
    return im
