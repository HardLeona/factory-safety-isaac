"""BodyGuard v2 시연 영상: 스토리보드 4장 + 내레이션(TTS) + 실제 Isaac Sim 시연 클립 + 관리자 화면.

    .venv-assistant/Scripts/python scripts/make_video2.py --out outputs/video/bodyguard_v2.mp4

흐름: 표지 이미지 → (스토리보드 이미지 + 내레이션 → 그 장면의 실제 시연 클립) × 4(지게차 → 끼임 → 주의·적재물 → 표지판 질의,
스토리보드에 박힌 번호 순서) → 관리자 화면(알림 피드 + 확인 상호작용) + 내레이션 → 마무리 이미지.
지게차·끼임 경보는 베트남어 고정 문구, 표지판 질문은 그 장면에서 실제로 생성된 베트남어 답변 음성을 쓴다.
내레이션은 scripts/make_narration.py, 시연 클립은 scripts/record_scenario.py --which {forklift,pinch,spill,sign},
관리자 화면은 scripts/render_manager_dashboard.py 가 미리 만들어 둔 결과를 그대로 읽어 붙인다 (이 스크립트는 합성만 한다).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_video import (  # noqa: E402
    FG, FPS, HEIGHT, KIND_COLOR, MUTED, WIDTH,
    add_audio, font, text, wrap,
)

# 리포(또는 이 워크트리)가 어디 있든, 스토리보드는 항상 AI_SW_CONTEST/agent_senario 에 있다
STORYBOARD_DIR = r"C:\Users\user\Desktop\AI_SW_CONTEST\agent_senario"
NARRATION_DIR = os.path.join(ROOT, "outputs", "video", "narration")
ALERT_DIR = os.path.join(NARRATION_DIR, "alerts_vi")
COVER_IMG = "BodyGuard_영상_표지.png"
OUTRO_IMG = "BodyGuard_영상_마무리.png"
MANAGER_IMG = "관리자 화면@1x.png"

# 기존 make_video.py 의 KIND_COLOR 에는 없는, 이 기능(끼임/지게차)에서 새로 쓰는 분류
EXTRA_KIND_COLOR = {"위험구역": (255, 110, 110), "끼임 경보": (232, 72, 72)}

# 스토리보드 이미지에 박힌 번호(①~④) 순서 그대로: 지게차 -> 끼임 -> 주의(바닥·적재물) -> 표지판 질문
SCENARIOS = [
    ("forklift", "image_1.png", "v2_forklift"),
    ("pinch", "image_2.png", "v2_pinch"),
    ("spill", "image_3.png", "v2_spill"),
    ("sign", "image_4.png", "v2_sign"),
]
CAPTION_DUR = 2.8   # 시연 클립 자막(에이전트 기록) 한 건을 화면에 띄우는 시간
TOAST_DUR = 3.0     # 관리자 화면 전송 토스트를 띄우는 시간


def kind_color(kind):
    return KIND_COLOR.get(kind) or EXTRA_KIND_COLOR.get(kind) or MUTED


# 자막(한국어, malgun.ttf)에 베트남어 고유 문자(ạ/ế/ệ/Đ 등)가 섞여 있으면 malgun 에 없는 글자라 네모로 깨진다
# (한글 가독성은 malgun 이 가장 좋아 기본 폰트는 그대로 두고, 베트남어 전용 문자만 segoeui 로 바꿔 그림)
VI_FONT_PATH = "C:/Windows/Fonts/segoeui.ttf"
VI_RANGES = [(0x0100, 0x024F), (0x1E00, 0x1EFF), (0x0300, 0x036F)]
_vi_fonts = {}


def vi_font(size):
    if size not in _vi_fonts:
        _vi_fonts[size] = ImageFont.truetype(VI_FONT_PATH, size)
    return _vi_fonts[size]


def _is_vi_only(ch):
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in VI_RANGES)


def draw_mixed(d, xy, s, size, fill=FG):
    """한 줄을 그리되, malgun 에 없는 베트남어 전용 문자 구간만 segoeui 로 바꿔 그린다."""
    x, y = xy
    f_ko = font(size)
    f_vi = vi_font(size)
    run, run_vi = "", False

    def flush(run, run_vi, x):
        if not run:
            return x
        f = f_vi if run_vi else f_ko
        d.text((x, y), run, font=f, fill=fill)
        return x + d.textlength(run, font=f)

    for ch in s:
        is_vi = _is_vi_only(ch)
        if run and is_vi != run_vi:
            x = flush(run, run_vi, x)
            run = ""
        run += ch
        run_vi = is_vi
    flush(run, run_vi, x)


def load_narration():
    manifest = json.load(open(os.path.join(NARRATION_DIR, "narration.json"), encoding="utf-8"))
    return {e["id"]: e for e in manifest}


def load_alerts_vi():
    """지게차 접근 / 끼임 경보용 베트남어 고정 음성 (scripts/make_narration.py 가 미리 만들어 둠)."""
    manifest = json.load(open(os.path.join(ALERT_DIR, "alerts_vi.json"), encoding="utf-8"))
    return {e["id"]: e for e in manifest}


def full_bleed(path):
    return Image.open(path).convert("RGB").resize((WIDTH, HEIGHT), Image.LANCZOS)


def draw_clip_caption(im, kind, txt):
    """시연 클립 재생 중, 그 순간 에이전트 기록(위험구역 설정·경보·판정 등)을 화면 아래 자막으로."""
    d = ImageDraw.Draw(im, "RGBA")
    color = kind_color(kind)
    bar_h = 150
    y0 = HEIGHT - bar_h
    d.rectangle([0, y0, WIDTH, HEIGHT], fill=(10, 14, 20, 225))
    d.rectangle([0, y0, WIDTH, y0 + 5], fill=color)
    bw = 160
    d.rounded_rectangle([40, y0 + 22, 40 + bw, y0 + 60], 8, fill=color)
    text(d, (40 + bw / 2, y0 + 41), kind, 19, (15, 15, 15), True, anchor="mm")
    lines = wrap(d, txt, 25, WIDTH - bw - 40 - 60, 2)
    yy = y0 + 18
    for ln in lines:
        draw_mixed(d, (40 + bw + 24, yy), ln, 25, FG)
        yy += 36


def draw_manager_toast(im, note):
    """시연 클립 재생 중, 에이전트가 그 순간 관리자 화면으로 보낸 알림을 오른쪽 위 토스트로 잠깐 띄운다
    (관리자 화면 쪽은 뒤에 따로 보여주지만, '이 사건이 지금 그쪽으로 전송됐다' 는 걸 그 자리에서 바로 보여줌)."""
    d = ImageDraw.Draw(im, "RGBA")
    color = (232, 72, 72) if note["level"] == "critical" else (190, 150, 60)
    w, pad, bw = 600, 20, 134
    lines = wrap(d, note["summary"], 19, w - pad * 2, 2)
    h = 50 + 24 * len(lines)
    x1, y0 = WIDTH - 36, 36
    x0, y1 = x1 - w, 36 + h
    d.rounded_rectangle([x0, y0, x1, y1], 12, fill=(10, 14, 20, 235))
    d.rounded_rectangle([x0, y0, x0 + 6, y1], 3, fill=color)
    d.rounded_rectangle([x0 + pad, y0 + 14, x0 + pad + bw, y0 + 42], 7, fill=color)
    text(d, (x0 + pad + bw / 2, y0 + 28), "관리자 전송", 16, (15, 15, 15), True, anchor="mm")
    text(d, (x0 + pad + bw + 14, y0 + 28), note["zone"], 16, MUTED, anchor="lm")
    yy = y0 + 48
    for ln in lines:
        text(d, (x0 + pad, yy), ln, 19, FG)
        yy += 24


def play_clip(emit, now, rec_dir, sid, alerts_vi):
    """record_scenario.py 가 남긴 state.jsonl/프레임을 그대로, 실제 경과 시간에 맞춰 재생한다 (가짜 타이밍 아님).
    음성 경보: 지게차·끼임은 베트남어 고정 문구(alerts_vi), 표지판 질문은 그 장면에서 실제로 생성된
    베트남어 답변 음성(report['assistant'][i]['wav'])을 쓴다. '주의'(바닥 미끄러움 등)는 경보 음성이 없다
    (관리자 토스트만 뜬다) — 애초에 작업자에게 큰 소리로 알릴 일이 아니라서."""
    states = [json.loads(ln) for ln in open(os.path.join(rec_dir, "state.jsonl"), encoding="utf-8") if ln.strip()]
    final = json.load(open(os.path.join(rec_dir, "final.json"), encoding="utf-8"))
    r = final["report"]
    events = sorted([(e["t"], e["kind"], e["text"]) for e in r["timeline"] if e["kind"] != "계획"], key=lambda x: x[0])
    notifications = sorted(r.get("manager_notifications", []), key=lambda note: note["t"])
    assist_events = sorted(r.get("assistant", []), key=lambda e: e["t"])
    if sid == "forklift":
        alert_times = [v["t"] for v in r.get("voice", []) if v["level"] == "hazard"]
    elif sid == "pinch":
        alert_times = [e["t"] for e in r.get("pinch_events", [])]
    else:
        alert_times = []
    alert_wav = (alerts_vi.get(sid) or {}).get("path")

    ei = ni = ai = ci = 0
    active, active_toast = None, None
    audio_clips = []
    n = len(states)
    for i, st in enumerate(states):
        t = st["t"]
        im = Image.open(os.path.join(rec_dir, f"body_{st['k']:05d}.jpg")).convert("RGB")
        if im.size != (WIDTH, HEIGHT):
            im = im.resize((WIDTH, HEIGHT), Image.LANCZOS)
        while ei < len(events) and events[ei][0] <= t:
            _, kind, txt = events[ei]
            active = (kind, txt, t + CAPTION_DUR)
            ei += 1
        if active and t <= active[2]:
            draw_clip_caption(im, active[0], active[1])
        else:
            active = None
        while ni < len(notifications) and notifications[ni]["t"] <= t:
            active_toast = (notifications[ni], t + TOAST_DUR)
            ni += 1
        if active_toast and t <= active_toast[1]:
            draw_manager_toast(im, active_toast[0])
        else:
            active_toast = None
        while ci < len(alert_times) and alert_times[ci] <= t:
            if alert_wav:
                audio_clips.append((now(), alert_wav))
            ci += 1
        while ai < len(assist_events) and assist_events[ai]["t"] <= t:
            w = assist_events[ai].get("wav")
            if w:
                audio_clips.append((now(), w))
            ai += 1
        nxt = states[i + 1]["t"] if i + 1 < n else t + 0.1
        dt = max(0.0, nxt - t)
        emit(im, frames=max(1, int(round(dt * FPS))))
    return audio_clips, notifications


def play_manager_clip(emit, rec_dir):
    """render_manager_dashboard.py 가 이미 30fps 로 구워 둔 프레임을 그대로 이어 붙인다."""
    for p in sorted(glob.glob(os.path.join(rec_dir, "body_*.jpg"))):
        im = Image.open(p).convert("RGB")
        if im.size != (WIDTH, HEIGHT):
            im = im.resize((WIDTH, HEIGHT), Image.LANCZOS)
        emit(im, frames=1)


def main():
    p = argparse.ArgumentParser(description="BodyGuard v2 시연 영상 합성 (스토리보드+내레이션+실제 시연)")
    p.add_argument("--record-dir", default=os.path.join(ROOT, "outputs", "record"))
    p.add_argument("--storyboard-dir", default=STORYBOARD_DIR)
    p.add_argument("--out", default=os.path.join(ROOT, "outputs", "video", "bodyguard_v2.mp4"))
    a = p.parse_args()

    narr = load_narration()
    alerts_vi = load_alerts_vi()
    missing = [rec_name for _, _, rec_name in SCENARIOS + [(None, None, "v2_manager")]
               if not os.path.isdir(os.path.join(a.record_dir, rec_name))]
    if missing:
        raise SystemExit(f"[오류] 녹화 폴더가 없습니다: {missing} (scripts/record_scenario.py, render_manager_dashboard.py 먼저 실행)")

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

    def now():
        return n[0] / FPS

    clips = []   # (영상 시각 초, wav 경로) — 내레이션 + 시연 중 음성, 끝에서 한 번에 섞어 넣음

    intro = narr["intro"]
    clips.append((now(), intro["path"]))
    emit(full_bleed(os.path.join(a.storyboard_dir, COVER_IMG)), seconds=intro["dur"] + 0.6)

    for sid, img_name, rec_name in SCENARIOS:
        rec_dir = os.path.join(a.record_dir, rec_name)
        meta = narr[sid]
        clips.append((now(), meta["path"]))
        emit(full_bleed(os.path.join(a.storyboard_dir, img_name)), seconds=meta["dur"] + 0.5)
        audio_clips, _ = play_clip(emit, now, rec_dir, sid, alerts_vi)
        clips.extend(audio_clips)
        print(f"[{sid}] 스토리보드 {meta['dur']:.1f}초 + 시연 클립 재생 완료 (누적 {now():.1f}초)")

    mgr = narr["manager"]
    clips.append((now(), mgr["path"]))
    emit(full_bleed(os.path.join(a.storyboard_dir, MANAGER_IMG)), seconds=mgr["dur"] + 0.5)
    play_manager_clip(emit, os.path.join(a.record_dir, "v2_manager"))
    print(f"[manager] 관리자 화면 재생 완료 (누적 {now():.1f}초)")

    outro = narr["outro"]
    clips.append((now(), outro["path"]))
    emit(full_bleed(os.path.join(a.storyboard_dir, OUTRO_IMG)), seconds=outro["dur"] + 1.0)

    writer.close()
    total = n[0] / FPS
    add_audio(a.out, clips, total)
    print(f"[완료] {a.out}  ({total:.0f}초, 내레이션 {len(narr)}개)")


if __name__ == "__main__":
    main()
