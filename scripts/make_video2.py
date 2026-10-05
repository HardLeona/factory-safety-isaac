"""BodyGuard v2 시연 영상: 스토리보드 4장 + 내레이션(TTS) + 실제 Isaac Sim 시연 클립 + 관리자 화면.

    .venv-assistant/Scripts/python scripts/make_video2.py --out outputs/video/bodyguard_v2.mp4

흐름: 인트로 타이틀 → (스토리보드 이미지 + 내레이션 → 그 장면의 실제 시연 클립) × 4(지게차/주의·적재물/표지판 질의/끼임 경보)
→ 관리자 화면(알림 피드 + 확인 상호작용) + 내레이션 → 아웃트로.
내레이션은 scripts/make_narration.py, 시연 클립은 scripts/record_scenario.py --which {forklift,spill,sign,pinch},
관리자 화면은 scripts/render_manager_dashboard.py 가 미리 만들어 둔 결과를 그대로 읽어 붙인다 (이 스크립트는 합성만 한다).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_video import (  # noqa: E402
    BG, FG, FPS, FONT, FONT_B, HEIGHT, KIND_COLOR, MUTED, WIDTH,
    add_audio, card, fit, font, text, wrap,
)

# 리포(또는 이 워크트리)가 어디 있든, 스토리보드는 항상 AI_SW_CONTEST/agent_senario 에 있다
STORYBOARD_DIR = r"C:\Users\user\Desktop\AI_SW_CONTEST\agent_senario"
NARRATION_DIR = os.path.join(ROOT, "outputs", "video", "narration")

# 기존 make_video.py 의 KIND_COLOR 에는 없는, 이 기능(끼임/지게차)에서 새로 쓰는 분류
EXTRA_KIND_COLOR = {"위험구역": (255, 110, 110), "끼임 경보": (232, 72, 72)}

SCENARIOS = [
    ("forklift", "image.png", "v2_forklift"),
    ("spill", "image (1).png", "v2_spill"),
    ("sign", "image (2).png", "v2_sign"),
    ("pinch", "image (3).png", "v2_pinch"),
]
CAPTION_DUR = 2.8   # 시연 클립 자막(에이전트 기록) 한 건을 화면에 띄우는 시간


def kind_color(kind):
    return KIND_COLOR.get(kind) or EXTRA_KIND_COLOR.get(kind) or MUTED


def load_narration():
    manifest = json.load(open(os.path.join(NARRATION_DIR, "narration.json"), encoding="utf-8"))
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
        text(d, (40 + bw + 24, yy), ln, 25, FG)
        yy += 36


def play_clip(emit, now, rec_dir):
    """record_scenario.py 가 남긴 state.jsonl/프레임을 그대로, 실제 경과 시간에 맞춰 재생한다 (가짜 타이밍 아님)."""
    states = [json.loads(ln) for ln in open(os.path.join(rec_dir, "state.jsonl"), encoding="utf-8") if ln.strip()]
    final = json.load(open(os.path.join(rec_dir, "final.json"), encoding="utf-8"))
    r = final["report"]
    events = sorted([(e["t"], e["kind"], e["text"]) for e in r["timeline"] if e["kind"] != "계획"], key=lambda x: x[0])
    voice_events = sorted(r.get("voice", []), key=lambda v: v["t"])
    ei = vi = 0
    active = None
    voice_marks = []
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
        while vi < len(voice_events) and voice_events[vi]["t"] <= t:
            voice_marks.append((now(), voice_events[vi]["level"]))
            vi += 1
        nxt = states[i + 1]["t"] if i + 1 < n else t + 0.1
        dt = max(0.0, nxt - t)
        emit(im, frames=max(1, int(round(dt * FPS))))
    return voice_marks, r.get("manager_notifications", [])


def play_manager_clip(emit, rec_dir):
    """render_manager_dashboard.py 가 이미 30fps 로 구워 둔 프레임을 그대로 이어 붙인다."""
    for p in sorted(glob.glob(os.path.join(rec_dir, "body_*.jpg"))):
        im = Image.open(p).convert("RGB")
        if im.size != (WIDTH, HEIGHT):
            im = im.resize((WIDTH, HEIGHT), Image.LANCZOS)
        emit(im, frames=1)


def manager_title_card():
    return card([
        ("에이전트가 스스로 판단해 보내는 알림입니다 — 작업자가 손짓으로 부르지 않아도 쌓입니다", 27, FG, False),
        ("", 14, FG, False),
        ("끼임 경보 · 지게차 접근 · '주의'로 남긴 애매한 판정 → 관리자 화면에 자동 기록", 24, MUTED, False),
        ("관리자가 '확인'을 누르면 처리 완료로 바뀝니다", 24, MUTED, False),
    ], title="관리자 화면")


def outro_card():
    return card([
        ("말이 안 통해도, 첫날부터 안전하게.", 38, (120, 220, 255), True),
        ("", 24, FG, False),
        ("BodyGuard", 30, FG, True),
    ], title="")


def intro_card():
    return card([
        ("외국인 근로자를 위한 바디캠 안전 관리 에이전트", 30, (120, 220, 255), True),
        ("", 16, FG, False),
        ("말이 안 통해도 괜찮습니다 — 위험하면 작업자에게 바로, 애매하면 관리자에게, 궁금한 건 손짓 하나로", 26, FG, False),
        ("NVIDIA Isaac Sim 디지털 트윈 · LangGraph + 로컬 LLM(Qwen2.5-7B) · YOLO26", 22, MUTED, False),
    ], title="BodyGuard")


def main():
    p = argparse.ArgumentParser(description="BodyGuard v2 시연 영상 합성 (스토리보드+내레이션+실제 시연)")
    p.add_argument("--record-dir", default=os.path.join(ROOT, "outputs", "record"))
    p.add_argument("--storyboard-dir", default=STORYBOARD_DIR)
    p.add_argument("--out", default=os.path.join(ROOT, "outputs", "video", "bodyguard_v2.mp4"))
    a = p.parse_args()

    narr = load_narration()
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

    from factory_safety.config import VOICE_TEXT_CAUTION
    from factory_safety.voice import VOICE_CAUTION_WAV, VOICE_WAV, ensure_voice
    ensure_voice()
    ensure_voice(VOICE_CAUTION_WAV, VOICE_TEXT_CAUTION)

    clips = []   # (영상 시각 초, wav 경로) — 내레이션 + 시연 중 음성 경고, 끝에서 한 번에 섞어 넣음

    intro = narr["intro"]
    clips.append((now(), intro["path"]))
    emit(intro_card(), seconds=intro["dur"] + 0.6)

    for sid, img_name, rec_name in SCENARIOS:
        rec_dir = os.path.join(a.record_dir, rec_name)
        meta = narr[sid]
        clips.append((now(), meta["path"]))
        emit(full_bleed(os.path.join(a.storyboard_dir, img_name)), seconds=meta["dur"] + 0.5)
        vmarks, _ = play_clip(emit, now, rec_dir)
        for vt, lvl in vmarks:
            clips.append((vt, VOICE_WAV if lvl == "hazard" else VOICE_CAUTION_WAV))
        print(f"[{sid}] 스토리보드 {meta['dur']:.1f}초 + 시연 클립 재생 완료 (누적 {now():.1f}초)")

    mgr = narr["manager"]
    clips.append((now(), mgr["path"]))
    emit(manager_title_card(), seconds=mgr["dur"] + 0.5)
    play_manager_clip(emit, os.path.join(a.record_dir, "v2_manager"))
    print(f"[manager] 관리자 화면 재생 완료 (누적 {now():.1f}초)")

    outro = narr["outro"]
    clips.append((now(), outro["path"]))
    emit(outro_card(), seconds=outro["dur"] + 1.0)

    writer.close()
    total = n[0] / FPS
    add_audio(a.out, clips, total)
    print(f"[완료] {a.out}  ({total:.0f}초, 내레이션 {len(narr)}개)")


if __name__ == "__main__":
    main()
