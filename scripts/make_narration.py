"""BodyGuard v2 시연 영상 내레이션 (한국어, edge-tts 신경망 음성). .venv-assistant 에서 실행.

    .venv-assistant/Scripts/python scripts/make_narration.py

각 대본을 WAV 로 만들어 outputs/video/narration/ 에 저장하고, (id, text, path, dur) 목록을
narration.json 에 적는다. make_video2.py 가 이 목록 순서대로 장면 앞에 내레이션을 깔고 길이를 읽는다.
"""
import asyncio
import json
import os
import subprocess
import sys
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT_DIR = os.path.join(ROOT, "outputs", "video", "narration")
VOICE = "ko-KR-SunHiNeural"
SPEED = "+8%"     # 다큐멘터리 내레이션이라 작업자 안내(+20%)보다 조금 차분하게
RATE = 22050

# (id, 대본) — 스토리보드 4장과 1:1 대응 + 인트로/아웃트로/관리자 화면
SCRIPT = [
    ("intro", "BodyGuard, 외국인 근로자를 위한 바디캠 안전 관리 에이전트입니다. "
              "말이 안 통해도 괜찮습니다. 위험하면 작업자에게 바로, 애매하면 관리자에게, "
              "궁금한 건 손짓 하나로 답합니다."),
    ("forklift", "입사 사흘째, 베트남에서 온 민 씨가 통로를 걷는 사이 뒤에서 지게차가 다가옵니다. "
                "BodyGuard는 위치 신호로 이를 즉시 알아채고, 작업자가 못 봐도 큰 소리로 멈추라고 경고합니다."),
    ("spill", "잠깐 스쳐 지나간 적재물, 안전한지 위험한지 섣불리 단정하지 않습니다. "
             "BodyGuard는 '주의' 로 남겨 두고, 관리자 화면에 자동으로 알립니다."),
    ("sign", "표지판이 궁금하면 손가락 하나만 들면 됩니다. BodyGuard는 검증된 안전 매뉴얼에서 찾은 "
            "근거로만 답하고, 모르면 추측 대신 관리자를 부릅니다."),
    ("pinch", "컨베이어가 돌아가는 중에 손이 끼임점에 가까워지면, BodyGuard는 최고 등급 경보로 "
             "즉시 손을 빼라고 알립니다."),
    ("manager", "현장에서 일어난 일은 관리자 화면에도 그대로 남습니다. 확인을 누르면 처리 완료로 바뀌어, "
               "무엇을 처리했고 무엇이 남았는지 한눈에 보입니다."),
    ("outro", "말이 안 통해도, 첫날부터 안전하게. BodyGuard입니다."),
]


def _ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def tts(text, path):
    import edge_tts
    mp3 = path[:-4] + ".mp3"
    asyncio.run(edge_tts.Communicate(text, VOICE, rate=SPEED).save(mp3))
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-i", mp3, "-ac", "1", "-ar", str(RATE), path], check=True)
    os.remove(mp3)


def wav_duration(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = []
    for nid, text in SCRIPT:
        path = os.path.join(OUT_DIR, f"{nid}.wav")
        if not os.path.exists(path):
            tts(text, path)
        dur = round(wav_duration(path), 2)
        manifest.append({"id": nid, "text": text, "path": path, "dur": dur})
        print(f"[내레이션] {nid}: {dur}초  {text[:30]}...")
    with open(os.path.join(OUT_DIR, "narration.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"[완료] {len(manifest)}개 -> {OUT_DIR}")


if __name__ == "__main__":
    main()
