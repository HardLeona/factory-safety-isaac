"""손 인식 + 다국어 음성 도우미 (별도 가상환경 .venv-assistant 에서 실행).

Isaac Sim 파이썬에는 MediaPipe 를 같이 깔기 어려워서 따로 띄우고, 표준 입출력으로 JSON 을 한 줄씩 주고받는다
(factory_safety/assistant_client.py 가 띄움).

    {"op": "ping"}                                   -> {"ok": true}
    {"op": "hand", "path": "frame.jpg"}              -> {"count": 3, "score": 0.95, "pts": [[x, y], ...]}   손이 없으면 count 0
    {"op": "tts", "text": "...", "lang": "zh"}       -> {"wav": ".../zh_ab12.wav", "dur": 4.2, "engine": "edge"}

음성: edge-tts (Microsoft 온라인 신경망 음성, 인터넷 필요). 안 되면 Windows 음성 (설치된 한국어·영어·일본어만).
만든 음성은 문장별로 assets/generated/tts 에 저장해 두고 다시 쓴다.

설치:
    py -3.12 -m venv .venv-assistant
    .venv-assistant\\Scripts\\python -m pip install mediapipe edge-tts imageio-ffmpeg
"""
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from factory_safety.hand_count import count_fingers  # noqa: E402

MODEL = os.path.join(ROOT, "assets", "models", "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
CACHE = os.path.join(ROOT, "assets", "generated", "tts")
VOICES = {"ko": "ko-KR-SunHiNeural", "en": "en-US-JennyNeural", "zh": "zh-CN-XiaoxiaoNeural", "ja": "ja-JP-NanamiNeural"}
SAPI = {"ko": "ko-KR", "en": "en-US", "ja": "ja-JP"}
SPEED = "+20%"          # 안내를 조금 빠르게 (순찰 중 짧게)
RATE = 22050
DETECT_CONF = 0.3       # 손 찾기 문턱 (낮춰도 '멈춘 손에서 3번 연속' 조건이 잘못 실행을 막음)
DARK = 0.4              # 화면 평균 밝기가 이보다 낮으면 밝기 보정 후 한 번 더 찾음
_det = None


def detector():
    global _det
    if _det is None:
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
        if not os.path.exists(MODEL):
            import urllib.request
            os.makedirs(os.path.dirname(MODEL), exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL, MODEL)
        opt = vision.HandLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=MODEL), num_hands=1,
                                           min_hand_detection_confidence=DETECT_CONF, min_hand_presence_confidence=DETECT_CONF)
        _det = vision.HandLandmarker.create_from_options(opt)
    return _det


def _detect(arr):
    import mediapipe as mp
    r = detector().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(arr)))
    return r if r.hand_landmarks else None


def _brighten(arr):
    """어두운 화면을 밝게 (실제 바디캠의 자동 노출 역할). 충분히 밝으면 None."""
    m = float(arr.mean()) / 255
    if m >= DARK:
        return None
    g = np.log(0.45) / np.log(max(m, 0.02))
    return (255 * (arr / 255.0) ** g).clip(0, 255).astype(np.uint8)


def hand(path):
    from PIL import Image
    arr = np.asarray(Image.open(path).convert("RGB"))
    r, bright = _detect(arr), False
    if r is None:
        b = _brighten(arr)
        if b is not None:
            r, bright = _detect(b), True
    if r is None:
        return {"count": 0}
    h, w = arr.shape[:2]
    pts = [[p.x * w, p.y * h] for p in r.hand_landmarks[0]]
    return {"count": count_fingers(pts), "score": round(float(r.handedness[0][0].score), 3), "bright": bright,
            "pts": [[round(x, 1), round(y, 1)] for x, y in pts]}


def _ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _sapi(text, lang, path):
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"$v = $s.GetInstalledVoices() | Where-Object {{ $_.VoiceInfo.Culture.Name -eq '{SAPI[lang]}' }} | Select-Object -First 1;"
        "if (-not $v) { exit 3 };"
        "$s.SelectVoice($v.VoiceInfo.Name); $s.Rate = 1;"
        f"$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo({RATE}, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono);"
        f"$s.SetOutputToWaveFile('{path}', $fmt);"
        "$s.Speak([Console]::In.ReadToEnd());"
        "$s.Dispose();"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], input=text, text=True, encoding="utf-8", check=True, capture_output=True)


def tts(text, lang):
    os.makedirs(CACHE, exist_ok=True)
    key = hashlib.sha1(f"{lang}|{VOICES[lang]}|{SPEED}|{text}".encode("utf-8")).hexdigest()[:16]
    wav = os.path.join(CACHE, f"{lang}_{key}.wav")
    engine = "cache"
    if not os.path.exists(wav):
        try:
            import edge_tts
            mp3 = wav[:-4] + ".mp3"
            asyncio.run(edge_tts.Communicate(text, VOICES[lang], rate=SPEED).save(mp3))
            subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-i", mp3, "-ac", "1", "-ar", str(RATE), wav], check=True)
            os.remove(mp3)
            engine = "edge"
        except Exception as e:
            if lang not in SAPI:
                return {"wav": None, "error": f"edge-tts 실패, 이 언어의 Windows 음성 없음: {e}"}
            _sapi(text, lang, wav)
            engine = "sapi"
    with wave.open(wav, "rb") as w:
        dur = w.getnframes() / w.getframerate()
    return {"wav": wav, "dur": round(dur, 2), "engine": engine}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        op = req.get("op")
        try:
            if op == "ping":
                res = {"ok": True}
            elif op == "hand":
                res = hand(req["path"])
            elif op == "tts":
                res = tts(req["text"], req["lang"])
            elif op == "quit":
                break
            else:
                res = {"error": f"모르는 op {op}"}
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {e}"}
        res["_id"] = req.get("_id")
        sys.stdout.write(json.dumps(res, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
