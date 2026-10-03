"""음성 경고: "경고! 경고! 위험 요소가 식별되었습니다." (짧은 경보음 + 한국어 음성).

Windows 기본 음성 합성 (System.Speech, 한국어 음성 Microsoft Heami) 으로 한 번 만들어 WAV 로 저장하고,
순찰 중에는 winsound 로 비동기 재생한다. 영상에는 같은 WAV 를 경고 시각에 맞춰 소리 트랙으로 넣는다.
"""
import math
import os
import struct
import subprocess
import wave

from .config import VOICE_TEXT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICE_WAV = os.path.join(ROOT, "assets", "generated", "voice_warning.wav")
RATE = 22050


def _tts(text, path, rate=2):
    """Windows 음성 합성으로 WAV (한국어 음성이 있으면 그걸로)."""
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$v = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -eq 'ko-KR' } | Select-Object -First 1;"
        "if ($v) { $s.SelectVoice($v.VoiceInfo.Name) };"
        f"$s.Rate = {rate};"
        f"$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo({RATE}, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono);"
        f"$s.SetOutputToWaveFile('{path}', $fmt);"
        f"$s.Speak('{text}');"
        "$s.Dispose();"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True)


def _read(path):
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        data = struct.unpack(f"<{n}h", w.readframes(n))
    return list(data)


def _alarm(seconds=0.6):
    """두 음 사이렌 (880 Hz / 660 Hz 번갈아)."""
    out = []
    n = int(RATE * seconds)
    for i in range(n):
        t = i / RATE
        f = 880.0 if int(t / 0.15) % 2 == 0 else 660.0
        env = min(1.0, i / 400, (n - i) / 400)
        out.append(int(0.45 * 32767 * env * math.sin(2 * math.pi * f * t)))
    return out


def ensure_voice(path=VOICE_WAV, text=VOICE_TEXT):
    """경보음 + 음성 WAV 를 만들어 두고 경로를 돌려준다 (이미 있으면 그대로)."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tts.wav"
    _tts(text, tmp)
    samples = _alarm() + [0] * int(RATE * 0.1) + _read(tmp)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    os.remove(tmp)
    return path


def duration(path=VOICE_WAV):
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


def play_async(path=VOICE_WAV):
    """순찰 중 재생 (Windows). 다른 곳에서는 조용히 넘어감."""
    try:
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
        return True
    except Exception:
        return False


def with_alarm(path, seconds=1.2):
    """경보음 + path 음성 (SOS 안내용). 새 WAV 경로."""
    out = path[:-4] + "_alarm.wav"
    if not os.path.exists(out):
        samples = _alarm(seconds) + [0] * int(RATE * 0.15) + _read(path)
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return out
