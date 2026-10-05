"""손 관절 21점 (MediaPipe Hands 순서) 으로 편 손가락 수를 센다.

손을 가로로 내밀어도 되게 화면 방향(위/아래)은 쓰지 않고, 마디가 곧은지와 손바닥에서 얼마나 떨어졌는지만 본다.
손바닥을 카메라에 보이는 동작이라 화면 좌표 (픽셀) 로 센다.
    0 손목 | 1~4 엄지 (CMC, MCP, IP, 끝) | 5~8 검지 (MCP, PIP, DIP, 끝) | 9~12 중지 | 13~16 약지 | 17~20 새끼
명령: 검지 = 1, 검지+중지 = 2, +약지 = 3. 그 밖의 모양(+새끼, 다섯 손가락 모두 포함)은 0 (명령 아님).
"""
import numpy as np

FINGERS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
BEND_MAX = 50.0         # 편 손가락: MCP→PIP 와 PIP→끝 사이 각도 (도) 가 이보다 작음
REACH_MIN = 1.15        # 편 손가락: 손목~끝 거리가 손목~PIP 거리의 이 배 이상
THUMB_OUT = 0.7         # 편 엄지: 엄지 끝~약지 MCP 거리가 손바닥 크기 (손목~중지 MCP) 의 이 배 이상 (접으면 손바닥을 가로질러 약지 쪽에 닿음)
PATTERNS = {(1, 0, 0, 0): 1, (1, 1, 0, 0): 2, (1, 1, 1, 0): 3}


def _angle(a, b):
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def finger_states(pts):
    """pts: (21, 2) 화면 좌표 (또는 (21, 3)). 손가락별 편 상태 {이름: bool} 와 근거 값."""
    p = np.asarray(pts, float)
    wrist = p[0]
    out, info = {}, {}
    for name, (mcp, pip, dip, tip) in FINGERS.items():
        bend = _angle(p[pip] - p[mcp], p[tip] - p[pip])
        reach = np.linalg.norm(p[tip] - wrist) / (np.linalg.norm(p[pip] - wrist) + 1e-9)
        out[name] = bend < BEND_MAX and reach > REACH_MIN
        info[name] = (round(bend, 1), round(float(reach), 2))
    palm = np.linalg.norm(p[9] - p[0]) + 1e-9
    out_d = float(np.linalg.norm(p[4] - p[13]) / palm)
    out["thumb"] = out_d > THUMB_OUT
    info["thumb"] = round(out_d, 2)
    return out, info


def count_fingers(pts):
    """명령 번호 1~3, 정해진 모양이 아니면 0."""
    s, _ = finger_states(pts)
    key = tuple(int(s[k]) for k in ("index", "middle", "ring", "pinky"))
    return PATTERNS.get(key, 0)


class GestureFilter:
    """같은 숫자가 손이 멈춘 채로 연속 n 번 보이면 명령으로 확정 (손을 올리거나 내리는 중의 화면은 안 셈).
    확정 뒤에는 손을 내리거나 모양을 바꿔야 다시 받고, cooldown 초 동안은 아예 안 받음."""
    STILL = 0.15        # 앞 화면과 관절 평균 이동이 손바닥 크기 (손목~중지 MCP) 의 이만큼 안이면 멈춘 손

    def __init__(self, need=3, cooldown=4.0):
        self.need, self.cooldown = need, cooldown
        self.last, self.run, self.t_fire, self.fired, self.prev = 0, 0, -1e9, 0, None

    def _still(self, pts):
        prev, self.prev = self.prev, pts
        if pts is None:
            return True                  # 관절 좌표를 안 주면 움직임은 안 봄
        if prev is None:
            return False
        p, q = np.asarray(pts, float), np.asarray(prev, float)
        palm = np.linalg.norm(p[9] - p[0]) + 1e-9
        return float(np.mean(np.linalg.norm(p - q, axis=1))) / palm < self.STILL

    def update(self, t, count, pts=None):
        if count > 0:
            still = self._still(pts)
        else:
            still, self.prev = False, None
        if count != self.fired:
            self.fired = 0
        if count > 0 and still and count == self.last:
            self.run += 1
        else:
            self.run = 1 if (count > 0 and still) else 0
        self.last = count
        if count > 0 and count != self.fired and self.run >= self.need and t - self.t_fire >= self.cooldown:
            self.t_fire, self.run, self.fired = t, 0, count
            return count
        return 0
