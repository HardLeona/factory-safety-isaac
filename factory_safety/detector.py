"""YOLO 래퍼 (ultralytics 필요). 바디캠 추적(같은 물체에 번호를 이어 붙임)."""
import numpy as np


class YoloDetector:
    """rgb: (H, W, 3|4) uint8 RGB."""

    def __init__(self, weights, conf=0.25, imgsz=960, device=None):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("ultralytics가 필요해요. Isaac Sim 파이썬에 설치: python.sh -m pip install --no-deps ultralytics") from e
        self.weights = weights
        self.conf, self.imgsz, self.device = conf, imgsz, device
        self.model = YOLO(weights)
        self._trackers = {}
        self.names = self.model.names

    @staticmethod
    def _bgr(rgb):
        return np.ascontiguousarray(np.asarray(rgb)[..., :3][..., ::-1])   # ultralytics numpy 입력은 BGR

    def _out(self, res):
        out = []
        for b in res.boxes:
            tid = int(b.id[0]) if getattr(b, "id", None) is not None else None
            out.append((self.names[int(b.cls[0])], float(b.conf[0]), [float(v) for v in b.xyxy[0].tolist()], tid))
        return out

    def track(self, rgb, stream="bodycam"):
        """같은 영상 흐름(stream)끼리 추적 번호를 이어간다. [(클래스, 신뢰도, xyxy, 추적 번호)], 결과 객체."""
        if stream not in self._trackers:
            from ultralytics import YOLO
            self._trackers[stream] = YOLO(self.weights)     # 흐름마다 추적기 상태를 따로
        m = self._trackers[stream]
        res = m.track(self._bgr(rgb), conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False,
                      persist=True, tracker="bytetrack.yaml")[0]
        return self._out(res), res
