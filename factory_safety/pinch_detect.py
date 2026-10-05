"""손가락 끝·끼임점의 3D 위치 추정: 바디캠 depth(distance_to_image_plane)로 2D 픽셀을 3D 로 역투영한다.

MediaPipe 손 관절(hand_count 의 21점, 검지 끝 = 인덱스 8)의 픽셀 좌표와, 별도 경량 YOLO(pinch_point
1클래스) 박스 중심의 픽셀 좌표를 같은 프레임의 깊이맵으로 3D 로 바꿔서 거리를 잴 수 있게 한다
(geometry.Projector.unproject 사용). 끼임점 탐지가 없거나 깊이를 못 읽으면 None 을 돌려주고,
호출 측(agent.check_pinch)이 등록된 끼임점 위치로 폴백한다.
"""
import numpy as np

INDEX_TIP = 8     # MediaPipe 21점: 5~8 검지 (MCP, PIP, DIP, 끝)


def _sample(projector, u, v, depth):
    h, w = depth.shape[:2]
    iu, iv = int(round(u)), int(round(v))
    if not (0 <= iu < w and 0 <= iv < h):
        return None
    z = float(depth[iv, iu])
    if not np.isfinite(z) or z <= 0:
        return None
    return projector.unproject([[u, v]], [z])[0]


def fingertip_xyz(projector, pts, depth):
    """pts: MediaPipe 21점 픽셀 좌표 (손이 없으면 None). depth: (H, W) distance_to_image_plane 맵.
    검지 끝 픽셀의 깊이로 3D 위치를 구한다. 손이 없거나 그 픽셀 깊이를 못 읽으면 None."""
    if pts is None or depth is None:
        return None
    u, v = pts[INDEX_TIP]
    return _sample(projector, u, v, depth)


def pinch_point_xyz(projector, xyxy, depth):
    """pinch_point YOLO 박스(xyxy) 중심 픽셀의 깊이로 3D 위치를 구한다. 깊이를 못 읽으면 None (호출 측 폴백)."""
    if xyxy is None or depth is None:
        return None
    u, v = (xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0
    return _sample(projector, u, v, depth)


def best_pinch_box(dets, machine_uv, max_px=120.0):
    """pinch_point 탐지들([(cls, conf, xyxy), ...]) 중 등록된 끼임점의 화면 투영 위치(machine_uv)에
    가장 가까운 것 (max_px 픽셀 안). 없으면 None. (기계가 여러 대가 돼도 끼임점을 헷갈리지 않게)"""
    best, bd = None, None
    for name, conf, xyxy, *_ in dets:
        if name != "pinch_point":
            continue
        cx, cy = (xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0
        d = float(np.hypot(cx - machine_uv[0], cy - machine_uv[1]))
        if d <= max_px and (bd is None or d < bd):
            best, bd = xyxy, d
    return best
