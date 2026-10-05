"""끼임 위험 기계의 작동 상태 (실제 공장 설비의 IoT/PLC 신호 조회를 대신하는 개념).

기계의 모양과 고정 자리(끼임점 3D 위치 포함)는 warehouse.MACHINES 에 있다. 여기서는 on/off 상태값만 들고
있다가, 에이전트가 호출하는 get_machine_state(machine_id) 로 내준다. 판단 규칙(agent.py)은 이 상태만
보고 위험구역을 켜고 끈다 (LLM 판단 없음, T0 규칙).
"""
import numpy as np

from . import warehouse as W


class MachineRegistry:
    """기계 id -> 작동 상태(on/off). 기본값은 모두 off (안전측)."""

    def __init__(self, machines=W.MACHINES, on_change=None):
        self.machines = {m["id"]: m for m in machines}
        self.state = {m["id"]: False for m in machines}
        self.on_change = on_change    # 상태가 바뀔 때 (machine_id, on) 으로 불림 (장면 표시등 갱신 등에 씀)

    def get(self, machine_id):
        """에이전트가 호출하는 도구: 기계 작동 상태 조회. {"id", "on"}."""
        return {"id": machine_id, "on": bool(self.state.get(machine_id, False))}

    def set(self, machine_id, on):
        """상태를 바꾼다 (시뮬레이션에서는 시연 스크립트나 테스트가 호출; 실제로는 PLC 신호에 해당)."""
        on = bool(on)
        if self.state.get(machine_id) == on:
            return
        self.state[machine_id] = on
        if self.on_change:
            self.on_change(machine_id, on)

    def pinch_xyz(self, machine_id):
        """등록된 끼임점 3D 위치 (설비 대장에 등록된 위치 개념). 탐지 실패 폴백에 씀."""
        m = self.machines[machine_id]
        return np.array([m["x"], m["y"], m["pinch_z"]], float)

    def __iter__(self):
        return iter(self.machines.values())

    def __len__(self):
        return len(self.machines)
