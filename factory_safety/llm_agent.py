"""작업자 요청을 처리하는 LLM 에이전트 (LangGraph + 로컬 Qwen2.5-7B, Ollama).

작업자가 손동작으로 명령하면 (1 장비 설명, 2 공장 위험 스캔, 3 TBM, 4 관리자 호출, 5 SOS) 에이전트가
지금 상황 (바디캠에 보인 물체, 위험물 대장, 위험 영역, TBM, 작업자 위치) 을 도구로 조회하고, 무엇을 알려 줄지 고른 뒤
finish 로 결정을 낸다.

    START → agent (LLM, 도구 고르기) ⇄ tools (도구 실행) → finish 가 불리면 END
                 agent 가 도구 없이 글로만 답하면 remind (finish 를 부르라고 다시 요청) → agent

안전 문장은 LLM 이 직접 쓰지 않는다. LLM 은 무엇을 말할지 (물체·위험 번호) 와 판단 근거, 관리자에게 보낼 한국어 메시지를 정하고,
작업자에게 들려주는 문장은 검수한 4개 언어 문장 틀 (i18n.py) 로 만든다 (번역 모델처럼 현장 용어를 틀릴 위험을 피하려고).
LLM 이 없거나 실패하면 호출한 쪽이 규칙으로 대신 정한다.

이 모듈은 손 인식·음성 도우미 프로세스 (.venv-assistant, scripts/assistant_worker.py) 안에서 돈다.
"""
import json
import time
from typing import Annotated, TypedDict

MODEL = "qwen2.5:7b"
MAX_STEPS = 6
COMMAND_GOAL = {
    1: "explain the piece of equipment the worker was looking at (the most centered and closest equipment in view)",
    2: "scan the whole factory for hazards using the hazard log and tell the worker the most important ones",
    3: "brief the worker on today's TBM (toolbox meeting): work, risks and rules",
    4: "call the supervisor: write a short Korean message for the supervisor with the worker's location and situation",
    5: "SOS emergency: alert the supervisor and safety team with the worker's location immediately",
}
SYSTEM = """You are the safety agent of a warehouse patrol system. A worker wearing a chest bodycam showed a hand gesture command.
Your goal: {goal}.
Use the tools to look at the real situation first. Only use ids that tools returned; never invent objects or hazards.
When you have decided, call `finish` exactly once with:
- say_ids: ids of the items to tell the worker, most important first (object ids for equipment, hazard ids for a scan, empty otherwise)
- reason_ko: one Korean sentence explaining why you chose them (distance, priority, today's work), for the agent log
- manager_ko: a short Korean message for the supervisor (only for supervisor calls and SOS, else empty)
Write reason_ko and manager_ko in Korean and copy names and areas exactly from the tools' name_ko / what_ko / area_ko fields.
Do not mention anything the tools did not return. The worker's language is {lang}. Use at most 4 tool calls before finish."""


def _tools(snap, out):
    """스냅샷을 조회하는 도구들 (부작용은 out 에 기록)."""
    from langchain_core.tools import tool

    @tool
    def look_around() -> str:
        """List the objects the bodycam saw in the last few seconds (id, name, distance in m, direction, how centered 0=center)."""
        rows = [{k: o.get(k) for k in ("id", "name", "name_ko", "kind", "dist", "dir", "center")} for o in snap.get("view", [])]
        return json.dumps(rows or "nothing recognized", ensure_ascii=False)

    @tool
    def equipment_info(object_id: str) -> str:
        """Get what an object in view is and how to use it safely."""
        o = next((o for o in snap.get("view", []) if o["id"] == object_id), None)
        return json.dumps({"id": object_id, "name": o["name"], "name_ko": o.get("name_ko"), "safety": o.get("info", "")} if o else "unknown id",
                          ensure_ascii=False)

    @tool
    def hazard_log(limit: int = 6) -> str:
        """List hazards recorded so far in the whole factory, highest priority first (id, what, area, distance from worker, priority)."""
        rows = snap.get("hazards", [])[:max(1, int(limit))]
        zones = snap.get("zones", [])
        return json.dumps({"hazards": rows or "none", "danger_zones": len(zones)}, ensure_ascii=False)

    @tool
    def todays_tbm() -> str:
        """Get today's TBM (toolbox meeting) content."""
        return json.dumps(snap.get("tbm", {}), ensure_ascii=False)

    @tool
    def worker_status() -> str:
        """Get the worker's location, area and what the agent knows nearby."""
        return json.dumps(snap.get("worker", {}), ensure_ascii=False)

    @tool
    def finish(say_ids: list[str], reason_ko: str, manager_ko: str = "") -> str:
        """Finish with your decision. say_ids: items to tell the worker. reason_ko: Korean reason. manager_ko: Korean message to the supervisor."""
        valid = {o["id"] for o in snap.get("view", [])} | {h["id"] for h in snap.get("hazards", [])}
        out["say_ids"] = [i for i in say_ids if i in valid]
        out["reason_ko"] = reason_ko.strip()
        out["manager_ko"] = (manager_ko or "").strip()
        out["done"] = True
        return "ok"

    return [look_around, equipment_info, hazard_log, todays_tbm, worker_status, finish]


def build_graph(snap, out, model=MODEL, host=None):
    from langchain_core.messages import HumanMessage
    from langchain_ollama import ChatOllama
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode

    class S(TypedDict):
        messages: Annotated[list, add_messages]
        steps: int

    tools = _tools(snap, out)
    llm = ChatOllama(model=model, temperature=0, seed=7, num_ctx=4096, keep_alive="30m", base_url=host).bind_tools(tools)

    def agent(state):
        msg = llm.invoke(state["messages"])
        for c in getattr(msg, "tool_calls", []) or []:
            out.setdefault("trace", []).append(f"{c['name']}({', '.join(f'{k}={v}' for k, v in c['args'].items() if k != 'reason_ko' and k != 'manager_ko')})")
        return {"messages": [msg], "steps": state["steps"] + 1}

    def after_agent(state):
        if state["steps"] > MAX_STEPS:
            return "end"
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else "remind"

    def remind(state):
        out.setdefault("trace", []).append("(다시 요청)")
        return {"messages": [HumanMessage("You must answer by calling the `finish` tool now (no plain text).")]}

    def after_tools(state):
        return "end" if out.get("done") or state["steps"] >= MAX_STEPS else "agent"

    g = StateGraph(S)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(tools))
    g.add_node("remind", remind)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", after_agent, {"tools": "tools", "remind": "remind", "end": END})
    g.add_edge("remind", "agent")
    g.add_conditional_edges("tools", after_tools, {"agent": "agent", "end": END})
    return g.compile()


def decide(snap, model=MODEL, host=None):
    """스냅샷 -> {say_ids, reason_ko, manager_ko, trace, llm, sec}. 실패하면 {"llm": False, "error"}."""
    t0 = time.time()
    out = {}
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        graph = build_graph(snap, out, model, host)
        count = int(snap["count"])
        sys_msg = SYSTEM.format(goal=COMMAND_GOAL[count], lang=snap.get("lang_name", "English"))
        user = f"Gesture command: {count} fingers. Time {snap.get('t', 0):.0f} s into the patrol. Decide and finish."
        graph.invoke({"messages": [SystemMessage(sys_msg), HumanMessage(user)], "steps": 0}, {"recursion_limit": 4 * MAX_STEPS})
    except Exception as e:      # Ollama 가 꺼져 있거나 도구 호출 형식이 틀림
        return {"llm": False, "error": f"{type(e).__name__}: {e}", "sec": round(time.time() - t0, 2), "trace": out.get("trace", [])}
    if not out.get("done"):
        return {"llm": False, "error": "finish 를 안 부름", "sec": round(time.time() - t0, 2), "trace": out.get("trace", [])}
    return {"llm": True, "model": model, "say_ids": out["say_ids"], "reason_ko": out["reason_ko"], "manager_ko": out["manager_ko"],
            "trace": out.get("trace", []), "sec": round(time.time() - t0, 2)}


def graph_mermaid():
    """보고서·README 용 그래프 그림 (mermaid)."""
    return "\n".join([
        "flowchart LR",
        "  S([손동작 명령]) --> A[agent<br/>Qwen2.5-7B]",
        "  A -- 도구 호출 --> T[tools<br/>look_around · equipment_info · hazard_log<br/>todays_tbm · worker_status]",
        "  T --> A",
        "  A -. 글로만 답함 .-> M[remind<br/>finish 다시 요청] -.-> A",
        "  A -- finish --> R[결정<br/>말할 항목 · 근거 · 관리자 메시지]",
        "  R --> V[검수한 문장 틀로 작업자 언어 음성]",
    ])
