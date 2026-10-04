"""조치 지시서의 조치 방법을 매뉴얼 근거로 보강하는 LLM 에이전트 (LangGraph + 로컬 Qwen2.5-7B, Ollama).

agent.py 의 report() 는 위험물마다 고정 테이블(ACTIONS)의 조치 문구를 쓴다. 이 모듈은 순찰이 다 끝난 뒤,
그 문구를 검증된 안전 매뉴얼(data/manuals/*.md, factory_safety/manuals.py) 에 비춰 더 구체적인 근거가 있으면
보강하고, 없으면 고정 문구를 그대로 둔다 (LLM 이 문구를 지어내지 않음).

실시간 순찰 중이 아니라 순찰이 끝난 뒤 조치 지시서를 만들 때 한 번만 쓰여서, 다른 모듈보다 지연에 여유가 있다.

    agent.report() 가 위험물마다 -> decide(snapshot) 한 번 호출
        grounded=true  -> 매뉴얼 근거를 더해 조치 문구를 보강 (기존 문구와 안 어긋나게)
        grounded=false 또는 LLM 실패 -> 기존 고정 문구(ACTIONS) 그대로
"""
import json
import time

MODEL = "qwen2.5:7b"
MAX_STEPS = 3
SYSTEM = """You are writing one action item for a warehouse safety action report. A hazard of class {label_ko} was found
in {zone}. The standard action (already reviewed, safe to use as-is) is: "{standard_action}".
Call `retrieve_manual` to see if the verified safety manual has more specific or additional guidance for this exact hazard.
Then call `finish` exactly once:
- grounded: true only if the manual passage adds genuinely useful, specific detail beyond the standard action
  (e.g. a concrete step, a number, a precaution it does not already mention). false if the manual has nothing extra useful,
  or only repeats the standard action in different words.
- action_ko: only used when grounded=true. The standard action plus the manual's extra specifics, in Korean, as one or two
  short sentences. Never contradict or remove anything from the standard action.
- source: only used when grounded=true. Copy the manual passage's title/section exactly.
Do not invent manual content. If `retrieve_manual` found nothing relevant, call finish with grounded=false."""


def _tools(snap, out):
    from langchain_core.tools import tool

    @tool
    def retrieve_manual(query: str) -> str:
        """Search the verified safety manuals for passages about this hazard."""
        from . import manuals
        hits = manuals.retrieve(query)
        return json.dumps(hits or "no relevant manual passage", ensure_ascii=False)

    @tool
    def finish(grounded: bool, action_ko: str = "", source: str = "") -> str:
        """Finish. grounded: true only if the manual added specific useful detail. action_ko/source: only when grounded=true."""
        out["grounded"] = bool(grounded)
        out["action_ko"] = action_ko.strip() if grounded else ""
        out["source"] = source.strip() if grounded else ""
        out["done"] = True
        return "ok"

    return [retrieve_manual, finish]


def build_graph(snap, out, model=MODEL, host=None):
    from langchain_core.messages import HumanMessage
    from langchain_ollama import ChatOllama
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode
    from typing import Annotated, TypedDict

    class S(TypedDict):
        messages: Annotated[list, add_messages]
        steps: int

    tools = _tools(snap, out)
    llm = ChatOllama(model=model, temperature=0, seed=7, num_ctx=4096, keep_alive="30m", base_url=host).bind_tools(tools)

    def agent(state):
        msg = llm.invoke(state["messages"])
        for c in getattr(msg, "tool_calls", []) or []:
            out.setdefault("trace", []).append(f"{c['name']}({', '.join(f'{k}={v}' for k, v in c['args'].items() if k != 'action_ko')})")
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


def decide(snapshot, model=MODEL, host=None):
    """스냅샷 -> {llm, grounded, action_ko, source, trace, sec}. 실패하면 {"llm": False, "error"}."""
    t0 = time.time()
    out = {}
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        graph = build_graph(snapshot, out, model, host)
        sys_msg = SYSTEM.format(label_ko=snapshot.get("label_ko", ""), zone=snapshot.get("zone", ""),
                                standard_action=snapshot.get("standard_action", ""))
        graph.invoke({"messages": [SystemMessage(sys_msg), HumanMessage("Write the action item now.")], "steps": 0},
                     {"recursion_limit": 4 * MAX_STEPS})
    except Exception as e:      # Ollama 가 꺼져 있거나 도구 호출 형식이 틀림
        return {"llm": False, "error": f"{type(e).__name__}: {e}", "sec": round(time.time() - t0, 2), "trace": out.get("trace", [])}
    if not out.get("done"):
        return {"llm": False, "error": "finish 를 안 부름", "sec": round(time.time() - t0, 2), "trace": out.get("trace", [])}
    return {"llm": True, "grounded": out["grounded"], "action_ko": out["action_ko"], "source": out["source"],
            "trace": out.get("trace", []), "sec": round(time.time() - t0, 2)}
