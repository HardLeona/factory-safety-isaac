"""'주의' 상태 물체를 바디캠이 자연스럽게 다시 지나칠 때 재판단하는 LLM 에이전트 (LangGraph + 로컬 Qwen2.5-7B, Ollama).

agent.py 의 SafetyAgent 는 바디캠 추적·투표 집계·즉시 위험/주의 분류(규칙, LLM 없음)는 그대로 하고,
"주의 상태로 남은 물체를 작업자가 다시 지나쳤을 때 이걸 위험으로 볼지 안전으로 볼지"만 이 모듈에 맡긴다.
CCTV 확대 촬영 같은 능동적 재확인 수단이 없으므로, 트리거는 오직 "바디캠이 같은 자리에서 같은 물체를 다시 포착함"뿐이다.

    주의(主義) 상태 Finding 을 바디캠이 재관측하면 -> decide(snapshot) 한 번 호출
        confident=true  -> verdict_class 로 확정 (위험 또는 안전)
        confident=false 또는 LLM 실패/없음 -> agent.py 의 규칙 폴백:
            이전+이번 관측에 고위험 후보가 있으면 위험으로 확정, 없으면 다음 재관측까지 '주의' 유지

LLM 은 finish 로 받은 클래스 이름만 쓸 수 있다 (evidence 가 실제로 본 클래스만, 지어내지 않음).
이 모듈은 손 인식·음성 도우미 프로세스 (.venv-assistant, scripts/assistant_worker.py) 안에서 돈다.
"""
import json
import time

MODEL = "qwen2.5:7b"
MAX_STEPS = 4
SYSTEM = """You are the safety agent of a warehouse patrol system. A worker walking their usual patrol route with a chest
bodycam has just walked past the same spot again, and the bodycam re-detected an object that was previously left in an
uncertain "caution" state (category: {kind}) — the first time it was seen too briefly or too ambiguously to tell whether
it is a hazard or a safe state of the same kind (e.g. a fire extinguisher that is either fine, fallen, or blocked).
Call `evidence` first to see what was observed the first time and what the bodycam sees now, on this new pass.
You may call `retrieve_manual` if a safety manual passage would help you judge how serious or plausible a candidate class is.
Then call `finish` exactly once with:
- confident: true only if the combined evidence (prior + new pass) clearly points to one class; false if still genuinely ambiguous
- verdict_class: the class name you believe is correct (must be one you saw via `evidence`, never invented)
- reason_ko: one Korean sentence explaining your judgment (what evidence, why confident or not), for the agent log
Do not mention anything `evidence` did not return. Use at most 3 tool calls before finish."""


def _tools(snap, out):
    from langchain_core.tools import tool

    candidates = set(snap.get("prior_observations", {})) | set(snap.get("new_observations", {}))

    @tool
    def evidence() -> str:
        """See what was observed the first time (prior_observations) and on this new pass (new_observations),
        the object's category, its zone, and how long ago the first observation was (elapsed_s)."""
        return json.dumps({k: snap.get(k) for k in ("target_kind", "zone", "prior_observations", "new_observations",
                                                     "elapsed_s")}, ensure_ascii=False)

    @tool
    def retrieve_manual(query: str) -> str:
        """Search the verified safety manuals for passages about a candidate class or hazard."""
        from . import manuals
        hits = manuals.retrieve(query)
        return json.dumps(hits or "no relevant manual passage", ensure_ascii=False)

    @tool
    def finish(confident: bool, verdict_class: str, reason_ko: str) -> str:
        """Finish with your judgment. confident: true only if clearly resolved. verdict_class: your best class
        (must be one seen via evidence). reason_ko: Korean reason for the agent log."""
        out["confident"] = bool(confident)
        out["verdict"] = verdict_class if verdict_class in candidates else None
        out["reason_ko"] = reason_ko.strip()
        out["done"] = True
        return "ok"

    return [evidence, retrieve_manual, finish]


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
            out.setdefault("trace", []).append(f"{c['name']}({', '.join(f'{k}={v}' for k, v in c['args'].items() if k != 'reason_ko')})")
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
    """스냅샷 -> {llm, confident, verdict, reason_ko, trace, sec}. 실패하면 {"llm": False, "error"}."""
    t0 = time.time()
    out = {}
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        graph = build_graph(snapshot, out, model, host)
        sys_msg = SYSTEM.format(kind=snapshot.get("target_kind", "object"))
        user = "Judge this re-observation now."
        graph.invoke({"messages": [SystemMessage(sys_msg), HumanMessage(user)], "steps": 0}, {"recursion_limit": 4 * MAX_STEPS})
    except Exception as e:      # Ollama 가 꺼져 있거나 도구 호출 형식이 틀림
        return {"llm": False, "error": f"{type(e).__name__}: {e}", "sec": round(time.time() - t0, 2), "trace": out.get("trace", [])}
    if not out.get("done") or out.get("verdict") is None:
        return {"llm": False, "error": "finish 를 안 부름 (또는 근거 없는 클래스)", "sec": round(time.time() - t0, 2),
                "trace": out.get("trace", [])}
    return {"llm": True, "model": model, "confident": out["confident"], "verdict": out["verdict"], "reason_ko": out["reason_ko"],
            "trace": out.get("trace", []), "sec": round(time.time() - t0, 2)}
