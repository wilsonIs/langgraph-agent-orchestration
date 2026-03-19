from __future__ import annotations

from typing import TypedDict

try:
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:
    raise SystemExit(
        "未找到 `langgraph`。这个目录需要使用专用的 Python 3.10 虚拟环境运行。\n"
        "请执行：\n"
        "1. cd /Users/wilson.zhang/Desktop/agent_engineering_lessons/langgraph-agent-orchestration\n"
        "2. source .venv/bin/activate\n"
        "3. python hello_world.py\n"
        "或者直接运行：\n"
        "   ./run_hello_world.sh"
    ) from exc


class HelloState(TypedDict, total=False):
    user_input: str
    message: str


def build_greeting(state: HelloState) -> HelloState:
    print("Node `build_greeting` received state:")
    print(state)
    print()

    user_input = state.get("user_input", "LangGraph")
    return {
        "message": f"Hello, {user_input}! This message was created inside a LangGraph node."
    }


def build_graph():
    graph_builder = StateGraph(HelloState)
    graph_builder.add_node("build_greeting", build_greeting)
    graph_builder.add_edge(START, "build_greeting")
    graph_builder.add_edge("build_greeting", END)
    return graph_builder.compile()


if __name__ == "__main__":
    graph = build_graph()
    initial_state: HelloState = {"user_input": "LangGraph"}

    print("Input state:")
    print(initial_state)
    print()

    result = graph.invoke(initial_state)

    print("Final state returned by graph.invoke():")
    print(result)
