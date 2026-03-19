from __future__ import annotations

import sys
from typing import Literal

try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    from langgraph.graph import END, START, MessagesState, StateGraph
except ImportError as exc:
    raise SystemExit(
        "未找到 LangGraph 相关依赖。请先进入本目录并激活 .venv：\n"
        "1. cd /Users/wilson.zhang/Desktop/agent_engineering_lessons/langgraph-agent-orchestration\n"
        "2. source .venv/bin/activate\n"
        "3. python conditional_router.py"
    ) from exc


class RouterState(MessagesState, total=False):
    intent: str


def message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)


def detect_intent(state: RouterState) -> RouterState:
    last_message = state["messages"][-1]
    text = message_text(last_message).strip()
    lowered = text.lower()

    if any(token in text for token in ("你好", "您好")) or any(
        token in lowered for token in ("hello", "hi")
    ):
        intent = "greeting"
    elif any(token in text for token in ("天气", "什么", "怎么", "?")) or any(
        token in lowered for token in ("weather", "what", "how", "?")
    ):
        intent = "question"
    else:
        intent = "fallback"

    print("Node `detect_intent` decided intent:")
    print({"text": text, "intent": intent})
    print()

    return {"intent": intent}


def route_by_intent(
    state: RouterState,
) -> Literal["greeting_node", "question_node", "fallback_node"]:
    intent = state.get("intent", "fallback")
    if intent == "greeting":
        return "greeting_node"
    if intent == "question":
        return "question_node"
    return "fallback_node"


def greeting_node(state: RouterState) -> RouterState:
    return {
        "messages": [
            AIMessage(
                content="你好，我已经收到你的消息。这一步展示的是：条件路由把流程分到了 greeting 节点。"
            )
        ]
    }


def question_node(state: RouterState) -> RouterState:
    user_text = message_text(state["messages"][-1]).strip()
    return {
        "messages": [
            AIMessage(
                content=(
                    f"你刚才的问题是：{user_text}\n"
                    "这一步展示的是：我们把用户消息保存在 messages 状态里，"
                    "路由节点读它，回答节点也能继续读它。"
                )
            )
        ]
    }


def fallback_node(state: RouterState) -> RouterState:
    return {
        "messages": [
            AIMessage(
                content="我暂时把这条输入归类为 fallback。后面接入 LLM 时，这里通常会变成更通用的兜底节点。"
            )
        ]
    }


def build_graph():
    graph_builder = StateGraph(RouterState)

    graph_builder.add_node("detect_intent", detect_intent)
    graph_builder.add_node("greeting_node", greeting_node)
    graph_builder.add_node("question_node", question_node)
    graph_builder.add_node("fallback_node", fallback_node)

    graph_builder.add_edge(START, "detect_intent")
    graph_builder.add_conditional_edges(
        "detect_intent",
        route_by_intent,
        {
            "greeting_node": "greeting_node",
            "question_node": "question_node",
            "fallback_node": "fallback_node",
        },
    )
    graph_builder.add_edge("greeting_node", END)
    graph_builder.add_edge("question_node", END)
    graph_builder.add_edge("fallback_node", END)

    return graph_builder.compile()


def run_demo(user_text: str) -> None:
    graph = build_graph()
    initial_state: RouterState = {"messages": [HumanMessage(content=user_text)]}

    print("=" * 72)
    print(f"User input: {user_text}")
    print()

    result = graph.invoke(initial_state)
    final_message = result["messages"][-1]

    print("Final state:")
    print(result)
    print()

    print("Message history:")
    for index, message in enumerate(result["messages"], start=1):
        print(f"{index}. {message.__class__.__name__}: {message_text(message)}")
    print()

    print("Final reply:")
    print(message_text(final_message))
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_demo(" ".join(sys.argv[1:]))
    else:
        run_demo("你好，LangGraph")
        run_demo("这个图的条件路由是怎么工作的？")
        run_demo("帮我记一下明天要做什么")
