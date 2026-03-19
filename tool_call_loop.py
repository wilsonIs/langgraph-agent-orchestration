from __future__ import annotations

import ast
import operator
import re
import sys
from typing import Literal

try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
    from langgraph.graph import END, START, MessagesState, StateGraph
except ImportError as exc:
    raise SystemExit(
        "未找到 LangGraph 相关依赖。请先进入本目录并激活 .venv：\n"
        "1. cd /Users/wilson.zhang/Desktop/agent_engineering_lessons/langgraph-agent-orchestration\n"
        "2. source .venv/bin/activate\n"
        "3. python tool_call_loop.py"
    ) from exc


def message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)


class QAAgentState(MessagesState, total=False):
    user_question: str
    extracted_expression: str
    selected_tool: str
    tool_result: str
    final_answer: str


class SafeMathEvaluator(ast.NodeVisitor):
    ALLOWED_BINARY_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
    }
    ALLOWED_UNARY_OPERATORS = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def evaluate(self, expression: str) -> float:
        tree = ast.parse(expression, mode="eval")
        return float(self.visit(tree.body))

    def visit_BinOp(self, node: ast.BinOp) -> float:
        operator_type = type(node.op)
        if operator_type not in self.ALLOWED_BINARY_OPERATORS:
            raise ValueError(f"不支持的运算符: {operator_type.__name__}")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return self.ALLOWED_BINARY_OPERATORS[operator_type](left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        operator_type = type(node.op)
        if operator_type not in self.ALLOWED_UNARY_OPERATORS:
            raise ValueError(f"不支持的单目运算符: {operator_type.__name__}")
        operand = self.visit(node.operand)
        return self.ALLOWED_UNARY_OPERATORS[operator_type](operand)

    def visit_Constant(self, node: ast.Constant) -> float:
        if not isinstance(node.value, (int, float)):
            raise ValueError("只允许数字常量。")
        return float(node.value)

    def visit_Num(self, node: ast.Num) -> float:  # pragma: no cover - Python 3.10 compatibility
        return float(node.n)

    def generic_visit(self, node: ast.AST):
        raise ValueError(f"表达式包含不允许的语法: {type(node).__name__}")


def extract_expression(text: str) -> str | None:
    cleaned = text.replace("（", "(").replace("）", ")")
    match = re.search(r"([0-9\.\s\+\-\*\/\(\)]+)", cleaned)
    if not match:
        return None

    expression = re.sub(r"\s+", "", match.group(1))
    if not expression or not re.search(r"\d", expression):
        return None
    if not any(operator_token in expression for operator_token in "+-*/"):
        return None
    return expression


def safe_calculator(expression: str) -> str:
    value = SafeMathEvaluator().evaluate(expression)
    if value.is_integer():
        return str(int(value))
    return str(round(value, 4))


def planner_node(state: QAAgentState) -> QAAgentState:
    last_message = state["messages"][-1]

    if isinstance(last_message, HumanMessage):
        user_text = message_text(last_message).strip()
        expression = extract_expression(user_text)

        if expression is None:
            print("Node `planner_node` decided: no tool call needed")
            print()
            final_answer = (
                "这次我没有调用工具，因为我没有识别到明确的数学表达式。"
                "这条路径会直接结束。"
            )
            return {
                "user_question": user_text,
                "selected_tool": "",
                "extracted_expression": "",
                "final_answer": final_answer,
                "messages": [
                    AIMessage(
                        content=final_answer
                    )
                ]
            }

        print("Node `planner_node` decided: call calculator tool")
        print({"expression": expression})
        print()
        return {
            "user_question": user_text,
            "selected_tool": "safe_calculator",
            "extracted_expression": expression,
            "messages": [
                AIMessage(
                    content=f"我准备调用 safe_calculator 来计算 `{expression}`。",
                    tool_calls=[
                        {
                            "name": "safe_calculator",
                            "args": {"expression": expression},
                            "id": "call_calculator_1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }

    if isinstance(last_message, ToolMessage):
        print("Node `planner_node` received tool result and is generating final answer")
        print({"tool_result": message_text(last_message)})
        print()
        final_answer = "工具已经执行完成。\n" f"最终结果是：{message_text(last_message)}"
        return {
            "tool_result": message_text(last_message),
            "final_answer": final_answer,
            "messages": [
                AIMessage(
                    content=final_answer
                )
            ]
        }

    return {
        "final_answer": "我已经完成这轮处理。",
        "messages": [
            AIMessage(content="我已经完成这轮处理。")
        ]
    }


def tool_node(state: QAAgentState) -> QAAgentState:
    last_message = state["messages"][-1]
    tool_results = []
    latest_result = ""

    for tool_call in last_message.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]
        if name != "safe_calculator":
            result_text = f"未知工具: {name}"
        else:
            result_text = safe_calculator(args["expression"])
        latest_result = result_text

        print("Node `tool_node` executed tool")
        print({"tool": name, "args": args, "result": result_text})
        print()

        tool_results.append(
            ToolMessage(
                content=result_text,
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": tool_results, "tool_result": latest_result}


def should_continue(state: QAAgentState) -> Literal["tool_node", END]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    return END


def build_graph():
    graph_builder = StateGraph(QAAgentState)
    graph_builder.add_node("planner_node", planner_node)
    graph_builder.add_node("tool_node", tool_node)

    graph_builder.add_edge(START, "planner_node")
    graph_builder.add_conditional_edges(
        "planner_node",
        should_continue,
        {
            "tool_node": "tool_node",
            END: END,
        },
    )
    graph_builder.add_edge("tool_node", "planner_node")

    return graph_builder.compile()


def run_demo(user_text: str) -> None:
    graph = build_graph()
    initial_state: QAAgentState = {
        "messages": [HumanMessage(content=user_text)],
        "user_question": user_text,
    }

    print("=" * 72)
    print(f"User input: {user_text}")
    print()

    result = graph.invoke(initial_state)

    print("Message history:")
    for index, message in enumerate(result["messages"], start=1):
        print(f"{index}. {message.__class__.__name__}: {message_text(message)}")
        if isinstance(message, AIMessage) and message.tool_calls:
            print(f"   tool_calls={message.tool_calls}")
    print()

    print("Final state:")
    print(result)
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_demo(" ".join(sys.argv[1:]))
    else:
        run_demo("请帮我计算 12 / (3 + 1)")
        run_demo("你好，今天怎么样？")
