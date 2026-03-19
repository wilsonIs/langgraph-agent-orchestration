from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import uuid4

import httpx

try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.errors import GraphInterrupt
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.runtime import Runtime
    from langgraph.store.memory import InMemoryStore
    from langgraph.types import Command, RetryPolicy, interrupt
except ImportError as exc:
    raise SystemExit(
        "未找到 LangGraph 相关依赖。请先进入本目录并激活 .venv：\n"
        "1. cd /Users/wilson.zhang/Desktop/agent_engineering_lessons/langgraph-agent-orchestration\n"
        "2. source .venv/bin/activate\n"
        "3. python advanced_qa_agent.py"
    ) from exc


IntentName = Literal[
    "smalltalk",
    "product_consultation",
    "complaint",
    "order_query",
    "human_handoff",
]
MemoryAction = Literal["write_memory", "recall_memory", "none"]

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_PATH = LOG_DIR / "advanced_qa_agent.log"
PROFILE_NAMESPACE = ("customer_profile",)
COMPLAINT_NAMESPACE = ("complaint_ticket",)
HANDOFF_NAMESPACE = ("human_handoff",)
CASE_HISTORY_NAMESPACE = ("customer_case_history",)

PRODUCT_KNOWLEDGE_BASE = [
    {
        "id": "kb-001",
        "product": "AeroBuds X1",
        "title": "AeroBuds X1 真无线降噪耳机",
        "content": (
            "AeroBuds X1 支持 42dB 主动降噪，单次续航 8 小时，搭配充电盒总续航 36 小时，"
            "支持双设备连接，整机达到 IP54 防尘防水等级，提供 2 年保修。"
        ),
        "tags": ["耳机", "降噪", "续航", "保修", "蓝牙"],
    },
    {
        "id": "kb-002",
        "product": "HomeHub Mini",
        "title": "HomeHub Mini 智能音箱",
        "content": (
            "HomeHub Mini 支持 Matter 协议与语音控制，可联动灯光、门锁和传感器。"
            "内置 360 度扬声器，适合客厅和卧室场景，整机提供 1 年保修。"
        ),
        "tags": ["音箱", "智能家居", "Matter", "保修"],
    },
    {
        "id": "kb-003",
        "product": "SwiftClean S2",
        "title": "SwiftClean S2 无线洗地机",
        "content": (
            "SwiftClean S2 拥有 60 分钟续航和自清洁底座，支持干湿两用，"
            "污水箱与净水箱分离，适合家庭大户型日常清洁。"
        ),
        "tags": ["洗地机", "清洁", "续航", "家电"],
    },
    {
        "id": "kb-004",
        "product": "TravelMate 65W",
        "title": "TravelMate 65W 氮化镓充电器",
        "content": (
            "TravelMate 65W 配备 2C1A 接口，支持 PD 快充，单口最高 65W 输出，"
            "适合手机、平板和轻薄本混合充电，提供 18 个月保修。"
        ),
        "tags": ["充电器", "快充", "65W", "保修"],
    },
]

ORDER_DB = {
    "SO-20260318-1001": {
        "status": "已发货",
        "product": "AeroBuds X1 曜石黑",
        "carrier": "顺丰速运",
        "latest_update": "包裹已于 2026-03-19 09:20 到达上海青浦转运中心。",
        "estimated_delivery": "2026-03-21",
    },
    "SO-20260317-2048": {
        "status": "已签收",
        "product": "HomeHub Mini 月白色",
        "carrier": "京东物流",
        "latest_update": "包裹已于 2026-03-18 18:32 由前台代收签收。",
        "estimated_delivery": "2026-03-18",
    },
    "SO-20260316-3008": {
        "status": "待发货",
        "product": "SwiftClean S2",
        "carrier": "待分配",
        "latest_update": "订单已支付成功，仓库预计于 2026-03-20 12:00 前完成出库。",
        "estimated_delivery": "2026-03-23",
    },
}


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("advanced_qa_agent")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False

    return logger


LOGGER = setup_logging()


class TransientServiceError(RuntimeError):
    """A retryable service error used to demonstrate RetryPolicy."""


@dataclass
class AgentContext:
    user_id: str
    stream_writer: Any | None = None


@dataclass
class LLMConfig:
    provider: str
    api_key: str
    base_url: str
    model: str


@dataclass
class PlannerDecision:
    intent: IntentName
    reason: str
    knowledge_query: str
    order_number: str
    complaint_summary: str
    handoff_reason: str
    memory_action: MemoryAction
    memory_fact: str
    memory_category: str
    needs_human_review: bool


class CustomerServiceState(MessagesState, total=False):
    user_question: str
    intent: str
    intent_reason: str
    selected_tool: str
    tool_args: dict[str, str]
    tool_result: str
    final_answer: str
    approval_status: str
    memory_facts: list[str]
    knowledge_hits: list[dict[str, str]]
    order_number: str
    complaint_ticket_id: str
    handoff_ticket_id: str
    retry_count: int
    last_error: str
    memory_action: str
    memory_candidate: str
    memory_category: str


def message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)


def serialize_message(message: BaseMessage) -> dict[str, Any]:
    payload = {
        "type": message.__class__.__name__,
        "content": message_text(message),
    }
    if isinstance(message, AIMessage) and message.tool_calls:
        payload["tool_calls"] = message.tool_calls
    if isinstance(message, ToolMessage):
        payload["tool_call_id"] = message.tool_call_id
        payload["status"] = getattr(message, "status", "success")
    return payload


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseMessage):
        return serialize_message(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return str(value)


def latest_assistant_reply(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return message_text(message)
    return ""


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def resolve_llm_config() -> LLMConfig | None:
    openai_key = _env("OPENAI_API_KEY")
    deepseek_key = _env("DEEPSEEK_API_KEY")

    if openai_key:
        return LLMConfig(
            provider="openai",
            api_key=openai_key,
            base_url=_env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=_env("OPENAI_MODEL", "gpt-4o-mini"),
        )

    if deepseek_key:
        return LLMConfig(
            provider="deepseek",
            api_key=deepseek_key,
            base_url=_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=_env("DEEPSEEK_MODEL", "deepseek-chat"),
        )

    return None


def normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", stripped)
        stripped = re.sub(r"\n```$", "", stripped)
    return stripped.strip()


def active_proxy_settings() -> dict[str, str]:
    proxy_settings: dict[str, str] = {}
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        value = _env(name)
        if value:
            proxy_settings[name] = value
    return proxy_settings


def build_llm_fallback_reason(exc: Exception) -> str:
    message = str(exc).strip()
    proxy_settings = active_proxy_settings()
    if "Using SOCKS proxy" in message or "socksio" in message:
        return (
            "LLM 决策失败，已回退规则逻辑：检测到当前环境正在使用 SOCKS 代理，"
            "但虚拟环境里还没有安装 `socksio`。"
            "请先执行 `pip install -r requirements.txt`，"
            "或单独执行 `pip install \"httpx[socks]\"`。"
            f" 原始错误: {message}"
        )
    if "Connection reset by peer" in message and proxy_settings:
        proxy_summary = ", ".join(f"{key}={value}" for key, value in proxy_settings.items())
        return (
            "LLM 决策失败，已回退规则逻辑：请求在代理链路上被对端重置。"
            f" 当前检测到代理配置：{proxy_summary}。"
            "请先确认 ClashX 或本地代理是否正常工作，"
            "或者在启动前临时关闭本项目对环境代理的继承："
            " `LLM_HTTP_TRUST_ENV=false python customer_service_server.py`。"
            f" 原始错误: {message}"
        )
    return f"LLM 决策失败，已回退规则逻辑: {message}"


def call_planner_llm(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    trust_env: bool,
) -> PlannerDecision:
    with httpx.Client(timeout=30.0, trust_env=trust_env) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    return parse_planner_decision_payload(content)


def should_retry_without_proxy(exc: Exception, *, trust_env: bool) -> bool:
    if not trust_env:
        return False
    if not active_proxy_settings():
        return False
    return "Connection reset by peer" in str(exc)


def emit_runtime_event(
    runtime: Runtime[AgentContext],
    payload: dict[str, Any],
) -> None:
    stream_writer = getattr(runtime.context, "stream_writer", None)
    if callable(stream_writer):
        stream_writer(to_jsonable(payload))


def build_customer_response_prompt(
    state: CustomerServiceState,
    *,
    draft_answer: str,
) -> list[dict[str, str]]:
    system_prompt = (
        "你是品牌官方智能客服。你的任务是基于给定事实，用自然、专业、温和的中文回复用户。\n"
        "规则：\n"
        "1. 严格基于提供的 draft_answer 和 context，不要新增事实。\n"
        "2. 可以润色语气，但不要改动订单状态、工单号、保修时长等关键信息。\n"
        "3. 回复尽量直接、清楚，不写标题，不使用多层列表。\n"
        "4. 如果 context 中有知识库命中，只能使用命中的内容，不要自行补充产品参数。\n"
        "5. 输出纯文本，不要解释你如何生成答案。"
    )
    payload = {
        "intent": state.get("intent", ""),
        "user_question": state.get("user_question", ""),
        "memory_facts": state.get("memory_facts", []),
        "selected_tool": state.get("selected_tool", ""),
        "knowledge_hits": state.get("knowledge_hits", []),
        "order_number": state.get("order_number", ""),
        "complaint_ticket_id": state.get("complaint_ticket_id", ""),
        "handoff_ticket_id": state.get("handoff_ticket_id", ""),
        "draft_answer": draft_answer,
    }
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]


def call_response_llm_stream(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    trust_env: bool,
    on_token: Callable[[str], None] | None = None,
) -> str:
    accumulated: list[str] = []
    timeout = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)
    with httpx.Client(timeout=timeout, trust_env=trust_env) as client:
        with client.stream("POST", url, headers=headers, json=payload) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    break

                event = json.loads(data)
                choices = event.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                token = delta.get("content")
                if not token:
                    continue
                accumulated.append(token)
                if on_token is not None:
                    on_token(token)

    return "".join(accumulated).strip()


def generate_customer_reply(
    state: CustomerServiceState,
    runtime: Runtime[AgentContext],
    *,
    draft_answer: str,
) -> str:
    config = resolve_llm_config()
    if config is None:
        LOGGER.warning("No API key configured for response LLM. Falling back to draft answer.")
        return draft_answer

    url = f"{normalize_base_url(config.base_url)}/chat/completions"
    payload = {
        "model": config.model,
        "temperature": 0.3,
        "stream": True,
        "messages": build_customer_response_prompt(state, draft_answer=draft_answer),
    }
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    trust_env = _env_flag("LLM_HTTP_TRUST_ENV", True)
    streamed_tokens: list[str] = []

    def on_token(token: str) -> None:
        streamed_tokens.append(token)
        emit_runtime_event(runtime, {"type": "token", "delta": token})

    LOGGER.info(
        "Calling response LLM provider=%s model=%s trust_env=%s",
        config.provider,
        config.model,
        trust_env,
    )
    try:
        answer = call_response_llm_stream(
            url=url,
            headers=headers,
            payload=payload,
            trust_env=trust_env,
            on_token=on_token,
        )
        return answer or draft_answer
    except Exception as exc:
        if should_retry_without_proxy(exc, trust_env=trust_env):
            LOGGER.warning(
                "Response LLM request failed while using env proxy. Retrying once with trust_env=False."
            )
            streamed_tokens.clear()
            try:
                answer = call_response_llm_stream(
                    url=url,
                    headers=headers,
                    payload=payload,
                    trust_env=False,
                    on_token=on_token,
                )
                return answer or draft_answer
            except Exception:
                LOGGER.exception("Response LLM retry without env proxy failed")
        LOGGER.exception("Response LLM failed. Falling back to draft answer.")
        return draft_answer


def parse_order_number(text: str) -> str:
    pattern = re.compile(r"\b(?:SO|ORD)-\d{8}-\d{4}\b", re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return ""
    return match.group(0).upper()


def normalize_intent(value: str) -> IntentName:
    normalized = value.strip().lower()
    if normalized not in {
        "smalltalk",
        "product_consultation",
        "complaint",
        "order_query",
        "human_handoff",
    }:
        return "smalltalk"
    return normalized  # type: ignore[return-value]


def normalize_memory_action(value: str) -> MemoryAction:
    normalized = value.strip().lower()
    if normalized not in {"write_memory", "recall_memory", "none"}:
        return "none"
    return normalized  # type: ignore[return-value]


def extract_memory_fact(user_text: str) -> tuple[str, str]:
    stripped = user_text.strip().strip("。")
    if stripped.startswith("记住"):
        fact = stripped.removeprefix("记住").strip(" ：:，,。")
        return fact, "preference"

    preference_markers = ["以后请", "默认", "我更喜欢", "我喜欢", "我不希望", "我不接受"]
    if any(marker in stripped for marker in preference_markers):
        return stripped, "preference"

    profile_markers = ["我是", "我在", "我住在", "我的公司是", "我是企业客户"]
    if any(marker in stripped for marker in profile_markers):
        return stripped, "profile"

    return "", ""


def fallback_planner_decision(user_text: str, memory_facts: list[str]) -> PlannerDecision:
    stripped = user_text.strip()
    lower = stripped.lower()
    order_number = parse_order_number(stripped)
    memory_fact, memory_category = extract_memory_fact(stripped)

    if any(token in stripped for token in ("你记得我", "你还记得我", "我的偏好", "记住了什么")):
        return PlannerDecision(
            intent="smalltalk",
            reason="识别为跨会话记忆召回请求。",
            knowledge_query="",
            order_number="",
            complaint_summary="",
            handoff_reason="",
            memory_action="recall_memory",
            memory_fact="",
            memory_category="",
            needs_human_review=False,
        )

    if any(token in stripped for token in ("人工", "转人工", "真人客服", "升级处理", "专员跟进")):
        return PlannerDecision(
            intent="human_handoff",
            reason="用户明确要求转人工或升级处理。",
            knowledge_query="",
            order_number=order_number,
            complaint_summary="",
            handoff_reason=stripped,
            memory_action="none",
            memory_fact="",
            memory_category="",
            needs_human_review=True,
        )

    if any(token in stripped for token in ("投诉", "差评", "质量问题", "破损", "漏发", "不满", "退款")):
        return PlannerDecision(
            intent="complaint",
            reason="用户表达强烈不满或投诉诉求。",
            knowledge_query="",
            order_number=order_number,
            complaint_summary=stripped,
            handoff_reason="",
            memory_action="none",
            memory_fact="",
            memory_category="",
            needs_human_review=True,
        )

    if order_number or any(token in stripped for token in ("订单", "物流", "发货", "配送", "签收", "快递")):
        return PlannerDecision(
            intent="order_query",
            reason="用户在查询订单进度或物流状态。",
            knowledge_query="",
            order_number=order_number,
            complaint_summary="",
            handoff_reason="",
            memory_action="none",
            memory_fact="",
            memory_category="",
            needs_human_review=False,
        )

    product_keywords = {
        "耳机",
        "音箱",
        "洗地机",
        "充电器",
        "续航",
        "保修",
        "参数",
        "价格",
        "推荐",
        "降噪",
        "matter",
        "aerobuds",
        "homehub",
        "swiftclean",
        "travelmate",
    }
    if any(keyword in lower for keyword in product_keywords):
        return PlannerDecision(
            intent="product_consultation",
            reason="检测到产品名或产品参数咨询关键词。",
            knowledge_query=stripped,
            order_number=order_number,
            complaint_summary="",
            handoff_reason="",
            memory_action="none",
            memory_fact="",
            memory_category="",
            needs_human_review=False,
        )

    if memory_fact and not memory_facts:
        return PlannerDecision(
            intent="smalltalk",
            reason="识别为稳定偏好或用户画像，可写入长期记忆。",
            knowledge_query="",
            order_number=order_number,
            complaint_summary="",
            handoff_reason="",
            memory_action="write_memory",
            memory_fact=memory_fact,
            memory_category=memory_category,
            needs_human_review=True,
        )

    return PlannerDecision(
        intent="smalltalk",
        reason="未命中业务关键词，按闲聊处理。",
        knowledge_query="",
        order_number=order_number,
        complaint_summary="",
        handoff_reason="",
        memory_action="none",
        memory_fact="",
        memory_category="",
        needs_human_review=False,
    )


def build_planner_prompt(user_text: str, memory_facts: list[str]) -> list[dict[str, str]]:
    system_prompt = (
        "你是一个智能客服 Agent 的路由与记忆决策器。"
        "你的任务不是直接回答用户，而是输出一份 JSON 决策，告诉系统下一步该走哪个客服能力。\n\n"
        "可选 intent 只有：\n"
        "1. smalltalk\n"
        "2. product_consultation\n"
        "3. complaint\n"
        "4. order_query\n"
        "5. human_handoff\n\n"
        "另外，你还要判断这条消息是否涉及长期记忆：\n"
        "- write_memory：用户表达了长期偏好、稳定身份、长期约束\n"
        "- recall_memory：用户在询问你记得什么\n"
        "- none：不需要长期记忆动作\n\n"
        "决策原则：\n"
        "- 普通寒暄归到 smalltalk\n"
        "- 产品参数、保修、适配、推荐归到 product_consultation\n"
        "- 抱怨、质量问题、退款、差评、体验差归到 complaint\n"
        "- 物流、发货、签收、订单号归到 order_query\n"
        "- 明确要求人工客服、升级处理、专员跟进归到 human_handoff\n"
        "- 关键动作 complaint、human_handoff、write_memory 默认 needs_human_review=true\n"
        "- 只输出 JSON，不要输出任何额外解释。\n\n"
        "JSON schema:\n"
        "{\n"
        '  "intent": "smalltalk|product_consultation|complaint|order_query|human_handoff",\n'
        '  "reason": "string",\n'
        '  "knowledge_query": "string",\n'
        '  "order_number": "string",\n'
        '  "complaint_summary": "string",\n'
        '  "handoff_reason": "string",\n'
        '  "memory_action": "write_memory|recall_memory|none",\n'
        '  "memory_fact": "string",\n'
        '  "memory_category": "preference|profile|constraint|task|",\n'
        '  "needs_human_review": true\n'
        "}"
    )
    user_prompt = json.dumps(
        {
            "user_text": user_text,
            "existing_memory_facts": memory_facts,
        },
        ensure_ascii=False,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_planner_decision_payload(content: str) -> PlannerDecision:
    stripped = strip_json_fence(content)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("模型返回中未找到 JSON 对象。")

    payload = json.loads(stripped[start : end + 1])
    intent = normalize_intent(str(payload.get("intent", "smalltalk")))
    memory_action = normalize_memory_action(str(payload.get("memory_action", "none")))
    needs_human_review = bool(payload.get("needs_human_review", False))
    if intent in {"complaint", "human_handoff"} or memory_action == "write_memory":
        needs_human_review = True

    return PlannerDecision(
        intent=intent,
        reason=str(payload.get("reason", "")).strip(),
        knowledge_query=str(payload.get("knowledge_query", "")).strip(),
        order_number=str(payload.get("order_number", "")).strip().upper(),
        complaint_summary=str(payload.get("complaint_summary", "")).strip(),
        handoff_reason=str(payload.get("handoff_reason", "")).strip(),
        memory_action=memory_action,
        memory_fact=str(payload.get("memory_fact", "")).strip(),
        memory_category=str(payload.get("memory_category", "")).strip(),
        needs_human_review=needs_human_review,
    )


def decide_customer_action_with_llm(
    user_text: str,
    memory_facts: list[str],
) -> PlannerDecision:
    config = resolve_llm_config()
    if config is None:
        LOGGER.warning("No API key configured for planner LLM. Falling back to heuristic.")
        return fallback_planner_decision(user_text, memory_facts)

    url = f"{normalize_base_url(config.base_url)}/chat/completions"
    payload = {
        "model": config.model,
        "temperature": 0,
        "messages": build_planner_prompt(user_text, memory_facts),
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    trust_env = _env_flag("LLM_HTTP_TRUST_ENV", True)

    LOGGER.info(
        "Calling planner LLM provider=%s model=%s trust_env=%s",
        config.provider,
        config.model,
        trust_env,
    )
    try:
        decision = call_planner_llm(
            url=url,
            headers=headers,
            payload=payload,
            trust_env=trust_env,
        )
        LOGGER.info(
            "Planner LLM resolved intent=%s memory_action=%s",
            decision.intent,
            decision.memory_action,
        )
        return decision
    except Exception as exc:
        if should_retry_without_proxy(exc, trust_env=trust_env):
            LOGGER.warning(
                "Planner LLM request failed while using env proxy. Retrying once with trust_env=False."
            )
            try:
                decision = call_planner_llm(
                    url=url,
                    headers=headers,
                    payload=payload,
                    trust_env=False,
                )
                LOGGER.info(
                    "Planner LLM retry without env proxy succeeded intent=%s memory_action=%s",
                    decision.intent,
                    decision.memory_action,
                )
                return decision
            except Exception as retry_exc:
                LOGGER.exception("Planner LLM retry without env proxy failed")
                exc = retry_exc
        LOGGER.exception("Planner LLM failed. Falling back to heuristic.")
        fallback = fallback_planner_decision(user_text, memory_facts)
        fallback.reason = build_llm_fallback_reason(exc)
        return fallback


def tokenize_for_search(text: str) -> list[str]:
    normalized = text.lower()
    raw_tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", normalized)
    tokens: list[str] = []
    for token in raw_tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            if len(token) <= 2:
                tokens.append(token)
            else:
                tokens.extend(token[index : index + 2] for index in range(len(token) - 1))
        else:
            tokens.append(token)
    deduped: list[str] = []
    for token in tokens:
        if token and token not in deduped:
            deduped.append(token)
    return deduped


def retrieve_product_documents(query: str) -> list[dict[str, str]]:
    tokens = tokenize_for_search(query)
    ranked: list[tuple[int, dict[str, str]]] = []
    lowered_query = query.lower()

    for document in PRODUCT_KNOWLEDGE_BASE:
        score = 0
        title = f"{document['product']} {document['title']}".lower()
        content = document["content"].lower()
        tags = " ".join(document["tags"]).lower()

        if document["product"].lower() in lowered_query:
            score += 8

        for token in tokens:
            if token in title:
                score += 5
            if token in tags:
                score += 3
            if token in content:
                score += 2

        if score > 0:
            ranked.append(
                (
                    score,
                    {
                        "id": document["id"],
                        "product": document["product"],
                        "title": document["title"],
                        "snippet": document["content"],
                        "score": str(score),
                    },
                )
            )

    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        return []

    top_score = ranked[0][0]
    threshold = max(4, int(top_score * 0.45))
    filtered = [item for score, item in ranked if score >= threshold]
    return filtered[:3]


def build_smalltalk_reply(user_text: str, memory_facts: list[str]) -> str:
    stripped = user_text.strip()
    if any(token in stripped for token in ("你好", "您好", "在吗", "哈喽")):
        return "你好，我在这边。你可以直接问我产品参数、订单进度、售后投诉，或者让我帮你转人工客服。"

    if any(token in stripped for token in ("谢谢", "感谢", "辛苦了")):
        return "不客气，这是我应该做的。如果你还想继续查订单、看产品资料或升级售后，我可以继续帮你。"

    if memory_facts:
        return (
            "我可以继续协助你。当前我还记得这些跨会话偏好："
            + "；".join(memory_facts[:3])
            + "。如果你的需求有变化，也可以直接告诉我。"
        )

    return "我可以继续协助你。你可以问我产品咨询、订单查询、投诉处理，或者直接要求人工客服介入。"


def build_product_answer(hits: list[dict[str, str]]) -> str:
    if not hits:
        return (
            "我先在产品知识库里查了一轮，但还没有找到足够匹配的资料。"
            "你可以告诉我更具体的产品名、预算，或者你关心的是续航、保修还是兼容性。"
        )

    lines = ["我已经根据知识库检索到这些信息："]
    for hit in hits[:2]:
        lines.append(f"- {hit['product']}：{hit['snippet']}")
    lines.append("如果你告诉我你的使用场景，我可以继续帮你缩小选择范围。")
    return "\n".join(lines)


def build_order_answer(order_number: str, order_info: str) -> str:
    if not order_info:
        return "我暂时没有查到这笔订单。请确认订单号是否正确，例如 `SO-20260318-1001`。"

    payload = json.loads(order_info)
    if not payload:
        return "我暂时没有查到这笔订单。请确认订单号是否正确，例如 `SO-20260318-1001`。"

    return (
        f"订单 {order_number} 当前状态是：{payload['status']}。\n"
        f"商品：{payload['product']}。\n"
        f"物流：{payload['carrier']}。\n"
        f"最新进度：{payload['latest_update']}\n"
        f"预计送达日期：{payload['estimated_delivery']}。"
    )


def build_complaint_answer(ticket_id: str, summary: str) -> str:
    return (
        f"我已经为你创建投诉工单 {ticket_id}。\n"
        f"问题摘要：{summary}\n"
        "售后专员会在 2 小时内跟进。如果你愿意，我也可以继续帮你转人工客服。"
    )


def build_handoff_answer(ticket_id: str, reason: str) -> str:
    return (
        f"我已经提交人工客服接入申请 {ticket_id}。\n"
        f"转接原因：{reason}\n"
        "当前预计排队时间约 5 到 10 分钟，请保持当前会话在线。"
    )


def build_memory_answer(fact: str) -> str:
    return f"我已经记录你的长期偏好：{fact}。后续新的会话里，我会优先参考这条信息。"


def parse_approval(resume_value: Any) -> tuple[bool, str]:
    if isinstance(resume_value, dict):
        approved = bool(resume_value.get("approved"))
        reviewer_note = str(resume_value.get("reviewer_note", "")).strip()
        return approved, reviewer_note

    normalized = str(resume_value).strip().lower()
    approved = normalized in {"y", "yes", "true", "1", "approve", "approved"}
    return approved, str(resume_value).strip()


def generate_ticket_id(prefix: str) -> str:
    date_part = datetime.now().strftime("%Y%m%d")
    suffix = uuid4().hex[:6].upper()
    return f"{prefix}-{date_part}-{suffix}"


SERVICE_ATTEMPTS: dict[str, int] = {}
CHECKPOINTER = MemorySaver()
STORE = InMemoryStore()


def maybe_raise_transient_service_error(key: str, trigger: bool) -> int:
    attempt = SERVICE_ATTEMPTS.get(key, 0) + 1
    SERVICE_ATTEMPTS[key] = attempt
    if trigger and attempt == 1:
        raise TransientServiceError("模拟一次临时服务故障，重试后应成功。")
    return attempt


def load_memory_node(
    state: CustomerServiceState,
    runtime: Runtime[AgentContext],
) -> CustomerServiceState:
    try:
        facts: list[str] = []
        if runtime.store:
            item = runtime.store.get(PROFILE_NAMESPACE, runtime.context.user_id)
            if item:
                facts = list(item.value.get("facts", []))

        LOGGER.info(
            "Loaded customer profile memory user_id=%s fact_count=%s",
            runtime.context.user_id,
            len(facts),
        )
        return {
            "memory_facts": facts,
            "last_error": "",
            "approval_status": "",
        }
    except Exception as exc:
        LOGGER.exception("load_memory_node failed")
        return {"memory_facts": [], "last_error": str(exc)}


def planner_node(
    state: CustomerServiceState,
    runtime: Runtime[AgentContext],
) -> CustomerServiceState:
    try:
        last_message = state["messages"][-1]

        if isinstance(last_message, HumanMessage):
            user_text = message_text(last_message).strip()
            memory_facts = state.get("memory_facts", [])
            decision = decide_customer_action_with_llm(user_text, memory_facts)

            LOGGER.info(
                "Planner decided intent=%s memory_action=%s reason=%s",
                decision.intent,
                decision.memory_action,
                decision.reason,
            )

            if decision.memory_action == "recall_memory":
                if memory_facts:
                    draft_answer = "我当前记得这些跨会话偏好：" + "；".join(memory_facts)
                else:
                    draft_answer = "我还没有记录到你的长期偏好。你可以告诉我你的偏好或约束，我会在确认后保存。"
                response_state: CustomerServiceState = {
                    **state,
                    "user_question": user_text,
                    "intent": decision.intent,
                    "intent_reason": decision.reason,
                }
                final_answer = generate_customer_reply(
                    response_state,
                    runtime,
                    draft_answer=draft_answer,
                )
                return {
                    "user_question": user_text,
                    "intent": decision.intent,
                    "intent_reason": decision.reason,
                    "selected_tool": "",
                    "tool_args": {},
                    "final_answer": final_answer,
                    "memory_action": decision.memory_action,
                    "memory_candidate": "",
                    "memory_category": "",
                    "last_error": "",
                    "messages": [AIMessage(content=final_answer)],
                }

            if (
                decision.memory_action == "write_memory"
                and decision.memory_fact
                and decision.intent == "smalltalk"
            ):
                return {
                    "user_question": user_text,
                    "intent": decision.intent,
                    "intent_reason": decision.reason,
                    "selected_tool": "save_customer_memory",
                    "tool_args": {
                        "fact": decision.memory_fact,
                        "category": decision.memory_category or "preference",
                    },
                    "approval_status": "pending" if decision.needs_human_review else "not_needed",
                    "final_answer": "",
                    "memory_action": decision.memory_action,
                    "memory_candidate": decision.memory_fact,
                    "memory_category": decision.memory_category or "preference",
                    "last_error": "",
                    "messages": [
                        AIMessage(
                            content=f"我建议把这条长期偏好保存下来：{decision.memory_fact}",
                            tool_calls=[
                                {
                                    "name": "save_customer_memory",
                                    "args": {
                                        "fact": decision.memory_fact,
                                        "category": decision.memory_category or "preference",
                                    },
                                    "id": "call_save_customer_memory_1",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    ],
                }

            if decision.intent == "smalltalk":
                draft_answer = build_smalltalk_reply(user_text, memory_facts)
                response_state: CustomerServiceState = {
                    **state,
                    "user_question": user_text,
                    "intent": decision.intent,
                    "intent_reason": decision.reason,
                }
                final_answer = generate_customer_reply(
                    response_state,
                    runtime,
                    draft_answer=draft_answer,
                )
                return {
                    "user_question": user_text,
                    "intent": decision.intent,
                    "intent_reason": decision.reason,
                    "selected_tool": "",
                    "tool_args": {},
                    "final_answer": final_answer,
                    "memory_action": decision.memory_action,
                    "memory_candidate": decision.memory_fact,
                    "memory_category": decision.memory_category,
                    "last_error": "",
                    "messages": [AIMessage(content=final_answer)],
                }

            if decision.intent == "product_consultation":
                query = decision.knowledge_query or user_text
                return {
                    "user_question": user_text,
                    "intent": decision.intent,
                    "intent_reason": decision.reason,
                    "selected_tool": "search_product_knowledge",
                    "tool_args": {"query": query},
                    "approval_status": "not_needed",
                    "final_answer": "",
                    "memory_action": decision.memory_action,
                    "memory_candidate": decision.memory_fact,
                    "memory_category": decision.memory_category,
                    "last_error": "",
                    "messages": [
                        AIMessage(
                            content=f"我先去知识库检索与你问题相关的产品资料：{query}",
                            tool_calls=[
                                {
                                    "name": "search_product_knowledge",
                                    "args": {"query": query},
                                    "id": "call_search_product_knowledge_1",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    ],
                }

            if decision.intent == "order_query":
                order_number = decision.order_number or parse_order_number(user_text)
                if not order_number:
                    draft_answer = (
                        "我可以帮你查订单。请直接发我订单号，例如 `SO-20260318-1001`，"
                        "我会继续帮你看发货和配送进度。"
                    )
                    response_state: CustomerServiceState = {
                        **state,
                        "user_question": user_text,
                        "intent": decision.intent,
                        "intent_reason": decision.reason,
                    }
                    final_answer = generate_customer_reply(
                        response_state,
                        runtime,
                        draft_answer=draft_answer,
                    )
                    return {
                        "user_question": user_text,
                        "intent": decision.intent,
                        "intent_reason": decision.reason,
                        "selected_tool": "",
                        "tool_args": {},
                        "final_answer": final_answer,
                        "memory_action": decision.memory_action,
                        "memory_candidate": decision.memory_fact,
                        "memory_category": decision.memory_category,
                        "last_error": "",
                        "messages": [AIMessage(content=final_answer)],
                    }
                return {
                    "user_question": user_text,
                    "intent": decision.intent,
                    "intent_reason": decision.reason,
                    "selected_tool": "lookup_order",
                    "tool_args": {"order_number": order_number},
                    "order_number": order_number,
                    "approval_status": "not_needed",
                    "final_answer": "",
                    "memory_action": decision.memory_action,
                    "memory_candidate": decision.memory_fact,
                    "memory_category": decision.memory_category,
                    "last_error": "",
                    "messages": [
                        AIMessage(
                            content=f"我现在开始查询订单 {order_number} 的状态。",
                            tool_calls=[
                                {
                                    "name": "lookup_order",
                                    "args": {"order_number": order_number},
                                    "id": "call_lookup_order_1",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    ],
                }

            if decision.intent == "complaint":
                summary = decision.complaint_summary or user_text
                return {
                    "user_question": user_text,
                    "intent": decision.intent,
                    "intent_reason": decision.reason,
                    "selected_tool": "create_complaint_ticket",
                    "tool_args": {"summary": summary},
                    "approval_status": "pending",
                    "final_answer": "",
                    "memory_action": decision.memory_action,
                    "memory_candidate": decision.memory_fact,
                    "memory_category": decision.memory_category,
                    "last_error": "",
                    "messages": [
                        AIMessage(
                            content="我可以先帮你创建投诉工单，但这一步会先等待人工确认。",
                            tool_calls=[
                                {
                                    "name": "create_complaint_ticket",
                                    "args": {"summary": summary},
                                    "id": "call_create_complaint_ticket_1",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    ],
                }

            reason = decision.handoff_reason or user_text
            return {
                "user_question": user_text,
                "intent": decision.intent,
                "intent_reason": decision.reason,
                "selected_tool": "handoff_to_human",
                "tool_args": {"reason": reason},
                "approval_status": "pending",
                "final_answer": "",
                "memory_action": decision.memory_action,
                "memory_candidate": decision.memory_fact,
                "memory_category": decision.memory_category,
                "last_error": "",
                "messages": [
                    AIMessage(
                        content="我可以帮你转人工客服，这一步会先等待人工确认。",
                        tool_calls=[
                            {
                                "name": "handoff_to_human",
                                "args": {"reason": reason},
                                "id": "call_handoff_to_human_1",
                                "type": "tool_call",
                            }
                        ],
                    )
                ],
            }

        if isinstance(last_message, ToolMessage):
            selected_tool = state.get("selected_tool", "")
            tool_result = message_text(last_message)
            if state.get("last_error"):
                draft_answer = f"处理阶段发生异常：{state['last_error']}"
            elif selected_tool == "search_product_knowledge":
                draft_answer = build_product_answer(state.get("knowledge_hits", []))
            elif selected_tool == "lookup_order":
                draft_answer = build_order_answer(state.get("order_number", ""), tool_result)
            elif selected_tool == "create_complaint_ticket":
                draft_answer = build_complaint_answer(
                    state.get("complaint_ticket_id", ""),
                    tool_result,
                )
            elif selected_tool == "handoff_to_human":
                draft_answer = build_handoff_answer(
                    state.get("handoff_ticket_id", ""),
                    tool_result,
                )
            elif selected_tool == "save_customer_memory":
                draft_answer = build_memory_answer(tool_result)
            else:
                draft_answer = f"工具执行完成：{tool_result}"

            final_answer = generate_customer_reply(
                state,
                runtime,
                draft_answer=draft_answer,
            )

            return {
                "tool_result": tool_result,
                "final_answer": final_answer,
                "last_error": "",
                "messages": [AIMessage(content=final_answer)],
            }

        return {"messages": [AIMessage(content="这轮处理已经完成。")]}
    except Exception as exc:
        LOGGER.exception("planner_node failed")
        final_answer = f"规划阶段发生异常：{exc}"
        return {
            "last_error": str(exc),
            "final_answer": final_answer,
            "messages": [AIMessage(content=final_answer)],
        }


def route_after_planner(
    state: CustomerServiceState,
) -> Literal["human_review_node", "tool_node", END]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        if state.get("approval_status") == "pending":
            return "human_review_node"
        return "tool_node"
    return END


def human_review_node(state: CustomerServiceState) -> CustomerServiceState:
    try:
        tool_name = state.get("selected_tool", "")
        tool_args = state.get("tool_args", {})
        review_payload = {
            "kind": "approval_required",
            "tool": tool_name,
            "tool_args": tool_args,
            "intent": state.get("intent", ""),
            "question": state.get("user_question", ""),
            "reason": state.get("intent_reason", ""),
            "message": "是否允许执行这次关键客服操作？请返回 approved=true/false。",
        }

        LOGGER.info("Interrupting for human review tool=%s tool_args=%s", tool_name, tool_args)
        human_decision = interrupt(review_payload)
        approved, reviewer_note = parse_approval(human_decision)

        if approved:
            return {"approval_status": "approved", "last_error": ""}

        final_answer = "人工未批准，本次关键操作已取消。"
        return {
            "approval_status": "rejected",
            "final_answer": final_answer,
            "last_error": reviewer_note,
            "messages": [AIMessage(content=final_answer)],
        }
    except GraphInterrupt:
        raise
    except Exception as exc:
        LOGGER.exception("human_review_node failed")
        final_answer = f"人工审核阶段发生异常：{exc}"
        return {
            "approval_status": "rejected",
            "final_answer": final_answer,
            "last_error": str(exc),
            "messages": [AIMessage(content=final_answer)],
        }


def route_after_review(state: CustomerServiceState) -> Literal["tool_node", END]:
    if state.get("approval_status") == "approved":
        return "tool_node"
    return END


def extract_tool_call_id(state: CustomerServiceState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return str(last_message.tool_calls[0].get("id", "tool_call_unknown"))
    return "tool_call_unknown"


def tool_node(
    state: CustomerServiceState,
    runtime: Runtime[AgentContext],
) -> CustomerServiceState:
    selected_tool = state.get("selected_tool", "")
    tool_args = state.get("tool_args", {})
    tool_call_id = extract_tool_call_id(state)

    try:
        if selected_tool == "search_product_knowledge":
            query = tool_args.get("query", "")
            attempt = maybe_raise_transient_service_error(
                f"kb:{query}",
                "重试演示" in query,
            )
            hits = retrieve_product_documents(query)
            LOGGER.info(
                "tool_node search_product_knowledge query=%s hit_count=%s attempt=%s",
                query,
                len(hits),
                attempt,
            )
            return {
                "knowledge_hits": hits,
                "tool_result": json.dumps(hits, ensure_ascii=False),
                "retry_count": attempt,
                "last_error": "",
                "messages": [
                    ToolMessage(
                        content=json.dumps(hits, ensure_ascii=False),
                        tool_call_id=tool_call_id,
                    )
                ],
            }

        if selected_tool == "lookup_order":
            order_number = tool_args.get("order_number", "").upper()
            attempt = maybe_raise_transient_service_error(
                f"order:{order_number}",
                order_number.endswith("9999"),
            )
            order_info = ORDER_DB.get(order_number, {})
            LOGGER.info(
                "tool_node lookup_order order_number=%s found=%s attempt=%s",
                order_number,
                bool(order_info),
                attempt,
            )
            return {
                "order_number": order_number,
                "tool_result": json.dumps(order_info, ensure_ascii=False),
                "retry_count": attempt,
                "last_error": "",
                "messages": [
                    ToolMessage(
                        content=json.dumps(order_info, ensure_ascii=False),
                        tool_call_id=tool_call_id,
                    )
                ],
            }

        if selected_tool == "create_complaint_ticket":
            summary = tool_args.get("summary", "").strip()
            ticket_id = generate_ticket_id("CS")
            if runtime.store:
                runtime.store.put(
                    COMPLAINT_NAMESPACE,
                    ticket_id,
                    {
                        "ticket_id": ticket_id,
                        "user_id": runtime.context.user_id,
                        "summary": summary,
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "status": "open",
                    },
                )
                history_item = runtime.store.get(CASE_HISTORY_NAMESPACE, runtime.context.user_id)
                history = [] if history_item is None else list(history_item.value.get("tickets", []))
                history.append(ticket_id)
                runtime.store.put(
                    CASE_HISTORY_NAMESPACE,
                    runtime.context.user_id,
                    {"tickets": history},
                )
            LOGGER.info(
                "tool_node created complaint ticket user_id=%s ticket_id=%s",
                runtime.context.user_id,
                ticket_id,
            )
            return {
                "complaint_ticket_id": ticket_id,
                "tool_result": summary,
                "last_error": "",
                "messages": [
                    ToolMessage(
                        content=summary,
                        tool_call_id=tool_call_id,
                    )
                ],
            }

        if selected_tool == "handoff_to_human":
            reason = tool_args.get("reason", "").strip()
            ticket_id = generate_ticket_id("HUMAN")
            if runtime.store:
                runtime.store.put(
                    HANDOFF_NAMESPACE,
                    ticket_id,
                    {
                        "ticket_id": ticket_id,
                        "user_id": runtime.context.user_id,
                        "reason": reason,
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "status": "queued",
                    },
                )
            LOGGER.info(
                "tool_node created human handoff user_id=%s ticket_id=%s",
                runtime.context.user_id,
                ticket_id,
            )
            return {
                "handoff_ticket_id": ticket_id,
                "tool_result": reason,
                "last_error": "",
                "messages": [
                    ToolMessage(
                        content=reason,
                        tool_call_id=tool_call_id,
                    )
                ],
            }

        if selected_tool == "save_customer_memory":
            fact = tool_args.get("fact", "").strip()
            facts: list[str] = []
            if runtime.store:
                item = runtime.store.get(PROFILE_NAMESPACE, runtime.context.user_id)
                if item:
                    facts = list(item.value.get("facts", []))
                if fact and fact not in facts:
                    facts.append(fact)
                runtime.store.put(PROFILE_NAMESPACE, runtime.context.user_id, {"facts": facts})
            LOGGER.info("tool_node saved customer memory user_id=%s fact=%s", runtime.context.user_id, fact)
            return {
                "tool_result": fact,
                "memory_facts": facts,
                "last_error": "",
                "messages": [
                    ToolMessage(
                        content=fact,
                        tool_call_id=tool_call_id,
                    )
                ],
            }

        error_text = f"未知工具: {selected_tool}"
        LOGGER.error(error_text)
        return {
            "tool_result": "",
            "last_error": error_text,
            "messages": [
                ToolMessage(
                    content=error_text,
                    tool_call_id=tool_call_id,
                    status="error",
                )
            ],
        }
    except TransientServiceError:
        raise
    except Exception as exc:
        LOGGER.exception("tool_node failed selected_tool=%s", selected_tool)
        error_text = f"工具执行异常：{exc}"
        return {
            "tool_result": "",
            "last_error": str(exc),
            "messages": [
                ToolMessage(
                    content=error_text,
                    tool_call_id=tool_call_id,
                    status="error",
                )
            ],
        }


def build_graph():
    graph_builder = StateGraph(
        state_schema=CustomerServiceState,
        context_schema=AgentContext,
    )
    graph_builder.add_node("load_memory_node", load_memory_node)
    graph_builder.add_node("planner_node", planner_node)
    graph_builder.add_node("human_review_node", human_review_node)
    graph_builder.add_node(
        "tool_node",
        tool_node,
        retry_policy=RetryPolicy(
            max_attempts=2,
            initial_interval=0.1,
            backoff_factor=1.0,
            jitter=False,
            retry_on=(TransientServiceError,),
        ),
    )

    graph_builder.add_edge(START, "load_memory_node")
    graph_builder.add_edge("load_memory_node", "planner_node")
    graph_builder.add_conditional_edges(
        "planner_node",
        route_after_planner,
        {
            "human_review_node": "human_review_node",
            "tool_node": "tool_node",
            END: END,
        },
    )
    graph_builder.add_conditional_edges(
        "human_review_node",
        route_after_review,
        {
            "tool_node": "tool_node",
            END: END,
        },
    )
    graph_builder.add_edge("tool_node", "planner_node")

    return graph_builder.compile(checkpointer=CHECKPOINTER, store=STORE)


def snapshot_to_response(snapshot: Any, events: list[dict[str, Any]]) -> dict[str, Any]:
    values = snapshot.values or {}
    raw_messages = values.get("messages", [])
    messages: list[BaseMessage] = raw_messages if isinstance(raw_messages, list) else []
    interrupts = [
        to_jsonable(getattr(item, "value", item))
        for item in getattr(snapshot, "interrupts", ())
    ]
    reply = latest_assistant_reply(messages) or values.get("final_answer", "")
    if interrupts and not reply:
        reply = "这一步需要人工确认后才能继续。"

    return {
        "status": "needs_review" if interrupts else "completed",
        "reply": reply,
        "interrupts": interrupts,
        "messages": [serialize_message(message) for message in messages],
        "state": to_jsonable(values),
        "events": events,
    }


def stream_graph(
    graph,
    payload: Any,
    *,
    thread_id: str,
    user_id: str,
    emit_event: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    config = {"configurable": {"thread_id": thread_id}}
    events: list[dict[str, Any]] = []

    def publish_event(event: dict[str, Any], *, persist: bool = True) -> None:
        json_event = to_jsonable(event)
        if persist:
            events.append(json_event)
        if emit_event is not None:
            emit_event(json_event)

    context = AgentContext(user_id=user_id, stream_writer=publish_event)

    LOGGER.info(
        "Starting graph stream thread_id=%s user_id=%s payload_type=%s",
        thread_id,
        user_id,
        type(payload).__name__,
    )
    try:
        for chunk in graph.stream(
            payload,
            config=config,
            context=context,
            stream_mode="updates",
        ):
            json_chunk = to_jsonable(chunk)
            LOGGER.info("Stream chunk thread_id=%s chunk=%s", thread_id, json_chunk)
            publish_event({"type": "update", "chunk": json_chunk})
        snapshot = graph.get_state(config)
        result = snapshot_to_response(snapshot, events)
        if emit_event is not None:
            emit_event({"type": "final", "data": result})
        return result
    except Exception as exc:
        if emit_event is not None:
            emit_event({"type": "error", "error": str(exc)})
        LOGGER.exception("run_stream failed thread_id=%s", thread_id)
        raise


def run_stream(
    graph,
    *,
    thread_id: str,
    user_id: str,
    message: str,
) -> dict[str, Any]:
    return stream_customer_turn(
        graph,
        thread_id=thread_id,
        user_id=user_id,
        message=message,
    )


def stream_customer_turn(
    graph,
    *,
    thread_id: str,
    user_id: str,
    message: str,
    emit_event: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    payload = {"messages": [HumanMessage(content=message)]}
    return stream_graph(
        graph,
        payload,
        thread_id=thread_id,
        user_id=user_id,
        emit_event=emit_event,
    )


def run_customer_turn(
    graph,
    *,
    thread_id: str,
    user_id: str,
    message: str,
) -> dict[str, Any]:
    return stream_customer_turn(
        graph,
        thread_id=thread_id,
        user_id=user_id,
        message=message,
    )


def stream_resume_turn(
    graph,
    *,
    thread_id: str,
    user_id: str,
    approved: bool,
    reviewer_note: str = "",
    emit_event: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    payload = Command(
        resume={
            "approved": approved,
            "reviewer_note": reviewer_note,
        }
    )
    return stream_graph(
        graph,
        payload,
        thread_id=thread_id,
        user_id=user_id,
        emit_event=emit_event,
    )


def resume_customer_turn(
    graph,
    *,
    thread_id: str,
    user_id: str,
    approved: bool,
    reviewer_note: str = "",
) -> dict[str, Any]:
    return stream_resume_turn(
        graph,
        thread_id=thread_id,
        user_id=user_id,
        approved=approved,
        reviewer_note=reviewer_note,
    )


def print_result(label: str, result: dict[str, Any]) -> None:
    print("=" * 80)
    print(label)
    print(f"status: {result['status']}")
    print(f"reply: {result['reply']}")
    if result["interrupts"]:
        print("interrupts:")
        print(json.dumps(result["interrupts"], ensure_ascii=False, indent=2))
    print("events:")
    print(json.dumps(result["events"], ensure_ascii=False, indent=2))
    print()


def run_demo(graph) -> None:
    print_result(
        "Scenario 1: Smalltalk",
        run_customer_turn(
            graph,
            thread_id="demo-smalltalk",
            user_id="user-wilson",
            message="你好，今天辛苦了",
        ),
    )

    print_result(
        "Scenario 2: Product consultation",
        run_customer_turn(
            graph,
            thread_id="demo-product",
            user_id="user-wilson",
            message="AeroBuds X1 的续航和保修是怎样的？",
        ),
    )

    print_result(
        "Scenario 3: Order query",
        run_customer_turn(
            graph,
            thread_id="demo-order",
            user_id="user-wilson",
            message="我的订单 SO-20260318-1001 现在到哪了？",
        ),
    )

    complaint_result = run_customer_turn(
        graph,
        thread_id="demo-complaint",
        user_id="user-wilson",
        message="我要投诉，昨天收到的 HomeHub Mini 外壳有裂痕。",
    )
    print_result("Scenario 4: Complaint requires review", complaint_result)

    print_result(
        "Scenario 4: Complaint approved and resumed",
        resume_customer_turn(
            graph,
            thread_id="demo-complaint",
            user_id="user-wilson",
            approved=True,
            reviewer_note="允许创建投诉工单。",
        ),
    )

    memory_result = run_customer_turn(
        graph,
        thread_id="demo-memory-save",
        user_id="user-wilson",
        message="以后请默认用中文回复。",
    )
    print_result("Scenario 5: Memory write requires review", memory_result)

    print_result(
        "Scenario 5: Memory write approved and resumed",
        resume_customer_turn(
            graph,
            thread_id="demo-memory-save",
            user_id="user-wilson",
            approved=True,
            reviewer_note="允许写入长期偏好。",
        ),
    )

    print_result(
        "Scenario 6: Cross-session memory recall",
        run_customer_turn(
            graph,
            thread_id="demo-memory-recall",
            user_id="user-wilson",
            message="你还记得我的偏好吗？",
        ),
    )

    handoff_result = run_customer_turn(
        graph,
        thread_id="demo-handoff",
        user_id="user-wilson",
        message="这个问题我想转人工客服处理。",
    )
    print_result("Scenario 7: Human handoff requires review", handoff_result)

    print_result(
        "Scenario 7: Human handoff approved and resumed",
        resume_customer_turn(
            graph,
            thread_id="demo-handoff",
            user_id="user-wilson",
            approved=True,
            reviewer_note="允许转人工。",
        ),
    )


if __name__ == "__main__":
    LOGGER.info("advanced_qa_agent customer-service demo started")
    graph = build_graph()

    if len(sys.argv) > 1:
        result = run_customer_turn(
            graph,
            thread_id="cli-session",
            user_id="user-cli",
            message=" ".join(sys.argv[1:]),
        )
        print_result("CLI Scenario", result)
    else:
        run_demo(graph)

    LOGGER.info("advanced_qa_agent customer-service demo finished")
