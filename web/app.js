const chatLog = document.querySelector("#chat-log");
const composerForm = document.querySelector("#composer-form");
const messageInput = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-btn");
const statusText = document.querySelector("#status-text");
const userIdInput = document.querySelector("#user-id-input");
const threadIdInput = document.querySelector("#thread-id-input");
const newThreadButton = document.querySelector("#new-thread-btn");
const reviewPanel = document.querySelector("#review-panel");
const reviewSummary = document.querySelector("#review-summary");
const reviewNoteInput = document.querySelector("#review-note-input");
const approveButton = document.querySelector("#approve-btn");
const rejectButton = document.querySelector("#reject-btn");
const debugIntent = document.querySelector("#debug-intent");
const debugTool = document.querySelector("#debug-tool");
const debugApproval = document.querySelector("#debug-approval");
const debugEvents = document.querySelector("#debug-events");
const messageTemplate = document.querySelector("#message-template");

const STORAGE_KEYS = {
  userId: "support-studio-user-id",
  threadId: "support-studio-thread-id",
};

const state = {
  pendingReview: null,
  liveEvents: [],
};

function createId(prefix) {
  const suffix = Math.random().toString(36).slice(2, 10);
  return `${prefix}-${suffix}`;
}

function ensureSessionFields() {
  const savedUserId = localStorage.getItem(STORAGE_KEYS.userId) || "user-wilson";
  const savedThreadId = localStorage.getItem(STORAGE_KEYS.threadId) || createId("thread");
  userIdInput.value = savedUserId;
  threadIdInput.value = savedThreadId;
}

function persistSessionFields() {
  localStorage.setItem(STORAGE_KEYS.userId, userIdInput.value.trim());
  localStorage.setItem(STORAGE_KEYS.threadId, threadIdInput.value.trim());
}

function setStatus(text, busy = false) {
  statusText.textContent = text;
  sendButton.disabled = busy;
  approveButton.disabled = busy;
  rejectButton.disabled = busy;
}

function appendMessage(role, text, meta = "") {
  const fragment = messageTemplate.content.cloneNode(true);
  const article = fragment.querySelector(".message");
  const metaNode = fragment.querySelector(".message-meta");
  const bubbleNode = fragment.querySelector(".message-bubble");

  article.classList.add(`message-${role}`);
  metaNode.textContent = meta || (role === "user" ? "你" : "客服 Agent");
  bubbleNode.textContent = text;

  chatLog.appendChild(fragment);
  chatLog.scrollTop = chatLog.scrollHeight;
  return chatLog.lastElementChild;
}

function updateMessage(node, text, meta = null) {
  if (!node) {
    return;
  }

  const metaNode = node.querySelector(".message-meta");
  const bubbleNode = node.querySelector(".message-bubble");
  if (meta !== null) {
    metaNode.textContent = meta;
  }
  bubbleNode.textContent = text;
  chatLog.scrollTop = chatLog.scrollHeight;
}

function renderReviewPanel(interrupt) {
  if (!interrupt) {
    reviewPanel.classList.add("hidden");
    reviewSummary.innerHTML = "";
    state.pendingReview = null;
    return;
  }

  state.pendingReview = interrupt;
  reviewSummary.innerHTML = `
    <p><strong>工具：</strong>${interrupt.tool || "-"}</p>
    <p><strong>意图：</strong>${interrupt.intent || "-"}</p>
    <p><strong>原始问题：</strong>${interrupt.question || "-"}</p>
    <pre>${JSON.stringify(interrupt.tool_args || {}, null, 2)}</pre>
  `;
  reviewPanel.classList.remove("hidden");
}

function updateDebugPanel(result) {
  const snapshot = result.state || {};
  debugIntent.textContent = snapshot.intent || "-";
  debugTool.textContent = snapshot.selected_tool || "-";
  debugApproval.textContent = snapshot.approval_status || "-";
  debugEvents.textContent = JSON.stringify(result.events || state.liveEvents || [], null, 2);
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "请求失败");
  }
  return data;
}

async function postNdjsonStream(url, payload, onEvent) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let message = "请求失败";
    try {
      const data = await response.json();
      message = data.error || message;
    } catch {
      const text = await response.text();
      if (text.trim()) {
        message = text;
      }
    }
    throw new Error(message);
  }

  if (!response.body) {
    throw new Error("当前浏览器不支持流式读取响应。");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }
      onEvent(JSON.parse(trimmed));
    }
  }

  buffer += decoder.decode();
  const tail = buffer.trim();
  if (tail) {
    onEvent(JSON.parse(tail));
  }
}

function summarizeStreamEvent(event) {
  if (event.type !== "update") {
    return event.type === "final" ? "本轮处理完成" : "处理过程中发生异常";
  }

  const chunk = event.chunk || {};
  const nodeName = Object.keys(chunk)[0] || "";
  const labels = {
    load_memory_node: "正在读取客户画像",
    planner_node: "正在规划客服意图",
    tool_node: "正在执行客服工具",
    human_review_node: "正在处理人工确认",
    __interrupt__: "已暂停，等待人工确认",
  };
  return labels[nodeName] || `正在执行 ${nodeName}`;
}

function pushLiveEvent(event) {
  state.liveEvents.push(event);
  debugEvents.textContent = JSON.stringify(state.liveEvents, null, 2);
}

async function sendMessage(message) {
  const userId = userIdInput.value.trim();
  const threadId = threadIdInput.value.trim();

  if (!message) {
    return;
  }

  persistSessionFields();
  appendMessage("user", message, "当前用户");
  messageInput.value = "";
  state.liveEvents = [];
  setStatus("客服处理中...", true);
  const assistantNode = appendMessage("assistant", "正在连接 LangGraph 工作流...", "流式返回");
  let tokenBuffer = "";
  let hasTokenStream = false;

  try {
    await postNdjsonStream("/api/chat/stream", {
      user_id: userId,
      thread_id: threadId,
      message,
    }, (event) => {
      pushLiveEvent(event);

      if (event.type === "token") {
        hasTokenStream = true;
        tokenBuffer += event.delta || "";
        updateMessage(assistantNode, tokenBuffer || " ", "客服 Agent");
        setStatus("正在生成回复...", true);
        return;
      }

      if (!hasTokenStream) {
        updateMessage(assistantNode, summarizeStreamEvent(event), "流式返回");
      }
      setStatus(hasTokenStream ? "正在生成回复..." : summarizeStreamEvent(event), true);

      if (event.type === "final") {
        const result = event.data;
        updateDebugPanel(result);
        updateMessage(
          assistantNode,
          result.reply || tokenBuffer || "本轮没有可展示的回复。",
          "客服 Agent"
        );

        if (result.status === "needs_review") {
          renderReviewPanel(result.interrupts?.[0] || null);
          setStatus("等待人工确认", false);
          return;
        }

        renderReviewPanel(null);
        setStatus("服务就绪", false);
      }

      if (event.type === "error") {
        updateMessage(assistantNode, "流式执行失败，请查看日志或重试。", "系统");
        setStatus("流式执行失败", false);
      }
    });
  } catch (error) {
    updateMessage(assistantNode, `请求失败：${error.message}`, "系统");
    setStatus("请求失败", false);
  }
}

async function submitReview(approved) {
  const userId = userIdInput.value.trim();
  const threadId = threadIdInput.value.trim();
  const reviewerNote = reviewNoteInput.value.trim();

  if (!state.pendingReview) {
    return;
  }

  state.liveEvents = [];
  setStatus("正在继续执行...", true);
  const assistantNode = appendMessage("assistant", "正在继续执行中断节点...", approved ? "审批通过" : "审批拒绝");
  let tokenBuffer = "";
  let hasTokenStream = false;

  try {
    await postNdjsonStream("/api/review/stream", {
      user_id: userId,
      thread_id: threadId,
      approved,
      reviewer_note: reviewerNote,
    }, (event) => {
      pushLiveEvent(event);

      if (event.type === "token") {
        hasTokenStream = true;
        tokenBuffer += event.delta || "";
        updateMessage(assistantNode, tokenBuffer || " ", "客服 Agent");
        setStatus("正在生成回复...", true);
        return;
      }

      if (!hasTokenStream) {
        updateMessage(
          assistantNode,
          summarizeStreamEvent(event),
          approved ? "审批通过" : "审批拒绝"
        );
      }
      setStatus(hasTokenStream ? "正在生成回复..." : summarizeStreamEvent(event), true);

      if (event.type === "final") {
        const result = event.data;
        updateDebugPanel(result);
        updateMessage(
          assistantNode,
          result.reply || tokenBuffer || "操作已继续执行。",
          hasTokenStream ? "客服 Agent" : approved ? "审批通过" : "审批拒绝"
        );
        renderReviewPanel(result.status === "needs_review" ? result.interrupts?.[0] || null : null);
        reviewNoteInput.value = "";
        setStatus(result.status === "needs_review" ? "等待人工确认" : "服务就绪", false);
      }

      if (event.type === "error") {
        updateMessage(assistantNode, "继续执行失败，请查看日志或重试。", "系统");
        setStatus("继续执行失败", false);
      }
    });
  } catch (error) {
    updateMessage(assistantNode, `继续执行失败：${error.message}`, "系统");
    setStatus("继续执行失败", false);
  }
}

composerForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await sendMessage(messageInput.value.trim());
});

document.querySelectorAll(".prompt-chip").forEach((button) => {
  button.addEventListener("click", async () => {
    const prompt = button.dataset.prompt || "";
    messageInput.value = prompt;
    await sendMessage(prompt);
  });
});

approveButton.addEventListener("click", async () => {
  await submitReview(true);
});

rejectButton.addEventListener("click", async () => {
  await submitReview(false);
});

newThreadButton.addEventListener("click", () => {
  threadIdInput.value = createId("thread");
  persistSessionFields();
  renderReviewPanel(null);
  chatLog.innerHTML = "";
  state.liveEvents = [];
  debugEvents.textContent = "等待请求...";
  debugIntent.textContent = "-";
  debugTool.textContent = "-";
  debugApproval.textContent = "-";
  setStatus("已创建新会话", false);
});

userIdInput.addEventListener("change", persistSessionFields);
threadIdInput.addEventListener("change", persistSessionFields);

ensureSessionFields();
appendMessage(
  "assistant",
  "欢迎来到 Support Studio。你可以直接测试产品咨询、订单查询、投诉处理、转人工和跨会话记忆。",
  "系统引导"
);
