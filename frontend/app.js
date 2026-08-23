import { createSseParser, runSseParserSelfTests } from "/assets/sse-parser.js";

const STORAGE_USER_ID = "ai-agent-learning.user-id";
const STORAGE_THREAD_ID = "ai-agent-learning.thread-id";

const elements = {
  userId: document.querySelector("#user-id"),
  threadId: document.querySelector("#thread-id"),
  newSession: document.querySelector("#new-session"),
  messageList: document.querySelector("#message-list"),
  form: document.querySelector("#message-form"),
  input: document.querySelector("#message-input"),
  send: document.querySelector("#send-button"),
  stop: document.querySelector("#stop-button"),
  statusText: document.querySelector("#status-text"),
  statusDot: document.querySelector("#status-dot")
};

const state = {
  threadId: loadThreadId(),
  busy: false,
  pendingApproval: false,
  controller: null,
  assistantMessage: null,
  progressMessage: null,
  receivedToken: false,
  terminalReceived: false
};

function generateThreadId() {
  if (globalThis.crypto?.randomUUID) {
    return `web_${globalThis.crypto.randomUUID()}`;
  }
  if (globalThis.crypto?.getRandomValues) {
    const random = new Uint32Array(4);
    globalThis.crypto.getRandomValues(random);
    return `web_${Date.now()}_${Array.from(random, (value) => value.toString(16)).join("")}`;
  }
  return `web_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function loadThreadId() {
  return localStorage.getItem(STORAGE_THREAD_ID) || generateThreadId();
}

function currentUserId() {
  return elements.userId.value.trim();
}

function setStatus(text, kind = "idle") {
  elements.statusText.textContent = text;
  elements.statusDot.dataset.kind = kind;
}

function updateControls() {
  elements.send.disabled = state.busy || state.pendingApproval;
  elements.stop.disabled = state.controller === null;
  elements.input.disabled = state.busy || state.pendingApproval;
  elements.userId.disabled = state.busy || state.pendingApproval;
  elements.newSession.disabled = state.busy;
}

function setBusy(busy) {
  state.busy = busy;
  updateControls();
}

function scrollToLatest() {
  elements.messageList.scrollTop = elements.messageList.scrollHeight;
}

function addMessage(role, label, text) {
  const article = document.createElement("article");
  article.className = `message message-${role}`;

  const heading = document.createElement("div");
  heading.className = "message-label";
  heading.textContent = label;

  const body = document.createElement("p");
  body.className = "message-content";
  body.textContent = text;

  article.append(heading, body);
  elements.messageList.append(article);
  scrollToLatest();
  return { article, body };
}

function addSystemMessage(text) {
  return addMessage("progress", "系统", text);
}

function showError(text) {
  addMessage("error", "错误", text);
  setStatus("发生错误", "error");
}

function updateProgress(data) {
  const text = typeof data?.message === "string" ? data.message : "Agent 正在执行";
  if (!state.progressMessage) {
    state.progressMessage = addMessage("progress", "执行进度", text);
  } else {
    state.progressMessage.body.textContent = text;
  }
  setStatus(text, "working");
  scrollToLatest();
}

function appendToken(content) {
  if (!state.assistantMessage || typeof content !== "string") {
    return;
  }
  if (!state.receivedToken) {
    state.assistantMessage.body.textContent = "";
    state.receivedToken = true;
  }
  // Deliberately use textContent: model output is never interpreted as HTML.
  state.assistantMessage.body.textContent += content;
  scrollToLatest();
}

function renderSources(message, sources) {
  if (!message?.article || !Array.isArray(sources) || sources.length === 0) {
    return;
  }
  message.article.querySelector(".message-sources")?.remove();
  const section = document.createElement("section");
  section.className = "message-sources";
  const title = document.createElement("strong");
  title.textContent = "来源";
  const list = document.createElement("ul");
  for (const source of sources) {
    if (!source || typeof source !== "object") {
      continue;
    }
    const item = document.createElement("li");
    const fileName = typeof source.source === "string" ? source.source : "未知文件";
    const page = Number.isInteger(source.page) ? `，第${source.page}页` : "";
    const chunk = typeof source.chunk_id === "string" ? `，${source.chunk_id}` : "";
    item.textContent = `${fileName}${page}${chunk}`;
    list.append(item);
  }
  if (list.childElementCount === 0) {
    return;
  }
  section.append(title, list);
  message.article.append(section);
}

function completeAnswer(data) {
  const answer = typeof data?.answer === "string" ? data.answer : "";
  if (state.assistantMessage && answer) {
    state.assistantMessage.body.textContent = answer;
  }
  renderSources(state.assistantMessage, data?.sources);
  state.terminalReceived = true;
  setStatus("回答完成", "success");
  scrollToLatest();
}

function safeJson(value) {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return "无法显示结构化参数";
  }
}

function renderApproval(interrupts) {
  const interruptInfo = Array.isArray(interrupts) ? interrupts[0] : null;
  const payload = interruptInfo?.payload;
  if (!payload || typeof payload !== "object") {
    showError("服务器返回了无法识别的审批信息，请新建会话后重试。");
    return;
  }

  state.pendingApproval = true;
  updateControls();

  const card = document.createElement("section");
  card.className = "approval-card";
  card.dataset.submitted = "false";

  const title = document.createElement("h2");
  title.textContent = "需要人工确认";

  const message = document.createElement("p");
  message.textContent =
    typeof payload.message === "string" ? payload.message : "是否继续执行此操作？";

  const operation = document.createElement("p");
  operation.className = "approval-operation";
  const toolName = typeof payload.tool_name === "string" ? payload.tool_name : "工具操作";
  operation.textContent = `操作：${toolName}`;

  const details = document.createElement("pre");
  details.textContent = safeJson(payload.arguments ?? {});

  const actions = document.createElement("div");
  actions.className = "approval-actions";

  const isFailureReview = payload.action === "tool_failure_review";
  const approve = document.createElement("button");
  approve.type = "button";
  approve.className = "primary-button";
  approve.textContent = isFailureReview ? "重试" : "批准";

  const reject = document.createElement("button");
  reject.type = "button";
  reject.className = "secondary-button";
  reject.textContent = isFailureReview ? "取消" : "拒绝";

  const buttons = [approve, reject];
  approve.addEventListener("click", () =>
    submitApproval(isFailureReview ? "retry" : "approve", card, buttons)
  );
  reject.addEventListener("click", () =>
    submitApproval(isFailureReview ? "cancel" : "reject", card, buttons)
  );

  actions.append(approve, reject);
  card.append(title, message, operation, details, actions);
  elements.messageList.append(card);
  scrollToLatest();
}

async function submitApproval(decision, card, buttons) {
  if (card.dataset.submitted === "true") {
    return;
  }
  card.dataset.submitted = "true";
  for (const button of buttons) {
    button.disabled = true;
  }

  setBusy(true);
  setStatus("正在恢复暂停任务", "working");
  try {
    const response = await fetch("/api/v1/agent/resume", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-User-ID": currentUserId()
      },
      body: JSON.stringify({
        thread_id: state.threadId,
        decision,
        reason: decision === "reject" || decision === "cancel" ? "用户在页面中拒绝" : null
      })
    });

    const result = await readJsonResponse(response);
    state.pendingApproval = false;
    if (result.status === "completed") {
      if (state.assistantMessage) {
        state.assistantMessage.body.textContent = result.answer || "操作已经完成。";
      }
      renderSources(state.assistantMessage, result.sources);
      setStatus("审批处理完成", "success");
    } else if (result.status === "interrupted") {
      renderApproval(result.interrupts);
    } else {
      throw new Error("恢复接口返回了未知状态");
    }
  } catch (error) {
    showError(humanError(error));
    addSystemMessage("本次审批按钮已锁定，避免不确定状态下重复执行。请检查状态或新建会话。");
  } finally {
    setBusy(false);
    updateControls();
  }
}

class HttpError extends Error {
  constructor(status, payload) {
    super(`HTTP ${status}`);
    this.status = status;
    this.payload = payload;
  }
}

async function readJsonResponse(response) {
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    throw new HttpError(response.status, payload);
  }
  return payload;
}

function humanError(error) {
  if (error instanceof HttpError) {
    if (error.status === 401 || error.status === 403) {
      return "当前用户无权访问这个会话，请检查用户 ID 或新建会话。";
    }
    if (error.status === 409) {
      return "会话状态发生冲突，可能是旧线程或尚有待审批任务。建议处理审批或新建会话。";
    }
    if (error.status === 422) {
      return "请求参数不符合接口要求，请检查用户 ID、消息和 thread_id。";
    }
    if (error.status >= 500) {
      return "Agent 服务暂时不可用，请稍后重试。";
    }
    return `请求失败（HTTP ${error.status}）。`;
  }
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return "发生未知错误。";
}

function humanStreamError(data) {
  switch (data?.code) {
    case "thread_forbidden":
      return "当前用户无权访问这个会话。请检查用户 ID，或点击“新建会话”。";
    case "legacy_thread_conflict":
      return "这是没有网页用户归属的旧线程，不能自动认领。请点击“新建会话”。";
    case "pending_interrupt":
      return "当前会话仍有待审批操作。请先完成审批，或点击“新建会话”。";
    case "request_rejected":
      return "本次请求被安全策略拒绝，请检查会话状态。";
    default:
      return typeof data?.message === "string" ? data.message : "Agent 流执行失败，请重试。";
  }
}

function handlePublicEvent(item) {
  const data = item.data ?? {};
  switch (item.event) {
    case "started":
      setStatus("任务已经开始", "working");
      break;
    case "progress":
      updateProgress(data);
      break;
    case "token":
      appendToken(data.content);
      break;
    case "completed":
      completeAnswer(data);
      break;
    case "interrupted":
      state.terminalReceived = true;
      if (state.assistantMessage) {
        state.assistantMessage.body.textContent = "操作已暂停，等待你的选择。";
      }
      setStatus("等待人工确认", "paused");
      renderApproval(data.interrupts);
      break;
    case "error":
      state.terminalReceived = true;
      if (state.assistantMessage && !state.receivedToken) {
        state.assistantMessage.body.textContent = "本次请求未能完成。";
      }
      showError(humanStreamError(data));
      break;
    default:
      console.debug("忽略未知SSE事件", item.event);
  }
}

async function readAgentStream(message, signal) {
  const response = await fetch("/api/v1/agent/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-User-ID": currentUserId()
    },
    body: JSON.stringify({ message, thread_id: state.threadId }),
    signal
  });

  if (!response.ok) {
    await readJsonResponse(response);
  }
  if (!response.body) {
    throw new Error("浏览器没有提供可读取的流响应。");
  }

  const parserErrors = [];
  const parser = createSseParser({
    onEvent: handlePublicEvent,
    onInvalidJson: (error) => {
      parserErrors.push(error);
      console.error("SSE事件JSON解析失败", error.event);
      addMessage("error", "流解析警告", "收到一条格式异常的流事件，已安全忽略并继续读取。");
    }
  });
  const reader = response.body.getReader();

  try {
    while (!state.terminalReceived) {
      const { value, done } = await reader.read();
      if (done) {
        parser.finish();
        break;
      }
      parser.push(value);
    }

    if (state.terminalReceived) {
      await reader.cancel();
    } else if (parserErrors.length > 0) {
      throw new Error("收到无法解析的流事件，连接已停止。");
    } else {
      throw new Error("网络连接已经结束，但没有收到 completed、interrupted 或 error。");
    }
  } finally {
    reader.releaseLock();
  }
}

async function sendMessage(message) {
  state.assistantMessage = addMessage("agent", "Agent", "正在处理……");
  state.progressMessage = null;
  state.receivedToken = false;
  state.terminalReceived = false;
  state.controller = new AbortController();
  setBusy(true);
  setStatus("正在连接 Agent", "working");

  try {
    await readAgentStream(message, state.controller.signal);
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      if (state.assistantMessage && !state.receivedToken) {
        state.assistantMessage.body.textContent = "浏览器已停止接收本次回答。";
      }
      addSystemMessage("已停止浏览器读取；服务器可能已完成部分执行，不能视为业务撤销。建议新建会话再继续。");
      setStatus("已停止读取", "paused");
    } else {
      showError(humanError(error));
    }
  } finally {
    state.controller = null;
    setBusy(false);
    updateControls();
    elements.input.focus();
  }
}

function startNewSession() {
  state.threadId = generateThreadId();
  localStorage.setItem(STORAGE_THREAD_ID, state.threadId);
  state.pendingApproval = false;
  state.assistantMessage = null;
  state.progressMessage = null;
  elements.threadId.textContent = state.threadId;
  elements.messageList.replaceChildren();
  addSystemMessage("已创建新会话。旧 thread 的 Checkpoint 不会自动进入这里；同一用户仍可检索长期记忆。");
  setStatus("新会话就绪", "idle");
  updateControls();
}

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (state.busy || state.pendingApproval) {
    return;
  }
  const message = elements.input.value.trim();
  const userId = currentUserId();
  if (!message) {
    return;
  }
  if (!userId) {
    showError("开发用户 ID 不能为空。");
    elements.userId.focus();
    return;
  }
  localStorage.setItem(STORAGE_USER_ID, userId);
  addMessage("user", "你", message);
  elements.input.value = "";
  void sendMessage(message);
});

elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.form.requestSubmit();
  }
});

elements.stop.addEventListener("click", () => {
  state.controller?.abort();
});

elements.newSession.addEventListener("click", startNewSession);

elements.userId.addEventListener("change", () => {
  const userId = currentUserId();
  if (!userId) {
    return;
  }
  const previous = localStorage.getItem(STORAGE_USER_ID);
  localStorage.setItem(STORAGE_USER_ID, userId);
  if (previous && previous !== userId) {
    startNewSession();
  }
});

function bootstrap() {
  elements.userId.value = localStorage.getItem(STORAGE_USER_ID) || "moon";
  localStorage.setItem(STORAGE_USER_ID, elements.userId.value);
  localStorage.setItem(STORAGE_THREAD_ID, state.threadId);
  elements.threadId.textContent = state.threadId;
  addSystemMessage("页面已就绪。刷新会保留用户和 thread 标识，但不会自动重新渲染历史消息。");
  updateControls();

  try {
    const passed = runSseParserSelfTests();
    console.info("SSE parser browser self-tests passed", passed);
  } catch (error) {
    console.error(error);
    showError("浏览器SSE解析器自检失败，请查看Console并停止使用当前页面。");
  }
}

window.AiAgentFrontendTests = {
  createSseParser,
  runSseParserSelfTests
};

bootstrap();
