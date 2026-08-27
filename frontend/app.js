import { createSseParser, runSseParserSelfTests } from "/assets/sse-parser.js";
import {
  ApiError,
  apiErrorMessage,
  apiFetch,
  clearClientCsrfCookie,
  configureCsrfFromResponse,
  readJsonResponse,
  requestJson,
  runApiClientSelfTests,
  setAuthStatusHandler
} from "/assets/api.js";
import { runAuthStateSelfTests, validRegistration } from "/assets/auth-state.js";
import { initResearchFlow, runResearchFlowSelfTests } from "/assets/researchflow.js";

const STORAGE_THREAD_ID = "ai-agent-learning.thread-id";
const STORAGE_KNOWLEDGE_BASE_ID = "ai-agent-learning.knowledge-base-id";
const STORAGE_PROJECT_ID = "researchflow.selected-project";

const elements = {
  initializingView: document.querySelector("#initializing-view"),
  authView: document.querySelector("#auth-view"),
  workspaceRoot: document.querySelector("#workspace-root"),
  showLogin: document.querySelector("#show-login"),
  showRegister: document.querySelector("#show-register"),
  loginForm: document.querySelector("#login-form"),
  loginName: document.querySelector("#login-name"),
  loginPassword: document.querySelector("#login-password"),
  loginButton: document.querySelector("#login-button"),
  registerForm: document.querySelector("#register-form"),
  registerUsername: document.querySelector("#register-username"),
  registerEmail: document.querySelector("#register-email"),
  registerPassword: document.querySelector("#register-password"),
  registerPasswordConfirmation: document.querySelector("#register-password-confirmation"),
  registerButton: document.querySelector("#register-button"),
  authMessage: document.querySelector("#auth-message"),
  currentUsername: document.querySelector("#current-username"),
  logoutButton: document.querySelector("#logout-button"),
  threadId: document.querySelector("#thread-id"),
  newSession: document.querySelector("#new-session"),
  messageList: document.querySelector("#message-list"),
  form: document.querySelector("#message-form"),
  input: document.querySelector("#message-input"),
  send: document.querySelector("#send-button"),
  stop: document.querySelector("#stop-button"),
  statusText: document.querySelector("#status-text"),
  statusDot: document.querySelector("#status-dot"),
  navChat: document.querySelector("#nav-chat"),
  navKnowledge: document.querySelector("#nav-knowledge"),
  navOverview: document.querySelector("#nav-overview"),
  navTasks: document.querySelector("#nav-tasks"),
  navArtifacts: document.querySelector("#nav-artifacts"),
  chatView: document.querySelector("#chat-view"),
  knowledgeView: document.querySelector("#knowledge-view"),
  overviewView: document.querySelector("#overview-view"),
  tasksView: document.querySelector("#tasks-view"),
  artifactsView: document.querySelector("#artifacts-view"),
  knowledgeSelector: document.querySelector("#knowledge-selector"),
  showCreateKb: document.querySelector("#show-create-kb"),
  cancelCreateKb: document.querySelector("#cancel-create-kb"),
  createKbForm: document.querySelector("#create-kb-form"),
  kbName: document.querySelector("#kb-name"),
  kbDescription: document.querySelector("#kb-description"),
  knowledgeBaseList: document.querySelector("#knowledge-base-list"),
  knowledgeEmpty: document.querySelector("#knowledge-empty"),
  knowledgeContent: document.querySelector("#knowledge-content"),
  selectedKbName: document.querySelector("#selected-kb-name"),
  selectedKbDescription: document.querySelector("#selected-kb-description"),
  selectedKbStats: document.querySelector("#selected-kb-stats"),
  deleteKb: document.querySelector("#delete-kb"),
  uploadZone: document.querySelector("#upload-zone"),
  documentFiles: document.querySelector("#document-files"),
  uploadStatus: document.querySelector("#upload-status"),
  documentTableBody: document.querySelector("#document-table-body")
};

let researchFlow = null;

const state = {
  authUser: null,
  authBusy: false,
  authGeneration: 0,
  threadId: loadThreadId(),
  busy: false,
  pendingApproval: false,
  controller: null,
  assistantMessage: null,
  progressMessage: null,
  receivedToken: false,
  terminalReceived: false,
  knowledgeBases: [],
  selectedKnowledgeBaseId: localStorage.getItem(STORAGE_KNOWLEDGE_BASE_ID) || "",
  knowledgeDocuments: [],
  knowledgeBusy: false
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

function selectedKnowledgeBase() {
  return state.knowledgeBases.find(
    (item) => item.knowledge_base_id === state.selectedKnowledgeBaseId
  ) || null;
}

function setView(name) {
  const views = {
    overview: elements.overviewView,
    knowledge: elements.knowledgeView,
    tasks: elements.tasksView,
    artifacts: elements.artifactsView,
    chat: elements.chatView
  };
  const navItems = {
    overview: elements.navOverview,
    knowledge: elements.navKnowledge,
    tasks: elements.navTasks,
    artifacts: elements.navArtifacts,
    chat: elements.navChat
  };
  for (const [viewName, view] of Object.entries(views)) {
    view.hidden = viewName !== name;
    navItems[viewName].classList.toggle("active", viewName === name);
  }
  if (name === "knowledge" && state.authUser) {
    void loadKnowledgeBases();
  }
  researchFlow?.activate(name);
}

function setSelectedKnowledgeBase(knowledgeBaseId, { loadDocuments = true } = {}) {
  const previous = state.selectedKnowledgeBaseId;
  const exists = state.knowledgeBases.some(
    (item) => item.knowledge_base_id === knowledgeBaseId
  );
  state.selectedKnowledgeBaseId = exists ? knowledgeBaseId : "";
  if (previous !== state.selectedKnowledgeBaseId) {
    state.knowledgeDocuments = [];
  }
  localStorage.setItem(STORAGE_KNOWLEDGE_BASE_ID, state.selectedKnowledgeBaseId);
  elements.knowledgeSelector.value = state.selectedKnowledgeBaseId;
  renderKnowledgeBaseList();
  renderKnowledgeDetail();
  if (loadDocuments && state.selectedKnowledgeBaseId) {
    void loadKnowledgeDocuments();
  }
}

function renderKnowledgeSelector() {
  const selected = state.selectedKnowledgeBaseId;
  elements.knowledgeSelector.replaceChildren();
  const empty = document.createElement("option");
  empty.value = "";
  empty.textContent = "不使用知识库";
  elements.knowledgeSelector.append(empty);
  for (const item of state.knowledgeBases) {
    const option = document.createElement("option");
    option.value = item.knowledge_base_id;
    option.textContent = item.name;
    elements.knowledgeSelector.append(option);
  }
  elements.knowledgeSelector.value = state.knowledgeBases.some(
    (item) => item.knowledge_base_id === selected
  ) ? selected : "";
}

function renderKnowledgeBaseList() {
  elements.knowledgeBaseList.replaceChildren();
  if (state.knowledgeBases.length === 0) {
    const empty = document.createElement("p");
    empty.className = "subtitle";
    empty.textContent = "当前用户还没有知识库。";
    elements.knowledgeBaseList.append(empty);
    return;
  }
  for (const item of state.knowledgeBases) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "knowledge-base-item";
    button.classList.toggle(
      "active",
      item.knowledge_base_id === state.selectedKnowledgeBaseId
    );
    const name = document.createElement("strong");
    name.textContent = item.name;
    const description = document.createElement("span");
    description.textContent = item.description || item.knowledge_base_id;
    button.append(name, description);
    button.addEventListener("click", () => {
      setSelectedKnowledgeBase(item.knowledge_base_id);
    });
    elements.knowledgeBaseList.append(button);
  }
}

function renderKnowledgeDetail() {
  const current = selectedKnowledgeBase();
  elements.knowledgeEmpty.hidden = current !== null;
  elements.knowledgeContent.hidden = current === null;
  if (!current) {
    elements.documentTableBody.replaceChildren();
    return;
  }
  elements.selectedKbName.textContent = current.name;
  elements.selectedKbDescription.textContent = current.description || "暂无描述";
  const ready = state.knowledgeDocuments.filter((item) => item.status === "ready").length;
  elements.selectedKbStats.textContent = `知识库ID：${current.knowledge_base_id}；文档 ${state.knowledgeDocuments.length}，可检索 ${ready}`;
}

function formatBytes(size) {
  if (!Number.isFinite(size)) return "-";
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "-" : date.toLocaleString();
}

function renderDocuments() {
  elements.documentTableBody.replaceChildren();
  if (state.knowledgeDocuments.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 6;
    cell.textContent = "还没有文档。";
    row.append(cell);
    elements.documentTableBody.append(row);
    renderKnowledgeDetail();
    return;
  }
  const labels = { processing: "处理中", ready: "可检索", failed: "失败" };
  for (const item of state.knowledgeDocuments) {
    const row = document.createElement("tr");
    const filename = document.createElement("td");
    const filenameText = document.createElement("span");
    filenameText.textContent = item.original_filename;
    filename.append(filenameText);
    if (item.error_message) {
      const errorText = document.createElement("small");
      errorText.className = "document-error";
      errorText.textContent = item.error_message;
      filename.append(errorText);
    }
    const status = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = `document-status ${item.status}`;
    badge.textContent = labels[item.status] || item.status;
    status.append(badge);
    const size = document.createElement("td");
    size.textContent = formatBytes(item.size);
    const chunks = document.createElement("td");
    chunks.textContent = String(item.chunk_count);
    const created = document.createElement("td");
    created.textContent = formatDate(item.created_at);
    const action = document.createElement("td");
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "table-delete";
    remove.textContent = "删除";
    remove.addEventListener("click", () => void deleteDocument(item));
    action.append(remove);
    row.append(filename, status, size, chunks, created, action);
    elements.documentTableBody.append(row);
  }
  renderKnowledgeDetail();
}

async function loadKnowledgeBases() {
  if (!state.authUser) return;
  const generation = state.authGeneration;
  try {
    const response = await apiFetch("/api/v1/knowledge-bases");
    const items = await readJsonResponse(response);
    if (!state.authUser || generation !== state.authGeneration) return;
    state.knowledgeBases = Array.isArray(items) ? items : [];
    if (!state.knowledgeBases.some(
      (item) => item.knowledge_base_id === state.selectedKnowledgeBaseId
    )) {
      state.selectedKnowledgeBaseId = "";
      state.knowledgeDocuments = [];
      localStorage.setItem(STORAGE_KNOWLEDGE_BASE_ID, "");
    }
    renderKnowledgeSelector();
    renderKnowledgeBaseList();
    renderKnowledgeDetail();
    if (state.selectedKnowledgeBaseId) await loadKnowledgeDocuments();
  } catch (error) {
    showKnowledgeError(error);
  }
}

async function loadKnowledgeDocuments() {
  const knowledgeBaseId = state.selectedKnowledgeBaseId;
  if (!knowledgeBaseId) return;
  const generation = state.authGeneration;
  try {
    const response = await apiFetch(
      `/api/v1/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/documents`,
      {}
    );
    const items = await readJsonResponse(response);
    if (
      !state.authUser ||
      generation !== state.authGeneration ||
      knowledgeBaseId !== state.selectedKnowledgeBaseId
    ) return;
    state.knowledgeDocuments = Array.isArray(items) ? items : [];
    renderDocuments();
  } catch (error) {
    showKnowledgeError(error);
  }
}

function showKnowledgeError(error) {
  const detail = error instanceof ApiError && typeof error.payload?.detail === "string"
    ? error.payload.detail
    : humanError(error);
  elements.uploadStatus.textContent = detail;
  elements.uploadStatus.dataset.kind = "error";
}

function setStatus(text, kind = "idle") {
  elements.statusText.textContent = text;
  elements.statusDot.dataset.kind = kind;
}

function updateControls() {
  elements.send.disabled = state.busy || state.pendingApproval;
  elements.stop.disabled = state.controller === null;
  elements.input.disabled = state.busy || state.pendingApproval;
  elements.newSession.disabled = state.busy;
  elements.knowledgeSelector.disabled = state.busy || state.pendingApproval;
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
    const response = await apiFetch("/api/v1/agent/resume", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        thread_id: state.threadId,
        decision,
        knowledge_base_id: state.selectedKnowledgeBaseId || null,
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

function humanError(error) {
  if (error instanceof ApiError) {
    if (error.status === 401) return "登录会话已失效，请重新登录。";
    if (error.status === 403) return "安全校验失败，请刷新页面后重试。";
    if (error.status === 409) {
      return "会话状态发生冲突，可能是旧线程或尚有待审批任务。建议处理审批或新建会话。";
    }
    if (error.status === 422) {
      return "请求参数不符合接口要求，请检查消息和thread_id。";
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
      return "当前登录用户无权访问这个会话，请点击“新建会话”。";
    case "legacy_thread_conflict":
      return "这是没有网页用户归属的旧线程，不能自动认领。请点击“新建会话”。";
    case "pending_interrupt":
      return "当前会话仍有待审批操作。请先完成审批，或点击“新建会话”。";
    case "knowledge_base_not_found":
      return "所选知识库不存在或不属于当前用户，请刷新知识库列表后重试。";
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
  const response = await apiFetch("/api/v1/agent/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      message,
      thread_id: state.threadId,
      knowledge_base_id: state.selectedKnowledgeBaseId || null
    }),
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

async function createKnowledgeBase(event) {
  event.preventDefault();
  if (state.knowledgeBusy) return;
  const name = elements.kbName.value.trim();
  if (!name) return;
  state.knowledgeBusy = true;
  try {
    const response = await apiFetch("/api/v1/knowledge-bases", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        description: elements.kbDescription.value.trim()
      })
    });
    const created = await readJsonResponse(response);
    elements.createKbForm.reset();
    elements.createKbForm.hidden = true;
    await loadKnowledgeBases();
    await researchFlow?.refreshKnowledgeBases();
    setSelectedKnowledgeBase(created.knowledge_base_id);
  } catch (error) {
    showKnowledgeError(error);
  } finally {
    state.knowledgeBusy = false;
  }
}

async function uploadFiles(fileList) {
  if (state.knowledgeBusy || !state.selectedKnowledgeBaseId) return;
  const files = Array.from(fileList || []);
  if (files.length === 0) return;
  state.knowledgeBusy = true;
  elements.uploadStatus.dataset.kind = "working";
  elements.uploadStatus.textContent = "正在上传；服务器随后会解析文档、生成向量并建立索引……";
  const data = new FormData();
  for (const file of files) data.append("files", file, file.name);
  try {
    const response = await apiFetch(
      `/api/v1/knowledge-bases/${encodeURIComponent(state.selectedKnowledgeBaseId)}/documents`,
      {
        method: "POST",
        body: data
      }
    );
    const result = await readJsonResponse(response);
    const duplicates = Array.isArray(result.items)
      ? result.items.filter((item) => item.duplicate).length
      : 0;
    const failures = Array.isArray(result.items)
      ? result.items.filter((item) => item.document?.status === "failed").length
      : 0;
    elements.uploadStatus.dataset.kind = failures ? "error" : "success";
    elements.uploadStatus.textContent = failures
      ? `入库完成，但有 ${failures} 个文件解析或索引失败。`
      : `入库完成${duplicates ? `；跳过 ${duplicates} 个重复文档` : ""}。`;
    await loadKnowledgeDocuments();
  } catch (error) {
    showKnowledgeError(error);
  } finally {
    state.knowledgeBusy = false;
    elements.documentFiles.value = "";
  }
}

async function deleteDocument(documentRecord) {
  if (state.knowledgeBusy) return;
  const confirmed = globalThis.confirm(
    `确定删除“${documentRecord.original_filename}”吗？其全部检索片段和源文件都会删除。`
  );
  if (!confirmed) return;
  state.knowledgeBusy = true;
  try {
    const response = await apiFetch(
      `/api/v1/knowledge-bases/${encodeURIComponent(state.selectedKnowledgeBaseId)}/documents/${encodeURIComponent(documentRecord.document_id)}`,
      { method: "DELETE" }
    );
    if (!response.ok) await readJsonResponse(response);
    await loadKnowledgeDocuments();
  } catch (error) {
    showKnowledgeError(error);
  } finally {
    state.knowledgeBusy = false;
  }
}

async function deleteKnowledgeBase() {
  const current = selectedKnowledgeBase();
  if (!current || state.knowledgeBusy) return;
  const confirmed = globalThis.confirm(
    `确定删除知识库“${current.name}”吗？这会删除其中全部文档、向量片段和受控源文件。`
  );
  if (!confirmed) return;
  state.knowledgeBusy = true;
  try {
    const response = await apiFetch(
      `/api/v1/knowledge-bases/${encodeURIComponent(current.knowledge_base_id)}`,
      { method: "DELETE" }
    );
    if (!response.ok) await readJsonResponse(response);
    state.selectedKnowledgeBaseId = "";
    state.knowledgeDocuments = [];
    localStorage.setItem(STORAGE_KNOWLEDGE_BASE_ID, "");
    await loadKnowledgeBases();
    await researchFlow?.refreshKnowledgeBases();
  } catch (error) {
    showKnowledgeError(error);
  } finally {
    state.knowledgeBusy = false;
  }
}

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (state.busy || state.pendingApproval) {
    return;
  }
  const message = elements.input.value.trim();
  if (!message) {
    return;
  }
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

elements.navChat.addEventListener("click", () => setView("chat"));
elements.navKnowledge.addEventListener("click", () => setView("knowledge"));
elements.navOverview.addEventListener("click", () => setView("overview"));
elements.navTasks.addEventListener("click", () => setView("tasks"));
elements.navArtifacts.addEventListener("click", () => setView("artifacts"));
elements.knowledgeSelector.addEventListener("change", () => {
  setSelectedKnowledgeBase(elements.knowledgeSelector.value);
});
elements.showCreateKb.addEventListener("click", () => {
  elements.createKbForm.hidden = false;
  elements.kbName.focus();
});
elements.cancelCreateKb.addEventListener("click", () => {
  elements.createKbForm.reset();
  elements.createKbForm.hidden = true;
});
elements.createKbForm.addEventListener("submit", createKnowledgeBase);
elements.deleteKb.addEventListener("click", () => void deleteKnowledgeBase());
elements.uploadZone.addEventListener("click", () => elements.documentFiles.click());
elements.uploadZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    elements.documentFiles.click();
  }
});
elements.documentFiles.addEventListener("change", () => {
  void uploadFiles(elements.documentFiles.files);
});
elements.documentFiles.addEventListener("click", (event) => {
  event.stopPropagation();
});
for (const eventName of ["dragenter", "dragover"]) {
  elements.uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.uploadZone.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  elements.uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.uploadZone.classList.remove("dragging");
  });
}
elements.uploadZone.addEventListener("drop", (event) => {
  void uploadFiles(event.dataTransfer?.files);
});

function setAuthMode(mode, message = "", kind = "error") {
  const register = mode === "register";
  elements.showLogin.classList.toggle("active", !register);
  elements.showRegister.classList.toggle("active", register);
  elements.loginForm.hidden = register;
  elements.registerForm.hidden = !register;
  elements.authMessage.textContent = message;
  elements.authMessage.dataset.kind = kind;
  if (!state.authBusy) {
    (register ? elements.registerUsername : elements.loginName).focus();
  }
}

function setAuthBusy(value) {
  state.authBusy = value;
  elements.loginButton.disabled = value;
  elements.registerButton.disabled = value;
  elements.showLogin.disabled = value;
  elements.showRegister.disabled = value;
}

function clearBusinessState() {
  state.authGeneration += 1;
  state.controller?.abort();
  state.controller = null;
  state.busy = false;
  state.pendingApproval = false;
  state.assistantMessage = null;
  state.progressMessage = null;
  state.receivedToken = false;
  state.terminalReceived = false;
  state.knowledgeBases = [];
  state.knowledgeDocuments = [];
  state.selectedKnowledgeBaseId = "";
  state.threadId = generateThreadId();
  localStorage.removeItem(STORAGE_THREAD_ID);
  localStorage.removeItem(STORAGE_KNOWLEDGE_BASE_ID);
  localStorage.removeItem(STORAGE_PROJECT_ID);
  clearClientCsrfCookie();
  elements.threadId.textContent = state.threadId;
  elements.messageList.replaceChildren();
  elements.knowledgeBaseList.replaceChildren();
  elements.documentTableBody.replaceChildren();
  renderKnowledgeSelector();
  renderKnowledgeDetail();
  researchFlow?.reset();
  updateControls();
}

function showLoggedOut(message = "请登录后继续。") {
  state.authUser = null;
  elements.currentUsername.textContent = "";
  elements.loginPassword.value = "";
  elements.registerPassword.value = "";
  elements.registerPasswordConfirmation.value = "";
  clearBusinessState();
  elements.initializingView.hidden = true;
  elements.workspaceRoot.hidden = true;
  elements.authView.hidden = false;
  setAuthMode("login", message);
}

async function enterWorkspace(user) {
  state.authUser = user;
  elements.currentUsername.textContent = user.username;
  elements.initializingView.hidden = true;
  elements.authView.hidden = true;
  elements.workspaceRoot.hidden = false;
  localStorage.setItem(STORAGE_THREAD_ID, state.threadId);
  elements.threadId.textContent = state.threadId;
  elements.messageList.replaceChildren();
  addSystemMessage("登录会话已恢复。当前页面只会加载此账户拥有的数据。刷新不会自动重绘历史消息。");
  updateControls();
  setView("overview");
  await Promise.all([loadKnowledgeBases(), researchFlow?.start()]);
}

async function restoreSession() {
  elements.initializingView.hidden = false;
  elements.authView.hidden = true;
  elements.workspaceRoot.hidden = true;
  let user;
  try {
    const response = await apiFetch("/api/v1/auth/me", { handleAuthStatus: false });
    if (response.status === 401) {
      showLoggedOut("请登录或注册账户后使用ResearchFlow。" );
      return false;
    }
    configureCsrfFromResponse(response);
    user = await readJsonResponse(response);
  } catch (error) {
    showLoggedOut(apiErrorMessage(error, "恢复登录状态"));
    return false;
  }
  await enterWorkspace(user);
  return true;
}

function handleAuthStatus(status) {
  if (status === 401) {
    showLoggedOut("登录会话已过期或被撤销，请重新登录。" );
  } else if (status === 403 && state.authUser) {
    setStatus("安全校验失败，请刷新页面", "error");
  }
}

async function submitLogin(event) {
  event.preventDefault();
  if (state.authBusy) return;
  setAuthBusy(true);
  elements.authMessage.textContent = "正在登录……";
  elements.authMessage.dataset.kind = "";
  try {
    await requestJson("/api/v1/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      handleAuthStatus: false,
      body: JSON.stringify({
        login: elements.loginName.value.trim(),
        password: elements.loginPassword.value
      })
    });
    elements.loginPassword.value = "";
    await restoreSession();
  } catch (error) {
    elements.loginPassword.value = "";
    setAuthMode("login", apiErrorMessage(error, "登录"));
  } finally {
    setAuthBusy(false);
  }
}

async function submitRegistration(event) {
  event.preventDefault();
  if (state.authBusy) return;
  const values = {
    username: elements.registerUsername.value.trim(),
    email: elements.registerEmail.value.trim(),
    password: elements.registerPassword.value,
    passwordConfirmation: elements.registerPasswordConfirmation.value
  };
  if (!validRegistration(values)) {
    setAuthMode("register", "请填写有效用户名和邮箱；密码至少8位且两次输入必须一致。" );
    return;
  }
  setAuthBusy(true);
  try {
    await requestJson("/api/v1/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      handleAuthStatus: false,
      body: JSON.stringify({ username: values.username, email: values.email, password: values.password })
    });
    elements.registerForm.reset();
    elements.loginName.value = values.username;
    setAuthMode("login", "注册成功。6A后端不会自动登录，请使用新账户登录。", "success");
  } catch (error) {
    setAuthMode("register", apiErrorMessage(error, "注册"));
  } finally {
    elements.registerPassword.value = "";
    elements.registerPasswordConfirmation.value = "";
    setAuthBusy(false);
  }
}

async function logout() {
  if (!state.authUser || state.authBusy) return;
  setAuthBusy(true);
  elements.logoutButton.disabled = true;
  try {
    const response = await apiFetch("/api/v1/auth/logout", { method: "POST" });
    if (!response.ok) await readJsonResponse(response);
    showLoggedOut("已安全退出，服务端Session已经撤销。" );
  } catch (error) {
    if (state.authUser) showError(apiErrorMessage(error, "退出登录"));
  } finally {
    elements.logoutButton.disabled = false;
    setAuthBusy(false);
  }
}

elements.showLogin.addEventListener("click", () => setAuthMode("login"));
elements.showRegister.addEventListener("click", () => setAuthMode("register"));
elements.loginForm.addEventListener("submit", submitLogin);
elements.registerForm.addEventListener("submit", submitRegistration);
elements.logoutButton.addEventListener("click", () => void logout());

function bootstrap() {
  researchFlow = initResearchFlow({
    navigate: setView
  });
  setAuthStatusHandler(handleAuthStatus);

  try {
    const passed = runSseParserSelfTests();
    console.info("SSE parser browser self-tests passed", passed);
    const researchPassed = runResearchFlowSelfTests();
    console.info("ResearchFlow browser self-tests passed", researchPassed);
    const authPassed = runAuthStateSelfTests();
    console.info("Auth browser self-tests passed", authPassed);
    const apiPassed = runApiClientSelfTests();
    console.info("Authenticated API browser self-tests passed", apiPassed);
  } catch (error) {
    console.error(error);
    showError("浏览器SSE解析器自检失败，请查看Console并停止使用当前页面。");
  }
  void restoreSession();
}

window.AiAgentFrontendTests = {
  createSseParser,
  runSseParserSelfTests,
  runResearchFlowSelfTests,
  runAuthStateSelfTests,
  runApiClientSelfTests
};

bootstrap();
