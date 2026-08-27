import { apiErrorMessage, apiFetch, readJsonResponse, requestJson } from "/assets/api.js";
import { createSseParser } from "/assets/sse-parser.js";
import {
  appendStreamToken,
  isTaskStartable,
  isTerminalResearchEvent,
  runResearchFlowSelfTests
} from "/assets/researchflow-state.js";

export {
  appendStreamToken,
  isTaskStartable,
  isTerminalResearchEvent,
  runResearchFlowSelfTests
} from "/assets/researchflow-state.js";

const SELECTED_PROJECT_STORAGE = "researchflow.selected-project";

const EVENT_LABELS = {
  run_started: "已创建 AgentRun",
  task_status: "任务状态已更新",
  agent_progress: "Agent 进度",
  evidence_found: "已找到研究证据",
  artifact_created: "已创建成果草稿",
  run_completed: "运行完成",
  run_blocked: "运行阻塞",
  run_needs_review: "等待人工审阅",
  run_failed: "运行失败"
};

function query(selector) {
  return document.querySelector(selector);
}

function createElement(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== "") node.textContent = text;
  return node;
}

function formatDate(value) {
  if (!value) return "-";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "-" : date.toLocaleString();
}

function criteriaFromText(value) {
  return value.split(/\r?\n/).map((item) => item.trim()).filter(Boolean);
}

function encode(value) {
  return encodeURIComponent(value);
}

function readStoredProject() {
  return localStorage.getItem(SELECTED_PROJECT_STORAGE) || "";
}

export function initResearchFlow({ navigate }) {
  const elements = {
    selector: query("#project-selector"),
    serviceStatus: query("#service-status"),
    notice: query("#research-notice"),
    noticeText: query("#research-notice-text"),
    retry: query("#research-retry"),
    projectList: query("#project-list"),
    showCreateProject: query("#show-create-project"),
    createProjectForm: query("#create-project-form"),
    cancelCreateProject: query("#cancel-create-project"),
    newProjectName: query("#new-project-name"),
    newProjectQuestion: query("#new-project-question"),
    newProjectKb: query("#new-project-kb"),
    overviewEmpty: query("#overview-empty"),
    overviewContent: query("#overview-content"),
    overviewName: query("#overview-project-name"),
    overviewStats: query("#overview-stats"),
    overviewRecentRun: query("#overview-recent-run"),
    projectEditForm: query("#project-edit-form"),
    projectName: query("#project-name"),
    projectDescription: query("#project-description"),
    projectQuestion: query("#project-question"),
    projectStatus: query("#project-status"),
    projectKb: query("#project-kb"),
    deleteProject: query("#delete-project"),
    showCreateTask: query("#show-create-task"),
    createTaskForm: query("#create-task-form"),
    cancelCreateTask: query("#cancel-create-task"),
    newTaskTitle: query("#new-task-title"),
    newTaskType: query("#new-task-type"),
    newTaskObjective: query("#new-task-objective"),
    newTaskCriteria: query("#new-task-criteria"),
    taskStatusFilter: query("#task-status-filter"),
    taskTypeFilter: query("#task-type-filter"),
    taskList: query("#task-list"),
    taskEmpty: query("#task-empty"),
    taskContent: query("#task-content"),
    taskDetailTitle: query("#task-detail-title"),
    taskEditForm: query("#task-edit-form"),
    taskTitle: query("#task-title"),
    taskType: query("#task-type"),
    taskObjective: query("#task-objective"),
    taskCriteria: query("#task-criteria"),
    taskStatus: query("#task-status"),
    startTask: query("#start-task"),
    deleteTask: query("#delete-task"),
    runLivePanel: query("#run-live-panel"),
    runLiveStatus: query("#run-live-status"),
    runTimeline: query("#run-timeline"),
    runCandidateOutput: query("#run-candidate-output"),
    runList: query("#run-list"),
    artifactStatusFilter: query("#artifact-status-filter"),
    artifactTypeFilter: query("#artifact-type-filter"),
    artifactTaskFilter: query("#artifact-task-filter"),
    artifactList: query("#artifact-list"),
    artifactEmpty: query("#artifact-empty"),
    artifactContent: query("#artifact-content"),
    artifactDetailTitle: query("#artifact-detail-title"),
    artifactActions: query("#artifact-actions"),
    artifactMeta: query("#artifact-meta"),
    artifactEditForm: query("#artifact-edit-form"),
    artifactTitle: query("#artifact-title"),
    artifactType: query("#artifact-type"),
    artifactBody: query("#artifact-body"),
    saveArtifact: query("#save-artifact"),
    finalizeArtifact: query("#finalize-artifact"),
    deleteArtifact: query("#delete-artifact"),
    artifactSources: query("#artifact-sources"),
    artifactIssuesSection: query("#artifact-issues-section"),
    artifactIssues: query("#artifact-issues")
  };

  const state = {
    projects: [],
    selectedProjectId: "",
    knowledgeBases: [],
    tasks: [],
    artifacts: [],
    runsByTask: new Map(),
    selectedTaskId: "",
    selectedArtifactId: "",
    activeView: "overview",
    startInFlight: new Set(),
    busy: false,
    enabled: false,
    generation: 0,
    streamController: null
  };

  function project() {
    return state.projects.find((item) => item.project_id === state.selectedProjectId) || null;
  }

  function task() {
    return state.tasks.find((item) => item.task_id === state.selectedTaskId) || null;
  }

  function artifact() {
    return state.artifacts.find((item) => item.artifact_id === state.selectedArtifactId) || null;
  }

  function projectPath(suffix = "") {
    return `/api/v1/research/projects/${encode(state.selectedProjectId)}${suffix}`;
  }

  function showNotice(message, kind = "error", retry = false) {
    elements.notice.hidden = !message;
    elements.noticeText.textContent = message || "";
    elements.notice.dataset.kind = kind;
    elements.retry.hidden = !retry;
  }

  function setServiceStatus(text, kind) {
    elements.serviceStatus.textContent = text;
    elements.serviceStatus.dataset.kind = kind;
  }

  function setBusy(value) {
    state.busy = value;
    elements.selector.disabled = value;
    elements.showCreateProject.disabled = value;
    elements.deleteProject.disabled = value;
  }

  function populateKnowledgeSelect(select, selected = "", emptyLabel = "不绑定知识库") {
    select.replaceChildren();
    const empty = createElement("option", "", emptyLabel);
    empty.value = "";
    select.append(empty);
    for (const item of state.knowledgeBases) {
      const option = createElement("option", "", item.name);
      option.value = item.knowledge_base_id;
      select.append(option);
    }
    select.value = state.knowledgeBases.some((item) => item.knowledge_base_id === selected)
      ? selected
      : "";
  }

  async function checkHealth() {
    try {
      const response = await fetch("/health");
      if (!response.ok) throw new Error("health failed");
      setServiceStatus("在线", "success");
    } catch {
      setServiceStatus("不可用", "error");
    }
  }

  async function loadKnowledgeBases() {
    if (!state.enabled) return;
    const generation = state.generation;
    const items = await requestJson("/api/v1/knowledge-bases");
    if (!state.enabled || generation !== state.generation) return;
    state.knowledgeBases = Array.isArray(items) ? items : [];
    populateKnowledgeSelect(elements.newProjectKb, "", "暂不绑定");
    populateKnowledgeSelect(elements.projectKb, project()?.default_knowledge_base_id || "", "解除绑定");
  }

  function renderProjectSelector() {
    elements.selector.replaceChildren();
    const empty = createElement("option", "", "尚未选择项目");
    empty.value = "";
    elements.selector.append(empty);
    for (const item of state.projects) {
      const option = createElement("option", "", `${item.name} · ${item.status}`);
      option.value = item.project_id;
      elements.selector.append(option);
    }
    elements.selector.value = state.selectedProjectId;
  }

  function renderProjectList() {
    elements.projectList.replaceChildren();
    if (state.projects.length === 0) {
      elements.projectList.append(createElement("p", "empty-copy", "当前用户还没有项目，请创建第一个项目。"));
      return;
    }
    for (const item of state.projects) {
      const button = createElement("button", "entity-item");
      button.type = "button";
      button.classList.toggle("active", item.project_id === state.selectedProjectId);
      button.append(
        createElement("strong", "", item.name),
        createElement("span", "", `${item.status} · ${item.research_question || "尚未填写研究问题"}`)
      );
      button.addEventListener("click", () => void selectProject(item.project_id));
      elements.projectList.append(button);
    }
  }

  function renderStats() {
    elements.overviewStats.replaceChildren();
    const counts = new Map();
    for (const item of state.tasks) counts.set(item.status, (counts.get(item.status) || 0) + 1);
    const values = [
      ["Task 总数", state.tasks.length],
      ["pending", counts.get("pending") || 0],
      ["running / blocked", (counts.get("running") || 0) + (counts.get("blocked") || 0)],
      ["Artifact", state.artifacts.length]
    ];
    for (const [label, value] of values) {
      const card = createElement("div", "stat-card");
      card.append(createElement("span", "", label), createElement("strong", "", String(value)));
      elements.overviewStats.append(card);
    }
  }

  function allRuns() {
    return Array.from(state.runsByTask.values()).flat();
  }

  function renderRecentRun() {
    elements.overviewRecentRun.replaceChildren();
    const runs = allRuns().sort((left, right) => String(right.updated_at).localeCompare(String(left.updated_at)));
    if (!runs.length) {
      elements.overviewRecentRun.textContent = "当前项目还没有运行记录。";
      return;
    }
    const run = runs[0];
    const relatedTask = state.tasks.find((item) => item.task_id === run.task_id);
    const card = createElement("div", "run-row");
    card.append(
      createElement("strong", "", relatedTask?.title || run.task_id),
      statusBadge(run.outcome || run.status),
      createElement("span", "", `第 ${run.attempt_number} 次 · ${formatDate(run.updated_at)}`)
    );
    elements.overviewRecentRun.append(card);
  }

  function renderOverview() {
    const current = project();
    elements.overviewEmpty.hidden = Boolean(current);
    elements.overviewContent.hidden = !current;
    if (!current) return;
    elements.overviewName.textContent = current.name;
    elements.projectName.value = current.name;
    elements.projectDescription.value = current.description || "";
    elements.projectQuestion.value = current.research_question || "";
    elements.projectStatus.value = current.status;
    populateKnowledgeSelect(elements.projectKb, current.default_knowledge_base_id || "", "解除绑定");
    renderStats();
    renderRecentRun();
  }

  async function loadProjects({ restore = true } = {}) {
    if (!state.enabled) return;
    const generation = state.generation;
    showNotice("");
    elements.projectList.replaceChildren(createElement("p", "empty-copy", "正在加载科研项目……"));
    try {
      const [projects] = await Promise.all([
        requestJson("/api/v1/research/projects"),
        loadKnowledgeBases()
      ]);
      if (!state.enabled || generation !== state.generation) return;
      state.projects = Array.isArray(projects) ? projects : [];
      const restored = restore ? readStoredProject() : "";
      const candidate = state.selectedProjectId || restored;
      state.selectedProjectId = state.projects.some((item) => item.project_id === candidate)
        ? candidate
        : "";
      renderProjectSelector();
      renderProjectList();
      if (state.selectedProjectId) await loadProjectWorkspace();
      else clearProjectWorkspace();
    } catch (error) {
      showNotice(apiErrorMessage(error, "加载科研项目"), "error", true);
    }
  }

  function clearProjectWorkspace() {
    state.tasks = [];
    state.artifacts = [];
    state.runsByTask = new Map();
    state.selectedTaskId = "";
    state.selectedArtifactId = "";
    renderOverview();
    renderTasks();
    renderArtifacts();
  }

  async function selectProject(projectId) {
    const exists = state.projects.some((item) => item.project_id === projectId);
    state.selectedProjectId = exists ? projectId : "";
    state.selectedTaskId = "";
    state.selectedArtifactId = "";
    localStorage.setItem(SELECTED_PROJECT_STORAGE, state.selectedProjectId);
    renderProjectSelector();
    renderProjectList();
    if (state.selectedProjectId) await loadProjectWorkspace();
    else clearProjectWorkspace();
  }

  async function loadProjectWorkspace() {
    const projectId = state.selectedProjectId;
    if (!projectId) return clearProjectWorkspace();
    const generation = state.generation;
    elements.taskList.replaceChildren(createElement("p", "empty-copy", "正在加载科研任务……"));
    elements.artifactList.replaceChildren(createElement("p", "empty-copy", "正在加载研究成果……"));
    try {
      const [tasks, artifacts] = await Promise.all([
        requestJson(projectPath("/tasks")),
        requestJson(projectPath("/artifacts"))
      ]);
      if (!state.enabled || generation !== state.generation || projectId !== state.selectedProjectId) return;
      state.tasks = Array.isArray(tasks) ? tasks : [];
      state.artifacts = Array.isArray(artifacts) ? artifacts : [];
      const runEntries = await Promise.all(
        state.tasks.map(async (item) => [
          item.task_id,
          await requestJson(projectPath(`/tasks/${encode(item.task_id)}/runs`))
        ])
      );
      if (!state.enabled || generation !== state.generation || projectId !== state.selectedProjectId) return;
      state.runsByTask = new Map(runEntries);
      if (!state.tasks.some((item) => item.task_id === state.selectedTaskId)) state.selectedTaskId = "";
      if (!state.artifacts.some((item) => item.artifact_id === state.selectedArtifactId)) state.selectedArtifactId = "";
      renderOverview();
      renderTasks();
      renderArtifacts();
    } catch (error) {
      showNotice(apiErrorMessage(error, "加载项目工作区"), "error", true);
    }
  }

  function statusBadge(value) {
    const badge = createElement("span", `status-badge ${value || "unknown"}`, value || "unknown");
    return badge;
  }

  function renderTasks() {
    const currentProject = project();
    const statusFilter = elements.taskStatusFilter.value;
    const typeFilter = elements.taskTypeFilter.value;
    const visible = state.tasks.filter(
      (item) => (!statusFilter || item.status === statusFilter) && (!typeFilter || item.task_type === typeFilter)
    );
    elements.taskList.replaceChildren();
    elements.showCreateTask.disabled = !currentProject || currentProject.status === "archived" || state.busy;
    if (!currentProject) {
      elements.taskList.append(createElement("p", "empty-copy", "请先选择科研项目。"));
    } else if (!visible.length) {
      elements.taskList.append(createElement("p", "empty-copy", state.tasks.length ? "没有符合筛选条件的任务。" : "尚无任务，请创建第一项科研任务。"));
    } else {
      for (const item of visible) {
        const button = createElement("button", "entity-item");
        button.type = "button";
        button.classList.toggle("active", item.task_id === state.selectedTaskId);
        const line = createElement("span", "entity-line");
        line.append(statusBadge(item.status), createElement("span", "", item.task_type));
        button.append(createElement("strong", "", item.title), line);
        button.addEventListener("click", () => {
          state.selectedTaskId = item.task_id;
          renderTasks();
        });
        elements.taskList.append(button);
      }
    }
    renderTaskDetail();
    renderArtifactTaskFilter();
  }

  function renderTaskDetail() {
    const currentTask = task();
    const currentProject = project();
    elements.taskEmpty.hidden = Boolean(currentTask);
    elements.taskContent.hidden = !currentTask;
    if (!currentTask) return;
    elements.taskDetailTitle.textContent = currentTask.title;
    elements.taskTitle.value = currentTask.title;
    elements.taskType.value = currentTask.task_type;
    elements.taskObjective.value = currentTask.objective || "";
    elements.taskCriteria.value = (currentTask.acceptance_criteria || []).join("\n");
    elements.taskStatus.textContent = currentTask.status;
    elements.taskStatus.className = `status-badge ${currentTask.status}`;
    const archived = currentProject?.status === "archived";
    elements.startTask.hidden = currentTask.status !== "pending";
    elements.startTask.disabled = !isTaskStartable(currentTask, currentProject) || state.startInFlight.has(currentTask.task_id);
    elements.deleteTask.disabled = archived || !["pending", "cancelled"].includes(currentTask.status);
    for (const control of elements.taskEditForm.elements) control.disabled = archived || state.busy;
    renderRuns(currentTask.task_id);
  }

  function renderRuns(taskId) {
    elements.runList.replaceChildren();
    const runs = state.runsByTask.get(taskId) || [];
    if (!runs.length) {
      elements.runList.textContent = "该任务还没有 AgentRun。";
      return;
    }
    for (const run of runs) {
      const row = createElement("article", "run-row");
      const heading = createElement("div", "entity-line");
      heading.append(createElement("strong", "", `第 ${run.attempt_number} 次运行`), statusBadge(run.outcome || run.status));
      row.append(
        heading,
        createElement("span", "", `run_id：${run.run_id}`),
        createElement("span", "", `开始：${formatDate(run.started_at)}；结束：${formatDate(run.finished_at)}`)
      );
      if (run.error_message) row.append(createElement("p", "error-copy", run.error_message));
      if (run.output_artifact_id) {
        const open = createElement("button", "link-button", "查看输出成果");
        open.type = "button";
        open.addEventListener("click", () => {
          state.selectedArtifactId = run.output_artifact_id;
          renderArtifacts();
          navigate("artifacts");
        });
        row.append(open);
      }
      elements.runList.append(row);
    }
  }

  function renderArtifactTaskFilter() {
    const selected = elements.artifactTaskFilter.value;
    elements.artifactTaskFilter.replaceChildren();
    const all = createElement("option", "", "全部任务");
    all.value = "";
    elements.artifactTaskFilter.append(all);
    for (const item of state.tasks) {
      const option = createElement("option", "", item.title);
      option.value = item.task_id;
      elements.artifactTaskFilter.append(option);
    }
    elements.artifactTaskFilter.value = state.tasks.some((item) => item.task_id === selected) ? selected : "";
  }

  function renderArtifacts() {
    const statusFilter = elements.artifactStatusFilter.value;
    const typeFilter = elements.artifactTypeFilter.value;
    const taskFilter = elements.artifactTaskFilter.value;
    const visible = state.artifacts.filter(
      (item) => (!statusFilter || item.status === statusFilter)
        && (!typeFilter || item.artifact_type === typeFilter)
        && (!taskFilter || item.task_id === taskFilter)
    );
    elements.artifactList.replaceChildren();
    if (!project()) {
      elements.artifactList.append(createElement("p", "empty-copy", "请先选择科研项目。"));
    } else if (!visible.length) {
      elements.artifactList.append(createElement("p", "empty-copy", state.artifacts.length ? "没有符合筛选条件的成果。" : "尚无成果；运行科研任务后会生成 draft。"));
    } else {
      for (const item of visible) {
        const button = createElement("button", "entity-item");
        button.type = "button";
        button.classList.toggle("active", item.artifact_id === state.selectedArtifactId);
        const line = createElement("span", "entity-line");
        line.append(statusBadge(item.status), createElement("span", "", item.artifact_type));
        button.append(createElement("strong", "", item.title), line);
        button.addEventListener("click", () => {
          state.selectedArtifactId = item.artifact_id;
          renderArtifacts();
        });
        elements.artifactList.append(button);
      }
    }
    renderArtifactDetail();
  }

  function appendMetadata(label, value) {
    elements.artifactMeta.append(createElement("dt", "", label), createElement("dd", "", value || "-"));
  }

  function renderArtifactDetail() {
    const current = artifact();
    elements.artifactEmpty.hidden = Boolean(current);
    elements.artifactContent.hidden = !current;
    if (!current) return;
    elements.artifactDetailTitle.textContent = current.title;
    elements.artifactTitle.value = current.title;
    elements.artifactType.value = current.artifact_type;
    elements.artifactBody.value = current.content;
    const mutable = current.status === "draft" && project()?.status !== "archived";
    elements.artifactActions.hidden = !mutable;
    for (const control of elements.artifactEditForm.elements) control.disabled = !mutable || state.busy;
    elements.artifactMeta.replaceChildren();
    const relatedTask = state.tasks.find((item) => item.task_id === current.task_id);
    appendMetadata("状态", current.status);
    appendMetadata("创建者", current.created_by);
    appendMetadata("关联任务", relatedTask?.title || current.task_id || "项目级成果");
    appendMetadata("来源 Run", current.origin_run_id || "人工创建");
    appendMetadata("更新时间", formatDate(current.updated_at));
    elements.artifactSources.replaceChildren();
    if (!current.sources?.length) {
      elements.artifactSources.textContent = "该成果没有关联证据来源。";
    } else {
      for (const source of current.sources) {
        const card = createElement("article", "source-card");
        const title = source.page == null ? source.source : `${source.source} · 第 ${source.page} 页`;
        card.append(createElement("strong", "", title), createElement("p", "", source.excerpt || ""));
        elements.artifactSources.append(card);
      }
    }
    const issues = Array.isArray(current.unresolved_issues) ? current.unresolved_issues : [];
    elements.artifactIssuesSection.hidden = issues.length === 0;
    elements.artifactIssues.replaceChildren();
    for (const issue of issues) elements.artifactIssues.append(createElement("li", "", String(issue)));
  }

  async function createProject(event) {
    event.preventDefault();
    if (state.busy) return;
    setBusy(true);
    try {
      const created = await requestJson("/api/v1/research/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: elements.newProjectName.value.trim(),
          research_question: elements.newProjectQuestion.value.trim(),
          default_knowledge_base_id: elements.newProjectKb.value || null
        })
      });
      elements.createProjectForm.reset();
      elements.createProjectForm.hidden = true;
      await loadProjects({ restore: false });
      await selectProject(created.project_id);
      showNotice("科研项目已创建。", "success");
    } catch (error) {
      showNotice(apiErrorMessage(error, "创建项目"));
    } finally {
      setBusy(false);
    }
  }

  async function updateProject(event) {
    event.preventDefault();
    if (!project() || state.busy) return;
    setBusy(true);
    try {
      const updated = await requestJson(projectPath(), {
        method: "PATCH", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: elements.projectName.value.trim(),
          description: elements.projectDescription.value.trim(),
          research_question: elements.projectQuestion.value.trim(),
          status: elements.projectStatus.value,
          default_knowledge_base_id: elements.projectKb.value || null
        })
      });
      const index = state.projects.findIndex((item) => item.project_id === updated.project_id);
      state.projects[index] = updated;
      renderProjectSelector(); renderProjectList(); renderOverview(); renderTasks(); renderArtifactDetail();
      showNotice("项目资料已保存。", "success");
    } catch (error) {
      showNotice(apiErrorMessage(error, "更新项目"));
    } finally { setBusy(false); }
  }

  async function deleteProject() {
    const current = project();
    if (!current || state.busy) return;
    if (!globalThis.confirm(`确定删除科研项目“${current.name}”吗？存在 Task、Artifact 或 Run 时后端会拒绝删除。`)) return;
    setBusy(true);
    try {
      const response = await apiFetch(projectPath(), { method: "DELETE" });
      if (!response.ok) await readJsonResponse(response);
      state.selectedProjectId = "";
      localStorage.removeItem(SELECTED_PROJECT_STORAGE);
      await loadProjects({ restore: false });
      showNotice("项目已删除；知识库、Checkpoint 与长期记忆未被删除。", "success");
    } catch (error) { showNotice(apiErrorMessage(error, "删除项目")); }
    finally { setBusy(false); }
  }

  async function createTask(event) {
    event.preventDefault();
    if (!project() || state.busy) return;
    setBusy(true);
    try {
      const created = await requestJson(projectPath("/tasks"), {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: elements.newTaskTitle.value.trim(),
          objective: elements.newTaskObjective.value.trim(),
          task_type: elements.newTaskType.value,
          acceptance_criteria: criteriaFromText(elements.newTaskCriteria.value)
        })
      });
      elements.createTaskForm.reset(); elements.createTaskForm.hidden = true;
      await loadProjectWorkspace();
      state.selectedTaskId = created.task_id; renderTasks();
      showNotice("科研任务已创建，初始状态为 pending。", "success");
    } catch (error) { showNotice(apiErrorMessage(error, "创建任务")); }
    finally { setBusy(false); }
  }

  async function updateTask(event) {
    event.preventDefault();
    const current = task();
    if (!current || state.busy) return;
    setBusy(true);
    try {
      await requestJson(projectPath(`/tasks/${encode(current.task_id)}`), {
        method: "PATCH", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: elements.taskTitle.value.trim(), objective: elements.taskObjective.value.trim(), task_type: elements.taskType.value, acceptance_criteria: criteriaFromText(elements.taskCriteria.value) })
      });
      await loadProjectWorkspace();
      showNotice("任务可编辑字段已保存。", "success");
    } catch (error) { showNotice(apiErrorMessage(error, "更新任务")); }
    finally { setBusy(false); }
  }

  async function deleteTask() {
    const current = task();
    if (!current || state.busy) return;
    if (!globalThis.confirm(`确定删除任务“${current.title}”吗？只有 pending/cancelled 且没有成果或 Run 引用时允许删除。`)) return;
    setBusy(true);
    try {
      const response = await apiFetch(projectPath(`/tasks/${encode(current.task_id)}`), { method: "DELETE" });
      if (!response.ok) await readJsonResponse(response);
      state.selectedTaskId = ""; await loadProjectWorkspace();
      showNotice("任务已删除。", "success");
    } catch (error) { showNotice(apiErrorMessage(error, "删除任务")); }
    finally { setBusy(false); }
  }

  function addTimeline(eventName, message, kind = "") {
    const item = createElement("li", kind);
    item.append(createElement("strong", "", EVENT_LABELS[eventName] || eventName), createElement("span", "", message));
    elements.runTimeline.append(item);
  }

  function handleResearchEvent(item, execution) {
    const data = item.data || {};
    switch (item.event) {
      case "run_started":
        execution.runId = data.run_id || "";
        addTimeline(item.event, `run_id：${execution.runId}`);
        break;
      case "task_status":
        elements.runLiveStatus.textContent = data.status || "running";
        addTimeline(item.event, String(data.status || "running"));
        break;
      case "agent_progress":
        addTimeline(item.event, String(data.message || data.stage || "正在处理"));
        break;
      case "evidence_found": {
        const summaries = Array.isArray(data.sources)
          ? data.sources.map((source) => source.page == null ? source.source : `${source.source}（第 ${source.page} 页）`).join("；")
          : "";
        addTimeline(item.event, `${data.count || 0} 条${summaries ? `：${summaries}` : ""}`);
        break;
      }
      case "token":
        execution.output = appendStreamToken(execution.output, data.content);
        elements.runCandidateOutput.textContent = execution.output;
        break;
      case "artifact_created":
        execution.artifactId = data.artifact_id || "";
        addTimeline(item.event, `draft · ${execution.artifactId}`);
        break;
      default:
        if (isTerminalResearchEvent(item.event)) {
          execution.terminal = item.event;
          if (typeof data.answer === "string") {
            execution.output = data.answer;
            elements.runCandidateOutput.textContent = data.answer;
          }
          elements.runLiveStatus.textContent = data.outcome || data.status || item.event;
          elements.runLiveStatus.className = `status-badge ${data.outcome || data.status || "failed"}`;
          addTimeline(item.event, String(data.error || data.message || data.outcome || "已结束"), "terminal");
        }
    }
  }

  async function consumeResearchStream(currentTask) {
    state.streamController = new AbortController();
    const response = await apiFetch(projectPath(`/tasks/${encode(currentTask.task_id)}/runs/stream`), {
      method: "POST", headers: { Accept: "text/event-stream" }, signal: state.streamController.signal
    });
    if (!response.ok) await readJsonResponse(response);
    if (!response.body) throw new Error("浏览器没有提供可读取的 SSE 响应体。");
    const execution = { runId: "", artifactId: "", output: "", terminal: "" };
    const parser = createSseParser({
      onEvent: (item) => handleResearchEvent(item, execution),
      onInvalidJson: () => addTimeline("stream_warning", "收到格式异常事件，已忽略并继续读取。", "warning")
    });
    const reader = response.body.getReader();
    try {
      while (!execution.terminal) {
        const { value, done } = await reader.read();
        if (done) { parser.finish(); break; }
        parser.push(value);
      }
      if (execution.terminal) await reader.cancel();
      else throw new Error("SSE 连接已结束，但没有收到 ResearchFlow 终止事件。后端可能仍在执行，请刷新 Run 状态。" );
    } finally {
      reader.releaseLock();
      state.streamController = null;
    }
    return execution;
  }

  async function startTask() {
    const current = task();
    if (!isTaskStartable(current, project()) || state.startInFlight.has(current.task_id)) return;
    state.startInFlight.add(current.task_id);
    elements.runLivePanel.hidden = false;
    elements.runTimeline.replaceChildren();
    elements.runCandidateOutput.textContent = "";
    elements.runLiveStatus.textContent = "starting";
    elements.runLiveStatus.className = "status-badge running";
    renderTaskDetail();
    try {
      const execution = await consumeResearchStream(current);
      showNotice(EVENT_LABELS[execution.terminal] || "科研任务运行结束。", execution.terminal === "run_failed" ? "error" : "success");
      await loadProjectWorkspace();
      state.selectedTaskId = current.task_id;
      if (execution.artifactId) state.selectedArtifactId = execution.artifactId;
      renderTasks(); renderArtifacts();
    } catch (error) {
      showNotice(`${apiErrorMessage(error, "启动科研任务")} 浏览器中断不等于业务取消，正在重新查询持久化状态。`);
      if (state.enabled) {
        await loadProjectWorkspace();
        state.selectedTaskId = current.task_id; renderTasks();
      }
    } finally {
      state.startInFlight.delete(current.task_id);
      renderTaskDetail();
    }
  }

  async function updateArtifact(event) {
    event.preventDefault();
    const current = artifact();
    if (!current || current.status !== "draft" || state.busy) return;
    setBusy(true);
    try {
      await requestJson(projectPath(`/artifacts/${encode(current.artifact_id)}`), {
        method: "PATCH", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: elements.artifactTitle.value.trim(), artifact_type: elements.artifactType.value, content: elements.artifactBody.value.trim() })
      });
      await loadProjectWorkspace();
      showNotice("成果草稿已保存。", "success");
    } catch (error) { showNotice(apiErrorMessage(error, "保存成果")); }
    finally { setBusy(false); }
  }

  async function finalizeArtifact() {
    const current = artifact();
    if (!current || current.status !== "draft" || state.busy) return;
    if (!globalThis.confirm("定稿后不能直接修改或删除，是否继续？")) return;
    setBusy(true);
    try {
      await requestJson(projectPath(`/artifacts/${encode(current.artifact_id)}/finalize`), { method: "POST" });
      await loadProjectWorkspace();
      showNotice("成果已经人工定稿。", "success");
    } catch (error) { showNotice(apiErrorMessage(error, "定稿成果")); }
    finally { setBusy(false); }
  }

  async function deleteArtifact() {
    const current = artifact();
    if (!current || current.status !== "draft" || state.busy) return;
    if (!globalThis.confirm(`确定删除草稿“${current.title}”吗？证据文档和知识库不会被删除。`)) return;
    setBusy(true);
    try {
      const response = await apiFetch(projectPath(`/artifacts/${encode(current.artifact_id)}`), { method: "DELETE" });
      if (!response.ok) await readJsonResponse(response);
      state.selectedArtifactId = ""; await loadProjectWorkspace();
      showNotice("成果草稿已删除。", "success");
    } catch (error) { showNotice(apiErrorMessage(error, "删除成果")); }
    finally { setBusy(false); }
  }

  function activate(viewName) {
    state.activeView = viewName;
    if (state.enabled && ["overview", "tasks", "artifacts"].includes(viewName)) {
      if (!state.projects.length) void loadProjects();
      else if (state.selectedProjectId) void loadProjectWorkspace();
    }
  }

  function reset() {
    state.enabled = false;
    state.generation += 1;
    state.streamController?.abort();
    state.streamController = null;
    state.startInFlight.clear();
    state.selectedProjectId = "";
    localStorage.removeItem(SELECTED_PROJECT_STORAGE);
    state.projects = [];
    state.knowledgeBases = [];
    clearProjectWorkspace();
    elements.runLivePanel.hidden = true;
    elements.runTimeline.replaceChildren();
    elements.runCandidateOutput.textContent = "";
    elements.runLiveStatus.textContent = "";
    renderProjectSelector(); renderProjectList();
    showNotice("");
  }

  async function start() {
    state.enabled = true;
    state.generation += 1;
    await checkHealth();
    await loadProjects();
  }

  elements.selector.addEventListener("change", () => void selectProject(elements.selector.value));
  elements.showCreateProject.addEventListener("click", () => { elements.createProjectForm.hidden = false; elements.newProjectName.focus(); });
  elements.cancelCreateProject.addEventListener("click", () => { elements.createProjectForm.reset(); elements.createProjectForm.hidden = true; });
  elements.createProjectForm.addEventListener("submit", createProject);
  elements.projectEditForm.addEventListener("submit", updateProject);
  elements.deleteProject.addEventListener("click", () => void deleteProject());
  elements.showCreateTask.addEventListener("click", () => { elements.createTaskForm.hidden = false; elements.newTaskTitle.focus(); });
  elements.cancelCreateTask.addEventListener("click", () => { elements.createTaskForm.reset(); elements.createTaskForm.hidden = true; });
  elements.createTaskForm.addEventListener("submit", createTask);
  elements.taskEditForm.addEventListener("submit", updateTask);
  elements.deleteTask.addEventListener("click", () => void deleteTask());
  elements.startTask.addEventListener("click", () => void startTask());
  elements.taskStatusFilter.addEventListener("change", renderTasks);
  elements.taskTypeFilter.addEventListener("change", renderTasks);
  elements.artifactStatusFilter.addEventListener("change", renderArtifacts);
  elements.artifactTypeFilter.addEventListener("change", renderArtifacts);
  elements.artifactTaskFilter.addEventListener("change", renderArtifacts);
  elements.artifactEditForm.addEventListener("submit", updateArtifact);
  elements.finalizeArtifact.addEventListener("click", () => void finalizeArtifact());
  elements.deleteArtifact.addEventListener("click", () => void deleteArtifact());
  elements.retry.addEventListener("click", () => {
    showNotice("");
    if (state.selectedProjectId) void loadProjectWorkspace();
    else void loadProjects();
  });

  return {
    activate,
    start,
    reset,
    refreshKnowledgeBases: loadKnowledgeBases,
    refresh: loadProjectWorkspace,
    selfTests: runResearchFlowSelfTests
  };
}
