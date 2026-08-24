const TERMINAL_EVENTS = new Set([
  "run_completed",
  "run_blocked",
  "run_needs_review",
  "run_failed"
]);

export function isTaskStartable(task, project) {
  return Boolean(task && project && task.status === "pending" && project.status !== "archived");
}

export function appendStreamToken(current, fragment) {
  return `${current || ""}${typeof fragment === "string" ? fragment : ""}`;
}

export function isTerminalResearchEvent(eventName) {
  return TERMINAL_EVENTS.has(eventName);
}

function assert(condition, message) {
  if (!condition) throw new Error(`ResearchFlow frontend self-test failed: ${message}`);
}

export function runResearchFlowSelfTests() {
  assert(isTaskStartable({ status: "pending" }, { status: "active" }), "pending task starts");
  assert(!isTaskStartable({ status: "completed" }, { status: "active" }), "completed task disabled");
  assert(!isTaskStartable({ status: "pending" }, { status: "archived" }), "archived project disabled");
  assert(appendStreamToken("候选", "成果") === "候选成果", "tokens share one buffer");
  assert(isTerminalResearchEvent("run_needs_review"), "needs review terminates stream");
  return ["task-start-policy", "single-token-buffer", "terminal-event-contract"];
}
