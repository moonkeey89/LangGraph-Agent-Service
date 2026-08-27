export class ApiError extends Error {
  constructor(status, payload) {
    super(`HTTP ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

export const DEFAULT_CSRF_COOKIE_NAME = "researchflow_csrf";
export const DEFAULT_CSRF_HEADER_NAME = "X-CSRF-Token";
const CSRF_COOKIE_CONTRACT_HEADER = "X-ResearchFlow-CSRF-Cookie";
const CSRF_HEADER_CONTRACT_HEADER = "X-ResearchFlow-CSRF-Header";

const SAFE_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);
let authStatusHandler = null;
let csrfCookieName = DEFAULT_CSRF_COOKIE_NAME;
let csrfHeaderName = DEFAULT_CSRF_HEADER_NAME;

export function setAuthStatusHandler(handler) {
  authStatusHandler = typeof handler === "function" ? handler : null;
}

export function configureCsrfFromResponse(response) {
  const cookieName = response?.headers?.get(CSRF_COOKIE_CONTRACT_HEADER);
  const headerName = response?.headers?.get(CSRF_HEADER_CONTRACT_HEADER);
  if (cookieName) csrfCookieName = cookieName;
  if (headerName) csrfHeaderName = headerName;
}

export function readCookie(name, cookieSource = document.cookie) {
  const prefix = `${encodeURIComponent(name)}=`;
  for (const part of String(cookieSource || "").split(";")) {
    const value = part.trim();
    if (value.startsWith(prefix)) {
      return decodeURIComponent(value.slice(prefix.length));
    }
  }
  return "";
}

export function isMutationMethod(method) {
  return !SAFE_METHODS.has(String(method || "GET").toUpperCase());
}

export function authenticatedHeaders(method, headers = {}, cookieSource = document.cookie) {
  const result = new Headers(headers);
  if (isMutationMethod(method) && !result.has(csrfHeaderName)) {
    const token = readCookie(csrfCookieName, cookieSource);
    if (token) result.set(csrfHeaderName, token);
  }
  return result;
}

export function clearClientCsrfCookie() {
  document.cookie = `${encodeURIComponent(csrfCookieName)}=; Path=/; Max-Age=0; SameSite=Lax`;
}

export async function apiFetch(
  path,
  { headers = {}, handleAuthStatus = true, ...options } = {}
) {
  const method = String(options.method || "GET").toUpperCase();
  const response = await fetch(path, {
    ...options,
    method,
    credentials: "same-origin",
    headers: authenticatedHeaders(method, headers)
  });
  if (handleAuthStatus && (response.status === 401 || response.status === 403)) {
    authStatusHandler?.(response.status);
  }
  return response;
}

export async function readJsonResponse(response) {
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    throw new ApiError(response.status, payload);
  }
  return payload;
}

export async function requestJson(path, options = {}) {
  const response = await apiFetch(path, options);
  return readJsonResponse(response);
}

function validationMessage(detail) {
  if (!Array.isArray(detail)) return "";
  return detail
    .map((item) => {
      const location = Array.isArray(item?.loc)
        ? item.loc.filter((part) => part !== "body").join(".")
        : "";
      const message = typeof item?.msg === "string" ? item.msg : "参数无效";
      return location ? `${location}：${message}` : message;
    })
    .filter(Boolean)
    .join("；");
}

export function apiErrorMessage(error, context = "请求") {
  if (error instanceof ApiError) {
    const detail = error.payload?.detail;
    if (typeof detail === "string" && detail.trim()) return detail;
    const validation = validationMessage(detail);
    if (validation) return validation;
    if (error.status === 401) return "登录会话已失效，请重新登录。";
    if (error.status === 403) return "安全校验失败，请刷新页面后重试。";
    if (error.status === 404) return `${context}的资源不存在或当前用户无权访问。`;
    if (error.status === 409) return `${context}与当前业务状态冲突，请刷新后重试。`;
    if (error.status === 422) return `${context}参数不符合接口要求。`;
    if (error.status >= 500) return "服务暂时无法完成请求，请稍后重试。";
    return `${context}失败（HTTP ${error.status}）。`;
  }
  if (error instanceof TypeError) {
    return "无法连接服务，请确认 FastAPI 已启动后重试。";
  }
  if (error instanceof Error && error.message) return error.message;
  return `${context}失败。`;
}

function assert(condition, message) {
  if (!condition) throw new Error(`API frontend self-test failed: ${message}`);
}

export function runApiClientSelfTests() {
  const cookie = "example=1; researchflow_csrf=csrf-test-value";
  assert(readCookie(DEFAULT_CSRF_COOKIE_NAME, cookie) === "csrf-test-value", "reads CSRF cookie");
  assert(!authenticatedHeaders("GET", {}, cookie).has(DEFAULT_CSRF_HEADER_NAME), "GET has no CSRF header");
  assert(
    authenticatedHeaders("POST", {}, cookie).get(DEFAULT_CSRF_HEADER_NAME) === "csrf-test-value",
    "mutation receives CSRF header"
  );
  return ["csrf-cookie", "safe-method", "mutation-csrf"];
}
