export class ApiError extends Error {
  constructor(status, payload) {
    super(`HTTP ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

export function userHeaders(userId, extra = {}) {
  return { "X-User-ID": userId, ...extra };
}

export async function apiFetch(path, { userId, headers = {}, ...options } = {}) {
  return fetch(path, {
    ...options,
    headers: userHeaders(userId, headers)
  });
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
    if (typeof detail === "string" && detail.trim()) {
      return detail;
    }
    const validation = validationMessage(detail);
    if (validation) return validation;
    if (error.status === 404) return `${context}的资源不存在或当前用户无权访问。`;
    if (error.status === 409) return `${context}与当前业务状态冲突，请刷新后重试。`;
    if (error.status === 422) return `${context}参数不符合接口要求。`;
    if (error.status === 401 || error.status === 403) return "当前开发用户无权执行此操作。";
    if (error.status >= 500) return "服务暂时无法完成请求，请稍后重试。";
    return `${context}失败（HTTP ${error.status}）。`;
  }
  if (error instanceof TypeError) {
    return "无法连接服务，请确认 FastAPI 已启动后重试。";
  }
  if (error instanceof Error && error.message) return error.message;
  return `${context}失败。`;
}
