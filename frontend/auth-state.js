export const AUTH_VIEWS = Object.freeze({
  INITIALIZING: "initializing",
  LOGIN: "login",
  REGISTER: "register",
  AUTHENTICATED: "authenticated"
});

export function nextAuthView(status) {
  if (status === 200) return AUTH_VIEWS.AUTHENTICATED;
  if (status === 401) return AUTH_VIEWS.LOGIN;
  return AUTH_VIEWS.INITIALIZING;
}

export function validRegistration({ username, email, password, passwordConfirmation }) {
  if (!String(username || "").trim() || !String(email || "").trim()) return false;
  if (String(password || "").length < 8) return false;
  return password === passwordConfirmation;
}

function assert(condition, message) {
  if (!condition) throw new Error(`Auth frontend self-test failed: ${message}`);
}

export function runAuthStateSelfTests() {
  assert(nextAuthView(200) === AUTH_VIEWS.AUTHENTICATED, "200 restores workspace");
  assert(nextAuthView(401) === AUTH_VIEWS.LOGIN, "401 shows login");
  assert(
    validRegistration({ username: "moon", email: "moon@example.com", password: "password1", passwordConfirmation: "password1" }),
    "matching registration is valid"
  );
  assert(
    !validRegistration({ username: "moon", email: "moon@example.com", password: "password1", passwordConfirmation: "different" }),
    "password confirmation is checked"
  );
  return ["auth-restore", "registration-validation"];
}
