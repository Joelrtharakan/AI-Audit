window.LQMS_AI_CONFIG = {
  apiBaseUrl: "http://localhost:8010", // swap for prod URL later
  // TODO(prod-security): never ship a real key here. The browser can read this value
  // via devtools regardless of how it's stored client-side. In production, route this
  // call through a thin ASP.NET .ashx proxy (or equivalent) that injects the
  // X-Internal-Api-Key header server-side, so the key never reaches the browser.
  // This value is acceptable ONLY for local dev against a local backend.
  internalApiKey: "devkey123"
};
