const FIELDS = [
  "apiBaseUrl",
  "mode",
  "provider",
  "audioProvider",
  "solveDelayMs",
  "textImageSelector",
  "textInputSelector",
  "disabledHosts"
];
const CHECKS = [
  "autoSolve",
  "evalMath",
  "enableText",
  "enableRecaptcha",
  "enableHcaptcha",
  "enableAwsCaptcha",
  "enableFuncaptcha"
];

const statusEl = document.getElementById("status");

chrome.runtime.sendMessage({ type: "GET_SETTINGS" }, (settings) => {
  if (!settings) return;
  for (const id of FIELDS) {
    document.getElementById(id).value = settings[id] ?? "";
  }
  for (const id of CHECKS) {
    document.getElementById(id).checked = Boolean(settings[id]);
  }
});

async function startLocator(targetField) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) {
    statusEl.textContent = "No active tab.";
    return;
  }
  chrome.tabs.sendMessage(tab.id, { type: "START_LOCATOR", targetField }, () => {
    window.close();
  });
}

document.getElementById("pickImgBtn").addEventListener("click", () => startLocator("textImageSelector"));
document.getElementById("pickInputBtn").addEventListener("click", () => startLocator("textInputSelector"));

document.getElementById("saveBtn").addEventListener("click", async () => {
  const next = {};
  for (const id of FIELDS) {
    const val = document.getElementById(id).value.trim();
    next[id] = id === "solveDelayMs" ? Number(val || 600) : val;
  }
  if (!next.apiBaseUrl) next.apiBaseUrl = "http://127.0.0.1:8000";
  for (const id of CHECKS) {
    next[id] = document.getElementById(id).checked;
  }
  await chrome.storage.local.set(next);
  statusEl.textContent = "Settings saved.";
});

document.getElementById("healthBtn").addEventListener("click", () => {
  statusEl.textContent = "Checking...";
  chrome.runtime.sendMessage({ type: "CHECK_HEALTH" }, (resp) => {
    if (resp?.ok) {
      const models = (resp.data.models_loaded || []).join(", ") || "none";
      const caps = (resp.data.capabilities || []).length;
      const whisper = resp.data.audio_configured?.whisper_local ? "yes" : "no";
      statusEl.textContent = `OK | ONNX: ${models} | Caps: ${caps} | Whisper: ${whisper}`;
    } else {
      statusEl.textContent = `Error: ${resp?.error || "unreachable"}`;
    }
  });
});
