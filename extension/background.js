const DEFAULT_SETTINGS = {
  apiBaseUrl: "http://127.0.0.1:8000",
  mode: "hybrid",
  provider: "openai",
  audioProvider: "whisper_local",
  autoSolve: true,
  enableText: true,
  enableRecaptcha: true,
  enableHcaptcha: true,
  enableAwsCaptcha: true,
  enableFuncaptcha: true,
  evalMath: false,
  solveDelayMs: 600,
  textImageSelector: "",
  textInputSelector: "",
  disabledHosts: ""
};

async function getSettings() {
  const stored = await chrome.storage.local.get(DEFAULT_SETTINGS);
  return { ...DEFAULT_SETTINGS, ...stored };
}

async function postJson(path, payload) {
  const settings = await getSettings();
  const url = `${settings.apiBaseUrl.replace(/\/$/, "")}${path}`;
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!resp.ok) {
    const errText = await resp.text();
    throw new Error(`HTTP ${resp.status}: ${errText}`);
  }
  return resp.json();
}

async function fetchUrlAsBase64(url, defaultMime = "image/png") {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Fetch failed: HTTP ${resp.status}`);
  const mime = resp.headers.get("content-type")?.split(";")[0] || defaultMime;
  const buf = await resp.arrayBuffer();
  let binary = "";
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return `data:${mime};base64,${btoa(binary)}`;
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === "GET_SETTINGS") {
    getSettings().then(sendResponse);
    return true;
  }

  if (message.type === "FETCH_URL_B64") {
    fetchUrlAsBase64(message.url, message.defaultMime)
      .then((data_url) => sendResponse({ ok: true, data_url }))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  if (message.type === "SOLVE_TEXT") {
    getSettings()
      .then((settings) =>
        postJson("/api/v1/solve/text", {
          image_b64: message.image_b64,
          mode: message.mode || settings.mode,
          provider: message.provider || settings.provider,
          eval_math: message.eval_math ?? settings.evalMath
        })
      )
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  if (message.type === "SOLVE_GRID") {
    getSettings()
      .then((settings) =>
        postJson("/api/v1/solve/grid", {
          image_b64: message.image_b64 || null,
          tiles_b64: message.tiles_b64 || null,
          target_label: message.target_label,
          grid_size: message.grid_size || 3,
          provider: message.provider || settings.provider
        })
      )
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  if (message.type === "SOLVE_AUDIO") {
    getSettings()
      .then((settings) =>
        postJson("/api/v1/solve/audio", {
          audio_b64: message.audio_b64,
          provider: message.provider || settings.audioProvider,
          mime_type: message.mime_type || "audio/mpeg"
        })
      )
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  if (message.type === "SOLVE_CLICK") {
    getSettings()
      .then((settings) =>
        postJson("/api/v1/solve/click", {
          image_b64: message.image_b64,
          target_order: message.target_order || null,
          prompt: message.prompt || null,
          mode: message.mode || (settings.mode === "vision_ai" ? "vision_ai" : "local"),
          provider: message.provider || settings.provider
        })
      )
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  if (message.type === "SOLVE_ROTATE") {
    getSettings()
      .then((settings) =>
        postJson("/api/v1/solve/rotate", {
          inner_b64: message.inner_b64,
          outer_b64: message.outer_b64 || null,
          mode: message.mode || (settings.mode === "vision_ai" ? "vision_ai" : "local"),
          provider: message.provider || settings.provider
        })
      )
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  if (message.type === "CHECK_HEALTH") {
    getSettings()
      .then(async (settings) => {
        const url = `${settings.apiBaseUrl.replace(/\/$/, "")}/api/v1/health`;
        const resp = await fetch(url);
        return resp.json();
      })
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  }
});
