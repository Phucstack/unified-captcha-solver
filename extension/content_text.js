(() => {
  const SOLVED_ATTR = "data-ucs-text-solved";
  let scanTimer = null;

  function findNearbyInput(el, customInputSelector) {
    if (customInputSelector) {
      const custom = document.querySelector(customInputSelector);
      if (custom && !custom.disabled && !custom.readOnly) return custom;
    }
    const form = el.closest("form") || el.parentElement?.parentElement || document;
    const candidates = form.querySelectorAll(
      'input[name*="captcha" i], input[id*="captcha" i], input[placeholder*="captcha" i], input[type="text"], input:not([type])'
    );
    for (const input of candidates) {
      if (input.offsetParent !== null && !input.disabled && !input.readOnly && !input.value) {
        return input;
      }
    }
    return null;
  }

  async function scanAndSolve() {
    const settings = await window.UCSUtils.getSettings();
    if (!settings?.autoSolve || !settings.enableText) return;
    if (window.UCSUtils.isHostDisabled(settings.disabledHosts)) return;

    const selector = settings.textImageSelector?.trim()
      ? settings.textImageSelector
      : 'img[src*="captcha" i], img[alt*="captcha" i], img[id*="captcha" i], img[class*="captcha" i], canvas[id*="captcha" i], canvas[class*="captcha" i]';

    const elements = document.querySelectorAll(selector);
    for (const el of elements) {
      if (el.getAttribute(SOLVED_ATTR) === "1") continue;
      if (el instanceof HTMLImageElement && (!el.complete || el.naturalWidth === 0)) continue;

      const input = findNearbyInput(el, settings.textInputSelector?.trim());
      if (!input) continue;

      el.setAttribute(SOLVED_ATTR, "1");
      try {
        const image_b64 = await window.UCSUtils.elementToBase64(el);
        const resp = await chrome.runtime.sendMessage({ type: "SOLVE_TEXT", image_b64 });
        if (resp?.ok && resp.data?.text) {
          window.UCSUtils.setInputValue(input, resp.data.text);
        } else {
          el.removeAttribute(SOLVED_ATTR);
        }
      } catch (_err) {
        el.removeAttribute(SOLVED_ATTR);
      }
    }
  }

  function scheduleScan(delayMs = 600) {
    if (scanTimer) clearTimeout(scanTimer);
    scanTimer = setTimeout(scanAndSolve, delayMs);
  }

  const observer = new MutationObserver(() => scheduleScan(500));
  observer.observe(document.documentElement, { childList: true, subtree: true });
  window.UCSUtils.getSettings().then((s) => scheduleScan(Number(s?.solveDelayMs ?? 600)));
})();
