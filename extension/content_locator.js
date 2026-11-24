(() => {
  if (window.__ucsLocatorInstalled) return;
  window.__ucsLocatorInstalled = true;

  let active = false;
  let targetField = "textImageSelector";
  let overlay = null;

  function buildSelector(el) {
    if (!el || el.nodeType !== Node.ELEMENT_NODE) return "";
    if (el.id) return `#${CSS.escape(el.id)}`;
    const parts = [];
    let cur = el;
    while (cur && cur.nodeType === Node.ELEMENT_NODE && cur !== document.body && parts.length < 4) {
      let part = cur.tagName.toLowerCase();
      if (cur.id) {
        parts.unshift(`#${CSS.escape(cur.id)}`);
        break;
      }
      const cls = [...cur.classList].filter((c) => !c.startsWith("ucs-")).slice(0, 2);
      if (cls.length) {
        part += cls.map((c) => `.${CSS.escape(c)}`).join("");
      } else if (cur.parentElement) {
        const siblings = [...cur.parentElement.children].filter((s) => s.tagName === cur.tagName);
        if (siblings.length > 1) {
          part += `:nth-of-type(${siblings.indexOf(cur) + 1})`;
        }
      }
      parts.unshift(part);
      cur = cur.parentElement;
    }
    return parts.join(" > ");
  }

  function ensureOverlay() {
    if (overlay) return overlay;
    overlay = document.createElement("div");
    overlay.style.cssText =
      "position:fixed;pointer-events:none;z-index:2147483647;border:2px solid #38bdf8;background:rgba(56,189,248,0.18);display:none;";
    document.documentElement.appendChild(overlay);
    return overlay;
  }

  function stopPicker() {
    active = false;
    if (overlay) overlay.style.display = "none";
    document.removeEventListener("mousemove", onMove, true);
    document.removeEventListener("click", onPick, true);
    document.removeEventListener("keydown", onEsc, true);
  }

  function onMove(e) {
    if (!active) return;
    const el = document.elementFromPoint(e.clientX, e.clientY);
    if (!el || el === overlay) return;
    const rect = el.getBoundingClientRect();
    const box = ensureOverlay();
    box.style.display = "block";
    box.style.left = `${rect.left}px`;
    box.style.top = `${rect.top}px`;
    box.style.width = `${rect.width}px`;
    box.style.height = `${rect.height}px`;
  }

  async function onPick(e) {
    if (!active) return;
    e.preventDefault();
    e.stopPropagation();
    const el = document.elementFromPoint(e.clientX, e.clientY);
    const selector = buildSelector(el);
    stopPicker();
    if (selector) {
      await chrome.storage.local.set({ [targetField]: selector });
    }
  }

  function onEsc(e) {
    if (e.key === "Escape") stopPicker();
  }

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg?.type === "START_LOCATOR") {
      targetField = msg.targetField || "textImageSelector";
      active = true;
      ensureOverlay();
      document.addEventListener("mousemove", onMove, true);
      document.addEventListener("click", onPick, true);
      document.addEventListener("keydown", onEsc, true);
      sendResponse({ ok: true });
      return true;
    }
  });
})();
