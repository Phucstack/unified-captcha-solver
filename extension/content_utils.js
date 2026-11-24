(() => {
  if (window.UCSUtils) return;

  window.UCSUtils = {
    sleep: (ms) => new Promise((r) => setTimeout(r, ms)),

    getSettings: () => chrome.runtime.sendMessage({ type: "GET_SETTINGS" }),

    isHostDisabled(disabledHosts) {
      if (!disabledHosts) return false;
      const host = window.location.hostname.toLowerCase();
      return disabledHosts
        .split(/[\n,]+/)
        .map((s) => s.trim().toLowerCase())
        .filter(Boolean)
        .some((d) => host === d || host.endsWith(`.${d}`));
    },

    async fetchUrlBase64(url, defaultMime = "image/png") {
      const resp = await chrome.runtime.sendMessage({
        type: "FETCH_URL_B64",
        url,
        defaultMime
      });
      if (!resp?.ok) throw new Error(resp?.error || "Background fetch failed");
      return resp.data_url;
    },

    async elementToBase64(el) {
      if (el instanceof HTMLCanvasElement) {
        return el.toDataURL("image/png");
      }
      if (el instanceof HTMLImageElement) {
        const canvas = document.createElement("canvas");
        canvas.width = el.naturalWidth || el.width || 160;
        canvas.height = el.naturalHeight || el.height || 60;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(el, 0, 0, canvas.width, canvas.height);
        try {
          return canvas.toDataURL("image/png");
        } catch (_cors) {
          if (el.src) return this.fetchUrlBase64(el.src, "image/png");
          throw _cors;
        }
      }
      const bg = getComputedStyle(el).backgroundImage || el.style?.backgroundImage || "";
      const match = bg.match(/url\(["']?(.*?)["']?\)/);
      if (match?.[1]) {
        return this.fetchUrlBase64(match[1], "image/png");
      }
      throw new Error("Unsupported CAPTCHA element");
    },

    setInputValue(input, value) {
      input.focus();
      input.value = value;
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.dispatchEvent(new Event("change", { bubbles: true }));
    }
  };
})();
