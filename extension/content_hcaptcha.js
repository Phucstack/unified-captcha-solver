(() => {
  const HOMOGLYPH_MAP = {
    "0430": "a", "0441": "c", "0501": "d", "0065": "e", "0435": "e",
    "04bb": "h", "0069": "i", "0456": "i", "0458": "j", "03f3": "j",
    "04cf": "l", "03bf": "o", "043e": "o", "0440": "p", "0455": "s",
    "0445": "x", "0443": "y", "0335": "-"
  };
  let lastSig = null;
  let busy = false;

  function normalizeHomoglyphs(text) {
    return [...text]
      .map((ch) => {
        const code = ch.charCodeAt(0).toString(16).padStart(4, "0");
        return HOMOGLYPH_MAP[code] || ch;
      })
      .join("");
  }

  function extractBgUrl(el) {
    const bg = el?.style?.background || getComputedStyle(el)?.backgroundImage || "";
    const match = bg.match(/url\(["']?(.*?)["']?\)/);
    return match?.[1] || null;
  }

  async function solveLoop() {
    if (busy) return;
    busy = true;
    try {
      const settings = await window.UCSUtils.getSettings();
      if (!settings?.autoSolve || !settings.enableHcaptcha) return;
      if (window.UCSUtils.isHostDisabled(settings.disabledHosts)) return;

      const checkEl = document.querySelector("div.check");
      if (checkEl && checkEl.style.display !== "block") {
        document.querySelector("#checkbox")?.click();
        await window.UCSUtils.sleep(Number(settings.solveDelayMs ?? 600));
        return;
      }

      const rawPrompt = document.querySelector("h2.prompt-text")?.innerText?.replace(/\s+/g, " ")?.trim();
      const cells = [...document.querySelectorAll(".task-image")];
      if (!rawPrompt || cells.length !== 9) return;

      const urls = cells.map((c) => extractBgUrl(c.querySelector("div.image")));
      if (urls.some((u) => !u)) return;

      const prompt = normalizeHomoglyphs(rawPrompt);
      const sig = `${prompt}::${urls.join("|")}`;
      if (sig === lastSig) return;
      lastSig = sig;

      const tiles_b64 = await Promise.all(
        urls.map((u) => window.UCSUtils.fetchUrlBase64(u, "image/jpeg"))
      );
      const resp = await chrome.runtime.sendMessage({
        type: "SOLVE_GRID",
        tiles_b64,
        target_label: prompt,
        grid_size: 3
      });

      if (resp?.ok && Array.isArray(resp.data?.indices)) {
        for (const idx of resp.data.indices) {
          if (cells[idx]) {
            cells[idx].click();
            await window.UCSUtils.sleep(150);
          }
        }
        await window.UCSUtils.sleep(Number(settings.solveDelayMs ?? 600));
        document.querySelector(".button-submit")?.click();
      }
    } finally {
      busy = false;
    }
  }

  setInterval(solveLoop, 1500);
})();
