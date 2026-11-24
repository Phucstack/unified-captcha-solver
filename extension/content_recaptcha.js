(() => {
  let lastSignature = null;
  let busy = false;

  function isSolved() {
    const cb = document.querySelector(".recaptcha-checkbox");
    if (cb?.getAttribute("aria-checked") === "true") return true;
    const verify = document.querySelector("#recaptcha-verify-button");
    return Boolean(verify?.disabled);
  }

  async function solveLoop() {
    if (busy) return;
    busy = true;
    try {
      const settings = await window.UCSUtils.getSettings();
      if (!settings?.autoSolve || !settings.enableRecaptcha) return;
      if (window.UCSUtils.isHostDisabled(settings.disabledHosts)) return;
      if (isSolved()) return;

      const anchor = document.querySelector("#recaptcha-anchor");
      if (anchor && document.querySelector(".recaptcha-checkbox")) {
        await window.UCSUtils.sleep(Number(settings.solveDelayMs ?? 600));
        anchor.click();
        return;
      }

      const audioSrc = document.querySelector("#audio-source")?.src;
      const audioInput = document.querySelector("#audio-response");
      if (audioSrc && audioInput && !audioInput.value) {
        if (lastSignature === audioSrc) return;
        lastSignature = audioSrc;
        const audio_b64 = await window.UCSUtils.fetchUrlBase64(audioSrc, "audio/mpeg");
        const resp = await chrome.runtime.sendMessage({
          type: "SOLVE_AUDIO",
          audio_b64,
          mime_type: "audio/mpeg"
        });
        if (resp?.ok && resp.data?.text) {
          window.UCSUtils.setInputValue(audioInput, resp.data.text);
          await window.UCSUtils.sleep(300);
          document.querySelector("#recaptcha-verify-button")?.click();
        }
        return;
      }

      if (!document.querySelector("#rc-imageselect")) return;
      const instr = document.querySelector(".rc-imageselect-instructions")?.innerText?.replace(/\s+/g, " ")?.trim();
      const cells = [...document.querySelectorAll("table tr td")];
      if (!instr || (cells.length !== 9 && cells.length !== 16)) return;

      const imgs = cells.map((c) => c.querySelector("img")).filter(Boolean);
      if (imgs.length !== cells.length || imgs.some((im) => !im.src)) return;

      const sig = `${instr}::${imgs.map((i) => i.src).join("|")}`;
      if (sig === lastSignature) return;
      lastSignature = sig;

      const gridSize = cells.length === 16 ? 4 : 3;
      const fullBgImg = imgs.find((im) => im.naturalWidth >= 300);
      let payload;
      if (fullBgImg) {
        payload = {
          type: "SOLVE_GRID",
          image_b64: await window.UCSUtils.fetchUrlBase64(fullBgImg.src, "image/jpeg"),
          target_label: instr,
          grid_size: gridSize
        };
      } else {
        const tiles_b64 = await Promise.all(
          imgs.map((im) => window.UCSUtils.fetchUrlBase64(im.src, "image/jpeg"))
        );
        payload = {
          type: "SOLVE_GRID",
          tiles_b64,
          target_label: instr,
          grid_size: gridSize
        };
      }

      const resp = await chrome.runtime.sendMessage(payload);
      if (resp?.ok && Array.isArray(resp.data?.indices)) {
        for (const idx of resp.data.indices) {
          if (cells[idx]) {
            cells[idx].click();
            await window.UCSUtils.sleep(150);
          }
        }
        await window.UCSUtils.sleep(Number(settings.solveDelayMs ?? 600));
        document.querySelector("#recaptcha-verify-button")?.click();
      }
    } finally {
      busy = false;
    }
  }

  setInterval(solveLoop, 1500);
})();
