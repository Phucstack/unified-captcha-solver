(() => {
  let lastImgData = null;
  let busy = false;

  function clickWidgetControls() {
    const selectors = [
      'button[aria-describedby="descriptionVerify"]',
      "#wrong_children_button",
      "#wrongTimeout_children_button"
    ];
    for (const sel of selectors) {
      const btn = document.querySelector(sel);
      if (btn) {
        btn.click();
        return true;
      }
    }
    return false;
  }

  async function solveLoop() {
    if (busy) return;
    busy = true;
    try {
      const settings = await window.UCSUtils.getSettings();
      if (!settings?.autoSolve || !settings.enableFuncaptcha) return;
      if (window.UCSUtils.isHostDisabled(settings.disabledHosts)) return;

      if (clickWidgetControls()) {
        await window.UCSUtils.sleep(Number(settings.solveDelayMs ?? 600));
        return;
      }

      const task = document.querySelector("#game_children_text > h2")?.innerText?.trim();
      const cells = [...document.querySelectorAll("#game_children_challenge ul > li > a")];
      const img = document.querySelector("img#game_challengeItem_image");
      if (!task || cells.length !== 6 || !img?.src) return;

      const dataUrl = img.src.startsWith("data:")
        ? img.src
        : await window.UCSUtils.fetchUrlBase64(img.src, "image/png");
      if (!dataUrl || dataUrl === lastImgData) return;
      lastImgData = dataUrl;

      const resp = await chrome.runtime.sendMessage({
        type: "SOLVE_GRID",
        image_b64: dataUrl,
        target_label: `${task} (6 tiles in 2x3 grid indexed 0..5, pick the matching tile index)`,
        grid_size: 3
      });

      if (resp?.ok && Array.isArray(resp.data?.indices)) {
        await window.UCSUtils.sleep(Number(settings.solveDelayMs ?? 600));
        for (const idx of resp.data.indices) {
          if (cells[idx]) {
            cells[idx].click();
            await window.UCSUtils.sleep(150);
          }
        }
        lastImgData = null;
      }
    } finally {
      busy = false;
    }
  }

  if (window.location.pathname.includes("/fc/")) {
    setInterval(solveLoop, 1500);
  }
})();
