(() => {
  let lastAudio = null;
  let busy = false;

  async function solveLoop() {
    if (busy) return;
    busy = true;
    try {
      const settings = await window.UCSUtils.getSettings();
      if (!settings?.autoSolve || !settings.enableAwsCaptcha) return;
      if (window.UCSUtils.isHostDisabled(settings.disabledHosts)) return;

      const verifyBtn = document.querySelector("#captcha-container > #root #amzn-captcha-verify-button");
      if (verifyBtn) {
        verifyBtn.click();
        return;
      }

      const switchAudioBtn = document.querySelector('#captcha-container > #root #amzn-btn-audio-internal > img[title="Audio problem"]');
      if (switchAudioBtn) {
        document.querySelector("#captcha-container > #root #amzn-btn-audio-internal")?.click();
        return;
      }

      const input = document.querySelector('input[placeholder="Answer"]');
      const audioSrc = document.querySelector("audio")?.src;
      if (!input || input.value || !audioSrc || audioSrc === lastAudio) return;

      lastAudio = audioSrc;
      const resp = await chrome.runtime.sendMessage({
        type: "SOLVE_AUDIO",
        audio_b64: audioSrc,
        mime_type: "audio/aac"
      });
      if (resp?.ok && resp.data?.text) {
        window.UCSUtils.setInputValue(input, resp.data.text);
        await window.UCSUtils.sleep(Number(settings.solveDelayMs ?? 600));
        document.querySelector("#amzn-btn-verify-internal")?.click();
      } else {
        document.querySelector("#amzn-btn-refresh-internal")?.click();
        lastAudio = null;
      }
    } finally {
      busy = false;
    }
  }

  setInterval(solveLoop, 1500);
})();
