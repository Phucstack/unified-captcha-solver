from __future__ import annotations

import base64
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from unified_captcha.config import SolverConfig


def decode_base64_audio(audio_b64: str) -> bytes:
    if not audio_b64 or not isinstance(audio_b64, str):
        raise ValueError("Empty or invalid base64 audio string")
    if "," in audio_b64 and audio_b64.startswith("data:"):
        audio_b64 = audio_b64.split(",", 1)[1]
    return base64.b64decode(audio_b64)


@dataclass
class AudioSolveResult:
    text: str
    backend: str
    raw_transcript: str = ""


class AudioCaptchaSolver:
    """Audio CAPTCHA solver merging ai-captcha-bypass + nopecha audio challenges."""

    def __init__(self, cfg: SolverConfig | None = None) -> None:
        self.cfg = cfg or SolverConfig()
        self._whisper_model = None

    def is_configured(self, provider: str | None = None) -> bool:
        prov = (provider or self.cfg.audio_provider).lower()
        if prov == "whisper_local":
            try:
                import whisper  # noqa: F401
                return True
            except Exception:
                return False
        if prov == "gemini":
            return bool(self.cfg.gemini_api_key)
        return bool(self.cfg.openai_api_key)

    def _transcribe_whisper_local(self, audio_bytes: bytes, suffix: str = ".mp3") -> str:
        import whisper

        if self._whisper_model is None:
            self._whisper_model = whisper.load_model(self.cfg.whisper_model)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = Path(tmp.name)
        try:
            res = self._whisper_model.transcribe(str(tmp_path))
            return str(res.get("text", "")).strip()
        finally:
            tmp_path.unlink(missing_ok=True)

    def _transcribe_openai(self, audio_bytes: bytes, suffix: str = ".mp3") -> str:
        if not self.cfg.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        from openai import OpenAI

        kwargs = {"api_key": self.cfg.openai_api_key}
        if self.cfg.openai_base_url:
            kwargs["base_url"] = self.cfg.openai_base_url
        client = OpenAI(**kwargs)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = Path(tmp.name)
        try:
            with tmp_path.open("rb") as f:
                resp = client.audio.transcriptions.create(
                    model=self.cfg.openai_audio_model,
                    file=f,
                    prompt="what is the captcha answer?",
                )
            return (resp.text or "").strip()
        finally:
            tmp_path.unlink(missing_ok=True)

    def _transcribe_gemini(self, audio_bytes: bytes, mime_type: str = "audio/mpeg") -> str:
        if not self.cfg.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.cfg.gemini_api_key)
        resp = client.models.generate_content(
            model=self.cfg.gemini_model,
            contents=[
                "Transcribe only the clearly spoken CAPTCHA letters/digits in order.",
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            ],
        )
        return (resp.text or "").strip()

    def solve_audio(
        self,
        audio_b64: str,
        provider: str | None = None,
        mime_type: str = "audio/mpeg",
    ) -> AudioSolveResult:
        audio_bytes = decode_base64_audio(audio_b64)
        prov = (provider or self.cfg.audio_provider).lower()
        suffix = ".aac" if "aac" in mime_type else ".wav" if "wav" in mime_type else ".mp3"

        if prov == "whisper_local":
            raw = self._transcribe_whisper_local(audio_bytes, suffix=suffix)
        elif prov == "gemini":
            raw = self._transcribe_gemini(audio_bytes, mime_type=mime_type)
        else:
            raw = self._transcribe_openai(audio_bytes, suffix=suffix)

        cleaned = re.sub(r"[^A-Za-z0-9]", "", raw)
        return AudioSolveResult(
            text=cleaned or raw,
            backend=f"audio_{prov}",
            raw_transcript=raw,
        )
