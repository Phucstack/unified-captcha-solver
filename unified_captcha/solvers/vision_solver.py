from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass

from PIL import Image, ImageDraw

from unified_captcha.config import SolverConfig
from unified_captcha.solvers.onnx_solver import TextSolveResult, decode_base64_image


def stitch_tiles_grid(tiles_b64: list[str], grid_size: int = 3) -> Image.Image:
    """Stitch individual tile images into a numbered collage for unified Vision classification."""
    tile_imgs = [decode_base64_image(b).convert("RGB").resize((120, 120), Image.BILINEAR) for b in tiles_b64]
    cols = grid_size
    rows = max(1, (len(tile_imgs) + cols - 1) // cols)
    canvas = Image.new("RGB", (cols * 120, rows * 120), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, tile in enumerate(tile_imgs):
        r, c = divmod(idx, cols)
        x0, y0 = c * 120, r * 120
        canvas.paste(tile, (x0, y0))
        draw.rectangle([x0 + 2, y0 + 2, x0 + 24, y0 + 20], fill=(0, 0, 0))
        draw.text((x0 + 6, y0 + 4), str(idx), fill=(255, 255, 255))
    return canvas


@dataclass
class GridSolveResult:
    indices: list[int]
    backend: str
    raw_response: str = ""


def _image_to_base64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class VisionAISolver:
    """Upgraded Vision AI solver inspired by ai-captcha-bypass (OpenAI / Gemini)."""

    def __init__(self, cfg: SolverConfig | None = None) -> None:
        self.cfg = cfg or SolverConfig()

    def is_configured(self, provider: str | None = None) -> bool:
        prov = (provider or self.cfg.vision_provider).lower()
        if prov == "gemini":
            return bool(self.cfg.gemini_api_key)
        return bool(self.cfg.openai_api_key)

    def _call_openai(self, prompt: str, image: Image.Image) -> str:
        if not self.cfg.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        from openai import OpenAI

        kwargs = {"api_key": self.cfg.openai_api_key}
        if self.cfg.openai_base_url:
            kwargs["base_url"] = self.cfg.openai_base_url
        client = OpenAI(**kwargs)
        b64 = _image_to_base64_png(image)
        resp = client.chat.completions.create(
            model=self.cfg.openai_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
            max_tokens=150,
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()

    def _call_gemini(self, prompt: str, image: Image.Image) -> str:
        if not self.cfg.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        from google import genai

        client = genai.Client(api_key=self.cfg.gemini_api_key)
        resp = client.models.generate_content(
            model=self.cfg.gemini_model,
            contents=[prompt, image],
        )
        return (resp.text or "").strip()

    def _ask_vision(self, prompt: str, image: Image.Image, provider: str | None = None) -> str:
        prov = (provider or self.cfg.vision_provider).lower()
        if prov == "gemini":
            return self._call_gemini(prompt, image)
        return self._call_openai(prompt, image)

    def solve_text(self, image_b64: str, provider: str | None = None) -> TextSolveResult:
        img = decode_base64_image(image_b64)
        prompt = (
            "Read the exact characters shown in this CAPTCHA image. "
            "Return ONLY the CAPTCHA text with no spaces, quotes, or explanation."
        )
        raw = self._ask_vision(prompt, img, provider=provider)
        cleaned = re.sub(r"[^A-Za-z0-9]", "", raw)
        return TextSolveResult(
            text=cleaned or raw.strip(),
            confidence=0.9,
            backend=f"vision_{(provider or self.cfg.vision_provider).lower()}",
            model_name=self.cfg.gemini_model
            if (provider or self.cfg.vision_provider).lower() == "gemini"
            else self.cfg.openai_model,
        )

    def solve_grid(
        self,
        image_b64: str | None = None,
        target_label: str = "",
        grid_size: int = 3,
        provider: str | None = None,
        tiles_b64: list[str] | None = None,
    ) -> GridSolveResult:
        if tiles_b64:
            img = stitch_tiles_grid(tiles_b64, grid_size=grid_size)
            total_cells = len(tiles_b64)
        elif image_b64:
            img = decode_base64_image(image_b64)
            total_cells = grid_size * grid_size
        else:
            raise ValueError("Either image_b64 or tiles_b64 must be provided")

        prompt = (
            f"This is a CAPTCHA image grid indexed 0 to {total_cells - 1} "
            f"(left-to-right, top-to-bottom). Select all tile indices containing: '{target_label}'. "
            "Return ONLY a JSON array of integers, e.g. [0, 3, 7]."
        )
        raw = self._ask_vision(prompt, img, provider=provider)
        indices: list[int] = []
        match = re.search(r"\[[\d,\s]*\]", raw)
        if match:
            try:
                parsed = json.loads(match.group(0))
                indices = [int(i) for i in parsed if 0 <= int(i) < total_cells]
            except Exception:
                indices = []
        return GridSolveResult(
            indices=sorted(set(indices)),
            backend=f"vision_{(provider or self.cfg.vision_provider).lower()}",
            raw_response=raw,
        )

    def solve_slider_distance(self, image_b64: str, provider: str | None = None) -> int:
        img = decode_base64_image(image_b64)
        prompt = (
            "Calculate the horizontal pixel distance from the slider puzzle piece to its target slot. "
            "Return ONLY a single non-negative integer."
        )
        raw = self._ask_vision(prompt, img, provider=provider)
        match = re.search(r"\d+", raw)
        if not match:
            raise ValueError(f"Vision AI did not return a valid integer distance: {raw!r}")
        return int(match.group(0))

    def solve_click(
        self,
        image_b64: str,
        prompt_text: str,
        provider: str | None = None,
    ) -> list[dict]:
        img = decode_base64_image(image_b64)
        prompt = (
            f"Locate clickable targets in this {img.width}x{img.height} image for instruction: '{prompt_text}'. "
            'Return ONLY a JSON array of objects in click order, e.g. [{"x": 40, "y": 55, "label": "A"}].'
        )
        raw = self._ask_vision(prompt, img, provider=provider)
        match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
            out: list[dict] = []
            for item in parsed:
                if isinstance(item, dict) and "x" in item and "y" in item:
                    out.append(
                        {
                            "x": int(item["x"]),
                            "y": int(item["y"]),
                            "box": item.get("box", []),
                            "label": str(item.get("label", "")),
                        }
                    )
            return out
        except Exception:
            return []

    def solve_rotate(self, image_b64: str, provider: str | None = None) -> int:
        img = decode_base64_image(image_b64)
        prompt = (
            "Estimate how many clockwise degrees (0 to 359) this CAPTCHA image must be rotated "
            "so the subject stands completely upright. Return ONLY a single integer between 0 and 359."
        )
        raw = self._ask_vision(prompt, img, provider=provider)
        match = re.search(r"\d+", raw)
        if not match:
            raise ValueError(f"Vision AI did not return a valid rotation integer: {raw!r}")
        return int(match.group(0)) % 360
