from __future__ import annotations

import base64
from dataclasses import asdict
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import UnidentifiedImageError
from pydantic import BaseModel, Field

from unified_captcha.config import SolverConfig
from unified_captcha.solvers.router import HybridCaptchaRouter


class TextSolveRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded CAPTCHA image")
    mode: Literal["hybrid", "local_cnn", "vision_ai"] = "hybrid"
    model_name: str | None = None
    provider: Literal["openai", "gemini"] | None = None
    eval_math: bool = False


class PuzzleSolveRequest(BaseModel):
    background_b64: str
    piece_b64: str


class PuzzleComparisonRequest(BaseModel):
    full_bg_b64: str
    gap_bg_b64: str


class SliderSolveRequest(BaseModel):
    image_b64: str
    provider: Literal["openai", "gemini"] | None = None


class GridSolveRequest(BaseModel):
    image_b64: str | None = None
    tiles_b64: list[str] | None = None
    target_label: str
    grid_size: int = Field(default=3, ge=2, le=5)
    provider: Literal["openai", "gemini"] | None = None


class ClickSolveRequest(BaseModel):
    image_b64: str
    target_order: list[str] | None = None
    prompt: str | None = None
    mode: Literal["local", "vision_ai"] = "local"
    provider: Literal["openai", "gemini"] | None = None


class RotateSolveRequest(BaseModel):
    inner_b64: str
    outer_b64: str | None = None
    mode: Literal["local", "vision_ai"] = "local"
    provider: Literal["openai", "gemini"] | None = None


class AudioSolveRequest(BaseModel):
    audio_b64: str
    provider: Literal["whisper_local", "openai", "gemini"] | None = None
    mime_type: str = "audio/mpeg"


def create_app(cfg: SolverConfig | None = None) -> FastAPI:
    app = FastAPI(
        title="Unified CAPTCHA Solver API",
        version="1.2.0",
        description="Merges nopecha-extension, ai-captcha-bypass, cnn_captcha, captcha_trainer, ddddocr, and rotate-captcha-crack.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    router = HybridCaptchaRouter(cfg)
    app.state.router = router

    _DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Unified CAPTCHA Solver UI</title>
  <style>
    :root { color-scheme: dark; }
    body { font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 24px; background: #0b1120; color: #e2e8f0; }
    .wrap { max-width: 960px; margin: 0 auto; }
    h1 { margin: 0 0 6px; font-size: 22px; color: #38bdf8; }
    .sub { color: #94a3b8; font-size: 13px; margin-bottom: 18px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
    .card { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 16px; }
    .card h2 { margin: 0 0 10px; font-size: 15px; color: #f8fafc; }
    label { display: block; font-size: 12px; color: #94a3b8; margin-top: 8px; }
    input, select, button { width: 100%; box-sizing: border-box; margin-top: 4px; padding: 8px; border-radius: 6px; border: 1px solid #475569; background: #0f172a; color: #f8fafc; font-size: 13px; }
    button { background: #0284c7; border: 0; font-weight: 600; cursor: pointer; margin-top: 12px; }
    button:hover { background: #0369a1; }
    pre { background: #020617; border: 1px solid #334155; border-radius: 8px; padding: 12px; font-size: 12px; overflow: auto; max-height: 240px; }
    .row { display: flex; align-items: center; gap: 8px; margin-top: 8px; }
    .row input { width: auto; margin: 0; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Unified CAPTCHA Solver Dashboard</h1>
    <div class="sub">Local ONNX (CRNN/ResNet/DenseNet/MobileNet + ddddocr/RapidOCR) · OpenCV Puzzle/Rotate/Click · Whisper Audio · Vision AI</div>
    <div class="grid">
      <div class="card">
        <h2>Text / Math OCR</h2>
        <label>Upload CAPTCHA Image <input id="textFile" type="file" accept="image/*" /></label>
        <label>Mode
          <select id="textMode">
            <option value="hybrid">hybrid</option>
            <option value="local_cnn">local_cnn</option>
            <option value="vision_ai">vision_ai</option>
          </select>
        </label>
        <div class="row"><input id="evalMath" type="checkbox" /><label for="evalMath" style="margin:0;">Evaluate math expression</label></div>
        <button onclick="solveText()">Solve Text</button>
      </div>
      <div class="card">
        <h2>Click / Object Localization</h2>
        <label>Upload Click CAPTCHA Image <input id="clickFile" type="file" accept="image/*" /></label>
        <label>Target Order (comma separated, optional) <input id="clickOrder" type="text" placeholder="A, B, C" /></label>
        <button onclick="solveClick()">Detect Click Points</button>
      </div>
      <div class="card">
        <h2>Rotation Alignment (0-359°)</h2>
        <label>Inner / Rotated Image <input id="rotInner" type="file" accept="image/*" /></label>
        <label>Outer Ring Image (optional) <input id="rotOuter" type="file" accept="image/*" /></label>
        <button onclick="solveRotate()">Estimate Rotation Angle</button>
      </div>
    </div>
    <div class="card" style="margin-top:16px;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <h2 style="margin:0;">Server Health & Output</h2>
        <button style="width:auto;margin:0;padding:6px 12px;" onclick="loadHealth()">Refresh Health</button>
      </div>
      <pre id="out">Loading...</pre>
    </div>
  </div>
  <script>
    const out = document.getElementById("out");
    function toBase64(file) {
      return new Promise((res, rej) => {
        const r = new FileReader();
        r.onload = () => res(r.result);
        r.onerror = rej;
        r.readAsDataURL(file);
      });
    }
    async function callApi(path, body) {
      const resp = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const data = await resp.json();
      out.textContent = JSON.stringify({ status: resp.status, data }, null, 2);
    }
    async function loadHealth() {
      const resp = await fetch("/api/v1/health");
      out.textContent = JSON.stringify(await resp.json(), null, 2);
    }
    async function solveText() {
      const f = document.getElementById("textFile").files[0];
      if (!f) return (out.textContent = "Select an image first.");
      await callApi("/api/v1/solve/text", { image_b64: await toBase64(f), mode: document.getElementById("textMode").value, eval_math: document.getElementById("evalMath").checked });
    }
    async function solveClick() {
      const f = document.getElementById("clickFile").files[0];
      if (!f) return (out.textContent = "Select an image first.");
      const rawOrder = document.getElementById("clickOrder").value.trim();
      const target_order = rawOrder ? rawOrder.split(",").map(s => s.trim()).filter(Boolean) : null;
      await callApi("/api/v1/solve/click", { image_b64: await toBase64(f), target_order, mode: "local" });
    }
    async function solveRotate() {
      const fIn = document.getElementById("rotInner").files[0];
      if (!fIn) return (out.textContent = "Select inner/rotated image first.");
      const fOut = document.getElementById("rotOuter").files[0];
      await callApi("/api/v1/solve/rotate", { inner_b64: await toBase64(fIn), outer_b64: fOut ? await toBase64(fOut) : null, mode: "local" });
    }
    loadHealth();
  </script>
</body>
</html>"""

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return _DASHBOARD_HTML

    @app.get("/api/v1/health")
    def health() -> dict:
        return {
            "status": "ok",
            "models_loaded": router.list_models(),
            "pretrained_ocr_available": router.pretrained_ocr.is_available(),
            "capabilities": [
                "text",
                "puzzle",
                "puzzle_comparison",
                "slider",
                "grid",
                "click",
                "rotate",
                "audio",
            ],
            "vision_configured": {
                "openai": router.vision_solver.is_configured("openai"),
                "gemini": router.vision_solver.is_configured("gemini"),
            },
            "audio_configured": {
                "whisper_local": router.audio_solver.is_configured("whisper_local"),
                "openai": router.audio_solver.is_configured("openai"),
                "gemini": router.audio_solver.is_configured("gemini"),
            },
        }

    @app.get("/api/v1/models")
    def list_models() -> dict:
        return {"models": router.list_models()}

    @app.post("/api/v1/models/reload")
    def reload_models() -> dict:
        return {"models": router.reload_models()}

    @app.post("/api/v1/solve/text")
    def solve_text(req: TextSolveRequest) -> dict:
        try:
            res = router.solve_text(
                image_b64=req.image_b64,
                mode=req.mode,
                model_name=req.model_name,
                provider=req.provider,
                eval_math=req.eval_math,
            )
            return asdict(res)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, UnidentifiedImageError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/v1/solve/text-file")
    async def solve_text_file(
        file: UploadFile = File(...),
        mode: Literal["hybrid", "local_cnn", "vision_ai"] = Form("hybrid"),
        model_name: str | None = Form(None),
        provider: Literal["openai", "gemini"] | None = Form(None),
        eval_math: bool = Form(False),
    ) -> dict:
        try:
            raw = await file.read()
            image_b64 = base64.b64encode(raw).decode("ascii")
            res = router.solve_text(
                image_b64=image_b64,
                mode=mode,
                model_name=model_name,
                provider=provider,
                eval_math=eval_math,
            )
            return asdict(res)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, UnidentifiedImageError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/v1/solve/puzzle")
    def solve_puzzle(req: PuzzleSolveRequest) -> dict:
        try:
            res = router.solve_puzzle(req.background_b64, req.piece_b64)
            return asdict(res)
        except (ValueError, UnidentifiedImageError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/v1/solve/puzzle-comparison")
    def solve_puzzle_comparison(req: PuzzleComparisonRequest) -> dict:
        try:
            res = router.solve_puzzle_comparison(req.full_bg_b64, req.gap_bg_b64)
            return asdict(res)
        except (ValueError, UnidentifiedImageError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/v1/solve/slider")
    def solve_slider(req: SliderSolveRequest) -> dict:
        try:
            res = router.solve_slider_single(req.image_b64, provider=req.provider)
            return asdict(res)
        except (ValueError, UnidentifiedImageError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/v1/solve/grid")
    def solve_grid(req: GridSolveRequest) -> dict:
        try:
            res = router.solve_grid(
                image_b64=req.image_b64,
                target_label=req.target_label,
                grid_size=req.grid_size,
                provider=req.provider,
                tiles_b64=req.tiles_b64,
            )
            return asdict(res)
        except (ValueError, UnidentifiedImageError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/v1/solve/click")
    def solve_click(req: ClickSolveRequest) -> dict:
        try:
            res = router.solve_click(
                image_b64=req.image_b64,
                target_order=req.target_order,
                prompt=req.prompt,
                mode=req.mode,
                provider=req.provider,
            )
            return asdict(res)
        except (ValueError, UnidentifiedImageError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/v1/solve/rotate")
    def solve_rotate(req: RotateSolveRequest) -> dict:
        try:
            res = router.solve_rotate(
                inner_b64=req.inner_b64,
                outer_b64=req.outer_b64,
                mode=req.mode,
                provider=req.provider,
            )
            return asdict(res)
        except (ValueError, UnidentifiedImageError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/v1/solve/audio")
    def solve_audio(req: AudioSolveRequest) -> dict:
        try:
            res = router.solve_audio(
                audio_b64=req.audio_b64,
                provider=req.provider,
                mime_type=req.mime_type,
            )
            return asdict(res)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/v1/solve/audio-file")
    async def solve_audio_file(
        file: UploadFile = File(...),
        provider: Literal["whisper_local", "openai", "gemini"] | None = Form(None),
    ) -> dict:
        try:
            raw = await file.read()
            audio_b64 = base64.b64encode(raw).decode("ascii")
            res = router.solve_audio(
                audio_b64=audio_b64,
                provider=provider,
                mime_type=file.content_type or "audio/mpeg",
            )
            return asdict(res)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()
