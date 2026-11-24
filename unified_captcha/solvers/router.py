from __future__ import annotations

from pathlib import Path

from unified_captcha.config import SolverConfig
from unified_captcha.solvers.audio_solver import AudioCaptchaSolver, AudioSolveResult
from unified_captcha.solvers.click_solver import ClickSolveResult, ClickSolver
from unified_captcha.solvers.onnx_solver import (
    ONNXTextSolver,
    TextSolveResult,
    decode_base64_image,
    safe_eval_math,
)
from unified_captcha.solvers.pretrained_ocr import PretrainedLocalOCR
from unified_captcha.solvers.puzzle_solver import (
    PuzzleSolveResult,
    PuzzleSolver,
    human_drag_trajectory,
)
from unified_captcha.solvers.rotate_solver import RotateSolveResult, RotateSolver
from unified_captcha.solvers.vision_solver import GridSolveResult, VisionAISolver


class HybridCaptchaRouter:
    """Unified router across Text/OCR, Puzzle/Slider, Grid, Click, Rotate, and Audio CAPTCHAs."""

    def __init__(self, cfg: SolverConfig | None = None) -> None:
        self.cfg = cfg or SolverConfig()
        self.onnx_solvers: dict[str, ONNXTextSolver] = {}
        self.pretrained_ocr = PretrainedLocalOCR()
        self.puzzle_solver = PuzzleSolver()
        self.click_solver = ClickSolver(ocr=self.pretrained_ocr)
        self.rotate_solver = RotateSolver()
        self.vision_solver = VisionAISolver(self.cfg)
        self.audio_solver = AudioCaptchaSolver(self.cfg)
        self.reload_models()

    def reload_models(self) -> list[str]:
        self.onnx_solvers.clear()
        models_dir = Path(self.cfg.models_dir)
        if models_dir.exists():
            for onnx_file in sorted(models_dir.glob("*.onnx")):
                meta_file = onnx_file.with_suffix(".json")
                if meta_file.exists():
                    solver = ONNXTextSolver(onnx_file, meta_file)
                    self.onnx_solvers[solver.model_name] = solver
        return list(self.onnx_solvers.keys())

    def list_models(self) -> list[str]:
        return list(self.onnx_solvers.keys())

    def solve_text(
        self,
        image_b64: str,
        mode: str = "hybrid",
        model_name: str | None = None,
        provider: str | None = None,
        eval_math: bool = False,
    ) -> TextSolveResult:
        mode = mode.lower()
        target_model = model_name or self.cfg.default_model_name

        # Validate image upfront so invalid base64 always fails fast with ValueError
        decode_base64_image(image_b64)

        if mode == "vision_ai":
            res = self.vision_solver.solve_text(image_b64, provider=provider)
        else:
            local_solver = self.onnx_solvers.get(target_model)
            if local_solver is None and self.onnx_solvers:
                local_solver = next(iter(self.onnx_solvers.values()))

            res: TextSolveResult | None = None
            if local_solver is not None:
                res = local_solver.solve_base64(image_b64)

            if (res is None or res.confidence < self.cfg.confidence_threshold) and self.pretrained_ocr.is_available():
                pretrained_res = self.pretrained_ocr.solve_base64(image_b64)
                if res is None or (pretrained_res.text and pretrained_res.confidence > res.confidence):
                    res = pretrained_res

            if res is None:
                if mode == "local_cnn":
                    raise FileNotFoundError(
                        f"No ONNX model loaded in {self.cfg.models_dir} and no pretrained OCR available."
                    )
                res = self.vision_solver.solve_text(image_b64, provider=provider)
            elif (
                mode == "hybrid"
                and res.confidence < self.cfg.confidence_threshold
                and self.vision_solver.is_configured(provider)
            ):
                res = self.vision_solver.solve_text(image_b64, provider=provider)

        if eval_math:
            evaluated = safe_eval_math(res.text)
            if evaluated is not None:
                res = TextSolveResult(
                    text=evaluated,
                    confidence=res.confidence,
                    backend=f"{res.backend}+math",
                    model_name=res.model_name,
                )
        return res

    def solve_puzzle(self, background_b64: str, piece_b64: str) -> PuzzleSolveResult:
        return self.puzzle_solver.solve_base64(background_b64, piece_b64)

    def solve_puzzle_comparison(self, full_bg_b64: str, gap_bg_b64: str) -> PuzzleSolveResult:
        return self.puzzle_solver.solve_comparison_base64(full_bg_b64, gap_bg_b64)

    def solve_slider_single(
        self, image_b64: str, provider: str | None = None
    ) -> PuzzleSolveResult:
        distance = self.vision_solver.solve_slider_distance(image_b64, provider=provider)
        return PuzzleSolveResult(
            x=distance,
            y=0,
            confidence=0.85,
            backend=f"vision_{(provider or self.cfg.vision_provider).lower()}_slider",
            drag_steps=human_drag_trajectory(distance),
        )

    def solve_grid(
        self,
        image_b64: str | None = None,
        target_label: str = "",
        grid_size: int = 3,
        provider: str | None = None,
        tiles_b64: list[str] | None = None,
    ) -> GridSolveResult:
        return self.vision_solver.solve_grid(
            image_b64=image_b64,
            target_label=target_label,
            grid_size=grid_size,
            provider=provider,
            tiles_b64=tiles_b64,
        )

    def solve_click(
        self,
        image_b64: str,
        target_order: list[str] | None = None,
        prompt: str | None = None,
        mode: str = "local",
        provider: str | None = None,
    ) -> ClickSolveResult:
        if mode.lower() == "vision_ai":
            pts = self.vision_solver.solve_click(
                image_b64=image_b64,
                prompt_text=prompt or (", ".join(target_order) if target_order else "click targets"),
                provider=provider,
            )
            return ClickSolveResult(
                points=pts,
                backend=f"vision_{(provider or self.cfg.vision_provider).lower()}_click",
            )
        return self.click_solver.solve_base64(image_b64, target_order=target_order)

    def solve_rotate(
        self,
        inner_b64: str,
        outer_b64: str | None = None,
        mode: str = "local",
        provider: str | None = None,
    ) -> RotateSolveResult:
        if mode.lower() == "vision_ai" and not outer_b64:
            angle = self.vision_solver.solve_rotate(inner_b64, provider=provider)
            return RotateSolveResult(
                angle=angle,
                confidence=0.85,
                backend=f"vision_{(provider or self.cfg.vision_provider).lower()}_rotate",
            )
        return self.rotate_solver.solve_base64(inner_b64, outer_b64=outer_b64)

    def solve_audio(
        self,
        audio_b64: str,
        provider: str | None = None,
        mime_type: str = "audio/mpeg",
    ) -> AudioSolveResult:
        return self.audio_solver.solve_audio(
            audio_b64=audio_b64,
            provider=provider,
            mime_type=mime_type,
        )
