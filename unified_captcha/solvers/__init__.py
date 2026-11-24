from .audio_solver import AudioCaptchaSolver, AudioSolveResult
from .click_solver import ClickPoint, ClickSolveResult, ClickSolver
from .onnx_solver import ONNXTextSolver
from .pretrained_ocr import PretrainedLocalOCR
from .puzzle_solver import PuzzleSolveResult, PuzzleSolver
from .rotate_solver import RotateSolveResult, RotateSolver
from .router import HybridCaptchaRouter
from .vision_solver import VisionAISolver

__all__ = [
    "AudioCaptchaSolver",
    "AudioSolveResult",
    "ClickPoint",
    "ClickSolveResult",
    "ClickSolver",
    "ONNXTextSolver",
    "PretrainedLocalOCR",
    "PuzzleSolveResult",
    "PuzzleSolver",
    "RotateSolveResult",
    "RotateSolver",
    "VisionAISolver",
    "HybridCaptchaRouter",
]
