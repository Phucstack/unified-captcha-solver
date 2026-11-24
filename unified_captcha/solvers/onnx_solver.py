from __future__ import annotations

import ast
import base64
import io
import json
import operator
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from unified_captcha.training.pretreatment import (
    PretreatmentConfig,
    composite_white_bg,
    preprocess_gray_array,
)


def decode_base64_image(image_b64: str) -> Image.Image:
    if not image_b64 or not isinstance(image_b64, str):
        raise ValueError("Empty or invalid base64 image string")
    if "," in image_b64 and image_b64.startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(image_b64, validate=True)
    except Exception:
        raw = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(raw))
    img.load()
    return img


_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Div: operator.truediv,
}


def safe_eval_math(expr: str) -> str | None:
    """Safely evaluate simple arithmetic CAPTCHAs like '12+7=?' -> '19'."""
    cleaned = expr.replace("×", "*").replace("x", "*").replace("X", "*").replace("÷", "/")
    cleaned = re.sub(r"[^0-9+\-*/]", "", cleaned)
    if not cleaned or not any(op in cleaned for op in "+-*/"):
        return None
    try:
        node = ast.parse(cleaned, mode="eval").body
        def _eval(n: ast.AST) -> float:
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                return float(n.value)
            if isinstance(n, ast.BinOp) and type(n.op) in _SAFE_OPS:
                return _SAFE_OPS[type(n.op)](_eval(n.left), _eval(n.right))
            raise ValueError("Unsupported expression")
        val = _eval(node)
        return str(int(val)) if val.is_integer() else str(round(val, 2))
    except Exception:
        return None


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


@dataclass
class TextSolveResult:
    text: str
    confidence: float
    backend: str
    model_name: str


class ONNXTextSolver:
    """Fast local text CAPTCHA inference via ONNX Runtime."""

    def __init__(self, onnx_path: str | Path, meta_path: str | Path | None = None) -> None:
        self.onnx_path = Path(onnx_path)
        self.meta_path = Path(meta_path) if meta_path else self.onnx_path.with_suffix(".json")
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Model metadata not found: {self.meta_path}")

        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.model_name: str = meta["model_name"]
        self.charset: str = meta["charset"]
        self.img_width: int = int(meta["img_width"])
        self.img_height: int = int(meta["img_height"])
        self.pretreat_cfg = PretreatmentConfig(
            img_width=self.img_width,
            img_height=self.img_height,
            binarize_threshold=int(meta.get("binarize_threshold", 0)),
            median_ksize=int(meta.get("median_ksize", 0)),
            equalize_hist=bool(meta.get("equalize_hist", False)),
        )

        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, image: Image.Image) -> np.ndarray:
        arr = preprocess_gray_array(image, self.pretreat_cfg)
        return arr[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)

    def solve_image(self, image: Image.Image) -> TextSolveResult:
        inp = self.preprocess(image)
        logits = self.session.run(None, {self.input_name: inp})[0]  # (T, 1, C)
        probs = _softmax(logits[:, 0, :], axis=-1)  # (T, C)
        preds = np.argmax(probs, axis=-1)
        max_probs = np.max(probs, axis=-1)

        chars: list[str] = []
        confs: list[float] = []
        prev = 0
        for idx, p in zip(preds.tolist(), max_probs.tolist()):
            if idx != 0 and idx != prev and 1 <= idx <= len(self.charset):
                chars.append(self.charset[idx - 1])
                confs.append(float(p))
            prev = idx

        mean_conf = sum(confs) / len(confs) if confs else 0.0
        return TextSolveResult(
            text="".join(chars),
            confidence=round(mean_conf, 4),
            backend="onnx_local",
            model_name=self.model_name,
        )

    def solve_base64(self, image_b64: str) -> TextSolveResult:
        return self.solve_image(decode_base64_image(image_b64))
