from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from unified_captcha.config import BASE_DIR
from unified_captcha.solvers.onnx_solver import TextSolveResult, decode_base64_image
from unified_captcha.training.pretreatment import composite_white_bg


class PretrainedLocalOCR:
    """Zero-training local ONNX OCR fallback using ddddocr (if present) or RapidOCR."""

    def __init__(self) -> None:
        self._ddddocr_engine = None
        self._rapidocr_engine = None
        self._backend_name = ""

    def _init_ddddocr(self) -> bool:
        if self._ddddocr_engine is not None:
            return True
        dddd_root = BASE_DIR / "repos" / "ddddocr"
        if not (dddd_root / "ddddocr" / "common_old.onnx").exists():
            return False
        if str(dddd_root) not in sys.path:
            sys.path.insert(0, str(dddd_root))
        try:
            import ddddocr

            self._ddddocr_engine = ddddocr.DdddOcr(show_ad=False)
            self._backend_name = "pretrained_ddddocr"
            return True
        except Exception:
            self._ddddocr_engine = None
            return False

    def _init_rapidocr(self) -> bool:
        if self._rapidocr_engine is not None:
            return True
        try:
            from rapidocr_onnxruntime import RapidOCR

            self._rapidocr_engine = RapidOCR()
            self._backend_name = "pretrained_rapidocr"
            return True
        except Exception:
            self._rapidocr_engine = None
            return False

    def is_available(self) -> bool:
        return self._init_ddddocr() or self._init_rapidocr()

    def solve_image(self, image: Image.Image) -> TextSolveResult:
        white_img = composite_white_bg(image).convert("RGB")
        if self._init_ddddocr():
            buf = io.BytesIO()
            white_img.save(buf, format="PNG")
            text = str(self._ddddocr_engine.classification(buf.getvalue())).strip()
            return TextSolveResult(
                text=text,
                confidence=0.8 if text else 0.0,
                backend="pretrained_ddddocr",
                model_name="ddddocr_common_old",
            )

        if self._init_rapidocr():
            arr = np.asarray(white_img)
            result, _ = self._rapidocr_engine(arr)
            if result:
                texts = [str(item[1]).strip() for item in result]
                confs = [float(item[2]) for item in result]
                combined = "".join(texts).replace(" ", "")
                mean_conf = sum(confs) / len(confs) if confs else 0.0
                return TextSolveResult(
                    text=combined,
                    confidence=round(mean_conf, 4),
                    backend="pretrained_rapidocr",
                    model_name="rapidocr_onnx",
                )
            return TextSolveResult(
                text="",
                confidence=0.0,
                backend="pretrained_rapidocr",
                model_name="rapidocr_onnx",
            )

        raise RuntimeError("No pretrained local OCR backend available (neither ddddocr nor rapidocr_onnxruntime).")

    def solve_base64(self, image_b64: str) -> TextSolveResult:
        return self.solve_image(decode_base64_image(image_b64))
