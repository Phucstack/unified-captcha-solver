from __future__ import annotations

import io
import sys
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from PIL import Image

from unified_captcha.config import BASE_DIR
from unified_captcha.solvers.onnx_solver import decode_base64_image
from unified_captcha.solvers.pretrained_ocr import PretrainedLocalOCR


@dataclass
class ClickPoint:
    x: int
    y: int
    box: list[int]  # [x1, y1, x2, y2]
    label: str = ""


@dataclass
class ClickSolveResult:
    points: list[dict] = field(default_factory=list)
    backend: str = "opencv_box_detector"


class ClickSolver:
    """Click / Icon / Character point localization solver (ddddocr det + OpenCV fallback)."""

    def __init__(self, ocr: PretrainedLocalOCR | None = None) -> None:
        self._det_engine = None
        self.ocr = ocr or PretrainedLocalOCR()

    def _init_ddddocr_det(self) -> bool:
        if self._det_engine is not None:
            return True
        dddd_root = BASE_DIR / "repos" / "ddddocr"
        if not (dddd_root / "ddddocr" / "common_det.onnx").exists():
            return False
        if str(dddd_root) not in sys.path:
            sys.path.insert(0, str(dddd_root))
        try:
            import ddddocr

            self._det_engine = ddddocr.DdddOcr(det=True, ocr=False, show_ad=False)
            return True
        except Exception:
            self._det_engine = None
            return False

    @staticmethod
    def detect_boxes_opencv(image: Image.Image, min_area: int = 80) -> list[list[int]]:
        """Deterministic contour/foreground detector for glyphs or icons."""
        rgb = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Contrast against median background brightness handles both dark-on-light and light-on-dark
        bg_val = float(np.median(blurred))
        diff = np.abs(blurred.astype(np.float32) - bg_val).astype(np.uint8)
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[list[int]] = []
        img_area = float(rgb.shape[0] * rgb.shape[1])
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > img_area * 0.75:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w >= 8 and h >= 8:
                boxes.append([int(x), int(y), int(x + w), int(y + h)])
        boxes.sort(key=lambda b: (b[0], b[1]))
        return boxes

    def solve_image(
        self,
        image: Image.Image,
        target_order: list[str] | None = None,
        prefer_ddddocr: bool = True,
    ) -> ClickSolveResult:
        boxes: list[list[int]] = []
        backend = "opencv_box_detector"

        if prefer_ddddocr and self._init_ddddocr_det():
            try:
                buf = io.BytesIO()
                image.convert("RGB").save(buf, format="PNG")
                detected = self._det_engine.detection(buf.getvalue())
                if detected:
                    boxes = [[int(v) for v in b[:4]] for b in detected]
                    backend = "ddddocr_det"
            except Exception:
                boxes = []

        if not boxes:
            boxes = self.detect_boxes_opencv(image)
            backend = "opencv_box_detector"

        points: list[ClickPoint] = []
        ocr_ready = self.ocr.is_available() if target_order else False
        for box in boxes:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            label = ""
            if ocr_ready and x2 > x1 and y2 > y1:
                crop = image.crop((max(0, x1 - 2), max(0, y1 - 2), min(image.width, x2 + 2), min(image.height, y2 + 2)))
                try:
                    label = self.ocr.solve_image(crop).text
                except Exception:
                    label = ""
            points.append(ClickPoint(x=cx, y=cy, box=box, label=label))

        if target_order:
            ordered: list[ClickPoint] = []
            remaining = list(points)
            for target in target_order:
                t_low = target.strip().lower()
                idx = next(
                    (i for i, pt in enumerate(remaining) if t_low and t_low in pt.label.lower()),
                    None,
                )
                if idx is not None:
                    ordered.append(remaining.pop(idx))
            ordered.extend(remaining)
            points = ordered
        else:
            points.sort(key=lambda pt: (pt.x, pt.y))

        return ClickSolveResult(
            points=[asdict(pt) for pt in points],
            backend=backend,
        )

    def solve_base64(
        self,
        image_b64: str,
        target_order: list[str] | None = None,
    ) -> ClickSolveResult:
        return self.solve_image(decode_base64_image(image_b64), target_order=target_order)
