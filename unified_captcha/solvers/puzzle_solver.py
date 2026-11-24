from __future__ import annotations

import random
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from unified_captcha.solvers.onnx_solver import decode_base64_image


def human_drag_trajectory(distance: int) -> list[int]:
    """Generate human-like horizontal drag increments (inspired by ai-captcha-bypass)."""
    if distance <= 0:
        return []
    p1 = int(round(distance * random.uniform(0.68, 0.78)))
    p2 = int(round(distance * random.uniform(0.15, 0.24)))
    p3 = distance - p1 - p2
    steps = [s for s in (p1, p2, p3) if s != 0]
    return steps or [distance]


@dataclass
class PuzzleSolveResult:
    x: int
    y: int
    confidence: float
    backend: str = "opencv_edge_match"
    drag_steps: list[int] = field(default_factory=list)


class PuzzleSolver:
    """OpenCV edge-based template matching for slider/puzzle CAPTCHAs."""

    @staticmethod
    def _pil_to_bgr(img: Image.Image) -> np.ndarray:
        rgb = np.asarray(img.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def solve_images(self, background: Image.Image, piece: Image.Image) -> PuzzleSolveResult:
        bg_bgr = self._pil_to_bgr(background)
        piece_bgr = self._pil_to_bgr(piece)

        bg_edges = cv2.Canny(cv2.GaussianBlur(bg_bgr, (3, 3), 0), 80, 200)
        piece_edges = cv2.Canny(cv2.GaussianBlur(piece_bgr, (3, 3), 0), 80, 200)

        bg_edges_rgb = cv2.cvtColor(bg_edges, cv2.COLOR_GRAY2BGR)
        piece_edges_rgb = cv2.cvtColor(piece_edges, cv2.COLOR_GRAY2BGR)

        res = cv2.matchTemplate(bg_edges_rgb, piece_edges_rgb, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        return PuzzleSolveResult(
            x=int(max_loc[0]),
            y=int(max_loc[1]),
            confidence=round(float(max_val), 4),
            drag_steps=human_drag_trajectory(int(max_loc[0])),
        )

    def solve_base64(self, background_b64: str, piece_b64: str) -> PuzzleSolveResult:
        return self.solve_images(
            decode_base64_image(background_b64),
            decode_base64_image(piece_b64),
        )

    def solve_comparison(self, full_bg: Image.Image, gap_bg: Image.Image) -> PuzzleSolveResult:
        """Find missing slider slot by comparing full background and gap background (ddddocr style)."""
        full_rgb = np.asarray(full_bg.convert("RGB"))
        gap_rgb = np.asarray(gap_bg.convert("RGB").resize(full_bg.size, Image.BILINEAR))

        diff = cv2.absdiff(full_rgb, gap_rgb)
        max_diff = diff.max(axis=2)
        _, thresh = cv2.threshold(max_diff, 18, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return PuzzleSolveResult(x=0, y=0, confidence=0.0, backend="opencv_comparison", drag_steps=[])

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        area_ratio = float(cv2.contourArea(largest)) / max(1.0, float(full_rgb.shape[0] * full_rgb.shape[1]))
        conf = min(0.99, max(0.5, area_ratio * 20.0)) if w > 4 and h > 4 else 0.2
        return PuzzleSolveResult(
            x=int(x),
            y=int(y),
            confidence=round(conf, 4),
            backend="opencv_comparison",
            drag_steps=human_drag_trajectory(int(x)),
        )

    def solve_comparison_base64(self, full_bg_b64: str, gap_bg_b64: str) -> PuzzleSolveResult:
        return self.solve_comparison(
            decode_base64_image(full_bg_b64),
            decode_base64_image(gap_bg_b64),
        )
