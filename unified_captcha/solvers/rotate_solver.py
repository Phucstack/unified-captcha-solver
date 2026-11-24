from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from unified_captcha.solvers.onnx_solver import decode_base64_image


@dataclass
class RotateSolveResult:
    angle: int  # Clockwise degrees in 0..359 needed to align image upright / with outer ring
    confidence: float
    backend: str = "polar_rim_correlation"


class RotateSolver:
    """Rotation CAPTCHA solver for single-circle and double-ring alignment challenges."""

    @staticmethod
    def _polar_ring_strip(gray: np.ndarray, r_start: int, r_end: int, max_r: int) -> np.ndarray:
        h, w = gray.shape[:2]
        center = (w / 2.0, h / 2.0)
        polar = cv2.warpPolar(
            gray,
            (max_r, 360),
            center,
            max_r,
            cv2.WARP_POLAR_LINEAR | cv2.WARP_FILL_OUTLIERS,
        )
        r0 = max(0, min(max_r - 1, r_start))
        r1 = max(r0 + 1, min(max_r, r_end))
        return polar[:, r0:r1].astype(np.float32)

    def solve_double_ring(self, inner: Image.Image, outer: Image.Image) -> RotateSolveResult:
        """Estimate clockwise angle (0..359) to rotate `inner` to seamlessly match `outer`."""
        inner_gray = np.asarray(inner.convert("L"))
        outer_gray = np.asarray(outer.convert("L"))

        r_inner = min(inner_gray.shape[0], inner_gray.shape[1]) // 2
        max_r_outer = min(outer_gray.shape[0], outer_gray.shape[1]) // 2
        band = max(4, min(10, r_inner // 6))

        inner_strip = self._polar_ring_strip(
            inner_gray,
            r_start=max(1, r_inner - band - 2),
            r_end=max(2, r_inner - 2),
            max_r=r_inner,
        )
        outer_strip = self._polar_ring_strip(
            outer_gray,
            r_start=min(max_r_outer - 2, r_inner + 2),
            r_end=min(max_r_outer, r_inner + band + 2),
            max_r=max_r_outer,
        )

        # Compare mean radial intensity & angular gradient across all 360 clockwise shifts
        s_in = inner_strip.mean(axis=1)
        s_out = outer_strip.mean(axis=1)

        s_in_norm = (s_in - s_in.mean()) / (s_in.std() + 1e-6)
        s_out_norm = (s_out - s_out.mean()) / (s_out.std() + 1e-6)

        best_angle = 0
        best_score = -1e9
        for shift in range(360):
            # Shifting polar rows down by +shift corresponds to rotating inner clockwise by +shift deg
            shifted = np.roll(s_in_norm, shift)
            score = float(np.mean(shifted * s_out_norm))
            if score > best_score:
                best_score = score
                best_angle = shift

        conf = round(max(0.0, min(1.0, (best_score + 1.0) / 2.0)), 4)
        return RotateSolveResult(
            angle=int(best_angle % 360),
            confidence=conf,
            backend="polar_rim_correlation",
        )

    def solve_single(self, image: Image.Image) -> RotateSolveResult:
        """Estimate upright rotation angle (0..359) for a single rotated CAPTCHA via edge orientation."""
        gray = np.asarray(image.convert("L"))
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        # Build 360-bin weighted orientation histogram and find vertical/horizontal structural axis
        hist = np.zeros(90, dtype=np.float32)
        bins = (np.floor(ang) % 90).astype(np.int32)
        np.add.at(hist, bins, mag)
        dominant_axis = int(np.argmax(hist))
        # Closest clockwise correction within 0..89 -> upright
        correction = (360 - dominant_axis) % 360
        conf = round(float(hist[dominant_axis] / (hist.sum() + 1e-6)) * 10.0, 4)
        return RotateSolveResult(
            angle=correction,
            confidence=min(1.0, max(0.0, conf)),
            backend="opencv_gradient_orientation",
        )

    def solve_base64(
        self,
        inner_b64: str,
        outer_b64: str | None = None,
    ) -> RotateSolveResult:
        inner_img = decode_base64_image(inner_b64)
        if outer_b64:
            return self.solve_double_ring(inner_img, decode_base64_image(outer_b64))
        return self.solve_single(inner_img)
