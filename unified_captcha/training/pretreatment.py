from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class PretreatmentConfig:
    """Shared image preprocessing config saved into model metadata JSON."""

    img_width: int = 160
    img_height: int = 60
    binarize_threshold: int = 0  # 0 disables binary thresholding; e.g. 140 enables
    median_ksize: int = 0        # 0 disables; odd >= 3 enables denoising
    equalize_hist: bool = False


def composite_white_bg(image: Image.Image) -> Image.Image:
    """Composite transparent RGBA/LA/P images onto white before grayscale conversion."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = img_to_rgba(image)
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("L")
    return image.convert("L")


def img_to_rgba(image: Image.Image) -> Image.Image:
    return image if image.mode == "RGBA" else image.convert("RGBA")


def preprocess_gray_array(image: Image.Image, cfg: PretreatmentConfig) -> np.ndarray:
    """Apply deterministic pretreatment (white composite, denoise, threshold, normalize)."""
    gray = composite_white_bg(image).resize((cfg.img_width, cfg.img_height), Image.BILINEAR)
    arr = np.asarray(gray, dtype=np.uint8)

    if cfg.equalize_hist:
        arr = cv2.equalizeHist(arr)
    if cfg.median_ksize and cfg.median_ksize >= 3:
        k = cfg.median_ksize if cfg.median_ksize % 2 == 1 else cfg.median_ksize + 1
        arr = cv2.medianBlur(arr, k)
    if cfg.binarize_threshold and 0 < cfg.binarize_threshold < 255:
        _, arr = cv2.threshold(arr, cfg.binarize_threshold, 255, cv2.THRESH_BINARY)

    norm = arr.astype(np.float32) / 255.0
    return (norm - 0.5) / 0.5


def augment_gray_array(arr: np.ndarray, max_rotate_deg: float = 6.0) -> np.ndarray:
    """Lightweight augmentation inspired by captcha_trainer Pretreatment during training."""
    h, w = arr.shape
    out = arr
    if random.random() < 0.5 and max_rotate_deg > 0:
        angle = random.uniform(-max_rotate_deg, max_rotate_deg)
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        out = cv2.warpAffine(out, m, (w, h), borderValue=1.0)
    if random.random() < 0.3:
        noise = np.random.normal(0.0, 0.04, size=out.shape).astype(np.float32)
        out = np.clip(out + noise, -1.0, 1.0)
    return out


def config_to_dict(cfg: PretreatmentConfig) -> dict:
    return asdict(cfg)
