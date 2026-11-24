from __future__ import annotations

import random
import uuid
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import torch
from torch.utils.data import Dataset

from unified_captcha.config import DEFAULT_CHARSET
from unified_captcha.training.pretreatment import (
    PretreatmentConfig,
    augment_gray_array,
    preprocess_gray_array,
)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "arial.ttf",
        "DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def generate_synthetic_captcha(
    text: str,
    width: int = 160,
    height: int = 60,
    noise_lines: int = 4,
    noise_dots: int = 40,
) -> Image.Image:
    """Generate a synthetic text CAPTCHA image using Pillow (no external captcha pkg required)."""
    bg_color = (
        random.randint(225, 250),
        random.randint(225, 250),
        random.randint(225, 250),
    )
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    font_size = max(22, int(height * 0.58))
    font = _load_font(font_size)

    n = max(len(text), 1)
    step = (width - 20) / n
    for i, ch in enumerate(text):
        x = int(10 + i * step + random.randint(-2, 3))
        y = random.randint(4, max(6, height - font_size - 4))
        color = (
            random.randint(15, 110),
            random.randint(15, 110),
            random.randint(15, 110),
        )
        draw.text((x, y), ch, font=font, fill=color)

    for _ in range(noise_lines):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        line_color = (
            random.randint(80, 180),
            random.randint(80, 180),
            random.randint(80, 180),
        )
        draw.line([(x1, y1), (x2, y2)], fill=line_color, width=random.randint(1, 2))

    for _ in range(noise_dots):
        dx, dy = random.randint(0, width - 1), random.randint(0, height - 1)
        dot_color = (
            random.randint(40, 180),
            random.randint(40, 180),
            random.randint(40, 180),
        )
        draw.point((dx, dy), fill=dot_color)

    if random.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    return img


def generate_dataset(
    out_dir: str | Path,
    count: int = 1000,
    length: int | tuple[int, int] = 4,
    charset: str = DEFAULT_CHARSET,
    width: int = 160,
    height: int = 60,
) -> Path:
    """Generate a labeled synthetic dataset saved as `<label>_<uuid>.png`."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for _ in range(count):
        cur_len = random.randint(length[0], length[1]) if isinstance(length, tuple) else length
        text = "".join(random.choices(charset, k=cur_len))
        img = generate_synthetic_captcha(text, width=width, height=height)
        filename = f"{text}_{uuid.uuid4().hex[:8]}.png"
        img.save(out_path / filename)
    return out_path


def parse_label_from_filename(path: Path) -> str:
    stem = path.stem
    return stem.split("_", 1)[0]


class CaptchaDataset(Dataset):
    """Dataset for variable-length or fixed-length CAPTCHA images (`<label>_<id>.png`)."""

    def __init__(
        self,
        image_dir: str | Path,
        charset: str = DEFAULT_CHARSET,
        img_width: int = 160,
        img_height: int = 60,
        binarize_threshold: int = 0,
        median_ksize: int = 0,
        equalize_hist: bool = False,
        augment: bool = False,
        extensions: Sequence[str] = (".png", ".jpg", ".jpeg", ".webp", ".bmp"),
    ) -> None:
        self.image_dir = Path(image_dir)
        self.charset = charset
        self.char_to_idx = {ch: i + 1 for i, ch in enumerate(charset)}  # 0 reserved for CTC blank
        self.img_width = img_width
        self.img_height = img_height
        self.pretreat_cfg = PretreatmentConfig(
            img_width=img_width,
            img_height=img_height,
            binarize_threshold=binarize_threshold,
            median_ksize=median_ksize,
            equalize_hist=equalize_hist,
        )
        self.augment = augment

        self.samples: list[tuple[Path, list[int]]] = []
        for p in sorted(self.image_dir.iterdir()):
            if p.suffix.lower() not in extensions:
                continue
            label = parse_label_from_filename(p)
            if not label or any(c not in self.char_to_idx for c in label):
                continue
            encoded = [self.char_to_idx[c] for c in label]
            self.samples.append((p, encoded))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        path, target = self.samples[idx]
        with Image.open(path) as img:
            arr = preprocess_gray_array(img, self.pretreat_cfg)
        if self.augment:
            arr = augment_gray_array(arr)
        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
        return tensor, torch.tensor(target, dtype=torch.long), len(target)


def ctc_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, int]]):
    images, targets, lengths = zip(*batch)
    images_tensor = torch.stack(images, dim=0)
    targets_concat = torch.cat(targets, dim=0)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    return images_tensor, targets_concat, lengths_tensor
