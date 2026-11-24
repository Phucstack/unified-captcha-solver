from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("UCS_MODELS_DIR", BASE_DIR / "models"))
DATA_DIR = Path(os.getenv("UCS_DATA_DIR", BASE_DIR / "data"))

DEFAULT_CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@dataclass
class SolverConfig:
    models_dir: Path = MODELS_DIR
    default_model_name: str = os.getenv("UCS_DEFAULT_MODEL", "crnn_default")
    confidence_threshold: float = float(os.getenv("UCS_CONFIDENCE_THRESHOLD", "0.65"))
    vision_provider: str = os.getenv("UCS_VISION_PROVIDER", "openai")  # openai | gemini
    audio_provider: str = os.getenv("UCS_AUDIO_PROVIDER", "whisper_local")  # whisper_local | openai | gemini
    whisper_model: str = os.getenv("UCS_WHISPER_MODEL", "base")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")
    openai_model: str = os.getenv("UCS_OPENAI_MODEL", "gpt-4o-mini")
    openai_audio_model: str = os.getenv("UCS_OPENAI_AUDIO_MODEL", "gpt-4o-transcribe")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("UCS_GEMINI_MODEL", "gemini-2.0-flash")


@dataclass
class TrainConfig:
    arch: str = "crnn"  # crnn | resnet
    charset: str = DEFAULT_CHARSET
    img_width: int = 160
    img_height: int = 60
    binarize_threshold: int = 0
    median_ksize: int = 0
    equalize_hist: bool = False
    augment: bool = False
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    num_workers: int = 0
    device: str = field(default_factory=lambda: "cuda" if _has_cuda() else "cpu")
