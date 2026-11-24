from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from unified_captcha.config import MODELS_DIR, TrainConfig
from unified_captcha.training.dataset import CaptchaDataset, ctc_collate_fn, parse_label_from_filename
from unified_captcha.training.models import build_model
from unified_captcha.training.pretreatment import PretreatmentConfig


def ctc_greedy_decode(logits: torch.Tensor, charset: str) -> list[tuple[str, float]]:
    """Decode (T, B, C) CTC logits to [(text, mean_confidence), ...]."""
    probs = F.softmax(logits, dim=-1)
    max_probs, preds = probs.max(dim=-1)
    preds = preds.transpose(0, 1).cpu().tolist()
    max_probs = max_probs.transpose(0, 1).cpu().tolist()

    results: list[tuple[str, float]] = []
    for seq, seq_probs in zip(preds, max_probs):
        chars: list[str] = []
        confs: list[float] = []
        prev = 0
        for idx, p in zip(seq, seq_probs):
            if idx != 0 and idx != prev and 1 <= idx <= len(charset):
                chars.append(charset[idx - 1])
                confs.append(float(p))
            prev = idx
        mean_conf = sum(confs) / len(confs) if confs else 0.0
        results.append(("".join(chars), mean_conf))
    return results


def export_onnx(
    model: nn.Module,
    output_dir: str | Path,
    model_name: str,
    cfg: TrainConfig,
) -> tuple[Path, Path]:
    """Export trained PyTorch model to ONNX + metadata JSON (including pretreatment)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{model_name}.onnx"
    meta_path = out_dir / f"{model_name}.json"

    model_cpu = model.to("cpu").eval()
    dummy = torch.randn(1, 1, cfg.img_height, cfg.img_width, dtype=torch.float32)

    torch.onnx.export(
        model_cpu,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {1: "batch"}},
        opset_version=17,
    )

    meta = {
        "model_name": model_name,
        "architecture": cfg.arch,
        "charset": cfg.charset,
        "img_width": cfg.img_width,
        "img_height": cfg.img_height,
        "binarize_threshold": cfg.binarize_threshold,
        "median_ksize": cfg.median_ksize,
        "equalize_hist": cfg.equalize_hist,
        "num_classes": len(cfg.charset) + 1,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return onnx_path, meta_path


def train_crnn(
    data_dir: str | Path,
    model_name: str = "crnn_default",
    output_dir: str | Path = MODELS_DIR,
    cfg: TrainConfig | None = None,
) -> dict:
    """Train CRNN or ResNet-CTC on a labeled image folder and export ONNX + metadata."""
    cfg = cfg or TrainConfig()
    dataset = CaptchaDataset(
        data_dir,
        charset=cfg.charset,
        img_width=cfg.img_width,
        img_height=cfg.img_height,
        binarize_threshold=cfg.binarize_threshold,
        median_ksize=cfg.median_ksize,
        equalize_hist=cfg.equalize_hist,
        augment=cfg.augment,
    )
    if len(dataset) == 0:
        raise ValueError(f"No valid labeled CAPTCHA images found in {data_dir}")

    val_size = max(1, int(len(dataset) * 0.1)) if len(dataset) >= 10 else 0
    train_size = len(dataset) - val_size
    if val_size > 0:
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
    else:
        train_ds, val_ds = dataset, None

    train_loader = DataLoader(
        train_ds,
        batch_size=min(cfg.batch_size, max(1, len(train_ds))),
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=ctc_collate_fn,
    )

    device = torch.device(cfg.device)
    model = build_model(cfg.arch, num_classes=len(cfg.charset) + 1).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    last_loss = 0.0
    for _epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for images, targets, target_lengths in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            logits = model(images)
            log_probs = F.log_softmax(logits, dim=-1)
            input_lengths = torch.full(
                (images.size(0),), logits.size(0), dtype=torch.long, device=device
            )

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1
        last_loss = total_loss / max(steps, 1)

    val_acc = None
    if val_ds is not None:
        model.eval()
        val_loader = DataLoader(
            val_ds,
            batch_size=min(cfg.batch_size, len(val_ds)),
            shuffle=False,
            collate_fn=ctc_collate_fn,
        )
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets, target_lengths in val_loader:
                logits = model(images.to(device))
                decoded = ctc_greedy_decode(logits, cfg.charset)
                offset = 0
                for (pred_text, _), length in zip(decoded, target_lengths.tolist()):
                    gt_indices = targets[offset : offset + length].tolist()
                    offset += length
                    gt_text = "".join(cfg.charset[i - 1] for i in gt_indices)
                    if pred_text == gt_text:
                        correct += 1
                    total += 1
        val_acc = correct / max(total, 1)

    onnx_path, meta_path = export_onnx(model, output_dir, model_name, cfg)
    return {
        "model_name": model_name,
        "architecture": cfg.arch,
        "samples": len(dataset),
        "epochs": cfg.epochs,
        "final_loss": round(last_loss, 4),
        "val_accuracy": round(val_acc, 4) if val_acc is not None else None,
        "onnx_path": str(onnx_path),
        "meta_path": str(meta_path),
        "config": asdict(cfg),
    }


def evaluate_onnx(onnx_path: str | Path, data_dir: str | Path) -> dict:
    """Validate an exported ONNX model against a folder of `<label>_<id>.png` images."""
    from unified_captcha.solvers.onnx_solver import ONNXTextSolver

    solver = ONNXTextSolver(onnx_path)
    img_dir = Path(data_dir)
    correct = 0
    total = 0
    char_correct = 0
    char_total = 0

    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            continue
        gt = parse_label_from_filename(p)
        if not gt:
            continue
        with Image.open(p) as im:
            res = solver.solve_image(im)
        total += 1
        if res.text == gt:
            correct += 1
        for c_pred, c_gt in zip(res.text, gt):
            if c_pred == c_gt:
                char_correct += 1
        char_total += max(len(gt), len(res.text))

    return {
        "model_name": solver.model_name,
        "total_images": total,
        "exact_match_accuracy": round(correct / max(total, 1), 4),
        "char_accuracy": round(char_correct / max(char_total, 1), 4),
    }
