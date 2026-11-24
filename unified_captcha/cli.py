from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from unified_captcha.config import DATA_DIR, DEFAULT_CHARSET, MODELS_DIR, TrainConfig
from unified_captcha.solvers.router import HybridCaptchaRouter
from unified_captcha.training.dataset import generate_dataset
from unified_captcha.training.trainer import evaluate_onnx, train_crnn


def _file_to_b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified CAPTCHA Solver & Trainer CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    gen_p = sub.add_parser("generate", help="Generate synthetic CAPTCHA dataset")
    gen_p.add_argument("--out", default=str(DATA_DIR / "synthetic"))
    gen_p.add_argument("--count", type=int, default=500)
    gen_p.add_argument("--length", type=int, default=4)
    gen_p.add_argument("--charset", default=DEFAULT_CHARSET)

    train_p = sub.add_parser("train", help="Train CRNN/ResNet/DenseNet/MobileNet model and export to ONNX")
    train_p.add_argument("--data", required=True)
    train_p.add_argument("--name", default="crnn_default")
    train_p.add_argument("--arch", default="crnn", choices=["crnn", "resnet", "densenet", "mobilenet"])
    train_p.add_argument("--out", default=str(MODELS_DIR))
    train_p.add_argument("--epochs", type=int, default=20)
    train_p.add_argument("--batch-size", type=int, default=64)
    train_p.add_argument("--binarize", type=int, default=0)
    train_p.add_argument("--median-ksize", type=int, default=0)
    train_p.add_argument("--augment", action="store_true")
    train_p.add_argument("--charset", default=DEFAULT_CHARSET)

    val_p = sub.add_parser("validate", help="Validate an exported ONNX model on labeled images")
    val_p.add_argument("--model", required=True, help="Path to .onnx file")
    val_p.add_argument("--data", required=True, help="Directory of labeled images")

    serve_p = sub.add_parser("serve", help="Start Unified FastAPI Server")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)

    solve_p = sub.add_parser("solve-text", help="Solve a local CAPTCHA image file")
    solve_p.add_argument("image", help="Path to image file")
    solve_p.add_argument("--mode", default="hybrid", choices=["hybrid", "local_cnn", "vision_ai"])
    solve_p.add_argument("--model", default=None)
    solve_p.add_argument("--math", action="store_true")

    click_p = sub.add_parser("solve-click", help="Locate click coordinates/boxes on a CAPTCHA image")
    click_p.add_argument("image", help="Path to image file")
    click_p.add_argument("--order", nargs="*", default=None, help="Optional target characters/labels in order")
    click_p.add_argument("--mode", default="local", choices=["local", "vision_ai"])

    rot_p = sub.add_parser("solve-rotate", help="Estimate rotation angle (0..359 clockwise) for a CAPTCHA")
    rot_p.add_argument("inner", help="Path to inner/rotated image file")
    rot_p.add_argument("--outer", default=None, help="Optional outer ring image for double-ring alignment")
    rot_p.add_argument("--mode", default="local", choices=["local", "vision_ai"])

    audio_p = sub.add_parser("solve-audio", help="Solve a local audio CAPTCHA file")
    audio_p.add_argument("audio", help="Path to audio file (.mp3/.wav/.aac)")
    audio_p.add_argument("--provider", default="whisper_local", choices=["whisper_local", "openai", "gemini"])

    args = parser.parse_args()

    if args.command == "generate":
        out = generate_dataset(args.out, count=args.count, length=args.length, charset=args.charset)
        print(json.dumps({"generated": args.count, "dir": str(out)}, indent=2))
    elif args.command == "train":
        cfg = TrainConfig(
            arch=args.arch,
            charset=args.charset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            binarize_threshold=args.binarize,
            median_ksize=args.median_ksize,
            augment=args.augment,
        )
        res = train_crnn(args.data, model_name=args.name, output_dir=args.out, cfg=cfg)
        print(json.dumps(res, indent=2))
    elif args.command == "validate":
        res = evaluate_onnx(args.model, args.data)
        print(json.dumps(res, indent=2))
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("unified_captcha.server.app:app", host=args.host, port=args.port, reload=False)
    elif args.command == "solve-text":
        router = HybridCaptchaRouter()
        res = router.solve_text(
            _file_to_b64(args.image),
            mode=args.mode,
            model_name=args.model,
            eval_math=args.math,
        )
        print(json.dumps(res.__dict__, indent=2))
    elif args.command == "solve-click":
        router = HybridCaptchaRouter()
        res = router.solve_click(_file_to_b64(args.image), target_order=args.order, mode=args.mode)
        print(json.dumps(res.__dict__, indent=2))
    elif args.command == "solve-rotate":
        router = HybridCaptchaRouter()
        outer_b64 = _file_to_b64(args.outer) if args.outer else None
        res = router.solve_rotate(_file_to_b64(args.inner), outer_b64=outer_b64, mode=args.mode)
        print(json.dumps(res.__dict__, indent=2))
    elif args.command == "solve-audio":
        router = HybridCaptchaRouter()
        res = router.solve_audio(_file_to_b64(args.audio), provider=args.provider)
        print(json.dumps(res.__dict__, indent=2))


if __name__ == "__main__":
    main()
