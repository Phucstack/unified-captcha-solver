from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from unified_captcha.config import SolverConfig, TrainConfig
from unified_captcha.server.app import create_app
from unified_captcha.solvers.click_solver import ClickSolver
from unified_captcha.solvers.onnx_solver import ONNXTextSolver, safe_eval_math
from unified_captcha.solvers.pretrained_ocr import PretrainedLocalOCR
from unified_captcha.solvers.puzzle_solver import PuzzleSolver, human_drag_trajectory
from unified_captcha.solvers.rotate_solver import RotateSolver
from unified_captcha.training.dataset import generate_dataset, generate_synthetic_captcha
from unified_captcha.training.pretreatment import composite_white_bg
from unified_captcha.training.trainer import evaluate_onnx, train_crnn

EXT_DIR = Path(__file__).resolve().parent.parent / "extension"


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.mark.parametrize("arch", ["crnn", "resnet", "densenet", "mobilenet"])
def test_generate_train_export_validate_and_onnx_solve(tmp_path: Path, arch: str) -> None:
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"

    generate_dataset(data_dir, count=12, length=4, charset="0123456789")
    assert len(list(data_dir.glob("*.png"))) == 12

    cfg = TrainConfig(
        arch=arch,
        charset="0123456789",
        binarize_threshold=140,
        median_ksize=3,
        equalize_hist=True,
        augment=True,
        epochs=1,
        batch_size=6,
        device="cpu",
    )
    result = train_crnn(data_dir, model_name=f"test_{arch}", output_dir=models_dir, cfg=cfg)
    assert Path(result["onnx_path"]).exists()
    meta = json.loads(Path(result["meta_path"]).read_text(encoding="utf-8"))
    assert meta["architecture"] == arch
    assert meta["binarize_threshold"] == 140
    assert meta["median_ksize"] == 3

    solver = ONNXTextSolver(result["onnx_path"], result["meta_path"])
    sample = generate_synthetic_captcha("1234", width=160, height=60)
    pred = solver.solve_base64(_pil_to_b64(sample))
    assert pred.backend == "onnx_local"
    assert pred.model_name == f"test_{arch}"
    assert 0.0 <= pred.confidence <= 1.0

    eval_res = evaluate_onnx(result["onnx_path"], data_dir)
    assert eval_res["total_images"] == 12


def test_pretrained_ocr_puzzle_comparison_click_and_rotate() -> None:
    ocr = PretrainedLocalOCR()
    assert ocr.is_available()
    sample_ocr = generate_synthetic_captcha("789", width=160, height=60, noise_lines=0, noise_dots=0)
    ocr_res = ocr.solve_image(sample_ocr)
    assert ocr_res.backend in ("pretrained_ddddocr", "pretrained_rapidocr")

    # Puzzle two-piece match
    bg = Image.new("RGB", (240, 120), (20, 30, 40))
    draw = ImageDraw.Draw(bg)
    draw.rectangle([90, 35, 130, 75], fill=(220, 180, 60), outline=(255, 255, 255), width=2)
    draw.ellipse([98, 43, 122, 67], fill=(40, 120, 220))
    piece = bg.crop((90, 35, 131, 76))
    solver = PuzzleSolver()
    res = solver.solve_base64(_pil_to_b64(bg), _pil_to_b64(piece))
    assert abs(res.x - 90) <= 2
    assert abs(res.y - 35) <= 2
    assert res.confidence > 0.5
    assert sum(res.drag_steps) == res.x

    # Puzzle comparison mode (full bg vs gap bg)
    gap_bg = bg.copy()
    gap_draw = ImageDraw.Draw(gap_bg)
    gap_draw.rectangle([140, 40, 178, 78], fill=(0, 0, 0))
    cmp_res = solver.solve_comparison(bg, gap_bg)
    assert abs(cmp_res.x - 140) <= 2
    assert abs(cmp_res.y - 40) <= 2
    assert cmp_res.backend == "opencv_comparison"
    assert sum(cmp_res.drag_steps) == cmp_res.x

    # Click solver (two distinct foreground objects on clean canvas)
    click_img = Image.new("RGB", (220, 100), (245, 245, 245))
    cd = ImageDraw.Draw(click_img)
    cd.rectangle([25, 25, 60, 65], fill=(20, 40, 180))
    cd.rectangle([140, 30, 180, 70], fill=(180, 30, 30))
    click_solver = ClickSolver(ocr=ocr)
    click_res = click_solver.solve_image(click_img, prefer_ddddocr=False)
    assert len(click_res.points) == 2
    assert abs(click_res.points[0]["x"] - 42) <= 3
    assert abs(click_res.points[1]["x"] - 160) <= 3

    # Double-ring rotate alignment: textured asymmetric angular pattern
    size = 220
    cx, cy = size // 2, size // 2
    outer = Image.new("RGB", (size, size), (128, 128, 128))
    od = ImageDraw.Draw(outer)
    od.pieslice([10, 10, 210, 210], start=0, end=55, fill=(245, 245, 245))
    od.pieslice([10, 10, 210, 210], start=55, end=140, fill=(20, 20, 20))
    od.pieslice([10, 10, 210, 210], start=140, end=250, fill=(180, 90, 40))
    od.pieslice([10, 10, 210, 210], start=250, end=360, fill=(60, 140, 210))

    r_cut = 70
    orig_inner = outer.crop((cx - r_cut, cy - r_cut, cx + r_cut, cy + r_cut))
    # Rotate counter-clockwise 120 deg -> clockwise recovery needed is 120 deg
    rotated_inner = orig_inner.rotate(120, resample=Image.BILINEAR)
    rot_res = RotateSolver().solve_double_ring(rotated_inner, outer)
    diff_deg = min(abs(rot_res.angle - 120), 360 - abs(rot_res.angle - 120))
    assert diff_deg <= 5, f"Expected ~120 deg, got {rot_res.angle}"


def test_fastapi_server_all_endpoints(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    generate_dataset(data_dir, count=10, length=4, charset="ABCD")
    train_crnn(
        data_dir,
        model_name="api_crnn",
        output_dir=models_dir,
        cfg=TrainConfig(charset="ABCD", epochs=1, batch_size=5, device="cpu"),
    )

    app = create_app(SolverConfig(models_dir=models_dir, default_model_name="api_crnn"))
    client = TestClient(app)

    ui = client.get("/")
    assert ui.status_code == 200
    assert "Unified CAPTCHA Solver Dashboard" in ui.text

    health = client.get("/api/v1/health")
    assert health.status_code == 200
    body = health.json()
    assert "api_crnn" in body["models_loaded"]
    assert body["pretrained_ocr_available"] is True
    for cap in ("text", "puzzle", "puzzle_comparison", "slider", "grid", "click", "rotate", "audio"):
        assert cap in body["capabilities"]

    sample_b64 = _pil_to_b64(generate_synthetic_captcha("ABCD", width=160, height=60))
    solve_resp = client.post(
        "/api/v1/solve/text",
        json={"image_b64": sample_b64, "mode": "local_cnn", "model_name": "api_crnn"},
    )
    assert solve_resp.status_code == 200

    bg = Image.new("RGB", (200, 100), (30, 30, 30))
    d = ImageDraw.Draw(bg)
    d.rectangle([60, 20, 100, 60], fill=(240, 90, 90), outline=(255, 255, 255), width=2)
    piece = bg.crop((60, 20, 101, 61))
    puzzle_resp = client.post(
        "/api/v1/solve/puzzle",
        json={"background_b64": _pil_to_b64(bg), "piece_b64": _pil_to_b64(piece)},
    )
    assert puzzle_resp.status_code == 200
    assert abs(puzzle_resp.json()["x"] - 60) <= 2

    gap_bg = bg.copy()
    ImageDraw.Draw(gap_bg).rectangle([120, 25, 160, 65], fill=(250, 250, 250))
    cmp_resp = client.post(
        "/api/v1/solve/puzzle-comparison",
        json={"full_bg_b64": _pil_to_b64(bg), "gap_bg_b64": _pil_to_b64(gap_bg)},
    )
    assert cmp_resp.status_code == 200
    assert abs(cmp_resp.json()["x"] - 120) <= 2

    click_img = Image.new("RGB", (180, 80), (255, 255, 255))
    ImageDraw.Draw(click_img).rectangle([30, 20, 70, 60], fill=(10, 10, 10))
    click_resp = client.post(
        "/api/v1/solve/click",
        json={"image_b64": _pil_to_b64(click_img), "mode": "local"},
    )
    assert click_resp.status_code == 200
    assert len(click_resp.json()["points"]) >= 1

    outer = Image.new("RGB", (160, 160), (100, 100, 100))
    od = ImageDraw.Draw(outer)
    od.pieslice([10, 10, 150, 150], start=0, end=120, fill=(250, 250, 250))
    od.pieslice([10, 10, 150, 150], start=120, end=240, fill=(20, 20, 20))
    inner_rot = outer.crop((30, 30, 130, 130)).rotate(90, resample=Image.BILINEAR)
    rot_resp = client.post(
        "/api/v1/solve/rotate",
        json={"inner_b64": _pil_to_b64(inner_rot), "outer_b64": _pil_to_b64(outer), "mode": "local"},
    )
    assert rot_resp.status_code == 200
    ang = rot_resp.json()["angle"]
    assert min(abs(ang - 90), 360 - abs(ang - 90)) <= 5

    with patch.object(app.state.router.vision_solver, "_ask_vision", return_value="[0, 4, 8]"):
        grid_resp = client.post(
            "/api/v1/solve/grid",
            json={"image_b64": sample_b64, "target_label": "traffic lights", "grid_size": 3},
        )
        assert grid_resp.status_code == 200
        assert grid_resp.json()["indices"] == [0, 4, 8]

        tile_img = _pil_to_b64(Image.new("RGB", (60, 60), (10, 100, 200)))
        tiles_resp = client.post(
            "/api/v1/solve/grid",
            json={"tiles_b64": [tile_img] * 9, "target_label": "bus", "grid_size": 3},
        )
        assert tiles_resp.status_code == 200
        assert tiles_resp.json()["indices"] == [0, 4, 8]

    with patch.object(app.state.router.vision_solver, "_ask_vision", return_value="134"):
        slider_resp = client.post("/api/v1/solve/slider", json={"image_b64": sample_b64})
        assert slider_resp.status_code == 200
        assert slider_resp.json()["x"] == 134
        assert sum(slider_resp.json()["drag_steps"]) == 134

    with patch.object(
        app.state.router.audio_solver,
        "_transcribe_whisper_local",
        return_value="A B C 1 2 3!",
    ):
        dummy_audio_b64 = base64.b64encode(b"ID3dummy-audio-bytes").decode("ascii")
        audio_resp = client.post(
            "/api/v1/solve/audio",
            json={"audio_b64": dummy_audio_b64, "provider": "whisper_local"},
        )
        assert audio_resp.status_code == 200
        assert audio_resp.json()["text"] == "ABC123"

        audio_file_resp = client.post(
            "/api/v1/solve/audio-file",
            files={"file": ("sample.mp3", io.BytesIO(b"ID3dummy"), "audio/mpeg")},
            data={"provider": "whisper_local"},
        )
        assert audio_file_resp.status_code == 200
        assert audio_file_resp.json()["text"] == "ABC123"

    buf = io.BytesIO()
    generate_synthetic_captcha("ABCD", width=160, height=60).save(buf, format="PNG")
    buf.seek(0)
    file_resp = client.post(
        "/api/v1/solve/text-file",
        files={"file": ("captcha.png", buf, "image/png")},
        data={"mode": "local_cnn", "model_name": "api_crnn"},
    )
    assert file_resp.status_code == 200

    bad_resp = client.post(
        "/api/v1/solve/text",
        json={"image_b64": "not-valid-base64!!!", "mode": "local_cnn"},
    )
    assert bad_resp.status_code == 400


def test_rgba_math_drag_and_extension_integrity() -> None:
    rgba = Image.new("RGBA", (40, 20), (0, 0, 0, 0))
    gray = composite_white_bg(rgba)
    assert gray.mode == "L"
    assert gray.getpixel((0, 0)) == 255

    assert safe_eval_math("12 + 7 = ?") == "19"
    assert safe_eval_math("6 × 8") == "48"
    assert safe_eval_math("__import__('os')") is None

    steps = human_drag_trajectory(150)
    assert sum(steps) == 150

    manifest = json.loads((EXT_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == 3
    for icon_rel in manifest["icons"].values():
        assert (EXT_DIR / icon_rel).exists(), f"Missing icon: {icon_rel}"
    assert (EXT_DIR / manifest["background"]["service_worker"]).exists()
    assert (EXT_DIR / manifest["action"]["default_popup"]).exists()
    for cs in manifest["content_scripts"]:
        for js_rel in cs["js"]:
            assert (EXT_DIR / js_rel).exists(), f"Missing content script: {js_rel}"
