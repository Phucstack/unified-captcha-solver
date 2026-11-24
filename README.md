# Unified CAPTCHA Solver & Trainer (v1.2)

Hệ thống hợp nhất toàn diện từ **6 repository mã nguồn mở**:
1. **`nopecha-extension` (`legacy-oss`)** → [extension/](file:///c:/Users/phucv/Downloads/web%20xem%20phim/unified_captcha_solver/extension/manifest.json): Chrome/Edge Extension MV3 cho **Text/Image/Canvas**, **reCAPTCHA v2/Audio**, **hCaptcha** (chuẩn hóa homoglyph), **FunCaptcha** (6 ô), **AWS WAF Audio**, kèm **CSS Element Locator** (`Pick Img` / `Pick Input`), Math mode và Disabled hosts.
2. **`ai-captcha-bypass`** → Vision AI (`openai` / `gemini`) cho text, grid, click, single-image slider, rotation và Audio fallback.
3. **`cnn_captcha`** → Sinh dữ liệu tổng hợp `<label>_<uuid>.png` và suy luận ONNX.
4. **`captcha_trainer`** → Huấn luyện PyTorch CTC (`crnn`, `resnet`, `densenet`, `mobilenet`), tiền xử lý (`PretreatmentConfig`: composite trắng RGBA, `equalizeHist`, `medianBlur`, `binarize_threshold`, augmentation), xuất `.onnx` + `.json`, kiểm định `evaluate_onnx`.
5. **`ddddocr`** → Tích hợp `PretrainedLocalOCR` (zero-training ONNX OCR qua `ddddocr` / `rapidocr_onnxruntime`), `PuzzleSolver.solve_comparison` (so khớp background gốc vs khuyết), và `ClickSolver` (`common_det.onnx` + OpenCV contour detector).
6. **`rotate-captcha-crack`** → Tích hợp `RotateSolver` (double-ring polar rim correlation `0..359°` và single-image orientation).

## Cài đặt & Chạy nhanh

```bash
# 1. Sinh dữ liệu mẫu & Train (hỗ trợ crnn | resnet | densenet | mobilenet)
py -3.10 -m unified_captcha.cli generate --out data/synthetic --count 500 --length 4
py -3.10 -m unified_captcha.cli train --data data/synthetic --name crnn_default --arch crnn --epochs 20 --augment
py -3.10 -m unified_captcha.cli validate --model models/crnn_default.onnx --data data/synthetic

# 2. Chạy giải trực tiếp qua CLI
py -3.10 -m unified_captcha.cli solve-text sample.png --mode hybrid
py -3.10 -m unified_captcha.cli solve-click click_captcha.png --order A B C
py -3.10 -m unified_captcha.cli solve-rotate inner.png --outer outer.png
py -3.10 -m unified_captcha.cli solve-audio challenge.mp3 --provider whisper_local

# 3. Khởi động FastAPI Server
py -3.10 -m unified_captcha.cli serve --host 127.0.0.1 --port 8000
```
