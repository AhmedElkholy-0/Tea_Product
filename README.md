# 🍵 Tea Product Quality Inspection API

A computer vision API that detects quality defects in tea product packaging using a fine-tuned YOLO11 model, served through a lightweight FastAPI application powered by ONNX Runtime.

[![Docker Pulls](https://img.shields.io/docker/pulls/ahmed0alkholy/tea-quality-api)](https://hub.docker.com/r/ahmed0alkholy/tea-quality-api)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Features

- **Object detection** on tea product images to flag defects: `Big`, `bad`, `red`, `text`
- **Lightweight inference backend** — runs on `onnxruntime` directly, no PyTorch or Ultralytics required at runtime
- **End-to-end exported model** — NMS and box decoding happen inside the ONNX graph itself
- **Config-driven** — model path, confidence threshold, and class names are read from `configs/config.yaml`, not hardcoded
- **Production-ready Docker image** — multi-stage build, non-root user, small final image size

## Tech Stack

| Layer | Tool |
|---|---|
| API framework | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| Inference | [ONNX Runtime](https://onnxruntime.ai/) |
| Image processing | OpenCV (headless), Pillow |
| Model training (optional) | [Ultralytics YOLO11](https://docs.ultralytics.com/), [Roboflow](https://roboflow.com/) |
| Package management | [uv](https://docs.astral.sh/uv/) |
| Containerization | Docker (multi-stage build) |

## Project Structure

```
Tea_Product/
├── configs/
│   └── config.yaml          # inference & training configuration
├── src/tea_product/
│   ├── api.py                # FastAPI app (onnxruntime inference)
│   ├── model.py               # YOLO training wrapper
│   ├── run_train.py           # training entry point
│   └── export_onnx.py         # PyTorch → ONNX export script
├── runs/detect/train-2/weights/
│   └── best.onnx               # exported model used by the API
├── Dockerfile
├── .dockerignore
├── pyproject.toml
└── uv.lock
```

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- Docker Desktop (for containerized usage)

## Local Development

Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-username>/tea-product.git
cd tea-product
uv sync
```

To also install training dependencies (Ultralytics, Roboflow):

```bash
uv sync --extra training
```

Run the API locally:

```bash
uv run uvicorn src.tea_product.api:app --reload
```

The API will be available at `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs`.

## Getting the Trained Model

The trained model file (`runs/detect/train-2/weights/best.onnx`, ~76 MB) is **not included in this repository** to keep it lightweight — large binary files don't belong in git history.

**If you want to try the API without training anything**, the easiest path is to use the pre-built Docker image, which already has the model baked in:

```bash
docker pull ahmed0alkholy/tea-quality-api
docker run -p 8000:8000 ahmed0alkholy/tea-quality-api
```

**If you cloned this repo and want to build the Docker image yourself**, `docker build` will fail at the `COPY runs/detect/train-2/weights/best.onnx` step unless you provide the model file first. You have two options:

1. Train your own model (see [Training a New Model](#training-a-new-model) below), which will generate `runs/detect/train-2/weights/best.onnx`.
2. Place a `best.onnx` file at that exact path yourself if you already have one.

> A future update may switch this repo to [Git LFS](https://git-lfs.com/) so the model ships with a plain `git clone`. For now, use one of the options above.

## Running with Docker

### Option 1 — Pull the pre-built image from Docker Hub

```bash
docker pull ahmed0alkholy/tea-quality-api
docker run -p 8000:8000 ahmed0alkholy/tea-quality-api
```

### Option 2 — Build the image yourself

```bash
docker build -t tea-quality-api .
docker run -p 8000:8000 tea-quality-api
```

Once running, visit `http://localhost:8000/docs` to try the API.

## API Reference

### `GET /`

Health check endpoint.

**Response**
```json
{
  "status": "Online",
  "message": "Welcome to Tea Product Quality Inspection API!",
  "backend": "onnxruntime"
}
```

### `POST /predict/`

Runs defect detection on an uploaded image.

**Request:** `multipart/form-data` with a `file` field containing the image.

**Response**
```json
{
  "filename": "sample.png",
  "detections": [
    { "class_name": "Big", "confidence": 0.9179 },
    { "class_name": "red", "confidence": 0.8846 },
    { "class_name": "text", "confidence": 0.8806 }
  ]
}
```

## Configuration

Inference settings live in `configs/config.yaml`:

```yaml
inference:
  onnx_path: "runs/detect/train-2/weights/best.onnx"
  imgsz: 640
  conf_threshold: 0.25
  letterbox_color: [114, 114, 114]
  class_names:
    0: "Big"
    1: "bad"
    2: "red"
    3: "text"
```

Adjust `conf_threshold` or `class_names` here — no code changes or image rebuilds required.

## Training a New Model

Training dependencies are kept separate from the runtime image to keep it lightweight. To train a new model:

```bash
uv sync --extra training
uv run python src/tea_product/run_train.py
```

Then export the trained weights to ONNX:

```bash
uv run python src/tea_product/export_onnx.py
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author

**Ahmed Elkholy**
📧 ahmedalkholy715@gmail.com
