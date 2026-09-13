import io
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import yaml
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field
from PIL import Image

# ==========================================
# 1. Load configuration from config.yaml
# ==========================================

# src/tea_product/api.py -> project root (3 levels up), same pattern as run_train.py
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"


def load_config(path: Path) -> dict:
    """Loads and validates the YAML configuration file."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path.absolute()}")

    with open(path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    if config_data is None:
        raise ValueError(f"The configuration file at '{path}' is empty!")

    return config_data


_config = load_config(CONFIG_PATH)
_inference_config = _config["inference"]

MODEL_PATH = str(BASE_DIR / _inference_config["onnx_path"])
INPUT_SIZE = _inference_config["imgsz"]
CONF_THRESHOLD = _inference_config["conf_threshold"]
LETTERBOX_COLOR = tuple(_inference_config["letterbox_color"])
CLASS_NAMES = {int(k): v for k, v in _inference_config["class_names"].items()}
# No IOU_THRESHOLD / manual NMS needed: this model was exported as an
# end-to-end detector, so duplicate-box removal already happens inside it.

app = FastAPI(
    title="Tea Quality Inspection API",
    description="API for detecting and inspecting tea product quality using YOLO11 (ONNX Runtime)",
    version="3.4",
)

# 2. Load the ONNX model once when the server starts
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name


# ==========================================
# 3. Pydantic models for validation
# ==========================================

class DetectionItem(BaseModel):
    class_name: str = Field(..., description="Name of the detected class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score, between 0 and 1")


class TeaPredictionResponse(BaseModel):
    filename: str = Field("img", description="Name of the uploaded image file")
    detections: List[DetectionItem] = Field(..., description="List of objects detected in the image")


# ==========================================
# 4. Preprocessing / Postprocessing functions
# ==========================================

def letterbox(image: np.ndarray, size: int = INPUT_SIZE) -> Tuple[np.ndarray, float, int, int]:
    """
    Resizes the image while preserving its aspect ratio, and pads the
    remaining space with gray borders.

    Args:
        image (np.ndarray): The original image in BGR format.
        size (int): Target size (a size x size square).

    Returns:
        Tuple: The processed image, the scale factor, horizontal padding, vertical padding.
    """
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(image, (new_w, new_h))

    pad_w = size - new_w
    pad_h = size - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=LETTERBOX_COLOR
    )
    return padded, scale, left, top


def preprocess(image_bytes: bytes) -> Tuple[np.ndarray, float, int, int, Tuple[int, int]]:
    """
    Converts the uploaded image bytes into the input format the model expects.

    Returns:
        Tuple: the ready input tensor, scale, pad_x, pad_y, (original height, original width).
    """
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = np.array(pil_image)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    original_shape = image_bgr.shape[:2]  # (h, w)

    padded, scale, pad_x, pad_y = letterbox(image_bgr, INPUT_SIZE)

    # BGR -> RGB, normalize [0,1], HWC -> CHW, add batch dimension
    img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[np.newaxis, :]

    return img, scale, pad_x, pad_y, original_shape


def postprocess(
    output: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    original_shape: Tuple[int, int],
) -> List[DetectionItem]:
    """
    Parses the model's output. This model was exported as an end-to-end
    detector, so NMS and box decoding are already done inside the model
    itself. The raw output has shape (1, max_dets, 6), where each row is
    [x1, y1, x2, y2, confidence, class_id], padded with zero-rows up to
    max_dets.

    Args:
        output (np.ndarray): Model output with shape (1, max_dets, 6).
        scale, pad_x, pad_y: Values produced by letterbox (kept for future
            use if box coordinates need to be returned/rescaled).
        original_shape: The original (height, width) of the image before any resizing.

    Returns:
        List[DetectionItem]: Final list of detections above the confidence threshold.
    """
    detections_raw = output[0]  # shape: (max_dets, 6)

    detections: List[DetectionItem] = []
    for x1, y1, x2, y2, conf, cls_id in detections_raw:
        if conf < CONF_THRESHOLD:
            continue
        detections.append(
            DetectionItem(class_name=CLASS_NAMES.get(int(cls_id), str(int(cls_id))), confidence=float(conf))
        )

    return detections


# ==========================================
# 5. Endpoints
# ==========================================

@app.get("/")
def home():
    return {
        "status": "Online",
        "message": "Welcome to Tea Product Quality Inspection API!",
        "backend": "onnxruntime",
    }


@app.post("/predict/", response_model=TeaPredictionResponse)
async def predict_tea(file: UploadFile = File(...)):
    image_bytes = await file.read()

    input_tensor, scale, pad_x, pad_y, original_shape = preprocess(image_bytes)

    outputs = session.run(None, {input_name: input_tensor})
    raw_output = outputs[0]  # shape: (1, max_dets, 6) -> [x1, y1, x2, y2, conf, class_id]

    detections = postprocess(raw_output, scale, pad_x, pad_y, original_shape)

    return {"filename": file.filename, "detections": detections}