from ultralytics import YOLO
import onnx


def export_model(weights_path: str = "runs/detect/train-2/weights/best.pt") -> None:
    """
    Exports a trained YOLO model to ONNX format and validates the result.

    Args:
        weights_path (str): Path to the trained .pt weights file.
                             Defaults to the best weights from the last training run.
    """
    # 1. Load the trained YOLO model weights
    model = YOLO(weights_path)

    # 2. Export to ONNX (ultralytics handles model.eval(), dummy input, dynamic_axes internally)
    onnx_path = model.export(format="onnx"
                             , imgsz=640,
                               dynamic=True,
                               opset=17,
                               nms=True)

    # 3. Verify the exported ONNX file
    checked_model = onnx.load(onnx_path)
    onnx.checker.check_model(checked_model)
    print(f"ONNX model exported and verified: {onnx_path}")


if __name__ == "__main__":
    export_model()