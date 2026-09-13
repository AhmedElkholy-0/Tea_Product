from pathlib import Path
from roboflow import Roboflow
from tea_product.model import QualityInspector

def main():
    print("1. Downloading dataset from Roboflow...")
    rf = Roboflow(api_key="wFaxTxTqNOH3vCTbQN2i")
    project = rf.workspace("ahmeds-workspace-y4s4y").project("lipton_task")
    version = project.version(2)
    dataset = version.download("yolov11")
    
    data_yaml_path = "Lipton_Task-2/data.yaml"

    print("2. Initializing Quality Inspector model...")
    
    # الصعود 3 مستويات من src/tea_product/run_train.py للوصول لمجلد المشروع الداخلي ثم configs
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    config_file = BASE_DIR / "configs" / "config.yaml"  # تأكد من اسم الملف yaml أو ymal بناءً على ما اخترته

    inspector = QualityInspector(config_path=str(config_file))

    print("3. Starting the training process...")
    inspector.train_model(data_path=data_yaml_path)

if __name__ == "__main__":
    main()