import yaml
from pathlib import Path
from ultralytics import YOLO

class QualityInspector:
    """
    A class to handle YOLO model training using an external configuration file.
    """
    
    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        """
        Initializes the inspector by loading configurations from a YAML file.
        
        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config: dict = self._load_config(config_path)
        
        # استخراج المتغيرات (Magic Constants) من ملف الإعدادات
        self.model_name: str = self.config["model"]["name"]
        self.imgsz: int = self.config["model"]["imgsz"]
        self.epochs: int = self.config["model"]["epochs"]
        self.seed: int = self.config["model"]["seed"]
        self.patience: int = self.config["model"]["patience"]
        
        self.model: YOLO = YOLO(self.model_name)

    def _load_config(self, path: str) -> dict:
        """Helper method to load YAML configuration file."""
        config_file_path = Path(path)
        print(f"Trying to load config from: {config_file_path.absolute()}")
        
        if not config_file_path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {config_file_path.absolute()}")
            
        with open(config_file_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
            
        if config_data is None:
            raise ValueError(f"The configuration file at '{config_file_path}' is empty!")
            
        return config_data

    def train_model(self, data_path: str) -> None:
        """
        Trains the YOLO model using parameters loaded from config.yaml.
        """
        print(f"Starting training with model: {self.model_name}...")
        
        self.model.train(
            data=data_path,
            epochs=self.epochs,
            imgsz=self.imgsz,
            seed=self.seed,
            patience=self.patience
        )
        print("Training completed successfully!")