from ultralytics import YOLO
import os

def train_yolo():
    data_yaml = r"C:\Users\HP\rooftop_pv_pipeline\data_yolo\data.yaml"

    os.makedirs("models", exist_ok=True)

    model = YOLO("yolov8n.pt")

    model.train(
        data=data_yaml,
        epochs=6,
        imgsz=640,
        batch=8,
        device="cpu",
        workers=4,
        name="solar_yolo",
        project="models",
        plots=False
    )

    print("\nTraining complete!")
    print("Best model saved at models/solar_yolo/weights/best.pt")

if __name__ == "__main__":
    train_yolo()
