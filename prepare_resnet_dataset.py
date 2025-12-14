import os
import shutil

YOLO_DATA_DIR = r"C:\Users\HP\rooftop_pv_pipeline\data"
RESNET_DATA_DIR = r"C:\Users\HP\rooftop_pv_pipeline\dataset_resnet"
os.makedirs(RESNET_DATA_DIR, exist_ok=True)

splits = ["train", "valid"]  # YOLO splits

for split in splits:
    os.makedirs(os.path.join(RESNET_DATA_DIR, split, "solar"), exist_ok=True)
    os.makedirs(os.path.join(RESNET_DATA_DIR, split, "no_solar"), exist_ok=True)

    images_path = os.path.join(YOLO_DATA_DIR, split, "images")
    labels_path = os.path.join(YOLO_DATA_DIR, split, "labels")

    for img_file in os.listdir(images_path):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(labels_path, label_file)

        # Check if the label file exists
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                content = f.read().strip()
            if content:  # any bounding boxes exist
                dest_folder = os.path.join(RESNET_DATA_DIR, split, "solar")
            else:
                dest_folder = os.path.join(RESNET_DATA_DIR, split, "no_solar")
        else:
            # If no label file, assume no solar
            dest_folder = os.path.join(RESNET_DATA_DIR, split, "no_solar")

        shutil.copy(os.path.join(images_path, img_file), dest_folder)

print("ResNet50 dataset prepared at", RESNET_DATA_DIR)
