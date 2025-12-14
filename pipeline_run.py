import os
import requests
import json
from ultralytics import YOLO
from PIL import Image, ImageDraw
import torch
from torchvision import models, transforms

# ------------------------------
# CONFIGURATION
# ------------------------------
MAPBOX_TOKEN = "pk.eyJ1Ijoic2h1YmhhaGFyaW5pIiwiYSI6ImNtajV1bDljZDFnd3czZXM0d3B6eWh3d2MifQ.jKWp7JP_4ds2uElUBzFdHQ"
IMAGE_FOLDER = "inputs"
OUTPUT_FOLDER = "outputs"
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# List of coordinates to process
coordinates = [
    {"sample_id": 1, "lat": 12.9716, "lon": 77.5946},
    {"sample_id": 2, "lat": 13.0350, "lon": 77.5970},
]

# Load models
resnet_model_path = r"C:\Users\HP\rooftop_pv_pipeline\models\resnet50_best.pth"
yolo_model_path = r"C:\Users\HP\rooftop_pv_pipeline\models\solar_yolo6\weights\best.pt"

# ------------------------------
# Load ResNet50
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model = models.resnet50(pretrained=False)
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, 2)  # binary classification
resnet_model.load_state_dict(torch.load(resnet_model_path, map_location=device))
resnet_model.to(device)
resnet_model.eval()

# Image transform for ResNet50
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load YOLO model
yolo_model = YOLO(yolo_model_path)

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def fetch_mapbox_image(lat, lon, zoom=20, size=512):
    url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},{zoom},0/{size}x{size}?access_token={MAPBOX_TOKEN}"
    response = requests.get(url)
    if response.status_code == 200:
        image_path = os.path.join(IMAGE_FOLDER, f"{lat}_{lon}.png")
        with open(image_path, "wb") as f:
            f.write(response.content)
        return image_path
    else:
        print(f"Failed to fetch image for {lat},{lon}: {response.status_code}")
        return None

def classify_resnet(image_path):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = resnet_model(input_tensor)
        prob = torch.softmax(output, dim=1)[0]
        confidence = float(prob[1])  # solar present class
    return confidence

def detect_yolo(image_path):
    results = yolo_model(image_path)
    return results

def save_audit_image(image_path, results, output_path):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    for r in results[0].boxes.xyxy:
        x1, y1, x2, y2 = r.tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    img.save(output_path)

# ------------------------------
# MAIN PIPELINE
# ------------------------------
all_outputs = []

for item in coordinates:
    sample_id, lat, lon = item["sample_id"], item["lat"], item["lon"]

    # 1. Fetch image
    image_path = fetch_mapbox_image(lat, lon)
    if image_path is None:
        continue

    # 2. ResNet50 classification
    confidence = classify_resnet(image_path)

    # Decision thresholds
    if confidence >= 0.25:
        run_yolo = True
    elif confidence >= 0.15:
        run_yolo = True  # borderline → still try YOLO
    else:
        run_yolo = False

    # Prepare output dictionary
    output = {
        "sample_id": sample_id,
        "lat": lat,
        "lon": lon,
        "has_solar": False,
        "confidence": confidence,
        "pv_area_sqm_est": None,
        "buffer_radius_sqft": None,
        "qc_status": "NOT_VERIFIABLE",
        "bbox_or_mask": None,
        "image_metadata": {"source": "Mapbox", "capture_date": None}
    }

    if run_yolo:
        results = detect_yolo(image_path)
        num_boxes = len(results[0].boxes)

        if num_boxes > 0:
            output["has_solar"] = True
            output["qc_status"] = "VERIFIABLE"

            boxes = results[0].boxes.xyxy.tolist()
            output["bbox_or_mask"] = boxes

            # Pixel-area estimate (rough scaling)
            pixel_area = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes)
            output["pv_area_sqm_est"] = round(pixel_area * 0.01, 2)

            audit_path = os.path.join(OUTPUT_FOLDER, f"{lat}_{lon}_audit.png")
            save_audit_image(image_path, results, audit_path)
        else:
            output["qc_status"] = "NOT_VERIFIABLE"

    all_outputs.append(output)

# Save JSON output
with open(os.path.join(OUTPUT_FOLDER, "results.json"), "w") as f:
    json.dump(all_outputs, f, indent=4)

print("Pipeline finished! Outputs saved in:", OUTPUT_FOLDER)
