from ultralytics import YOLO
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json

def load_yolo(weights_path):
    return YOLO(weights_path)

def yolo_detect_and_overlay(model, image_path, out_png_path, conf_thresh=0.25, iou=0.45):
    results = model.predict(source=image_path, conf=conf_thresh, iou=iou, save=False, imgsz=1024)
    # results is list-like; get first
    r = results[0]
    boxes = []
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    W, H = img.size
    for det in r.boxes.data.tolist():  # xyxy conf cls
        x1,y1,x2,y2,conf,cls = det
        boxes.append({'bbox':[x1,y1,x2,y2], 'confidence':float(conf), 'class':int(cls)})
        draw.rectangle([x1,y1,x2,y2], outline="red", width=3)
    # Save audit overlay
    Path(out_png_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png_path, quality=90)
    return boxes
