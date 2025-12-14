import json
from math import pi
from pathlib import Path

def estimate_area_from_pixels(total_pixels, sqm_per_pixel=0.01):
    """
    total_pixels -> estimated square-meters by multiplying with sqm_per_pixel.
    sqm_per_pixel must be calibrated for the tile size / zoom you requested.
    """
    return total_pixels * sqm_per_pixel

def bbox_area_in_pixels(bbox):
    x1,y1,x2,y2 = bbox
    w = max(0, x2-x1)
    h = max(0, y2-y1)
    return w*h

def save_json_record(out_folder, record):
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_folder) / f"{record['sample_id']}.json"
    with open(out_path, 'w', encoding='utf8') as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return str(out_path)

def qc_decision(resnet_prob, yolo_boxes, img_quality_ok=True, min_conf=0.5):
    """
    Return (qc_status, reason_code)
    Rules:
     - If image quality not OK => NOT_VERIFIABLE (stale/occluded)
     - If resnet_prob < min_conf => NOT_VERIFIABLE (low classifier confidence)
     - If resnet_prob >= min_conf and yolo found boxes => VERIFIABLE
     - If resnet_prob >= min_conf but no yolo boxes => NOT_VERIFIABLE (needs human check)
    """
    if not img_quality_ok:
        return "NOT_VERIFIABLE", "low_image_quality"
    if resnet_prob < min_conf:
        return "NOT_VERIFIABLE", "low_classifier_confidence"
    if len(yolo_boxes) > 0:
        return "VERIFIABLE", "detected_by_yolo"
    return "NOT_VERIFIABLE", "classifier_positive_no_detections"
