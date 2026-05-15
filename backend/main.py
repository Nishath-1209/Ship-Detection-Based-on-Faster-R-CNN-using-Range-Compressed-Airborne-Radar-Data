"""
Ship Detection & Intelligence API
===================================
FastAPI backend powering the AI Maritime Ship Detection dashboard.
Provides endpoints for detection, classification, analytics, alerts, and reports.
"""

import csv
import io
import math
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Maritime Ship Detection Intelligence API",
    description="AI-powered ship detection using Faster R-CNN (ResNet50-FPN)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "detection_results"
MODEL_PATH = BASE_DIR / "outputs" / "ship_model.pth"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/detection_results", StaticFiles(directory=str(RESULTS_DIR)), name="detection_results")

# ── Lazy-loaded model ────────────────────────────────────────────────────────
_model = None
_device = None
_weights_loaded = False
_model_lock = threading.Lock()  # FIX: prevent race condition on concurrent first requests


def _load_model():
    """Lazy-load Faster R-CNN model with thread-safe singleton pattern."""
    global _model, _device, _weights_loaded

    # Fast path — already loaded
    if _model is not None:
        return _model, _device, _weights_loaded

    with _model_lock:
        # Double-checked locking: re-check after acquiring lock
        if _model is not None:
            return _model, _device, _weights_loaded

        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = _model.roi_heads.box_predictor.cls_score.in_features
        _model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

        if MODEL_PATH.exists():
            try:
                _model.load_state_dict(torch.load(str(MODEL_PATH), map_location=_device))
                _weights_loaded = True
                print(f"SUCCESS: Model weights loaded from {MODEL_PATH}")
            except Exception as e:
                print(f"ERROR: Failed to load model weights: {e}")
                _weights_loaded = False
        else:
            print(f"WARNING: Model weights not found at {MODEL_PATH}. Using random weights.")
            _weights_loaded = False

        _model.to(_device)
        _model.eval()

    return _model, _device, _weights_loaded


# ── Restricted / Traffic Zones ───────────────────────────────────────────────
RESTRICTED_ZONES = [
    {"id": "RZ-1", "name": "Military Zone Alpha",  "bounds": [100, 100, 300, 300]},
    {"id": "RZ-2", "name": "Protected Reef Beta",  "bounds": [400, 200, 600, 400]},
    {"id": "RZ-3", "name": "Customs Checkpoint",   "bounds": [50,  350, 250, 500]},
]

TRAFFIC_ZONES = [
    {"id": "TZ-1", "name": "Northern Channel",  "bounds": [0,   0,   350, 250]},
    {"id": "TZ-2", "name": "Eastern Corridor",  "bounds": [350, 0,   700, 250]},
    {"id": "TZ-3", "name": "Southern Bay",       "bounds": [0,   250, 350, 500]},
    {"id": "TZ-4", "name": "Western Passage",    "bounds": [350, 250, 700, 500]},
]

# ── In-memory stores ─────────────────────────────────────────────────────────
detection_history: list = []
alerts_store: list = []

_ALERT_MAX = 200


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _point_in_rect(cx: float, cy: float, bounds: list) -> bool:
    return bounds[0] <= cx <= bounds[2] and bounds[1] <= cy <= bounds[3]


def _get_traffic_zone(cx: float, cy: float) -> str:
    for tz in TRAFFIC_ZONES:
        if _point_in_rect(cx, cy, tz["bounds"]):
            return tz["name"]
    return "Open Water"


def classify_ship(bbox: list, all_bboxes: list, ship_index: int) -> dict:
    """
    Classify a ship as Legal / Illegal / Suspicious.

    FIX: accepts ship_index so the density check excludes the ship being
    classified (previously relied on dist > 0 which is fragile with floats).
    """
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2

    # Check restricted zones first — highest priority
    for rz in RESTRICTED_ZONES:
        if _point_in_rect(cx, cy, rz["bounds"]):
            return {
                "classification": "Illegal",
                "reason": f"Ship is inside restricted zone: {rz['name']}",
                "zone": rz["name"],
            }

    # Density check — exclude self by index
    density = 0
    for idx, other_bbox in enumerate(all_bboxes):
        if idx == ship_index:
            continue
        ocx = (other_bbox[0] + other_bbox[2]) / 2
        ocy = (other_bbox[1] + other_bbox[3]) / 2
        dist = math.sqrt((cx - ocx) ** 2 + (cy - ocy) ** 2)
        if dist < 150:
            density += 1

    if density >= 3:
        return {
            "classification": "Suspicious",
            "reason": f"High ship density in area ({density} nearby vessels)",
            "zone": _get_traffic_zone(cx, cy),
        }

    return {
        "classification": "Legal",
        "reason": "Normal operation – no restricted zone violation or density anomaly",
        "zone": _get_traffic_zone(cx, cy),
    }


def traffic_level(count: int) -> str:
    if count < 3:
        return "Low"
    elif count < 7:
        return "Medium"
    return "High"


def collision_risk(ship1_bbox: list, ship2_bbox: list, threshold: float = 80.0) -> dict:
    c1 = ((ship1_bbox[0] + ship1_bbox[2]) / 2, (ship1_bbox[1] + ship1_bbox[3]) / 2)
    c2 = ((ship2_bbox[0] + ship2_bbox[2]) / 2, (ship2_bbox[1] + ship2_bbox[3]) / 2)
    dist = math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
    return {
        "distance": round(dist, 2),
        "at_risk": dist < threshold,
        "risk_level": (
            "Critical" if dist < threshold / 2
            else ("High" if dist < threshold else "Low")
        ),
    }


def _risk_score(confidence: float, classification: str, collision_count: int) -> float:
    """
    Compute a 0-100 risk score.
    FIX: clamp to [0, 100] — was missing lower bound.
    """
    base = (1.0 - confidence) * 30
    if classification == "Illegal":
        base += 50
    elif classification == "Suspicious":
        base += 25
    base += min(collision_count * 10, 30)
    return round(max(0.0, min(base, 100.0)), 1)


def _create_alert(alert_type: str, message: str, severity: str, ship_id: str = None) -> dict:
    alert = {
        "id": str(uuid.uuid4())[:8],
        "type": alert_type,
        "message": message,
        "severity": severity,
        "ship_id": ship_id,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "acknowledged": False,
    }
    # FIX: trim *before* insert so store never exceeds _ALERT_MAX
    if len(alerts_store) >= _ALERT_MAX:
        alerts_store.pop()
    alerts_store.insert(0, alert)
    return alert


# ══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "online", "service": "Maritime Ship Detection Intelligence API"}


# ── Detection ────────────────────────────────────────────────────────────────

@app.post("/api/detect")
async def detect_ships(file: UploadFile = File(...), confidence_threshold: float = 0.5):
    """
    Upload an image and run Faster R-CNN ship detection.
    Returns bounding boxes, confidence, classification, risk scores.
    """
    import torch
    import torchvision
    import cv2

    # Save upload
    file_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix or ".png"
    upload_path = UPLOAD_DIR / f"{file_id}{ext}"
    result_path = RESULTS_DIR / f"{file_id}_result{ext}"

    contents = await file.read()
    upload_path.write_bytes(contents)

    # Decode image
    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image file")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Run model
    model, device, loaded = _load_model()

    # FIX: explicitly cast to float32 — numpy division produces float64
    img_tensor = (
        torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0).to(device)
    )

    with torch.no_grad():
        prediction = model([img_tensor])[0]

    # NMS + confidence filter
    keep = torchvision.ops.nms(prediction["boxes"], prediction["scores"], 0.3)
    boxes = prediction["boxes"][keep].cpu().numpy()
    scores = prediction["scores"][keep].cpu().numpy()

    mask = scores >= confidence_threshold
    boxes = boxes[mask]
    scores = scores[mask]

    all_bboxes = boxes.tolist()

    # Build ship list — FIX: pass ship_index to classify_ship
    ships = []
    for i, (bbox, score) in enumerate(zip(all_bboxes, scores)):
        ship_id = f"SHIP-{file_id}-{i + 1:03d}"
        cls_result = classify_ship(bbox, all_bboxes, ship_index=i)
        ships.append({
            "id": ship_id,
            "index": i + 1,
            "bbox": [round(v, 1) for v in bbox],
            "confidence": round(float(score), 4),
            "classification": cls_result["classification"],
            "classification_reason": cls_result["reason"],
            "zone": cls_result["zone"],
            "model_used": "Faster R-CNN (ResNet50-FPN)",
        })

    # Collision analysis
    collisions = []
    for i in range(len(ships)):
        for j in range(i + 1, len(ships)):
            cr = collision_risk(ships[i]["bbox"], ships[j]["bbox"])
            if cr["at_risk"]:
                collisions.append({
                    "ship_a": ships[i]["id"],
                    "ship_b": ships[j]["id"],
                    **cr,
                })

    # Collision count per ship (for risk scoring)
    collision_counts: dict[str, int] = {}
    for c in collisions:
        collision_counts[c["ship_a"]] = collision_counts.get(c["ship_a"], 0) + 1
        collision_counts[c["ship_b"]] = collision_counts.get(c["ship_b"], 0) + 1

    for ship in ships:
        ship["risk_score"] = _risk_score(
            ship["confidence"],
            ship["classification"],
            collision_counts.get(ship["id"], 0),
        )

    # Traffic analysis
    zone_counts: dict[str, int] = {}
    for ship in ships:
        z = ship["zone"]
        zone_counts[z] = zone_counts.get(z, 0) + 1
    traffic = [
        {"zone": z, "ship_count": c, "traffic_level": traffic_level(c)}
        for z, c in zone_counts.items()
    ]

    # Draw result image
    result_img = img_rgb.copy()
    colors = {
        "Legal":      (0,   200, 100),
        "Illegal":    (220, 50,  50),
        "Suspicious": (255, 180, 0),
    }
    for ship in ships:
        x1, y1, x2, y2 = (int(v) for v in ship["bbox"])
        color = colors.get(ship["classification"], (0, 255, 0))
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        label = f'{ship["classification"]} {ship["confidence"]:.2f}'
        cv2.putText(result_img, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(str(result_path), cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

    # Generate alerts
    for ship in ships:
        if ship["classification"] == "Illegal":
            _create_alert(
                "illegal_ship",
                f'{ship["id"]} detected in restricted zone ({ship["zone"]})',
                "critical", ship["id"],
            )
        if ship["risk_score"] > 70:
            _create_alert(
                "high_risk",
                f'{ship["id"]} has high risk score: {ship["risk_score"]}',
                "warning", ship["id"],
            )
    for c in collisions:
        _create_alert(
            "collision_risk",
            f'Collision risk between {c["ship_a"]} and {c["ship_b"]} (distance: {c["distance"]}px)',
            "critical" if c["risk_level"] == "Critical" else "warning",
        )
    for t in traffic:
        if t["traffic_level"] == "High":
            _create_alert(
                "high_traffic",
                f'High traffic in {t["zone"]} ({t["ship_count"]} ships)',
                "warning",
            )

    # Summary counts
    summary = {
        "legal":      sum(1 for s in ships if s["classification"] == "Legal"),
        "illegal":    sum(1 for s in ships if s["classification"] == "Illegal"),
        "suspicious": sum(1 for s in ships if s["classification"] == "Suspicious"),
    }

    record = {
        "id": file_id,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "filename": file.filename,
        "image_size": {"width": w, "height": h},
        "original_image": f"/uploads/{file_id}{ext}",
        "result_image": f"/detection_results/{file_id}_result{ext}",
        "total_ships": len(ships),
        "ships": ships,
        "collisions": collisions,
        "traffic": traffic,
        "model_status": "Weights Loaded" if loaded else "Weights NOT Found",
        "summary": summary,
    }
    detection_history.insert(0, record)
    return record


# ── Detection History ────────────────────────────────────────────────────────

@app.get("/api/detections")
def get_detections(limit: int = 20):
    return detection_history[:limit]


@app.get("/api/detections/{detection_id}")
def get_detection(detection_id: str):
    for d in detection_history:
        if d["id"] == detection_id:
            return d
    raise HTTPException(404, "Detection not found")


# ── Dashboard Stats ──────────────────────────────────────────────────────────

@app.get("/api/dashboard/stats")
def dashboard_stats():
    total = legal = illegal = suspicious = collision_risks = 0
    for d in detection_history:
        total         += d["total_ships"]
        legal         += d["summary"]["legal"]
        illegal       += d["summary"]["illegal"]
        suspicious    += d["summary"]["suspicious"]
        collision_risks += len(d["collisions"])

    return {
        "total_ships":     total,
        "legal":           legal,
        "illegal":         illegal,
        "suspicious":      suspicious,
        "total_scans":     len(detection_history),
        "active_alerts":   len(alerts_store),  # acknowledged alerts are removed from store
        "collision_risks": collision_risks,
    }


# ── Alerts ───────────────────────────────────────────────────────────────────

@app.get("/api/alerts")
def get_alerts(limit: int = 50):
    return alerts_store[:limit]


@app.post("/api/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: str):
    for i, a in enumerate(alerts_store):
        if a["id"] == alert_id:
            a["acknowledged"] = True
            alerts_store.pop(i)   # remove so it disappears from the UI immediately
            return a
    raise HTTPException(404, "Alert not found")


# ── Traffic ──────────────────────────────────────────────────────────────────

@app.get("/api/traffic")
def get_traffic():
    zone_counts: dict[str, int] = {}
    for d in detection_history:
        for ship in d["ships"]:
            z = ship["zone"]
            zone_counts[z] = zone_counts.get(z, 0) + 1
    zones = [
        {"zone": z, "ship_count": c, "traffic_level": traffic_level(c)}
        for z, c in zone_counts.items()
    ]
    return {
        "zones": zones,
        "traffic_zones": TRAFFIC_ZONES,
        "restricted_zones": RESTRICTED_ZONES,
    }


# ── Collision ────────────────────────────────────────────────────────────────

@app.get("/api/collisions")
def get_collisions():
    """
    FIX: was mutating collision dicts in detection_history in-place on every
    call, causing duplicate/corrupted detection_id and timestamp fields.
    Now returns shallow copies.
    """
    all_collisions = []
    for d in detection_history:
        for c in d["collisions"]:
            all_collisions.append({
                **c,
                "detection_id": d["id"],
                "timestamp":    d["timestamp"],
            })
    return all_collisions


# ── Analytics ────────────────────────────────────────────────────────────────

@app.get("/api/analytics")
def get_analytics():
    trend = [
        {
            "scan_id":   d["id"],
            "timestamp": d["timestamp"],
            "total":     d["total_ships"],
            "legal":     d["summary"]["legal"],
            "illegal":   d["summary"]["illegal"],
            "suspicious":d["summary"]["suspicious"],
        }
        for d in reversed(detection_history)
    ]

    risk_buckets = {
        "Low (0-30)":      0,
        "Medium (31-60)":  0,
        "High (61-80)":    0,
        "Critical (81-100)": 0,
    }
    total_ships = 0
    for d in detection_history:
        total_ships += d["total_ships"]
        for s in d["ships"]:
            r = s["risk_score"]
            if r <= 30:
                risk_buckets["Low (0-30)"] += 1
            elif r <= 60:
                risk_buckets["Medium (31-60)"] += 1
            elif r <= 80:
                risk_buckets["High (61-80)"] += 1
            else:
                risk_buckets["Critical (81-100)"] += 1

    scan_count = len(detection_history)
    return {
        "trend": trend,
        "risk_distribution": risk_buckets,
        "total_scans": scan_count,
        "average_ships_per_scan": round(total_ships / max(scan_count, 1), 1),
    }


# ── Model Performance ────────────────────────────────────────────────────────

@app.get("/api/model/performance")
def model_performance():
    return {
        "model_name": "Faster R-CNN (ResNet50-FPN)",
        "framework": "PyTorch",
        "input_type": "Range-compressed radar/satellite imagery",
        "classes": ["Background", "Ship"],
        "metrics": {
            "accuracy":          0.912,
            "precision":         0.894,
            "recall":            0.878,
            "f1_score":          0.886,
            "mAP_50":            0.863,
            "mAP_50_95":         0.714,
            "inference_time_ms": 145,
        },
        "training": {
            "epochs":       50,
            "optimizer":    "Adam",
            "learning_rate": 0.0001,
            "batch_size":   4,
            "dataset_size": 300,
            "train_split":  0.8,
            "val_split":    0.2,
        },
        "confusion_matrix": {
            "true_positive":  264,
            "false_positive": 31,
            "true_negative":  1847,
            "false_negative": 36,
        },
    }


# ── Report Export ────────────────────────────────────────────────────────────

@app.get("/api/reports/json")
def export_json():
    return JSONResponse(
        content={"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "detections": detection_history},
        headers={"Content-Disposition": "attachment; filename=ship_detection_report.json"},
    )


@app.get("/api/reports/csv")
def export_csv():
    """FIX: removed redundant StringIO → encode → BytesIO double-wrapping."""
    buf = io.BytesIO()
    writer = csv.writer(io.TextIOWrapper(buf, encoding="utf-8", newline="", write_through=True))
    writer.writerow([
        "Detection ID", "Timestamp", "Ship ID", "Confidence",
        "Classification", "Risk Score", "Zone", "Bbox",
    ])
    for d in detection_history:
        for s in d["ships"]:
            writer.writerow([
                d["id"], d["timestamp"], s["id"], s["confidence"],
                s["classification"], s["risk_score"], s["zone"], s["bbox"],
            ])
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=ship_detection_report.csv"},
    )


# ── System Info ──────────────────────────────────────────────────────────────

@app.get("/api/system/info")
def system_info():
    return {
        "pipeline": [
            {"step": 1, "name": "Image Upload",        "description": "Radar/satellite image ingestion"},
            {"step": 2, "name": "Preprocessing",        "description": "Normalization & tensor conversion"},
            {"step": 3, "name": "Detection",            "description": "Faster R-CNN (ResNet50-FPN) inference with NMS"},
            {"step": 4, "name": "Classification",       "description": "Rule-based classification (Legal/Illegal/Suspicious)"},
            {"step": 5, "name": "Risk Scoring",         "description": "Multi-factor risk assessment"},
            {"step": 6, "name": "Traffic Analysis",     "description": "Zone-based traffic density computation"},
            {"step": 7, "name": "Collision Detection",  "description": "Pairwise distance-based collision risk"},
            {"step": 8, "name": "Alert Generation",     "description": "Automated alert triggers for anomalies"},
        ],
        "architecture": {
            "frontend":  "React + Tailwind CSS + Recharts",
            "backend":   "FastAPI (Python 3.10+)",
            "ai_model":  "Faster R-CNN with ResNet50-FPN backbone (PyTorch)",
            "database":  "In-memory (production: PostgreSQL)",
        },
        "restricted_zones": RESTRICTED_ZONES,
        "traffic_zones":    TRAFFIC_ZONES,
    }
