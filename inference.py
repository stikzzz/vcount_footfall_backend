import cv2
import numpy as np
import os
import random
import time
import datetime

DETECTION_DEVICE = os.environ.get("DETECTION_DEVICE", "cpu")

backend_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(backend_dir)
raw_model_path = os.environ.get("MODEL_PATH", "my_model/my_model.pt")

if os.path.isabs(raw_model_path) and os.path.exists(raw_model_path):
    MODEL_PATH = raw_model_path
elif os.path.exists(os.path.abspath(raw_model_path)):
    MODEL_PATH = os.path.abspath(raw_model_path)
elif os.path.exists(os.path.join(backend_dir, raw_model_path)):
    MODEL_PATH = os.path.join(backend_dir, raw_model_path)
elif os.path.exists(os.path.join(backend_dir, "my_model", "my_model.pt")):
    MODEL_PATH = os.path.join(backend_dir, "my_model", "my_model.pt")
elif os.path.exists(os.path.join(workspace_root, raw_model_path)):
    MODEL_PATH = os.path.join(workspace_root, raw_model_path)
else:
    MODEL_PATH = os.path.join(workspace_root, "my_model", "my_model.pt")

MOCK_MODE = os.environ.get("MOCK_MODE", "").lower() == "true"

CLASS_NAMES = {
    0: "Kids",
    1: "Man",
    2: "Senior Citizen",
    3: "Woman"
}

mock_tracks = {}
mock_next_track_id = 1

models = {}
track_history = {}
counted_ids = {}
track_age = {}
global_counts = {}
last_5min_buckets = {}

def check_5min_counter_reset(camera_id):
    now_dt = datetime.datetime.now()
    curr_bucket = (now_dt.hour, now_dt.minute // 5)
    
    if camera_id not in last_5min_buckets:
        last_5min_buckets[camera_id] = curr_bucket
        return

    if curr_bucket != last_5min_buckets[camera_id]:
        print(f"⏱️ [5-Min Reset] Clock boundary ({now_dt.strftime('%H:%M')}). Resetting demographic counters to 0 for {camera_id}")
        if camera_id in global_counts:
            lane_id = "camera_view"
            global_counts[camera_id][lane_id] = {"Kids": 0, "Man": 0, "Senior Citizen": 0, "Woman": 0}
        if camera_id in counted_ids:
            counted_ids[camera_id].clear()
        if camera_id in track_history:
            track_history[camera_id].clear()
        last_5min_buckets[camera_id] = curr_bucket

def get_model(camera_id):
    global MODEL_PATH, DETECTION_DEVICE
    if camera_id not in models:
        from ultralytics import YOLO
        print(f"⏳ [YOLO] Loading model '{MODEL_PATH}' for camera {camera_id}...")
        try:
            loaded = YOLO(MODEL_PATH)
            models[camera_id] = loaded
            print(f"✅ [YOLO] Model loaded successfully: {loaded.names}")
        except Exception as e:
            print(f"❌ [YOLO] Error loading model '{MODEL_PATH}': {e}")
            raise e

        track_history[camera_id] = {}
        counted_ids[camera_id] = set()
        track_age[camera_id] = {}
        global_counts[camera_id] = {}
    return models[camera_id]

def run_inference(frame, camera_id="AVENUE_FEED", virtual_lines=None):
    global DETECTION_DEVICE, MODEL_PATH, MOCK_MODE

    if camera_id not in counted_ids:
        counted_ids[camera_id] = set()
    if camera_id not in track_history:
        track_history[camera_id] = {}
    if camera_id not in track_age:
        track_age[camera_id] = {}
    if camera_id not in global_counts:
        global_counts[camera_id] = {}

    check_5min_counter_reset(camera_id)

    history = track_history[camera_id]
    counted = counted_ids[camera_id]
    age = track_age[camera_id]
    counts = global_counts[camera_id]
    crossing_events = []
    detections = []

    lane_id = "camera_view"
    if lane_id not in counts:
        counts[lane_id] = {"Kids": 0, "Man": 0, "Senior Citizen": 0, "Woman": 0}

    # --- REAL YOLO INFERENCE WITH TRACKING ID VISUALIZATION ---
    if os.path.exists(MODEL_PATH) and not MOCK_MODE:
        try:
            model = get_model(camera_id)

            for tid in list(age.keys()):
                age[tid] += 1
                if age[tid] > 60:
                    age.pop(tid, None)
                    history.pop(tid, None)
                    counted.discard(tid)

            # Perform persistent object tracking
            results = model.track(frame, persist=True, verbose=False, device=DETECTION_DEVICE)[0]

            # Use YOLO official visualizer which renders bounding boxes with ID labels (e.g. "Man 0.95 ID: 1")
            annotated_frame = results.plot()
            height, width = annotated_frame.shape[:2]

            # Render virtual counting lines if present
            if virtual_lines:
                for idx, v_line in enumerate(virtual_lines):
                    px1, py1 = int(v_line["x1"] * width), int(v_line["y1"] * height)
                    px2, py2 = int(v_line["x2"] * width), int(v_line["y2"] * height)
                    line_label = v_line.get("id", f"Gate {idx+1}")
                    cv2.line(annotated_frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
                    cv2.putText(annotated_frame, line_label, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if results.boxes is not None and len(results.boxes) > 0:
                has_ids = results.boxes.id is not None
                box_list = results.boxes
                id_list = results.boxes.id if has_ids else [None] * len(box_list)

                for i, (box, track_id) in enumerate(zip(box_list, id_list)):
                    cls_idx = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    label = CLASS_NAMES.get(cls_idx, model.names.get(cls_idx, f"Person_{cls_idx}"))

                    # Assign persistent ID or fallback ID
                    tid = int(track_id.item()) if track_id is not None else (i + 1)

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx, cy = int((x1 + x2) / 2), int(y2)

                    # Draw movement trail (Green line + Red point)
                    if tid in history:
                        prev_cx, prev_cy = history[tid]
                        cv2.line(annotated_frame, (prev_cx, prev_cy), (cx, cy), (0, 255, 0), 2)
                        cv2.circle(annotated_frame, (cx, cy), 4, (0, 0, 255), -1)

                    # Register detection / count when first seen
                    if tid not in counted:
                        counted.add(tid)
                        counts[lane_id][label] = counts[lane_id].get(label, 0) + 1
                        crossing_events.append({
                            "lane_id": lane_id,
                            "person_type": label,
                            "confidence": conf,
                            "track_id": tid
                        })

                    # If YOLO plot didn't draw ID label (when boxes.id is None), draw custom overlay
                    if not has_ids:
                        color = (255, 200, 0)
                        if label == "Man": color = (255, 100, 0)
                        elif label == "Woman": color = (200, 0, 255)
                        elif label == "Senior Citizen": color = (0, 255, 100)
                        elif label == "Kids": color = (0, 220, 255)

                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        label_text = f"{label} {conf:.2f} (ID: {tid})"
                        cv2.putText(annotated_frame, label_text, (int(x1), max(15, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    history[tid] = (cx, cy)
                    age[tid] = 0

                    detections.append({
                        "class": label,
                        "confidence": conf,
                        "track_id": tid
                    })

            return {
                "detections": detections,
                "counts": counts,
                "crossing_events": crossing_events,
                "frame": annotated_frame
            }

        except Exception as e:
            print(f"⚠️ Real YOLO inference error: {e}")

    # Fallback annotated frame
    annotated_frame = frame.copy()
    height, width = annotated_frame.shape[:2]

    for tid in list(age.keys()):
        age[tid] += 1
        if age[tid] > 60:
            age.pop(tid, None)
            history.pop(tid, None)
            counted.discard(tid)

    global mock_tracks, mock_next_track_id
    if camera_id not in mock_tracks:
        mock_tracks[camera_id] = []

    active_tracks = []
    for track in mock_tracks[camera_id]:
        track["x"] += track["speed_x"]
        track["y"] += track["speed_y"]

        if track["y"] > height + 50 or track["x"] < -50 or track["x"] > width + 50:
            continue

        track_id = track["id"]
        label = track["class"]
        conf = track["confidence"]
        w, h = track["w"], track["h"]

        x1, y1 = int(track["x"] - w/2), int(track["y"] - h)
        x2, y2 = int(track["x"] + w/2), int(track["y"])
        cx, cy = int(track["x"]), int(track["y"])

        if track_id in history:
            prev_cx, prev_cy = history[track_id]
            cv2.line(annotated_frame, (prev_cx, prev_cy), (cx, cy), (0, 255, 0), 2)
            cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)

        if track_id not in counted:
            counted.add(track_id)
            counts[lane_id][label] = counts[lane_id].get(label, 0) + 1
            crossing_events.append({
                "lane_id": lane_id,
                "person_type": label,
                "confidence": conf,
                "track_id": track_id
            })

        history[track_id] = (cx, cy)
        age[track_id] = 0

        color = (255, 200, 0)
        if label == "Man": color = (255, 100, 0)
        elif label == "Woman": color = (200, 0, 255)
        elif label == "Senior Citizen": color = (0, 255, 100)
        elif label == "Kids": color = (0, 220, 255)

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        label_text = f"{label} {conf:.2f} (ID: {track_id})"
        cv2.putText(annotated_frame, label_text, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        detections.append({
            "class": label,
            "confidence": conf,
            "track_id": track_id
        })
        active_tracks.append(track)

    mock_tracks[camera_id] = active_tracks

    if len(mock_tracks[camera_id]) < 5 and random.random() < 0.18:
        label = random.choices(
            ["Man", "Woman", "Kids", "Senior Citizen"],
            weights=[0.40, 0.35, 0.15, 0.10],
            k=1
        )[0]
        w_factor = 0.08 if label in ["Man", "Woman"] else 0.05
        h_factor = 0.18 if label in ["Man", "Woman", "Senior Citizen"] else 0.12

        w, h = int(width * w_factor), int(height * h_factor)
        spawn_x = random.uniform(0.15 * width, 0.85 * width)
        spawn_y = int(0.10 * height)
        speed_y = random.uniform(2.5, 4.5)
        speed_x = random.uniform(-0.8, 0.8)

        new_track = {
            "id": mock_next_track_id,
            "class": label,
            "x": spawn_x, "y": spawn_y,
            "speed_x": speed_x, "speed_y": speed_y,
            "w": w, "h": h,
            "confidence": random.uniform(0.80, 0.96)
        }
        mock_next_track_id += 1
        mock_tracks[camera_id].append(new_track)

    return {
        "detections": detections,
        "counts": counts,
        "crossing_events": crossing_events,
        "frame": annotated_frame
    }
