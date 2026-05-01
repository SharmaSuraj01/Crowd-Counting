from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
import base64
import time
import os

app = FastAPI()
model = YOLO('yolov8n.pt')

# YOLO class IDs we care about
OBJECT_CLASSES = {
    0:  ('Person',     (0, 255, 0)),
    2:  ('Car',        (0, 165, 255)),
    3:  ('Motorcycle', (255, 0, 255)),
    5:  ('Bus',        (255, 255, 0)),
    7:  ('Truck',      (0, 128, 255)),
}

ALERT_THRESHOLD = 10

def get_density_level(count):
    if count == 0:     return "Empty"
    elif count <= 5:   return "Low"
    elif count <= 20:  return "Medium"
    elif count <= 50:  return "High"
    else:              return "Very High"

def detect_gender(img, x1, y1, x2, y2):
    try:
        h, w = img.shape[:2]
        box_h = y2 - y1
        box_w = x2 - x1
        # Tight face crop: top 28%, slight horizontal inset
        fy1 = max(0, y1)
        fy2 = min(h, y1 + int(box_h * 0.28))
        fx1 = max(0, x1 + int(box_w * 0.1))
        fx2 = min(w, x2 - int(box_w * 0.1))
        face_crop = img[fy1:fy2, fx1:fx2]
        if face_crop.size == 0 or face_crop.shape[0] < 40 or face_crop.shape[1] < 40:
            return "Unknown"
        result = DeepFace.analyze(face_crop, actions=['gender'],
                                  enforce_detection=True, detector_backend='opencv', silent=True)
        gender_scores = result[0]['gender']
        man_score = gender_scores['Man']
        woman_score = gender_scores['Woman']
        # Require strong confidence gap to avoid wrong predictions
        if man_score > 72 and man_score > woman_score + 20:
            return 'Male'
        elif woman_score > 72 and woman_score > man_score + 20:
            return 'Female'
        return 'Unknown'
    except:
        return "Unknown"

def generate_heatmap(img, boxes):
    h, w = img.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    for (x1, y1, x2, y2) in boxes:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
        heat[max(0, cy-bh):min(h, cy+bh), max(0, cx-bw):min(w, cx+bw)] += 1

    if heat.max() > 0:
        heat = heat / heat.max()

    ksize = max(51, (min(h, w) // 10))
    ksize = ksize if ksize % 2 == 1 else ksize + 1  # GaussianBlur needs odd ksize
    heat = cv2.GaussianBlur(heat, (ksize, ksize), 0)
    heatmap_color = cv2.applyColorMap(np.uint8(heat * 255), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.5, heatmap_color, 0.5, 0)

    for i, (label, color) in enumerate([("Low", (0,255,0)), ("Medium", (0,255,255)), ("High", (0,128,255)), ("Critical", (0,0,255))]):
        cv2.rectangle(overlay, (10, 10 + i*22), (30, 28 + i*22), color, -1)
        cv2.putText(overlay, label, (35, 24 + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    return overlay

def process_frame(img, conf_threshold=0.45):
    start_time = time.time()
    results = model(img, verbose=False, conf=conf_threshold)
    processing_time = round((time.time() - start_time) * 1000)

    person_count = 0
    male_count = 0
    female_count = 0
    total_confidence = 0.0
    person_boxes = []
    object_counts = {}
    annotated_img = img.copy()

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls not in OBJECT_CLASSES:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            obj_name, color = OBJECT_CLASSES[cls]

            if cls == 0:  # Person
                person_count += 1
                total_confidence += conf
                person_boxes.append((x1, y1, x2, y2))

                gender = detect_gender(img, x1, y1, x2, y2)
                if gender == 'Male':
                    male_count += 1
                    color = (255, 100, 0)    # blue for male
                elif gender == 'Female':
                    female_count += 1
                    color = (0, 100, 255)    # pink for female

                label = f'#{person_count} {gender} {conf:.0%}'
            else:
                object_counts[obj_name] = object_counts.get(obj_name, 0) + 1
                label = f'{obj_name} {conf:.0%}'

            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated_img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    if person_count > ALERT_THRESHOLD:
        cv2.rectangle(annotated_img, (0, 0), (annotated_img.shape[1], 50), (0, 0, 200), -1)
        cv2.putText(annotated_img, f'CROWD ALERT: {person_count} people detected!',
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    heatmap_img = generate_heatmap(img, person_boxes)
    avg_confidence = round((total_confidence / person_count) * 100) if person_count > 0 else 0

    return annotated_img, heatmap_img, person_count, male_count, female_count, object_counts, avg_confidence, processing_time

def encode_image(img):
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/webcam-frame")
async def webcam_frame(file: UploadFile = File(...), threshold: int = 10):
    try:
        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid frame")
        # Resize if too large for faster processing
        h, w = img.shape[:2]
        if w > 640:
            img = cv2.resize(img, (640, int(h * 640 / w)))
        global ALERT_THRESHOLD
        ALERT_THRESHOLD = threshold
        annotated_img, _, person_count, male_count, female_count, object_counts, avg_confidence, processing_time = process_frame(img)
        return JSONResponse({
            "person_count": person_count,
            "male_count": male_count,
            "female_count": female_count,
            "object_counts": object_counts,
            "avg_confidence": avg_confidence,
            "density_level": get_density_level(person_count),
            "processing_time_ms": processing_time,
            "alert": person_count > ALERT_THRESHOLD,
            "output_image": encode_image(annotated_img)
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_image(file: UploadFile = File(...), threshold: int = 10):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        global ALERT_THRESHOLD
        ALERT_THRESHOLD = threshold

        annotated_img, heatmap_img, person_count, male_count, female_count, object_counts, avg_confidence, processing_time = process_frame(img)

        return JSONResponse({
            "person_count": person_count,
            "male_count": male_count,
            "female_count": female_count,
            "object_counts": object_counts,
            "avg_confidence": avg_confidence,
            "density_level": get_density_level(person_count),
            "processing_time_ms": processing_time,
            "alert": person_count > ALERT_THRESHOLD,
            "alert_threshold": ALERT_THRESHOLD,
            "output_image": encode_image(annotated_img),
            "heatmap_image": encode_image(heatmap_img)
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")

        tmp_path = "temp_video.mp4"
        with open(tmp_path, 'wb') as f:
            f.write(await file.read())

        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        sample_interval = max(1, int(fps * 3))

        frame_results = []
        frame_idx = 0
        last_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            last_frame = frame
            if frame_idx % sample_interval == 0:
                h, w = frame.shape[:2]
                if w > 640:
                    frame = cv2.resize(frame, (640, int(h * 640 / w)))
                ann_frame, _, count, males, females, obj_counts, conf, proc_time = process_frame(frame)
                frame_results.append({
                    "second": round(frame_idx / fps, 1),
                    "count": count,
                    "males": males,
                    "females": females,
                    "objects": obj_counts,
                    "confidence": conf,
                    "processing_time_ms": proc_time,
                    "frame_image": encode_image(ann_frame)
                })
            frame_idx += 1

        cap.release()
        os.remove(tmp_path)

        max_count = max((r["count"] for r in frame_results), default=0)
        max_persons = max((r["count"] for r in frame_results), default=0)
        max_cars = max((r["objects"].get("Car", 0) for r in frame_results), default=0)
        avg_count = round(sum(r["count"] for r in frame_results) / len(frame_results)) if frame_results else 0

        preview = ""
        heatmap_preview = ""
        if last_frame is not None:
            annotated, heatmap, _, _, _, _, _, _ = process_frame(last_frame)
            preview = encode_image(annotated)
            heatmap_preview = encode_image(heatmap)

        return JSONResponse({
            "frame_results": frame_results,
            "max_count": max_count,
            "max_persons": max_persons,
            "max_cars": max_cars,
            "avg_count": avg_count,
            "density_level": get_density_level(max_count),
            "alert": max_count > ALERT_THRESHOLD,
            "alert_threshold": ALERT_THRESHOLD,
            "preview_image": preview,
            "heatmap_image": heatmap_preview
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
