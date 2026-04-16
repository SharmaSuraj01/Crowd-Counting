from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import time
import os
import uvicorn

app = FastAPI()
model = YOLO('yolov8n.pt')

ALERT_THRESHOLD = 10

def get_density_level(count):
    if count == 0:     return "Empty"
    elif count <= 5:   return "Low"
    elif count <= 20:  return "Medium"
    elif count <= 50:  return "High"
    else:              return "Very High"

def generate_heatmap(img, boxes):
    h, w = img.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    for (x1, y1, x2, y2) in boxes:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
        heat[max(0, cy-bh):min(h, cy+bh), max(0, cx-bw):min(w, cx+bw)] += 1

    if heat.max() > 0:
        heat = heat / heat.max()

    ksize = max(51, (min(h, w) // 10) | 1)
    heat = cv2.GaussianBlur(heat, (ksize, ksize), 0)
    heatmap_color = cv2.applyColorMap(np.uint8(heat * 255), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.5, heatmap_color, 0.5, 0)

    for i, (label, color) in enumerate([("Low", (0,255,0)), ("Medium", (0,255,255)), ("High", (0,128,255)), ("Critical", (0,0,255))]):
        cv2.rectangle(overlay, (10, 10 + i*22), (30, 28 + i*22), color, -1)
        cv2.putText(overlay, label, (35, 24 + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    return overlay

def process_frame(img):
    start_time = time.time()
    results = model(img, verbose=False)
    processing_time = round((time.time() - start_time) * 1000)

    person_count = 0
    car_count = 0
    total_confidence = 0.0
    all_boxes = []
    annotated_img = img.copy()

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls not in (0, 2):  # 0=person, 2=car
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            all_boxes.append((x1, y1, x2, y2))

            is_person = cls == 0
            if is_person:
                person_count += 1
                total_confidence += conf
                color = (0, 255, 0)   # green for person
                label = f'Person {conf:.0%}'
            else:
                car_count += 1
                color = (255, 165, 0)  # orange for car
                label = f'Car {conf:.0%}'

            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 1)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(annotated_img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated_img, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    total_count = person_count + car_count
    if total_count > ALERT_THRESHOLD:
        cv2.rectangle(annotated_img, (0, 0), (annotated_img.shape[1], 50), (0, 0, 200), -1)
        cv2.putText(annotated_img, f'ALERT: {person_count} people, {car_count} cars detected!',
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    heatmap_img = generate_heatmap(img, all_boxes)
    avg_confidence = round((total_confidence / person_count) * 100) if person_count > 0 else 0

    return annotated_img, heatmap_img, person_count, car_count, avg_confidence, processing_time

def encode_image(img):
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

@app.get("/health")
def health():
    return {"status": "ok"}

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

        annotated_img, heatmap_img, person_count, car_count, avg_confidence, processing_time = process_frame(img)

        return JSONResponse({
            "person_count": person_count,
            "car_count": car_count,
            "avg_confidence": avg_confidence,
            "density_level": get_density_level(person_count + car_count),
            "processing_time_ms": processing_time,
            "alert": (person_count + car_count) > ALERT_THRESHOLD,
            "alert_threshold": ALERT_THRESHOLD,
            "output_image": encode_image(annotated_img),
            "heatmap_image": encode_image(heatmap_img)
        })

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
                _, _, p_count, c_count, conf, proc_time = process_frame(frame)
                frame_results.append({
                    "second": round(frame_idx / fps, 1),
                    "count": p_count + c_count,
                    "person_count": p_count,
                    "car_count": c_count,
                    "confidence": conf,
                    "processing_time_ms": proc_time
                })
            frame_idx += 1

        cap.release()
        os.remove(tmp_path)

        max_count = max((r["count"] for r in frame_results), default=0)
        avg_count = round(sum(r["count"] for r in frame_results) / len(frame_results)) if frame_results else 0
        max_persons = max((r["person_count"] for r in frame_results), default=0)
        max_cars = max((r["car_count"] for r in frame_results), default=0)

        preview = ""
        heatmap_preview = ""
        if last_frame is not None:
            annotated, heatmap, _, _, _, _ = process_frame(last_frame)
            preview = encode_image(annotated)
            heatmap_preview = encode_image(heatmap)

        return JSONResponse({
            "frame_results": frame_results,
            "max_count": max_count,
            "avg_count": avg_count,
            "max_persons": max_persons,
            "max_cars": max_cars,
            "density_level": get_density_level(max_count),
            "alert": max_count > ALERT_THRESHOLD,
            "alert_threshold": ALERT_THRESHOLD,
            "preview_image": preview,
            "heatmap_image": heatmap_preview
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/webcam-frame")
async def webcam_frame(threshold: int = 10):
    try:
        global ALERT_THRESHOLD
        ALERT_THRESHOLD = threshold

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Cannot access webcam")

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise HTTPException(status_code=500, detail="Failed to capture frame")

        annotated_img, heatmap_img, person_count, car_count, avg_confidence, processing_time = process_frame(frame)

        return JSONResponse({
            "person_count": person_count,
            "car_count": car_count,
            "avg_confidence": avg_confidence,
            "density_level": get_density_level(person_count + car_count),
            "processing_time_ms": processing_time,
            "alert": (person_count + car_count) > ALERT_THRESHOLD,
            "alert_threshold": ALERT_THRESHOLD,
            "output_image": encode_image(annotated_img),
            "heatmap_image": encode_image(heatmap_img)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
