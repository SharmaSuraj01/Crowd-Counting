from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import time

app = FastAPI()

model = YOLO('yolov8n.pt')

def get_density_level(count):
    if count == 0:
        return "Empty"
    elif count <= 5:
        return "Low"
    elif count <= 20:
        return "Medium"
    elif count <= 50:
        return "High"
    else:
        return "Very High"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        start_time = time.time()
        results = model(img, verbose=False)
        processing_time = round((time.time() - start_time) * 1000)

        person_count = 0
        total_confidence = 0.0
        annotated_img = img.copy()

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 0:
                        person_count += 1
                        conf = float(box.conf[0])
                        total_confidence += conf

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                        label = f'#{person_count} {conf:.0%}'
                        font_scale = 0.6
                        thickness = 2
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                        cv2.rectangle(annotated_img, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 255, 0), -1)
                        cv2.putText(annotated_img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        avg_confidence = round((total_confidence / person_count) * 100) if person_count > 0 else 0

        _, buffer = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return JSONResponse({
            "person_count": person_count,
            "avg_confidence": avg_confidence,
            "density_level": get_density_level(person_count),
            "processing_time_ms": processing_time,
            "output_image": f"data:image/jpeg;base64,{img_base64}"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
