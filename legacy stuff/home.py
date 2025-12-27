import cv2
import json
import time
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "best.pt"
CONF_THRESH = 0.4
WEBCAM_IDX = 0
IMG_WIDTH = 640
IMG_HEIGHT = 480

RAW_IMAGE_PATH = "raw.jpg"
PROCESSED_IMAGE_PATH = "processed.jpg"
OUTPUT_JSON = "detections.json"
# ---------------------------------------

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(WEBCAM_IDX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

last_frame = None
last_results = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Live inference
    results = model.predict(frame, conf=CONF_THRESH, verbose=False)

    annotated = frame.copy()
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        detections.append({
            "class": class_name,
            "confidence": confidence,
            "bbox_xyxy": [x1, y1, x2, y2]
        })

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{class_name} {confidence:.2f}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # Cache last frame + results
    last_frame = frame.copy()
    last_results = detections

    cv2.imshow("YOLO Live Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# ---------------- BACKEND SNAPSHOT ----------------
if last_frame is not None:
    timestamp = time.time()

    # Save raw image
    cv2.imwrite(RAW_IMAGE_PATH, last_frame)

    # Run YOLO once more for clean output image
    final_results = model.predict(last_frame, conf=CONF_THRESH, verbose=False)

    # YOLO built-in plotting (this is what you were thinking of)
    processed_img = final_results[0].plot()

    cv2.imwrite(PROCESSED_IMAGE_PATH, processed_img)

    # Save JSON
    output = {
        "timestamp": timestamp,
        "detections": last_results
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print("Saved:")
    print(f"  Raw image       → {RAW_IMAGE_PATH}")
    print(f"  Processed image → {PROCESSED_IMAGE_PATH}")
    print(f"  JSON            → {OUTPUT_JSON}")
else:
    print("No frame captured")
