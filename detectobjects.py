from ultralytics import YOLO

# Load the pretrained YOLO model
model = YOLO("best.pt")

def detect_objects(image_path):
    results = model(image_path)
    detections = []

    # Loop through results (YOLO may return multiple result objects)
    for result in results:
        # Loop through each detected bounding box
        for box in result.boxes:
            class_id = int(box.cls[0])        # Class index
            confidence = float(box.conf[0])   # Confidence score
            object_name = model.names[class_id]

            detections.append({
                "object": object_name,
                "confidence": round(confidence, 2)
            })
    return detections
detect_objects("webcam_photo.jpg")