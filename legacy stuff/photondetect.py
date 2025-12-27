import cv2
from ultralytics import YOLO

camera = cv2.VideoCapture(0)
ret, image = camera.read()
camera.release()
IMAGE_PATH = "webcam_photo.jpg"
cv2.imwrite(IMAGE_PATH, image)
print(f"Photo saved at {IMAGE_PATH}")

model = YOLO("best.pt")

results = model(
    source=IMAGE_PATH,
    show=True,
    save=True,
    project="runs/detect",
    name="webcam_photo",
    exist_ok=True
)

print("done")
