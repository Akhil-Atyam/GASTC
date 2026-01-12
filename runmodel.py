from ultralytics import YOLO

model = YOLO('../runs/train/Battery model/weights/best.pt')
source = 0
results = model(source = source, show=True, save=True, project='runs/detect', name='webcam_photo',exist_ok=True, conf=0.6)