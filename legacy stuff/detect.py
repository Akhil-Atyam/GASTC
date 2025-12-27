from ultralytics import YOLO

model = YOLO('best.pt')
source = 'webcam_photo.jpg'
results = model(source = source, show=True, save=True, project='runs/detect', name='webcam_photo',exist_ok=True)