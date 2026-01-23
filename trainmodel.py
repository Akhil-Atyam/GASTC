from ultralytics import YOLO

# Load lightweight YOLOv11 model
model = YOLO("yolo11n.pt")

model.train(
    data="dataset/battery2/data.yaml",   # Yaml file
    imgsz=640,
    epochs=20,                        # this makes it longer
    batch=16,                          # idk
    workers=8,                        # How many cores do i wish to abuse
    device="cpu",

    optimizer="AdamW",                # better for small datasets apparently
    lr0=0.001,                        # lower LR = stability
    patience=5,                       # early stopping works now

    project="runs/train",
    name="Battery 2 model",
    exist_ok=True,

    cache=True,                       # speeds up CPU training
    pretrained=True,
    verbose=True
)
