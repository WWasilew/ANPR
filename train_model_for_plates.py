from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(data="training/config.yaml", epochs=50, imgsz=1280)
