from ultralytics import YOLO

# model = YOLO("ultralytics/cfg/models/v8/yolov8n.yaml")
model = YOLO("ultralytics/cfg/models/v8/yolov8x.yaml")
model.train(data="military_aircrafts.yaml", epochs=300, imgsz=512, patience=20)
