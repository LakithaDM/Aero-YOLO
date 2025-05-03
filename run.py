from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v8/yolov8x.yaml" , )
model.info()