from ultralytics import YOLO
from copy import deepcopy
from ultralytics.nn.tasks import yaml_model_load, parse_model

# Load model config
yaml_path = "ultralytics/cfg/models/v8/yolov8x.yaml"
yaml_dict = yaml_model_load(yaml_path)

# Extract backbone and head from YAML
backbone = yaml_dict['backbone']
head = yaml_dict['head']

# Count backbone and head layers
num_backbone_layers = len(backbone)

# Rebuild full model to extract detailed structure
model = YOLO(yaml_path).model

print("Full Model Architecture: ")
model.info(detailed = True)

print("\n Backbone Layers: ")
for i in range(num_backbone_layers):
    layer = model.model[i]
    print(f"[{i}] {layer.type} - args: {layer}")

print("Head Layers: ")
for i in range(num_backbone_layers, len(model.model)):
    layer = model.model[i]
    print(f"[{i}] {layer.type} - args: {layer}")