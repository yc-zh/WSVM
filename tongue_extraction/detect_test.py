# yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data/tongue.yaml device=0

from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('runs/detect/train/weights/best.pt')  # load a custom model

# Validate the model
metrics = model.val(data='data/tongue.yaml', workers=0)  # no arguments needed, dataset and settings remembered
# print(metrics.box.map)    # map50-95
# print(metrics.box.map50)  # map50
# print(metrics.box.map75)  # map75
# print(metrics.box.maps)   # a list contains map50-95 of each category