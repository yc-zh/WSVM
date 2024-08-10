# yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=data/images device=0
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('runs/detect/train/weights/best.pt')  # load a custom model

# Define path to the image file
source = 'data_unl/test'

# Run inference on the source
results = model.predict(source, workers=0, save=True, save_txt=True)  # list of Results objects

print(results[0].boxes)  # inference bounding boxes    