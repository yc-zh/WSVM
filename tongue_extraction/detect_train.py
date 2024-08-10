# yolo task=detect mode=train model=yolov8n.pt data=data/tongue.yaml batch=32 epochs=100 imgsz=640 workers=16 device=0
from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
# if __name__ == '__main__':
#     results = model.train(data='data/tongue.yaml', epochs=100, imgsz=640, workers=16) 
results = model.train(data='data/tongue.yaml', epochs=100, imgsz=640, workers=0) 