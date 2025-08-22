from ultralytics import YOLO

# Load model
model = YOLO("yolov8m.pt")

# Train the model
model.train(data="yolov8-data.yaml")
