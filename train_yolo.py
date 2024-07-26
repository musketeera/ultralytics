import comet_ml
from ultralytics import YOLO

comet_ml.login(project_name="comet-yolov8n-pose")
model = YOLO("/home/ehooph/tyop/ultralytics/ultralytics/weights/yolov8n-pose.pt")
results = model.train(
    data="/home/ehooph/tyop/ultralytics/ultralytics/cfg/datasets/space-pose.yaml", 
    project="comet-yolov8n-pose", 
    imgsz=640,
    batch=32, 
    optimizer="AdamW",
    save_period=1, 
    save_json=True, 
    epochs=300
)