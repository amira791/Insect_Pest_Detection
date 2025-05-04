import os
from ultralytics import YOLO

# Configuration
DATA_YAML = os.path.abspath("C:/Users/VECTUS/Insect_Pest_Detection/dataset.yaml")
MODEL_TYPE = "yolov8m-obb.pt"  # Medium model for better accuracy
EPOCHS = 300
IMGSZ = 640
BATCH = 16
PATIENCE = 50
NAME = "insect_pest_high_acc"

# Training
model = YOLO(MODEL_TYPE)

results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    patience=PATIENCE,
    name=NAME,
    
    # Optimization
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    
    # Augmentation
    degrees=45,
    flipud=0.5,
    fliplr=0.5,
    mixup=0.2,
    copy_paste=0.2,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    
    # Model
    box=7.5,
    cls=0.5,
    dfl=1.5,
    kobj=2.0,  # Increased objectness loss weight
)

# Validation
metrics = model.val()
print(f"mAP50-95: {metrics.box.map:.3f}")