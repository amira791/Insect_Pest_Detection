import os
import numpy as np
from ultralytics import YOLO

def validate_and_fix_labels(label_dir):
    """Ensure all coordinates are normalized (0-1) and have exactly 8 values per line"""
    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue
            
        filepath = os.path.join(label_dir, label_file)
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        valid_lines = []
        for line in lines:
            parts = line.strip().split()
            # Skip empty lines or malformed entries
            if len(parts) != 9:  # class_id + 8 coords
                continue
            
            try:
                class_id = int(parts[0])
                coords = np.array([float(x) for x in parts[1:]]).clip(0, 1)  # Force 0-1 range
                valid_lines.append(f"{class_id} {' '.join(['%.6f' % x for x in coords])}\n")
            except ValueError:
                continue  # Skip corrupt lines
        
        # Overwrite only if we have valid data
        if valid_lines:
            with open(filepath, 'w') as f:
                f.writelines(valid_lines)

def main():
    # 1. Clean labels first
    validate_and_fix_labels("./data/train/labels")
    validate_and_fix_labels("./data/val/labels")

    # 2. Model configuration
    model = YOLO("yolov8m-obb.pt")  # Medium model for better accuracy
    
    # 3. Training parameters (optimized for OBB)
    results = model.train(
        data=os.path.abspath("dataset.yaml"),  # Absolute path to avoid issues
        epochs=300,
        imgsz=640,
        batch=16,
        patience=50,
        workers=4,  # Reduced for Windows stability
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        
        # Augmentations (critical for OBB)
        degrees=45.0,  # Rotation
        flipud=0.5,    # Vertical flip
        fliplr=0.5,    # Horizontal flip
        mixup=0.2,     # Image mixing
        copy_paste=0.2, # Object pasting
        hsv_h=0.015,   # Hue variation
        hsv_s=0.7,     # Saturation variation
        hsv_v=0.4,     # Value variation
        
        # OBB-specific
        kobj=2.0,      # Higher objectness weight
        name="insect_pest_obb_final"
    )

    # 4. Post-training validation
    metrics = model.val()
    print(f"\nTraining completed! mAP50-95: {metrics.box.map:.3f}")

if __name__ == '__main__':
    # Windows-specific multiprocessing fix
    import torch
    torch.multiprocessing.freeze_support()
    main()