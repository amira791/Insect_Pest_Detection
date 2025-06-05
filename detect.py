import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

def draw_rotated_boxes(image, boxes, labels, colors, confidences):
    """Draw rotated bounding boxes with class labels and confidence"""
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Load a font (adjust path if needed)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for box, label, color, conf in zip(boxes, labels, colors, confidences):
        # Convert box to polygon points
        points = [(int(box[i]), int(box[i+1])) for i in range(0, len(box), 2)]
        
        # Draw rotated box
        draw.polygon(points, outline=color, width=3)
        
        # Draw label background
        text = f"{label} {conf:.2f}"
        text_width, text_height = font.getsize(text)
        draw.rectangle(
            [points[0][0], points[0][1], 
            points[0][0] + text_width + 6, 
            points[0][1] + text_height + 6
        ], fill=color)
        
        # Draw text
        draw.text(
            (points[0][0] + 3, points[0][1] + 3),
            text, 
            fill="white",
            font=font
        )
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def detect_image(model, image_path, output_dir="detection_results", conf=0.5):
    """Run detection on a single image"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Run detection
    results = model.predict(image, conf=conf, imgsz=640)
    result = results[0]
    
    # Prepare visualization
    class_names = model.names
    colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)
    
    boxes = []
    labels = []
    confs = []
    color_list = []
    
    for box in result.obb:
        # Get OBB coordinates (xyxyxyxy format)
        rotated_box = box.xyxyxyxy.cpu().numpy()[0].astype(int)
        class_id = int(box.cls)
        conf = float(box.conf)
        
        boxes.append(rotated_box)
        labels.append(class_names[class_id])
        confs.append(conf)
        color_list.append(tuple(map(int, colors[class_id])))
    
    # Draw detections
    vis_image, _ = draw_rotated_boxes(image, boxes, labels, color_list, confs)
    
    # Save results
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, vis_image)
    print(f"Saved results to {output_path}")
    
    # Print summary
    print(f"\nDetection Summary for {os.path.basename(image_path)}:")
    print(f"- Detected objects: {len(boxes)}")
    print(f"- Average confidence: {np.mean(confs):.2f}")
    
    return vis_image

def detect_batch(model, image_dir, output_dir="detection_results", conf=0.5):
    """Run detection on all images in a directory"""
    image_files = [f for f in os.listdir(image_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\nProcessing {len(image_files)} images in {image_dir}...")
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        detect_image(model, img_path, output_dir, conf)
    
    print(f"\nAll results saved to {output_dir}")

def main():
    # Load your trained OBB model
    model = YOLO("runs/obb/insect_pest_obb_final/weights/best.pt")
    
    # Choose detection mode
    print("1. Detect single image")
    print("2. Detect all images in a folder")
    choice = input("Enter choice (1/2): ")
    
    conf = float(input("Enter confidence threshold (0.1-0.9): ") or 0.5)
    
    if choice == "1":
        image_path = input("Enter image path: ").strip('"')
        detect_image(model, image_path, conf=conf)
    elif choice == "2":
        folder_path = input("Enter folder path: ").strip('"')
        detect_batch(model, folder_path, conf=conf)
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()