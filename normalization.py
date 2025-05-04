import os
import numpy as np

def fix_labels(label_dir):
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            filepath = os.path.join(label_dir, label_file)
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            fixed_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 9:  # class + 8 coords
                    continue
                
                # Convert to floats and clip to [0,1]
                coords = np.array([float(x) for x in parts[1:]]).clip(0, 1)
                fixed_line = f"{parts[0]} {' '.join(['%.6f' % x for x in coords])}\n"
                fixed_lines.append(fixed_line)
            
            with open(filepath, 'w') as f:
                f.writelines(fixed_lines)

# Apply to both train and val
fix_labels("./data/train/labels")
fix_labels("./data/val/labels")