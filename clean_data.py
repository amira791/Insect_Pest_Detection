import os

def clean_labels(label_dir):
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            filepath = os.path.join(label_dir, label_file)
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Keep only lines with numeric data
            cleaned = [l for l in lines if l[0].isdigit() or l.startswith('-')]
            
            with open(filepath, 'w') as f:
                f.writelines(cleaned)
            print(f"Cleaned {label_file}")

# Clean both train and val labels
clean_labels("./data/train/labels")
clean_labels("./data/val/labels")