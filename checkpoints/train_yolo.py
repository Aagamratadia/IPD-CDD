import os
import shutil
from ultralytics import YOLO
import torch
from pathlib import Path
import concurrent.futures

def move_files(src_files, src_dir, dst_dir):
    """Move files in parallel using a thread pool"""
    def move_single_file(filename):
        src = os.path.join(src_dir, filename)
        dst = os.path.join(dst_dir, filename)
        if os.path.isfile(src):
            shutil.move(src, dst)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(move_single_file, src_files)

def fix_directory_structure():
    base_dir = Path.cwd()
    data_dir = base_dir / 'data'
    
    # Create directories if they don't exist
    for split in ['train', 'val']:
        (data_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (data_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Fix training data location
    train_images_nested = data_dir / 'images' / 'train' / 'train'
    train_images_correct = data_dir / 'images' / 'train'
    train_labels_nested = data_dir / 'labels' / 'train' / 'train'
    train_labels_correct = data_dir / 'labels' / 'train'
    
    # Move files in parallel if needed
    if train_images_nested.exists():
        print("Moving training images...")
        move_files(os.listdir(train_images_nested), train_images_nested, train_images_correct)
        shutil.rmtree(train_images_nested, ignore_errors=True)
    
    if train_labels_nested.exists():
        print("Moving training labels...")
        move_files(os.listdir(train_labels_nested), train_labels_nested, train_labels_correct)
        shutil.rmtree(train_labels_nested, ignore_errors=True)

    # Create validation set if needed
    val_images_dir = data_dir / 'images' / 'val'
    if not any(val_images_dir.iterdir()):
        print("Creating validation set...")
        train_images = list(train_images_correct.glob('*.jpg')) + list(train_images_correct.glob('*.png'))
        val_size = int(len(train_images) * 0.2)
        val_files = train_images[:val_size]
        
        def copy_val_files(img_path):
            # Copy image
            shutil.copy2(img_path, val_images_dir / img_path.name)
            # Copy corresponding label
            label_path = data_dir / 'labels' / 'train' / (img_path.stem + '.txt')
            if label_path.exists():
                shutil.copy2(label_path, data_dir / 'labels' / 'val' / (img_path.stem + '.txt'))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(copy_val_files, val_files)

    # Update config
    config_content = f"""path: {data_dir}
train: images/train
val: images/val
nc: 8
names: ['damaged door', 'damaged window', 'damaged headlight', 'damaged mirror', 'dent', 'damaged hood', 'damaged bumper', 'damaged wind shield']"""
    
    with open('config.yaml', 'w') as f:
        f.write(config_content)
    
    return data_dir

def train_model():
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load and train model with optimized parameters
    try:
        model = YOLO('yolov8n.pt')  # Using nano model for faster training
        
        results = model.train(
            data='config.yaml',
            epochs=10,
            imgsz=640,
            batch=32,  # Increased batch size for faster training
            name='car_damage_detector',
            project='runs',
            device=device,
            patience=5,  # Reduced patience for faster early stopping
            verbose=True,
            exist_ok=True,
            workers=8,  # Optimized number of workers
            cache=True,  # Cache images in RAM
            close_mosaic=10,  # Disable mosaic augmentation in last 10 epochs
            amp=True,  # Automatic mixed precision for faster training
            optimizer='auto',
            single_cls=False,  # Set to True if all damages should be treated as one class
            overlap_mask=True,
            mask_ratio=4,
            seed=42,
            deterministic=False,  # Set to False for faster training
            plots=False  # Disable plotting for speed
        )
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")

if __name__ == "__main__":
    print("Setting up directory structure...")
    data_dir = fix_directory_structure()
    
    print("\nStarting training...")
    train_model()
