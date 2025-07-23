import torch
from ultralytics import YOLO
import os

def train_model():
    # Check GPU availability
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Get current working directory
    current_dir = os.getcwd()

    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Training arguments
    args = {
        'data': 'config.yaml',
        'epochs': 50,
        'imgsz': 640,
        'batch': 16,
        'name': 'car_damage_detector',
        'project': os.path.join(current_dir, 'runs'),
        'device': device,
        'patience': 10,  # Early stopping
        'verbose': True,
        'save': True,  # Save model after training
        'save_period': 10,  # Save checkpoint every 10 epochs
        'cache': True,  # Cache images for faster training
        'workers': 8,  # Number of worker threads
        'resume': False,  # Resume training from last checkpoint if available
    }

    try:
        # Train the model
        results = model.train(**args)
        print("Training completed successfully!")
        
        # Print final metrics
        print("\nFinal Metrics:")
        print(f"mAP50: {results.maps[50]:.3f}")
        print(f"mAP50-95: {results.maps[0]:.3f}")
        
        # Save the final model
        model.export(format='onnx')  # Export to ONNX format
        print("\nModel exported to ONNX format")
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")

if __name__ == "__main__":
    # First, make sure ultralytics is installed
    try:
        import ultralytics
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.run(["pip", "install", "ultralytics"])
        
    train_model()
