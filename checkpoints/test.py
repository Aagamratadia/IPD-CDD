from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/car_damage_detector/weights/best.pt")

# Image path
img_path = "test_images/test/0.jpg"

# Run inference
results = model(img_path)

# Display result with bounding boxes
results[0].show()

# Save the result
results[0].save(filename="output.jpg")

# --- Classifier Test Part ---

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths (adjust if needed)
test_img_path = 'validation'  # or the correct path to your validation/test set
model_path = 'best_densenet_model.h5'  # or 'best_densenet_model.h5'

# Image parameters (should match training)
img_height = 224
img_width = 224
batch_size = 32

# Prepare test data generator
test_data_gen = ImageDataGenerator(rescale=1.0 / 255.0)
test_ds = test_data_gen.flow_from_directory(
    test_img_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Load the trained classifier model
classifier_model = tf.keras.models.load_model(model_path)

# Evaluate on test set
test_loss, test_accuracy = classifier_model.evaluate(test_ds)
print(f"Classifier Final Test Accuracy: {test_accuracy:.4f}")

# --- Classify the same image with the classifier ---

from tensorflow.keras.preprocessing import image
import numpy as np

# Get class indices mapping (label -> index)
class_indices = test_ds.class_indices
# Reverse mapping (index -> label)
idx_to_class = {v: k for k, v in class_indices.items()}

# Load and preprocess the image
img = image.load_img(img_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # Rescale like training
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict
pred_probs = classifier_model.predict(img_array)
pred_class_idx = np.argmax(pred_probs, axis=1)[0]
pred_class_label = idx_to_class[pred_class_idx]

print(f"Classifier prediction for {img_path}: {pred_class_label} (confidence: {pred_probs[0][pred_class_idx]:.4f})")