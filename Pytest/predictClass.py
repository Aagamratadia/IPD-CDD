import cv2
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# --- CONFIGURATION ---
img_path = "test_images/test/3.jpg"
yolo_model_path = "best.pt"
classifier_model_path = "best_densenet_model.h5"
test_img_dir = "training"  # Directory with subfolders for each class

img_height = 224
img_width = 224
batch_size = 32

# --- YOLO DETECTION ---
yolo_model = YOLO(yolo_model_path)
results = yolo_model(source=img_path)
res_plotted = results[0].plot()
cv2.imshow("YOLO Detection Result", res_plotted)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- CLASSIFIER SETUP ---
# Prepare test data generator to get class indices
test_data_gen = ImageDataGenerator(rescale=1.0 / 255.0)
test_ds = test_data_gen.flow_from_directory(
    test_img_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)
class_indices = test_ds.class_indices
idx_to_class = {v: k for k, v in class_indices.items()}

# Load the trained classifier model
classifier_model = tf.keras.models.load_model(classifier_model_path)

# --- CLASSIFY THE SAME IMAGE ---
img = keras_image.load_img(img_path, target_size=(img_height, img_width))
img_array = keras_image.img_to_array(img)
img_array = img_array / 255.0  # Rescale like training
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

pred_probs = classifier_model.predict(img_array)
pred_class_idx = np.argmax(pred_probs, axis=1)[0]
pred_class_label = idx_to_class[pred_class_idx]
confidence = pred_probs[0][pred_class_idx]

print(f"Classifier prediction for {img_path}: {pred_class_label} (confidence: {confidence:.4f})")

# --- OPTIONAL: Evaluate on test set (uncomment if needed) ---
# test_loss, test_accuracy = classifier_model.evaluate(test_ds)
# print(f"Classifier Final Test Accuracy: {test_accuracy:.4f}")