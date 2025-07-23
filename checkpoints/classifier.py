import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import CosineDecay
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

# Set your data paths here
train_img_path = 'training'      # <-- CHANGE THIS
test_img_path = 'validation'     # <-- CHANGE THIS

# Hyperparameters
batch_size = 32
img_height = 224
img_width = 224
num_classes = 3  # <-- CHANGE THIS if you have a different number of classes
initial_epochs = 50
fine_tune_epochs = 20

# Enhanced Data Augmentation
train_data_gen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=45,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.2
)

train_ds = train_data_gen.flow_from_directory(
    train_img_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    seed=123,
    shuffle=True
)

valid_ds = train_data_gen.flow_from_directory(
    train_img_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    seed=123,
    shuffle=True
)

test_data_gen = ImageDataGenerator(rescale=1.0 / 255.0)
test_ds = test_data_gen.flow_from_directory(
    test_img_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Compute class weights for imbalanced data
labels = train_ds.classes
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}
print('Class weights:', class_weights_dict)

# DenseNet121 with ImageNet weights
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False  # Freeze the base model initially

# Build Model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# Compile Model
initial_learning_rate = 0.0001
lr_schedule = CosineDecay(initial_learning_rate, decay_steps=initial_epochs * len(train_ds))
model.compile(
    optimizer=optimizers.Adam(learning_rate=lr_schedule),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_densenet_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Initial Training
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=initial_epochs,
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
    class_weight=class_weights_dict
)

# Load Best Weights
model.load_weights('best_densenet_model.h5')

# Fine-Tuning
base_model.trainable = True
# Freeze earlier layers (e.g., first 300 layers)
for layer in base_model.layers[:300]:
    layer.trainable = False

# Recompile Model with Lower Learning Rate
fine_tune_lr = CosineDecay(1e-5, decay_steps=fine_tune_epochs * len(train_ds))
fine_tune_optimizer = optimizers.Adam(learning_rate=fine_tune_lr)
model.compile(
    optimizer=fine_tune_optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-Tune Model
fine_tune_history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=fine_tune_epochs,
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
    class_weight=class_weights_dict
)

# Evaluate on Test Set
print('Evaluating on test set...')
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Final Test Accuracy: {test_accuracy:.4f}")

# Confusion Matrix and Classification Report
print('Generating confusion matrix and classification report...')
test_ds.reset()
preds = model.predict(test_ds)
y_pred = np.argmax(preds, axis=1)
y_true = test_ds.classes
class_labels = list(test_ds.class_indices.keys())
cm = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:')
print(cm)
print('Classification Report:')
print(classification_report(y_true, y_pred, target_names=class_labels))

# Optionally, save the final model
model.save('final_densenet_dent_classifier.h5')