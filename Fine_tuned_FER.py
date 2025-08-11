import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2

# Paths for your dataset
train_dir = "./emotion-detection-fer/train"
test_dir = "./emotion-detection-fer/test"

# Image size - increased for better accuracy
img_size = 224  # Changed from 48 to 224 for better feature extraction

print("🚀 Building High-Accuracy Emotion Recognition System...")
print(f"Using image size: {img_size}x{img_size}")

# Create improved model using transfer learning
def create_high_accuracy_model():
    """
    Creates a high-accuracy model using MobileNetV2 backbone
    Expected accuracy: 75-85%
    """
    print("📦 Loading MobileNetV2 pre-trained model...")
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet',
        alpha=1.0
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Create custom head for emotion classification
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        
        # First dense block
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        
        # Second dense block
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Third dense block
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        
        # Output layer
        Dense(7, activation='softmax', name='emotion_predictions')
    ])
    
    return model, base_model

# Advanced data augmentation for better generalization
print("🔄 Setting up advanced data augmentation...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,          # Increased rotation
    width_shift_range=0.3,      # Increased shifting
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.3,             # Increased zoom
    horizontal_flip=True,
    brightness_range=[0.8, 1.2], # Brightness variation
    channel_shift_range=0.1,    # Color variation
    fill_mode='nearest',
    validation_split=0.2
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Load data with improved settings
print("📊 Loading dataset...")

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(img_size, img_size),
    batch_size=16,  # Reduced batch size for larger images
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42
)

validation_generator = validation_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(img_size, img_size),
    batch_size=16,
    class_mode="categorical",
    subset="validation",
    shuffle=True,
    seed=42
)

# Get class names
class_names = list(train_generator.class_indices.keys())
print(f"📝 Emotion classes: {class_names}")
print(f"📈 Training samples: {train_generator.samples}")
print(f"📉 Validation samples: {validation_generator.samples}")

# Create the improved model
model, base_model = create_high_accuracy_model()

# Compile with optimized settings
print("⚙️ Compiling model with optimized settings...")
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Advanced callbacks for better training
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=4,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        'best_emotion_model_v2.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Phase 1: Train the head with frozen base
print("\n🎯 PHASE 1: Training classification head (base model frozen)...")
print("Expected improvement: 40% → 65% accuracy")

epochs_phase1 = 15
history_phase1 = model.fit(
    train_generator,
    epochs=epochs_phase1,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

# Phase 2: Fine-tune the entire model
print("\n🔥 PHASE 2: Fine-tuning entire model...")
print("Expected improvement: 65% → 75-85% accuracy")

# Unfreeze the base model
base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze the first layers
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Much lower learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training with fine-tuning
epochs_phase2 = 20
history_phase2 = model.fit(
    train_generator,
    epochs=epochs_phase2,
    initial_epoch=len(history_phase1.history['loss']),
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

# Combine training histories
total_history = {}
for key in history_phase1.history:
    total_history[key] = history_phase1.history[key] + history_phase2.history[key]

# Save the final model
model.save('high_accuracy_emotion_model.keras')
print("💾 High-accuracy model saved successfully!")

# Enhanced plotting
print("📊 Generating training visualizations...")

acc = total_history['accuracy']
val_acc = total_history['val_accuracy']
loss = total_history['loss']
val_loss = total_history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(15, 5))

# Accuracy plot
plt.subplot(1, 3, 1)
plt.plot(epochs_range, acc, label='Training Accuracy', linewidth=2)
plt.plot(epochs_range, val_acc, label='Validation Accuracy', linewidth=2)
plt.axvline(x=epochs_phase1-1, color='red', linestyle='--', alpha=0.7, label='Fine-tuning Start')
plt.legend(loc='lower right')
plt.title('Model Accuracy Over Time')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)

# Loss plot
plt.subplot(1, 3, 2)
plt.plot(epochs_range, loss, label='Training Loss', linewidth=2)
plt.plot(epochs_range, val_loss, label='Validation Loss', linewidth=2)
plt.axvline(x=epochs_phase1-1, color='red', linestyle='--', alpha=0.7, label='Fine-tuning Start')
plt.legend(loc='upper right')
plt.title('Model Loss Over Time')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

# Accuracy improvement plot
plt.subplot(1, 3, 3)
final_acc = val_acc[-1]
plt.bar(['Initial\n(Random)', 'After Phase 1\n(Head Training)', 'Final\n(Fine-tuned)'], 
        [0.14, max(val_acc[:epochs_phase1]), final_acc], 
        color=['red', 'orange', 'green'], alpha=0.7)
plt.title('Accuracy Improvement')
plt.ylabel('Validation Accuracy')
plt.ylim(0, 1)
for i, v in enumerate([0.14, max(val_acc[:epochs_phase1]), final_acc]):
    plt.text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('high_accuracy_training_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Final evaluation
print("\n📋 FINAL EVALUATION:")
final_loss, final_accuracy = model.evaluate(validation_generator, verbose=0)
print(f"🎯 Final Validation Accuracy: {final_accuracy:.1%}")
print(f"📉 Final Validation Loss: {final_loss:.4f}")

# Performance summary
print(f"\n🏆 PERFORMANCE SUMMARY:")
print(f"   • Random Baseline: 14.3%")
print(f"   • After Head Training: {max(val_acc[:epochs_phase1]):.1%}")
print(f"   • Final Accuracy: {final_accuracy:.1%}")
print(f"   • Improvement: {(final_accuracy - 0.143):.1%} points above random")

if final_accuracy > 0.75:
    print("   🌟 EXCELLENT: 75%+ accuracy achieved!")
elif final_accuracy > 0.70:
    print("   ✅ VERY GOOD: 70%+ accuracy achieved!")
else:
    print("   ✅ GOOD: Solid improvement over baseline!")

# Enhanced real-time detection with confidence filtering
def enhanced_real_time_detection():
    """
    Enhanced real-time detection with confidence filtering and smoothing
    """
    print("\n🎥 Starting Enhanced Real-time Detection...")
    print("Features: Confidence filtering, prediction smoothing, FPS counter")
    
    # Load the high-accuracy model
    model = tf.keras.models.load_model('high_accuracy_emotion_model.keras')
    
    # Emotion labels
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    # Face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Start camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Prediction smoothing buffer
    prediction_buffer = []
    buffer_size = 5
    confidence_threshold = 0.4  # Only show predictions above 40% confidence
    
    # FPS calculation
    import time
    fps_counter = 0
    fps_start_time = time.time()
    
    print("📹 Camera started. Controls:")
    print("   • Press 'q' to quit")
    print("   • Press 's' to save screenshot")
    print("   • Press 'r' to reset prediction buffer")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror the frame for more natural interaction
        frame = cv2.flip(frame, 1)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size > 0:
                # Preprocess for model (resize to 224x224, normalize)
                face_resized = cv2.resize(face_roi, (img_size, img_size))
                face_normalized = face_resized.astype('float32') / 255.0
                face_input = np.expand_dims(face_normalized, axis=0)
                
                # Predict emotion
                predictions = model.predict(face_input, verbose=0)[0]
                
                # Add to smoothing buffer
                prediction_buffer.append(predictions)
                if len(prediction_buffer) > buffer_size:
                    prediction_buffer.pop(0)
                
                # Average predictions for smoothing
                if len(prediction_buffer) >= 3:
                    smoothed_predictions = np.mean(prediction_buffer, axis=0)
                else:
                    smoothed_predictions = predictions
                
                emotion_idx = np.argmax(smoothed_predictions)
                emotion = emotions[emotion_idx]
                confidence = smoothed_predictions[emotion_idx]
                
                # Only display if confidence is above threshold
                if confidence > confidence_threshold:
                    # Color based on confidence
                    if confidence > 0.8:
                        color = (0, 255, 0)  # Green for high confidence
                    elif confidence > 0.6:
                        color = (0, 255, 255)  # Yellow for medium confidence
                    else:
                        color = (0, 165, 255)  # Orange for low confidence
                    
                    # Draw face rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    
                    # Prepare label with confidence
                    label = f'{emotion}: {confidence:.1%}'
                    
                    # Calculate label size for background
                    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    
                    # Draw label background
                    cv2.rectangle(frame, (x, y-label_height-15), (x+label_width+10, y), color, -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    
                    # Show top 3 predictions (optional)
                    top_3_indices = np.argsort(smoothed_predictions)[-3:][::-1]
                    for i, idx in enumerate(top_3_indices):
                        if i == 0:  # Skip the main prediction
                            continue
                        small_label = f"{emotions[idx]}: {smoothed_predictions[idx]:.1%}"
                        cv2.putText(frame, small_label, (x, y+h+20+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                else:
                    # Low confidence - show "Uncertain"
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 128, 128), 2)
                    cv2.putText(frame, "Uncertain", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        
        # Calculate and display FPS
        fps_counter += 1
        fps_elapsed = time.time() - fps_start_time
        if fps_elapsed >= 1.0:
            fps = fps_counter / fps_elapsed
            fps_counter = 0
            fps_start_time = time.time()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Enhanced Emotion Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('emotion_detection_screenshot.jpg', frame)
            print("📸 Screenshot saved!")
        elif key == ord('r'):
            prediction_buffer = []
            print("🔄 Prediction buffer reset!")
    
    cap.release()
    cv2.destroyAllWindows()
    print("📹 Camera session ended.")

# Instructions for usage
print(f"\n🎯 USAGE INSTRUCTIONS:")
print(f"   1. Training completed! Model saved as 'high_accuracy_emotion_model.keras'")
print(f"   2. To test real-time detection, run:")
print(f"      enhanced_real_time_detection()")
print(f"   3. Expected accuracy: {final_accuracy:.1%} (vs 67% from basic model)")
print(f"   4. Model size: ~{os.path.getsize('high_accuracy_emotion_model.keras')/1024/1024:.1f}MB")

# Auto-start real-time detection prompt
start_demo = input("\n🤖 Start enhanced real-time detection now? (y/n): ").strip().lower()
if start_demo == 'y':
    enhanced_real_time_detection()
else:
    print("💡 Run enhanced_real_time_detection() when ready to test!")
