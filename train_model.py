# train_model.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# -----------------------------
# 1️⃣ Paths
# -----------------------------
frames_path = r"C:\Users\Urooj\Desktop\EmotionTimelineGenerator\dataset\frames"

# -----------------------------
# 2️⃣ Parameters
# -----------------------------
IMG_SIZE = 224  # Resize frames
EMOTIONS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# -----------------------------
# 3️⃣ Load frames
# -----------------------------
X = []
y = []

for emotion in EMOTIONS:
    emotion_folder = os.path.join(frames_path, emotion)
    if not os.path.exists(emotion_folder):
        continue
    for video_folder in os.listdir(emotion_folder):
        video_path = os.path.join(emotion_folder, video_folder)
        if not os.path.isdir(video_path):
            continue
        for frame_file in os.listdir(video_path):
            if frame_file.lower().endswith('.jpg'):
                frame_path = os.path.join(video_path, frame_file)
                img = cv2.imread(frame_path)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X.append(img)
                y.append(EMOTIONS.index(emotion))

X = np.array(X, dtype='float32') / 255.0
y = to_categorical(y, num_classes=len(EMOTIONS))

print(f"✅ Total frames loaded: {len(X)}")
print(f"Classes: {EMOTIONS}")

# -----------------------------
# 4️⃣ Train-Test Split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 5️⃣ Build Model
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(EMOTIONS), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------
# 6️⃣ Train Model
# -----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)

# -----------------------------
# 7️⃣ Save Model
# -----------------------------
model.save(r"C:\Users\Urooj\Desktop\EmotionTimelineGenerator\saved_model\emotion_model.h5")
print("✅ Model saved successfully!")
