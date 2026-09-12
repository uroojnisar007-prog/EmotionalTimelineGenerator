import cv2
import os
import sys
import numpy as np
import csv
from tensorflow.keras.models import load_model

# --------- CONFIG ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "saved_folder", "emotion_model.h5")

frames_temp_path = os.path.join(BASE_DIR, "temp_frames")
if not os.path.exists(frames_temp_path):
    os.makedirs(frames_temp_path)

timeline_output_path = os.path.join(BASE_DIR, "timeline_output")
if not os.path.exists(timeline_output_path):
    os.makedirs(timeline_output_path)

emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# --------- LOAD MODEL ----------
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}. Run train_model.py first!")
    sys.exit()
model = load_model(model_path)
print("Model loaded successfully!")

# --------- VIDEO INPUT ----------
if len(sys.argv) < 2:
    print("Usage: python generate_timeline.py <video_filename>")
    sys.exit()

video_filename = sys.argv[1]
video_path = os.path.join(BASE_DIR, "videos", video_filename)

if not os.path.exists(video_path):
    print(f"Video file not found: {video_path}")
    sys.exit()

# --------- EXTRACT FRAMES & PREDICT EMOTIONS ----------
cap = cv2.VideoCapture(video_path)
frame_count = 0
emotions_detected = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)
    emotion_idx = np.argmax(preds)
    emotion = emotion_labels[emotion_idx]
    emotions_detected.append(emotion)

    frame_count += 1

cap.release()

# --------- PRINT TIMELINE ----------
print(f"\nTotal frames processed: {frame_count}\n")
print("Frame-wise Emotion Timeline:")
for i, e in enumerate(emotions_detected):
    print(f"Frame {i:03d}: {e}")

# --------- SAVE CSV WITH TIME & EMOTION COUNTS ----------
fps = cap.get(cv2.CAP_PROP_FPS) or 30  # agar FPS na mile to default 30
csv_filename = os.path.join(timeline_output_path, f"{os.path.splitext(video_filename)[0]}_timeline.csv")

# Frame-wise CSV with timestamp
with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["frame_number", "time_sec", "emotion"])
    for i, e in enumerate(emotions_detected):
        time_sec = round(i / fps, 2)
        writer.writerow([i, time_sec, e])

print(f"Timeline CSV saved with timestamps: {csv_filename}")

# Emotion counts & percentages
from collections import Counter
emotion_counter = Counter(emotions_detected)
total_frames = len(emotions_detected)

summary_csv_filename = os.path.join(timeline_output_path, f"{os.path.splitext(video_filename)[0]}_summary.csv")
with open(summary_csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["emotion", "count", "percentage"])
    for emotion, count in emotion_counter.items():
        percentage = round((count / total_frames) * 100, 2)
        writer.writerow([emotion, count, percentage])

print(f"Summary CSV with emotion counts & percentages saved: {summary_csv_filename}")