from collections import Counter
import os
import cv2
import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')  # Fix for running matplot in web server
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from flask import Flask, render_template, request, send_file, url_for
from tensorflow.keras.models import load_model

app = Flask(__name__)

# --- CONFIGURATION ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_folder", "emotion_model.h5")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULTS_FOLDER = os.path.join(BASE_DIR, "static", "results")

# Create directories if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load Model Once Globally
try:
    print("Loading Model...")
    model = load_model(MODEL_PATH)
    emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    print("Model Loaded Successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    # Exit or handle error gracefully if model is essential
    model = None
    emotion_labels = []

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', model_loaded=(model is not None))

@app.route('/analyze', methods=['POST'])
def analyze():
    if model is None:
        return render_template('index.html', error="Model not loaded. Check saved_folder/emotion_model.h5 path.")
    
    if 'video' not in request.files:
        return render_template('index.html', error="No video uploaded.")
    
    file = request.files['video']
    if file.filename == '':
        return render_template('index.html', error="No selected file.")

    # 1. Save Video
    video_filename = file.filename
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    file.save(video_path)

    # 2. Process Video (Extract & Predict)
    cap = cv2.VideoCapture(video_path)
    timeline_data = [] # To store {'frame': X, 'emotion': Y} for CSV & HTML table
    raw_emotions = []  # To store just the labels for summary calculation
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize & Preprocess
        img = cv2.resize(frame, (224, 224))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        preds = model.predict(img, verbose=0)
        emotion_idx = np.argmax(preds)
        emotion_label = emotion_labels[emotion_idx]

        timeline_data.append({'frame': frame_count, 'emotion': emotion_label})
        raw_emotions.append(emotion_label) # Collecting raw emotions for summary
        frame_count += 1

    cap.release()
    
    # Check if any frames were processed
    if not timeline_data:
        return render_template('index.html', error="Could not process video. Check file format or video codec.")

    # 3. Calculate Summary (NEW LOGIC)
    emotion_counts = Counter(raw_emotions)
    emotion_summary = []
    
    if frame_count > 0:
        for emotion, count in emotion_counts.most_common():
            percentage = round((count / frame_count) * 100, 2)
            emotion_summary.append({
                'emotion': emotion,
                'count': count,
                'percentage': percentage
            })
    
    # Safely determine dominant emotion
    dominant_emotion = emotion_summary[0]['emotion'] if emotion_summary else 'neutral'


    # 4. Save CSV
    csv_filename = f"{os.path.splitext(video_filename)[0]}_timeline.csv"
    csv_path = os.path.join(RESULTS_FOLDER, csv_filename)
    
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Frame", "Emotion"])
        for item in timeline_data:
            writer.writerow([item['frame'], item['emotion']])

    # 5. Generate Graph
    graph_filename = f"{os.path.splitext(video_filename)[0]}_graph.png"
    graph_path = os.path.join(RESULTS_FOLDER, graph_filename)
    
    # Read CSV using Pandas
    df = pd.read_csv(csv_path)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(df['Frame'], df['Emotion'], marker='o', linestyle='-', markersize=2, color='#6c5ce7')
    plt.title(f"Emotion Timeline: {video_filename}", fontsize=16)
    plt.xlabel("Frame Number")
    plt.ylabel("Emotion")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()

    # Pass ALL data to HTML
    return render_template('index.html', 
                           processed=True,
                           video_name=video_filename,
                           csv_link=csv_filename,
                           graph_link=graph_filename,
                           timeline_data=timeline_data,      # For frame-wise table
                           emotion_summary=emotion_summary, # For summary table
                           dominant_emotion=dominant_emotion)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(RESULTS_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    # Increase max content length to handle large video uploads
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
    app.run(host='0.0.0.0', port=7860)