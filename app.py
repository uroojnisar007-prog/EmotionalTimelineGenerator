import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
import streamlit as st
from tensorflow.keras.models import load_model
import tempfile

st.set_page_config(page_title="Emotion Timeline Generator", layout="wide")

# --- CONFIGURATION ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_folder", "emotion_model.h5")

# Load Model Once Globally with caching
@st.cache_resource
def load_emotion_model():
    try:
        print("Loading Model...")
        model = load_model(MODEL_PATH)
        print("Model Loaded Successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_emotion_model()
emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

st.title("😊 Emotion Timeline Generator")
st.write("Upload a video to analyze facial emotions frame-by-frame.")

# File Uploader
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save uploaded video to a temporary file so OpenCV can read it
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(uploaded_file)
    
    if st.button("Start Analysis"):
        if model is None:
            st.error("Model not loaded. Check saved_folder/emotion_model.h5 path.")
        else:
            with st.spinner("Processing video frames and predicting emotions..."):
                cap = cv2.VideoCapture(video_path)
                timeline_data = []
                raw_emotions = []
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

                    timeline_data.append({'Frame': frame_count, 'Emotion': emotion_label})
                    raw_emotions.append(emotion_label)
                    frame_count += 1

                cap.release()

            if not timeline_data:
                st.error("Could not process video. Check file format or video codec.")
            else:
                st.success("Analysis Complete!")

                # Calculate Summary
                emotion_counts = Counter(raw_emotions)
                emotion_summary = []
                for emotion, count in emotion_counts.most_common():
                    percentage = round((count / frame_count) * 100, 2)
                    emotion_summary.append({
                        'Emotion': emotion,
                        'Count': count,
                        'Percentage (%)': percentage
                    })

                dominant_emotion = emotion_summary[0]['Emotion'] if emotion_summary else 'neutral'
                st.metric(label="Dominant Emotion", value=dominant_emotion.capitalize())

                # Display Summary Table
                st.subheader("Emotion Summary")
                df_summary = pd.DataFrame(emotion_summary)
                st.dataframe(df_summary, use_container_width=True)

                # Generate and Display Graph
                st.subheader("Emotion Timeline Graph")
                df_timeline = pd.DataFrame(timeline_data)
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df_timeline['Frame'], df_timeline['Emotion'], marker='o', linestyle='-', markersize=2, color='#6c5ce7')
                ax.set_title(f"Emotion Timeline: {uploaded_file.name}")
                ax.set_xlabel("Frame Number")
                ax.set_ylabel("Emotion")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                st.pyplot(fig)

                # CSV Download Button
                csv_data = df_timeline.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Timeline CSV",
                    data=csv_data,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_timeline.csv",
                    mime="text/csv"
                )
