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
import gdown

# Page Configuration
st.set_page_config(page_title="SENTIMENTIX - AI-Powered Emotion Timeline Generator", layout="wide")

# --- CUSTOM SENTIMENTIX CSS STYLING ---
st.markdown("""
<style>
    /* Global Background & Font */
    .stApp {
        background-color: #121212;
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Top Brand Title */
    .brand-title {
        font-size: 26px;
        font-weight: 800;
        color: #ffc107;
        letter-spacing: 2px;
    }
    
    /* Glowing Main Title */
    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: 800;
        color: #ffc107;
        text-shadow: 0 0 15px rgba(255, 193, 7, 0.5);
        margin-top: 10px;
        margin-bottom: 5px;
    }
    
    .subtitle {
        text-align: center;
        color: #b0b0b0;
        font-size: 16px;
        margin-bottom: 30px;
    }
    
    /* Card Container Styling */
    .custom-card {
        background-color: #1a1a1a;
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 30px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
        margin-bottom: 25px;
    }
    
    /* Section Headings */
    .section-header {
        color: #ffc107;
        font-size: 22px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 20px;
        border-bottom: 1px solid #333333;
        padding-bottom: 10px;
        letter-spacing: 1px;
    }
    
    /* Customizing Streamlit Buttons */
    .stButton>button {
        background-color: #ffc107 !important;
        color: #121212 !important;
        font-weight: bold !important;
        border-radius: 6px !important;
        width: 100%;
        padding: 12px;
        border: none;
        font-size: 16px;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        background-color: #e0a800 !important;
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        color: #ffc107 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- NAVIGATION HEADER ---
col1, col2 = st.columns([6, 4])
with col1:
    st.markdown('<div class="brand-title">SENTIMENTIX</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div style="text-align: right; color: #b0b0b0; padding-top: 5px;"><span style="margin-left: 20px; cursor: pointer;">Analyzer</span><span style="margin-left: 20px; cursor: pointer;">Results</span><span style="margin-left: 20px; cursor: pointer;">Contact</span></div>', unsafe_allow_html=True)

st.markdown('<div class="main-title">AI-Powered Emotion Timeline Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Instantly analyze frame-by-frame sentiment dynamics from any video using deep learning.</div>', unsafe_allow_html=True)

# --- CONFIGURATION & MODEL LOADING ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.h5")

@st.cache_resource
def load_emotion_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.info("Downloading model file from cloud storage... Please wait.")
            file_id = "10wnWscczkl1Jo8SO1dLp-9LvmTqbgxRV"
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
        
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_emotion_model()
emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# --- UPLOAD SECTION CARD ---
st.markdown('<div class="custom-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">1. Upload Video for Analysis</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.video(uploaded_file)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("ANALYZE NOW"):
        if model is None:
            st.error("Model not loaded. Check model download link.")
        else:
            with st.spinner("Processing frames and generating reports... Please wait."):
                cap = cv2.VideoCapture(video_path)
                timeline_data = []
                raw_emotions = []
                frame_count = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    img = cv2.resize(frame, (224, 224))
                    img = img.astype("float32") / 255.0
                    img = np.expand_dims(img, axis=0)

                    preds = model.predict(img, verbose=0)
                    emotion_idx = np.argmax(preds)
                    emotion_label = emotion_labels[emotion_idx]

                    timeline_data.append({'Frame': frame_count, 'Emotion': emotion_label})
                    raw_emotions.append(emotion_label)
                    frame_count += 1

                cap.release()

            if not timeline_data:
                st.error("Could not process video. Check file format or codec.")
            else:
                st.success("Analysis Complete!")

                # --- RESULTS SECTION CARD ---
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="section-header">2. Analysis Results (Video: {uploaded_file.name} | Total Frames: {frame_count})</div>', unsafe_allow_html=True)

                # Calculate Summary
                emotion_counts = Counter(raw_emotions)
                emotion_summary = []
                for emotion, count in emotion_counts.most_common():
                    percentage = round((count / frame_count) * 100, 2)
                    emotion_summary.append({
                        'Emotion': emotion.upper(),
                        'Count': count,
                        'Percentage (%)': f"{percentage}%"
                    })

                dominant_emotion = emotion_summary[0]['Emotion'] if emotion_summary else 'NEUTRAL'
                st.markdown(f"<h3 style='text-align: center; color: #ffc107;'>Primary Emotion: {dominant_emotion}</h3>", unsafe_allow_html=True)

                st.subheader("Dominant Sentiment Summary")
                df_summary = pd.DataFrame(emotion_summary)
                st.dataframe(df_summary, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # --- GRAPH SECTION CARD ---
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">Emotion Timeline Graph</div>', unsafe_allow_html=True)
                df_timeline = pd.DataFrame(timeline_data)
                
                fig, ax = plt.subplots(figsize=(10, 4))
                fig.patch.set_facecolor('#1a1a1a')
                ax.set_facecolor('#121212')
                
                ax.plot(df_timeline['Frame'], df_timeline['Emotion'], marker='o', linestyle='-', markersize=2, color='#ffc107')
                ax.set_title(f"Emotion Timeline: {uploaded_file.name}", color='#ffc107', fontsize=14)
                ax.set_xlabel("Frame Number", color='#b0b0b0')
                ax.set_ylabel("Emotion", color='#b0b0b0')
                ax.tick_params(colors='#b0b0b0')
                ax.grid(True, alpha=0.2, color='#444')
                plt.tight_layout()
                
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)

                # --- FRAME-WISE BREAKDOWN CARD ---
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">Frame-Wise Breakdown</div>', unsafe_allow_html=True)
                df_timeline_display = df_timeline.copy()
                df_timeline_display['Emotion'] = df_timeline_display['Emotion'].str.upper()
                st.dataframe(df_timeline_display, use_container_width=True, height=300)
                
                csv_data = df_timeline.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="DOWNLOAD FULL CSV REPORT",
                    data=csv_data,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_timeline.csv",
                    mime="text/csv"
                )
                st.markdown('</div>', unsafe_allow_html=True)
