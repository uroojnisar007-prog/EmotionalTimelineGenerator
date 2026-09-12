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
st.set_page_config(
    page_title="SENTIMENTIX - AI-Powered Emotion Timeline Generator",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- CUSTOM SENTIMENTIX CSS STYLING ---
# NOTE: color palette is unchanged from the original app (#121212, #ffc107,
# #1a1a1a, #333333, #b0b0b0, #e0a800) — only layout/spacing/typography/UX
# has been reworked.
st.markdown("""
<style>
    /* Global Background & Font */
    .stApp {
        background-color: #121212;
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Tighten default Streamlit top padding for a more "app-like" feel */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    /* --- NAVBAR --- */
    .navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 22px;
        background-color: #1a1a1a;
        border: 1px solid #333333;
        border-radius: 12px;
        margin-bottom: 28px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
    }
    .brand-title {
        font-size: 22px;
        font-weight: 800;
        color: #ffc107;
        letter-spacing: 2px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .nav-links span {
        margin-left: 26px;
        color: #b0b0b0;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.5px;
        cursor: pointer;
        padding-bottom: 4px;
        border-bottom: 2px solid transparent;
        transition: color 0.2s ease, border-color 0.2s ease;
    }
    .nav-links span:hover {
        color: #ffc107;
        border-bottom: 2px solid #ffc107;
    }

    /* --- HERO / TITLE --- */
    .hero {
        text-align: center;
        margin-bottom: 34px;
    }
    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #ffc107;
        text-shadow: 0 0 15px rgba(255, 193, 7, 0.5);
        margin-bottom: 8px;
        line-height: 1.2;
    }
    .subtitle {
        color: #b0b0b0;
        font-size: 15.5px;
        max-width: 640px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* --- STEP PILL (small numbered tag above section headers) --- */
    .step-pill {
        display: inline-block;
        background-color: rgba(255, 193, 7, 0.12);
        color: #ffc107;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 4px 12px;
        border-radius: 20px;
        border: 1px solid rgba(255, 193, 7, 0.35);
        margin-bottom: 10px;
    }

    /* Card Container Styling */
    .custom-card {
        background-color: #1a1a1a;
        border: 1px solid #333333;
        border-radius: 14px;
        padding: 28px 30px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
        margin-bottom: 22px;
        transition: border-color 0.2s ease;
    }
    .custom-card:hover {
        border-color: #4a4a4a;
    }

    /* Section Headings */
    .section-header {
        color: #ffc107;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 4px;
        letter-spacing: 0.5px;
    }
    .section-subtext {
        color: #808080;
        font-size: 13.5px;
        margin-bottom: 18px;
    }
    .section-divider {
        border: none;
        border-top: 1px solid #333333;
        margin: 0 0 20px 0;
    }

    /* Upload dropzone helper text */
    .upload-hint {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #808080;
        font-size: 13px;
        margin-top: 10px;
    }

    /* Customizing Streamlit Buttons */
    .stButton>button {
        background-color: #ffc107 !important;
        color: #121212 !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        width: 100%;
        padding: 12px;
        border: none;
        font-size: 15.5px;
        letter-spacing: 1px;
        transition: transform 0.15s ease, background-color 0.15s ease;
    }
    .stButton>button:hover {
        background-color: #e0a800 !important;
        transform: translateY(-1px);
    }
    .stButton>button:active {
        transform: translateY(0px);
    }

    /* Download button — keep same accent but slightly distinct outline style */
    .stDownloadButton>button {
        background-color: transparent !important;
        color: #ffc107 !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        width: 100%;
        padding: 12px;
        border: 1.5px solid #ffc107 !important;
        font-size: 14.5px;
        letter-spacing: 0.8px;
        transition: background-color 0.15s ease, color 0.15s ease;
    }
    .stDownloadButton>button:hover {
        background-color: #ffc107 !important;
        color: #121212 !important;
    }

    /* Metric styling */
    div[data-testid="stMetric"] {
        background-color: #121212;
        border: 1px solid #333333;
        border-radius: 10px;
        padding: 14px 10px;
    }
    div[data-testid="stMetricValue"] {
        color: #ffc107 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #b0b0b0 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #333333;
    }
    .stTabs [data-baseweb="tab"] {
        color: #b0b0b0;
        font-weight: 600;
        padding: 10px 16px;
    }
    .stTabs [aria-selected="true"] {
        color: #ffc107 !important;
        border-bottom: 2px solid #ffc107 !important;
    }

    /* Dominant emotion badge */
    .dominant-badge {
        text-align: center;
        margin: 6px auto 22px auto;
    }
    .dominant-badge .label {
        color: #808080;
        font-size: 13px;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .dominant-badge .value {
        display: inline-block;
        color: #121212;
        background-color: #ffc107;
        font-size: 22px;
        font-weight: 800;
        letter-spacing: 1px;
        padding: 8px 26px;
        border-radius: 30px;
        box-shadow: 0 0 20px rgba(255, 193, 7, 0.35);
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #555555;
        font-size: 12.5px;
        margin-top: 40px;
        padding-top: 18px;
        border-top: 1px solid #222222;
    }
</style>
""", unsafe_allow_html=True)

# --- NAVIGATION HEADER ---
st.markdown("""
<div class="navbar">
    <div class="brand-title">🎭 SENTIMENTIX</div>
    <div class="nav-links">
        <span>Analyzer</span><span>Results</span><span>Contact</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- HERO ---
st.markdown("""
<div class="hero">
    <div class="main-title">AI-Powered Emotion Timeline Generator</div>
    <div class="subtitle">Instantly analyze frame-by-frame sentiment dynamics from any video using deep learning.</div>
</div>
""", unsafe_allow_html=True)

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
st.markdown('<div class="step-pill">STEP 1</div>', unsafe_allow_html=True)
st.markdown('<div class="section-header">Upload Video for Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtext">Supported formats: MP4, MOV, AVI &nbsp;•&nbsp; Larger videos take longer to process</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-pill">STEP 2</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Preview & Run Analysis</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-subtext">File: {uploaded_file.name}</div>', unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Compact & Centered Video Layout
    vcol1, vcol2, vcol3 = st.columns([1, 2, 1])
    with vcol2:
        st.video(uploaded_file)
        st.write("")
        analyze_clicked = st.button("▶  ANALYZE NOW")

    st.markdown('</div>', unsafe_allow_html=True)

    if analyze_clicked:
        if model is None:
            st.error("Model not loaded. Check model download link.")
        else:
            cap = cv2.VideoCapture(video_path)
            total_frames_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

            progress_bar = st.progress(0, text="Starting analysis...")

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

                if total_frames_hint:
                    pct = min(frame_count / total_frames_hint, 1.0)
                    progress_bar.progress(pct, text=f"Processing frame {frame_count} / {total_frames_hint}")
                else:
                    progress_bar.progress(0, text=f"Processing frame {frame_count}...")

            cap.release()
            progress_bar.empty()

            if not timeline_data:
                st.error("Could not process video. Check file format or codec.")
            else:
                st.success("Analysis Complete!")

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

                # --- RESULTS SECTION CARD ---
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.markdown('<div class="step-pill">STEP 3</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="section-subtext">Video: {uploaded_file.name}</div>', unsafe_allow_html=True)
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

                # Key metrics row
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Frames Analyzed", frame_count)
                m2.metric("Unique Emotions Detected", len(emotion_summary))
                m3.metric("Dominant Emotion", dominant_emotion)

                st.markdown(f"""
                <div class="dominant-badge">
                    <div class="label">Primary Emotion Detected</div>
                    <div class="value">{dominant_emotion}</div>
                </div>
                """, unsafe_allow_html=True)

                tab_summary, tab_graph, tab_frames = st.tabs(
                    ["📊  Sentiment Summary", "📈  Timeline Graph", "🗂  Frame-Wise Breakdown"]
                )

                with tab_summary:
                    df_summary = pd.DataFrame(emotion_summary)
                    st.dataframe(df_summary, use_container_width=True, hide_index=True)

                with tab_graph:
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

                with tab_frames:
                    df_timeline_display = pd.DataFrame(timeline_data).copy()
                    df_timeline_display['Emotion'] = df_timeline_display['Emotion'].str.upper()
                    st.dataframe(df_timeline_display, use_container_width=True, height=320, hide_index=True)

                    csv_data = pd.DataFrame(timeline_data).to_csv(index=False).encode('utf-8')
                    st.write("")
                    st.download_button(
                        label="⬇  DOWNLOAD FULL CSV REPORT",
                        data=csv_data,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_timeline.csv",
                        mime="text/csv"
                    )

                st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown('<div class="footer">SENTIMENTIX &nbsp;•&nbsp; AI-Powered Emotion Timeline Generator</div>', unsafe_allow_html=True)
