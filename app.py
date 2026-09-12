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
    page_title="Sentimentix",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================================
# STYLING
# Color palette is unchanged from the original app: #121212, #1a1a1a,
# #333333, #ffc107, #e0a800, #b0b0b0. Everything else — type, spacing,
# iconography, layout — has been rebuilt.
# =========================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

:root {
    --bg: #121212;
    --panel: #1a1a1a;
    --panel-alt: #161616;
    --border: #333333;
    --border-soft: #262626;
    --amber: #ffc107;
    --amber-hover: #e0a800;
    --text: #ffffff;
    --muted: #b0b0b0;
    --faint: #6e6e6e;
}

html, body, .stApp {
    background-color: var(--bg);
    color: var(--text);
    font-family: 'Inter', -apple-system, sans-serif;
}

/* Streamlit's own top toolbar sits fixed above the page and was covering
   our custom nav — make it transparent and push content clear of it. */
header[data-testid="stHeader"] {
    background-color: transparent;
    height: 3.2rem;
}
div[data-testid="stToolbar"] { right: 1rem; }

.block-container {
    padding-top: 0.5rem;
    padding-bottom: 4rem;
    max-width: 1180px;
}

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }

/* ---------- NAVBAR ---------- */
.nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 22px;
    margin-bottom: 40px;
    background-color: var(--panel);
    border: 1px solid var(--border-soft);
    border-radius: 8px;
}
.nav-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 17px;
    letter-spacing: 0.2px;
    color: var(--text);
}
.nav-brand .mark {
    width: 9px;
    height: 22px;
    background: var(--amber);
    border-radius: 1px;
    display: inline-block;
}
.nav-links { display: flex; gap: 30px; }
.nav-links span {
    color: var(--muted);
    font-size: 13.5px;
    font-weight: 500;
    cursor: pointer;
    transition: color 0.15s ease;
}
.nav-links span:hover { color: var(--amber); }

/* ---------- HERO ---------- */
.eyebrow {
    color: var(--faint);
    font-size: 13px;
    font-weight: 500;
    margin-bottom: 14px;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 42px;
    line-height: 1.15;
    letter-spacing: -0.5px;
    color: var(--text);
    margin-bottom: 18px;
}
.hero-title .accent { color: var(--amber); }
.hero-copy {
    color: var(--muted);
    font-size: 15.5px;
    line-height: 1.65;
    max-width: 460px;
    margin-bottom: 32px;
}
.steps { display: flex; flex-direction: column; gap: 16px; }
.step-row { display: flex; align-items: flex-start; gap: 14px; }
.step-num {
    flex-shrink: 0;
    width: 24px; height: 24px;
    border: 1px solid var(--border);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 11.5px; font-weight: 600;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
}
.step-text { color: var(--muted); font-size: 14px; padding-top: 2px; }
.step-text b { color: var(--text); font-weight: 600; }

.hero-graphic {
    background-color: var(--panel);
    border: 1px solid var(--border-soft);
    border-radius: 8px;
    padding: 20px 22px 14px 22px;
    height: 100%;
}
.hero-graphic .g-label {
    color: var(--faint);
    font-size: 11.5px;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.3px;
    margin-bottom: 6px;
}

/* ---------- SECTION LABELS ---------- */
.section-divider {
    border: none;
    border-top: 1px solid var(--border-soft);
    margin: 46px 0 34px 0;
}
.section-label {
    color: var(--faint);
    font-size: 12.5px;
    font-weight: 500;
    margin-bottom: 6px;
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 24px;
}

/* ---------- PANELS ---------- */
.panel {
    background-color: var(--panel);
    border: 1px solid var(--border-soft);
    border-radius: 8px;
    padding: 26px 28px;
    margin-bottom: 20px;
}
.panel-icon-row { display: flex; gap: 18px; align-items: flex-start; margin-bottom: 18px; }
.icon-badge {
    flex-shrink: 0;
    width: 38px; height: 38px;
    border: 1px solid var(--border);
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    color: var(--amber);
}
.panel-heading { font-weight: 600; font-size: 16px; color: var(--text); margin-bottom: 3px; }
.panel-subtext { color: var(--muted); font-size: 13.5px; }

/* ---------- FILE UPLOADER ---------- */
section[data-testid="stFileUploaderDropzone"] {
    background-color: var(--bg) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 6px !important;
}
section[data-testid="stFileUploaderDropzone"] small { color: var(--faint) !important; }
section[data-testid="stFileUploaderDropzone"] button {
    background-color: var(--panel) !important;
    color: var(--amber) !important;
    border: 1px solid var(--border) !important;
    border-radius: 5px !important;
}

/* ---------- BUTTONS ---------- */
.stButton>button {
    background-color: var(--amber) !important;
    color: var(--bg) !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif;
    border-radius: 6px !important;
    width: 100%;
    padding: 11px;
    border: none;
    font-size: 14.5px;
    transition: background-color 0.15s ease;
}
.stButton>button:hover { background-color: var(--amber-hover) !important; }

.stDownloadButton>button {
    background-color: transparent !important;
    color: var(--muted) !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    width: 100%;
    padding: 11px;
    border: 1px solid var(--border) !important;
    font-size: 14px;
    transition: all 0.15s ease;
}
.stDownloadButton>button:hover {
    border-color: var(--amber) !important;
    color: var(--amber) !important;
}

/* ---------- PROGRESS BAR ---------- */
div[data-testid="stProgress"] div[role="progressbar"] > div { background-color: var(--amber) !important; }

/* ---------- METRICS ---------- */
div[data-testid="stMetric"] {
    background-color: var(--panel);
    border: 1px solid var(--border-soft);
    border-radius: 8px;
    padding: 16px 18px;
}
div[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 13px !important; }
div[data-testid="stMetricValue"] {
    color: var(--amber) !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ---------- TABS ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border-soft); }
.stTabs [data-baseweb="tab"] {
    color: var(--muted);
    font-weight: 500;
    font-size: 14px;
    padding: 8px 4px;
    margin-right: 24px;
}
.stTabs [aria-selected="true"] {
    color: var(--text) !important;
    border-bottom: 2px solid var(--amber) !important;
}

/* ---------- DATAFRAME ---------- */
div[data-testid="stDataFrame"] { border: 1px solid var(--border-soft); border-radius: 6px; }

/* ---------- FOOTER ---------- */
.footer {
    color: var(--faint);
    font-size: 12.5px;
    margin-top: 60px;
    padding-top: 18px;
    border-top: 1px solid var(--border-soft);
    display: flex;
    justify-content: space-between;
}
</style>
""", unsafe_allow_html=True)

# =========================================================================
# ICONS (inline SVG, inherit currentColor)
# =========================================================================
ICON_UPLOAD = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16V4"/><path d="M6 10l6-6 6 6"/><path d="M4 20h16"/></svg>"""
ICON_PLAY = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polygon points="6 4 20 12 6 20 6 4"/></svg>"""

# =========================================================================
# NAVBAR
# =========================================================================
st.markdown("""
<div class="nav">
    <div class="nav-brand"><span class="mark"></span>Sentimentix</div>
    <div class="nav-links"><span>Analyzer</span><span>Results</span><span>Docs</span></div>
</div>
""", unsafe_allow_html=True)

# =========================================================================
# HERO — asymmetric two-column: headline + process, and a real timeline
# graphic depicting what the tool produces.
# =========================================================================
hero_left, hero_right = st.columns([1.1, 0.9], gap="large")

with hero_left:
    st.markdown("""
    <div class="eyebrow">Video sentiment analysis</div>
    <div class="hero-title">Turn footage into an<br><span class="accent">emotional timeline.</span></div>
    <div class="hero-copy">
        Upload a video and Sentimentix scores every frame with a deep learning
        model, then maps the emotional arc of the footage from start to finish.
    </div>
    <div class="steps">
        <div class="step-row"><div class="step-num">1</div><div class="step-text"><b>Upload</b> a clip in MP4, MOV or AVI.</div></div>
        <div class="step-row"><div class="step-num">2</div><div class="step-text"><b>Every frame</b> is scored across eight emotion classes.</div></div>
        <div class="step-row"><div class="step-num">3</div><div class="step-text"><b>Read the result</b> as a timeline, table, or CSV export.</div></div>
    </div>
    """, unsafe_allow_html=True)

with hero_right:
    st.markdown("""
    <div class="hero-graphic">
        <div class="g-label">emotion_timeline.preview</div>
        <svg width="100%" height="220" viewBox="0 0 460 220" xmlns="http://www.w3.org/2000/svg">
            <line x1="0" y1="60" x2="460" y2="60" stroke="#262626" stroke-width="1"/>
            <line x1="0" y1="112" x2="460" y2="112" stroke="#333333" stroke-width="1" stroke-dasharray="3 4"/>
            <line x1="0" y1="164" x2="460" y2="164" stroke="#262626" stroke-width="1"/>
            <polyline fill="none" stroke="#ffc107" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"
                points="10,150 45,120 80,160 115,90 150,132 185,55 220,105 255,150 290,78 325,140 360,95 395,150 430,118"/>
            <circle cx="185" cy="55" r="4" fill="#ffc107"/>
            <circle cx="80" cy="160" r="4" fill="#666666"/>
            <circle cx="290" cy="78" r="4" fill="#ffc107"/>
            <text x="165" y="40" fill="#ffc107" font-size="11" font-family="IBM Plex Mono, monospace">happy</text>
            <text x="60" y="188" fill="#8a8a8a" font-size="11" font-family="IBM Plex Mono, monospace">sad</text>
            <text x="298" y="64" fill="#ffc107" font-size="11" font-family="IBM Plex Mono, monospace">surprised</text>
            <text x="4" y="108" fill="#666666" font-size="10" font-family="IBM Plex Mono, monospace">neutral</text>
        </svg>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# =========================================================================
# MODEL LOADING (unchanged logic)
# =========================================================================
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

# Per-emotion accent colors — used to color-code frame-wise results in the
# tables and the timeline graph. The core brand palette (background/panel/
# amber accent) is untouched; these are additional, purely for encoding
# which emotion is which at a glance.
EMOTION_COLORS = {
    'angry': '#ff5c5c',
    'calm': '#4dd0e1',
    'disgust': '#8bc34a',
    'fearful': '#b388ff',
    'happy': '#ffc107',
    'neutral': '#b0b0b0',
    'sad': '#5c7cfa',
    'surprised': '#ff8fb3',
}

def _style_emotion_column(val):
    color = EMOTION_COLORS.get(str(val).lower(), '#ffffff')
    return f'color: {color}; font-weight: 600;'

# =========================================================================
# UPLOAD PANEL
# =========================================================================
st.markdown('<div class="section-label">Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Add a video</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="panel">
    <div class="panel-icon-row">
        <div class="icon-badge">{ICON_UPLOAD}</div>
        <div>
            <div class="panel-heading">Choose a file to analyze</div>
            <div class="panel-subtext">MP4, MOV or AVI — shorter clips process faster</div>
        </div>
    </div>
""", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.markdown(f"""
    <div class="panel">
        <div class="panel-icon-row">
            <div class="icon-badge">{ICON_PLAY}</div>
            <div>
                <div class="panel-heading">{uploaded_file.name}</div>
                <div class="panel-subtext">Preview the clip, then run the analysis</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    vcol1, vcol2, vcol3 = st.columns([1, 2, 1])
    with vcol2:
        st.video(uploaded_file)
        st.write("")
        analyze_clicked = st.button("Analyze video")

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
                    progress_bar.progress(pct, text=f"Processing frame {frame_count} of {total_frames_hint}")
                else:
                    progress_bar.progress(0, text=f"Processing frame {frame_count}...")

            cap.release()
            progress_bar.empty()

            if not timeline_data:
                st.error("Could not process video. Check file format or codec.")
            else:
                st.success("Analysis complete.")

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

                # =========================================================
                # RESULTS
                # =========================================================
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                st.markdown('<div class="section-label">Analyzer</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)

                dominant_color = EMOTION_COLORS.get(dominant_emotion.lower(), '#ffc107')

                m1, m2, m3 = st.columns(3)
                m1.metric("Frames analyzed", frame_count)
                m2.metric("Emotions detected", len(emotion_summary))
                with m3:
                    st.markdown(f"""
                    <div style="background-color:#1a1a1a;border:1px solid #262626;border-radius:8px;padding:16px 18px;">
                        <div style="color:#b0b0b0;font-size:13px;margin-bottom:6px;">Dominant emotion</div>
                        <div style="display:flex;align-items:center;gap:8px;">
                            <span style="width:10px;height:10px;border-radius:50%;background-color:{dominant_color};display:inline-block;"></span>
                            <span style="font-family:'IBM Plex Mono',monospace;font-size:1.6rem;font-weight:500;color:{dominant_color};">{dominant_emotion}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.write("")
                tab_summary, tab_graph, tab_frames = st.tabs(["Summary", "Timeline", "Frame data"])

                with tab_summary:
                    df_summary = pd.DataFrame(emotion_summary)
                    styled_summary = df_summary.style.applymap(_style_emotion_column, subset=['Emotion'])
                    st.dataframe(styled_summary, use_container_width=True, hide_index=True)

                with tab_graph:
                    df_timeline = pd.DataFrame(timeline_data)
                    point_colors = df_timeline['Emotion'].map(lambda e: EMOTION_COLORS.get(e, '#ffffff'))

                    fig, ax = plt.subplots(figsize=(10, 4))
                    fig.patch.set_facecolor('#1a1a1a')
                    ax.set_facecolor('#121212')

                    # Thin connecting line for sequence, colored dots per detected emotion
                    ax.plot(df_timeline['Frame'], df_timeline['Emotion'], linestyle='-', linewidth=0.8, color='#3a3a3a', zorder=1)
                    ax.scatter(df_timeline['Frame'], df_timeline['Emotion'], c=point_colors, s=22, zorder=2, edgecolors='none')

                    ax.set_title(f"Emotion timeline — {uploaded_file.name}", color='#ffc107', fontsize=13)
                    ax.set_xlabel("Frame number", color='#b0b0b0')
                    ax.set_ylabel("Emotion", color='#b0b0b0')
                    ax.tick_params(colors='#b0b0b0')
                    ax.grid(True, alpha=0.15, color='#444')
                    for spine in ax.spines.values():
                        spine.set_color('#333333')

                    # Legend only for emotions actually present in this video
                    present_emotions = list(dict.fromkeys(raw_emotions))
                    legend_handles = [
                        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=EMOTION_COLORS.get(e, '#ffffff'),
                                   markersize=7, label=e.upper())
                        for e in present_emotions
                    ]
                    legend = ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.18),
                                        ncol=min(len(legend_handles), 8), frameon=False, fontsize=9)
                    for text in legend.get_texts():
                        text.set_color('#b0b0b0')

                    plt.tight_layout()

                    st.pyplot(fig)

                with tab_frames:
                    df_timeline_display = pd.DataFrame(timeline_data).copy()
                    df_timeline_display['Emotion'] = df_timeline_display['Emotion'].str.upper()
                    styled_frames = df_timeline_display.style.applymap(_style_emotion_column, subset=['Emotion'])
                    st.dataframe(styled_frames, use_container_width=True, height=320, hide_index=True)

                    csv_data = pd.DataFrame(timeline_data).to_csv(index=False).encode('utf-8')
                    st.write("")
                    st.download_button(
                        label="Download CSV report",
                        data=csv_data,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_timeline.csv",
                        mime="text/csv"
                    )

# =========================================================================
# FOOTER
# =========================================================================
st.markdown("""
<div class="footer">
    <span>Sentimentix</span>
    <span>Emotion timeline generator</span>
</div>
""", unsafe_allow_html=True)
