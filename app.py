import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime
import random
import string
import os
import uuid
import pandas as pd

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Mpox Health Checker | AI Diagnostic System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ RECORDS / "DATABASE" ------------------
# Simple CSV-based storage: one row per diagnosis (healthy or not).
# Images are saved alongside the CSV so they can be reviewed later.
APP_DIR     = os.path.dirname(os.path.abspath(__file__))
RECORDS_DIR = os.path.join(APP_DIR, "records")
IMAGES_DIR  = os.path.join(RECORDS_DIR, "images")
CSV_PATH    = os.path.join(RECORDS_DIR, "diagnosis_log.csv")

CSV_COLUMNS = [
    "record_id",        # unique ID for this record
    "scan_id",           # the SCAN-###### shown in the UI
    "timestamp",         # when the record was created
    "source",            # File Upload / Camera Capture
    "image_path",        # path to the saved image (relative to app folder)
    "skin_ratio",        # fraction of skin-tone pixels detected
    "image_verdict",     # the CNN's outcome label, e.g. "Healthy", "Monkeypox", "Inconclusive", "Review"
    "predicted_class",   # raw top predicted class from the CNN (may be blank if inconclusive)
    "confidence",        # CNN confidence (0-1) for the predicted class
    "mpox_probability",  # CNN's raw probability specifically for the Monkeypox class
    "symptoms_checked",  # True/False — whether the symptom form was submitted
    "symptom_answers",   # summary string of the symptom answers, blank if not checked
    "final_verdict",     # the overall final outcome shown to the user
]


def _ensure_storage():
    """Create the records folder structure and CSV header if they don't exist yet."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(CSV_PATH, index=False)


def save_diagnosis_record(
    scan_id,
    source,
    img_rgb,
    skin_ratio,
    image_verdict,
    predicted_class=None,
    confidence=None,
    mpox_probability=None,
    symptoms_checked=False,
    symptom_answers=None,
    final_verdict=None,
):
    """Append one diagnosis record to the CSV and save the associated image.
    Called for EVERY outcome — healthy and unhealthy alike — so the log
    reflects the full population of people who used the checker."""
    _ensure_storage()

    record_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save the image as a JPEG next to the CSV, named with the record_id
    # so it's easy to trace a row back to its photo.
    image_filename = f"{record_id}.jpg"
    image_path_abs = os.path.join(IMAGES_DIR, image_filename)
    image_path_rel = os.path.join("records", "images", image_filename)
    try:
        img_bgr_to_save = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path_abs, img_bgr_to_save)
    except Exception:
        image_path_rel = ""  # if saving the image fails, still log the rest of the record

    new_row = {
        "record_id": record_id,
        "scan_id": scan_id,
        "timestamp": timestamp,
        "source": source,
        "image_path": image_path_rel,
        "skin_ratio": round(float(skin_ratio), 4) if skin_ratio is not None else "",
        "image_verdict": image_verdict,
        "predicted_class": predicted_class or "",
        "confidence": round(float(confidence), 4) if confidence is not None else "",
        "mpox_probability": round(float(mpox_probability), 4) if mpox_probability is not None else "",
        "symptoms_checked": symptoms_checked,
        "symptom_answers": symptom_answers or "",
        "final_verdict": final_verdict or image_verdict,
    }

    df = pd.read_csv(CSV_PATH)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)
    return record_id


def load_records():
    """Load all saved diagnosis records, newest first."""
    _ensure_storage()
    df = pd.read_csv(CSV_PATH)
    if not df.empty and "timestamp" in df.columns:
        df = df.sort_values("timestamp", ascending=False)
    return df


# ------------------ GLOBAL STYLES ------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Base reset ─────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0A0F1E !important;
    color: #E2E8F0 !important;
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] .main .block-container {
    max-width: 880px;
    padding-top: 2rem;
}

[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #060B16 !important;
    border-right: 1px solid #1E3A5F;
}
[data-testid="stSidebar"] * {
    color: #CBD5E1 !important;
}
.sidebar-logo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    letter-spacing: 2px;
    color: #00C8B4 !important;
    text-transform: uppercase;
    padding: 4px 0 2px;
}
.sidebar-title {
    font-size: 20px;
    font-weight: 700;
    color: #F0F6FF !important;
    margin: 0 0 18px;
}
.sidebar-section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    color: #00C8B4 !important;
    text-transform: uppercase;
    margin: 22px 0 10px;
    border-top: 1px solid #1E3A5F;
    padding-top: 16px;
}
.sidebar-section:first-of-type { border-top: none; padding-top: 0; }
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    font-size: 12px;
}
.stat-row .stat-label { color: #7A9CC0 !important; }
.stat-row .stat-value {
    font-family: 'JetBrains Mono', monospace;
    color: #E2E8F0 !important;
    font-weight: 600;
}
.stat-value.good  { color: #34D399 !important; }
.stat-value.live  { color: #00C8B4 !important; }
.sidebar-p {
    font-size: 12px;
    line-height: 1.6;
    color: #94A3B8 !important;
}
.sidebar-disclaimer {
    background: #0D1B2E;
    border: 1px solid #1E3A5F;
    border-radius: 8px;
    padding: 12px 14px;
    font-size: 11px;
    line-height: 1.6;
    color: #7A9CC0 !important;
    margin-top: 8px;
}

/* ── Top banner / hero ──────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #0D1B3E 0%, #0A2240 50%, #061226 100%);
    border: 1px solid #1E3A5F;
    border-radius: 16px;
    padding: 36px 40px 28px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,200,180,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 40px;
    width: 140px; height: 140px;
    background: radial-gradient(circle, rgba(30,100,200,0.10) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    color: #00C8B4;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.hero-title {
    font-size: 32px;
    font-weight: 700;
    color: #F0F6FF;
    line-height: 1.2;
    margin: 0 0 8px;
}
.hero-title span { color: #00C8B4; }
.hero-sub {
    font-size: 14px;
    color: #7A9CC0;
    font-weight: 400;
    margin: 0;
}
.badge-row {
    display: flex;
    gap: 10px;
    margin-top: 20px;
    flex-wrap: wrap;
}
.badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 1px;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid;
}
.badge-teal  { color: #00C8B4; border-color: #00C8B430; background: #00C8B408; }
.badge-blue  { color: #60A5FA; border-color: #60A5FA30; background: #60A5FA08; }
.badge-green { color: #34D399; border-color: #34D39930; background: #34D39908; }

/* ── Section labels ─────────────────────────────── */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #00C8B4;
    margin-bottom: 10px;
}

/* ── Upload zone ────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #0D1B2E !important;
    border: 2px dashed #1E3A5F !important;
    border-radius: 12px !important;
    padding: 24px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #00C8B460 !important;
}
[data-testid="stFileUploader"] label {
    color: #7A9CC0 !important;
    font-size: 14px !important;
}

/* ── Camera input ───────────────────────────────── */
[data-testid="stCameraInput"] {
    background: #0D1B2E !important;
    border: 2px dashed #1E3A5F !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
[data-testid="stCameraInput"] label {
    color: #7A9CC0 !important;
    font-size: 14px !important;
}
[data-testid="stCameraInput"] button {
    background: linear-gradient(135deg, #00C8B4, #0891B2) !important;
    color: #0A0F1E !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
}

/* ── Tabs (Upload / Camera) ─────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 6px;
    background: #0D1B2E;
    border: 1px solid #1E3A5F;
    border-radius: 10px;
    padding: 4px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    color: #7A9CC0 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg, #00C8B4, #0891B2) !important;
    color: #0A0F1E !important;
}
[data-testid="stTabs"] [data-testid="stMarkdownContainer"] p { margin: 0; }

/* ── Scan metadata bar ──────────────────────────── */
.scan-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    background: #0D1B2E;
    border: 1px solid #1E3A5F;
    border-radius: 10px;
    padding: 10px 18px;
    margin-top: 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #7A9CC0;
}
.scan-live {
    color: #34D399;
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    font-weight: 600;
}
.scan-live-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #34D399;
    box-shadow: 0 0 0 0 rgba(52,211,153,0.7);
    animation: pulse-dot 1.8s infinite;
    display: inline-block;
}
@keyframes pulse-dot {
    0%   { box-shadow: 0 0 0 0 rgba(52,211,153,0.5); }
    70%  { box-shadow: 0 0 0 6px rgba(52,211,153,0); }
    100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); }
}

/* ── Result card ────────────────────────────────── */
.result-card {
    background: #0D1B2E;
    border: 1px solid #1E3A5F;
    border-radius: 14px;
    padding: 28px 32px;
    margin-top: 24px;
}
.result-card.danger {
    border-color: #C0392B50;
    background: #1A0A0A;
}
.result-card.safe {
    border-color: #00C8B440;
    background: #081A18;
}
.result-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    color: #7A9CC0;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.result-class {
    font-size: 28px;
    font-weight: 700;
    color: #F0F6FF;
    margin: 0;
    text-transform: capitalize;
}
.result-class.mpox { color: #F87171; }
.result-class.safe { color: #34D399; }
.confidence-bar-bg {
    background: #1E3A5F;
    border-radius: 6px;
    height: 8px;
    margin-top: 16px;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.6s ease;
}
.confidence-fill-safe   { background: linear-gradient(90deg, #00C8B4, #34D399); }
.confidence-fill-danger { background: linear-gradient(90deg, #F87171, #EF4444); }
.confidence-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 22px;
    font-weight: 600;
    margin-top: 8px;
}
.confidence-text.safe   { color: #34D399; }
.confidence-text.danger { color: #F87171; }

/* ── Symptom form ───────────────────────────────── */
.symptoms-header {
    background: #1A0A0A;
    border: 1px solid #C0392B40;
    border-radius: 12px;
    padding: 18px 24px;
    margin: 24px 0 20px;
}
.symptoms-header p {
    color: #F87171;
    font-weight: 600;
    font-size: 15px;
    margin: 0;
}
.symptoms-header small {
    color: #9CA3AF;
    font-size: 12px;
}

[data-testid="stSelectbox"] > div > div {
    background: #0D1B2E !important;
    border: 1px solid #1E3A5F !important;
    border-radius: 8px !important;
    color: #E2E8F0 !important;
    font-size: 13px !important;
}
[data-testid="stSelectbox"] label {
    color: #94A3B8 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
}

/* ── Analyze button ─────────────────────────────── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #00C8B4, #0891B2) !important;
    color: #0A0F1E !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 0.5px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 32px !important;
    width: 100% !important;
    margin-top: 12px !important;
    transition: opacity 0.2s !important;
}
[data-testid="stButton"] > button:hover {
    opacity: 0.88 !important;
}

/* ── Final diagnosis card ────────────────────────── */
.final-card {
    border-radius: 14px;
    padding: 28px 32px;
    margin-top: 20px;
    text-align: center;
}
.final-card.positive {
    background: linear-gradient(135deg, #1A0808, #200E0E);
    border: 1.5px solid #F8717160;
}
.final-card.negative {
    background: linear-gradient(135deg, #081A14, #0A1F18);
    border: 1.5px solid #34D39960;
}
.final-card.inconclusive {
    background: linear-gradient(135deg, #111208, #181A08);
    border: 1.5px solid #FBBF2460;
}
.final-card.rejected {
    background: linear-gradient(135deg, #150B1F, #1A0F26);
    border: 1.5px solid #A78BFA60;
}
.final-card.review {
    background: linear-gradient(135deg, #1A1408, #201808);
    border: 1.5px solid #FB923C60;
}
.final-card .verdict {
    font-size: 26px;
    font-weight: 700;
    margin: 0 0 6px;
}
.final-card .verdict.positive { color: #F87171; }
.final-card .verdict.negative { color: #34D399; }
.final-card .verdict.inconclusive { color: #FBBF24; }
.final-card .verdict.rejected { color: #A78BFA; }
.final-card .verdict.review { color: #FB923C; }
.final-card .verdict-sub {
    font-size: 13px;
    color: #94A3B8;
}

/* ── Pipeline info row ──────────────────────────── */
.pipeline-row {
    display: flex;
    gap: 12px;
    margin: 28px 0 0;
    flex-wrap: wrap;
}
.pipeline-step {
    flex: 1;
    min-width: 120px;
    background: #0D1B2E;
    border: 1px solid #1E3A5F;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.pipeline-step .step-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #00C8B4;
    letter-spacing: 1px;
}
.pipeline-step .step-name {
    font-size: 12px;
    font-weight: 600;
    color: #CBD5E1;
    margin-top: 4px;
}
.pipeline-step .step-desc {
    font-size: 10px;
    color: #64748B;
    margin-top: 2px;
}

/* ── Divider ────────────────────────────────────── */
.styled-divider {
    border: none;
    border-top: 1px solid #1E3A5F;
    margin: 28px 0;
}

/* ── Footer ─────────────────────────────────────── */
.footer {
    text-align: center;
    margin-top: 48px;
    padding: 24px 0 16px;
    border-top: 1px solid #1E3A5F;
}
.footer p {
    font-size: 11px;
    color: #334155;
    margin: 0;
}
</style>
""", unsafe_allow_html=True)


# ------------------ LOAD MODELS ------------------
@st.cache_resource
def load_models():
    vgg = load_model("vgg_mpox_model.h5")
    xgb = joblib.load("xgb_mpox_model.pkl")
    return vgg, xgb

vgg_model, xgb_model = load_models()

# Order MUST match training: {'Chickenpox':0,'Cowpox':1,'HFMD':2,'Healthy':3,'Measles':4,'Monkeypox':5}
classes = ["Chickenpox", "Cowpox", "HFMD", "Healthy", "Measles", "Monkeypox"]

# Minimum confidence to trust a prediction — below this = inconclusive.
# Lowered from 0.70 → 0.40: with 76% overall accuracy across 6 visually-similar
# classes, many CORRECT predictions sit in the 40-65% range. The skin-tone
# and review checks now handle "irrelevant image" cases, so this threshold's
# job is just to catch genuinely flat/uncertain predictions (~near 1/6 = 16.7%).
CONFIDENCE_THRESHOLD = 0.40

# Minimum fraction of skin-tone pixels for an image to be considered
# a plausible skin photo — filters out charts, documents, screenshots, etc.
SKIN_PIXEL_THRESHOLD = 0.05

# A skin-lesion close-up usually fills MUCH more of the frame with skin tone
# than something that just happens to contain orange/red/brown pixels
# (e.g. retina scans, wood, food). Below this, treat high-confidence
# results with suspicion.
SKIN_PIXEL_STRONG_THRESHOLD = 0.20

# If confidence is this high AND skin coverage is only "borderline"
# (between SKIN_PIXEL_THRESHOLD and SKIN_PIXEL_STRONG_THRESHOLD),
# flag the result for manual review instead of presenting it as fact —
# real predictions on this model rarely hit such extreme confidence
# on borderline-skin images.
EXTREME_CONFIDENCE_THRESHOLD = 0.85


def is_likely_skin_image(img, threshold=SKIN_PIXEL_THRESHOLD):
    """Returns True if the image contains enough skin-tone pixels
    to plausibly be a photo of skin (vs a chart, document, graph, etc.)"""
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # Skin tone range in YCrCb — works reasonably across skin tones
    lower = np.array([0, 135, 85], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)

    mask = cv2.inRange(img_ycrcb, lower, upper)
    skin_ratio = np.sum(mask > 0) / mask.size

    return skin_ratio >= threshold, skin_ratio


# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.markdown("""

    <div class="sidebar-title">SYSTEM OVERVIEW</div>

    <div class="sidebar-section">Common Symptoms</div>
    <div class="stat-row">
        <span class="stat-label">✓</span>
        <span class="stat-value">Skin Rash</span>
    </div>

    <div class="stat-row">
        <span class="stat-label">✓</span>
        <span class="stat-value">Fever</span>
    </div>

    <div class="stat-row">
        <span class="stat-label">✓</span>
        <span class="stat-value">Body Aches</span>
    </div>

    <div class="stat-row">
        <span class="stat-label">✓</span>
        <span class="stat-value">Swollen Glands</span>
    </div>

    <div class="sidebar-section">How to Use The System</div>
    <div class="stat-row">
        <span class="stat-label">Step 1</span>
        <span class="stat-value">Upload Photo</span>
    </div>
    <div class="stat-row">
        <span class="stat-label">Step 2</span>
        <span class="stat-value">Answer Questions</span>
    </div>
    <div class="stat-row">
        <span class="stat-label">Step 3</span>
        <span class="stat-value">View Results</span>
    </div>

    <div class="sidebar-section">Seek Medical Care If</div>

    <p class="sidebar-p">
    • Rash is spreading rapidly<br>
    • You have persistent fever<br>
    • You have swollen glands<br>
    • Symptoms are worsening
    </p>

    <div class="sidebar-section">About</div>
    <p class="sidebar-p">
        Mpox Health Checker fuses CNN-based image classification with a clinical
        symptom questionnaire to support differential diagnosis of
        Mpox against visually similar conditions (Chickenpox, Cowpox,
        HFMD, Measles).
    </p>
    <div class="sidebar-disclaimer">
        Your Health, Our Priority!!!.
    </div>
    """, unsafe_allow_html=True)


# ------------------ HERO ------------------
st.markdown("""
<div class="hero-banner">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
        <div class="hero-eyebrow">🧬 Multimodal AI Diagnostic Platform</div>
        <div class="scan-live"><span class="scan-live-dot"></span>SYSTEM ONLINE</div>
    </div>
    <h1 class="hero-title">Mpox<span> Health Checker</span></h1>
    <p class="hero-sub">
        Upload a photo of the affected skin area and answer a few simple
        questions. The system will help identify possible signs of Mpox
        and provide guidance on the next steps.
    </p>
    <div class="badge-row">
        <span class="badge badge-teal"> Photo Analysis</span>
        <span class="badge badge-blue"> Symptom Check</span>
        <span class="badge badge-green">Health Guidance</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ------------------ PIPELINE STEPS ------------------
st.markdown("""
<div class="pipeline-row">
    <div class="pipeline-step">
        <div class="step-num">STEP 01</div>
        <div class="step-name">Upload Image</div>
        <div class="step-desc">Skin lesion photo</div>
    </div>
    <div class="pipeline-step">
        <div class="step-num">STEP 02</div>
        <div class="step-name">Photo Analysis</div>
        <div class="step-desc">Analyze the Photo</div>
    </div>
    <div class="pipeline-step">
        <div class="step-num">STEP 03</div>
        <div class="step-name">Symptom Check</div>
        <div class="step-desc">If Mpox suspected</div>
    </div>
    <div class="pipeline-step">
        <div class="step-num">STEP 04</div>
        <div class="step-name">Get Results</div>
        <div class="step-desc">View Guidance</div>
    </div>
</div>
<hr class="styled-divider">
""", unsafe_allow_html=True)


# ------------------ UPLOAD / CAPTURE ------------------
st.markdown('<div class="section-label">STEP 01 — Provide Skin Image</div>', unsafe_allow_html=True)

tab_upload, tab_camera, tab_history = st.tabs(["  Upload Image", "📷  Take Photo", "🗂️  History"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a clear image of affected skin area",
        type=["jpg", "png"],
        label_visibility="visible"
    )

with tab_camera:
    camera_file = st.camera_input(
        "Use your device camera to capture the affected skin area",
        label_visibility="visible"
    )

with tab_history:
    st.markdown('<div class="section-label">Past Diagnosis Records</div>', unsafe_allow_html=True)

    records_df = load_records()

    if records_df.empty:
        st.info("No records yet. Records are saved automatically each time a photo is analyzed.")
    else:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            verdict_options = ["All"] + sorted(records_df["final_verdict"].dropna().unique().tolist())
            verdict_filter = st.selectbox("Filter by result", verdict_options)
        with col_f2:
            source_options = ["All"] + sorted(records_df["source"].dropna().unique().tolist())
            source_filter = st.selectbox("Filter by source", source_options)

        filtered_df = records_df.copy()
        if verdict_filter != "All":
            filtered_df = filtered_df[filtered_df["final_verdict"] == verdict_filter]
        if source_filter != "All":
            filtered_df = filtered_df[filtered_df["source"] == source_filter]

        st.caption(f"Showing {len(filtered_df)} of {len(records_df)} total record(s)")

        st.dataframe(
            filtered_df[[
                "timestamp", "scan_id", "source", "image_verdict",
                "confidence", "symptoms_checked", "final_verdict"
            ]],
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("🖼️ View record photos"):
            n_thumbs = min(len(filtered_df), 12)
            if n_thumbs == 0:
                st.caption("No photos to show for this filter.")
            else:
                thumb_cols = st.columns(4)
                for i in range(n_thumbs):
                    row = filtered_df.iloc[i]
                    img_p = os.path.join(APP_DIR, row["image_path"]) if isinstance(row["image_path"], str) and row["image_path"] else None
                    with thumb_cols[i % 4]:
                        if img_p and os.path.exists(img_p):
                            st.image(img_p, caption=f"{row['scan_id']} · {row['final_verdict']}", use_container_width=True)
                        else:
                            st.caption(f"{row['scan_id']} (image unavailable)")

        csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download records as CSV",
            data=csv_bytes,
            file_name="diagnosis_log.csv",
            mime="text/csv",
        )

# Use whichever input was provided (upload takes priority if both exist)
image_file = uploaded_file if uploaded_file is not None else camera_file


# ------------------ MAIN LOGIC ------------------
if image_file is not None:

    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Generate a scan ID for this session/image — adds a "real record" feel
    if "scan_id" not in st.session_state or st.session_state.get("last_file_id") != image_file.file_id:
        st.session_state["scan_id"] = "SCAN-" + "".join(random.choices(string.digits, k=6))
        st.session_state["last_file_id"] = image_file.file_id

    source_label = "Camera Capture" if image_file is camera_file else "File Upload"

    col_img, col_gap = st.columns([1, 0.05])
    with col_img:
        st.image(img_rgb, caption="Uploaded image", use_container_width=True)

    st.markdown(f"""
    <div class="scan-meta">
        <span>🆔 {st.session_state['scan_id']}</span>
        <span>📅 {datetime.now().strftime('%d %b %Y, %H:%M')}</span>
        <span>📥 Source: {source_label}</span>
        <span class="scan-live"><span class="scan-live-dot"></span>ANALYZING</span>
    </div>
    """, unsafe_allow_html=True)

    # ---------- SKIN-IMAGE PRE-CHECK ----------
    skin_ok, skin_ratio = is_likely_skin_image(img)

    if not skin_ok:
        st.markdown(f"""
        <div class="final-card rejected">
            <div class="verdict rejected">📷 Photo Not Suitable</div>
            <div class="verdict-sub">
                The current image does not provide enough detail
                for reliable analysis.
            
            
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Preprocess & predict
    img_resized = cv2.resize(img, (224, 224))
    img_array   = np.expand_dims(img_resized, axis=0)
    img_input   = tf.keras.applications.vgg16.preprocess_input(img_array)  # ✅ matches training

    with st.spinner("Running CNN analysis…"):
        prediction = vgg_model.predict(img_input)
        probs      = prediction[0]

        predicted_idx   = int(np.argmax(probs))
        predicted_class = classes[predicted_idx]
        confidence      = float(probs[predicted_idx])

        # Safety-net: still flag for symptom check if Mpox crosses 35%,
        # even when it isn't the model's top guess — but don't force
        # the DISPLAYED result to "Monkeypox" when it isn't.
        mpox_idx  = classes.index("Monkeypox")
        mpox_prob = float(probs[mpox_idx])

    is_mpox         = (predicted_class == "Monkeypox") or (mpox_prob >= 0.35)
    is_inconclusive = confidence < CONFIDENCE_THRESHOLD

    # Borderline skin coverage + suspiciously high confidence = likely
    # a non-skin image (retina scans, food, wood grain, etc.) that happens
    # to fall in the skin-tone color range. Flag for manual review.
    is_review = (
        not is_inconclusive
        and skin_ratio < SKIN_PIXEL_STRONG_THRESHOLD
        and confidence >= EXTREME_CONFIDENCE_THRESHOLD
    )

    bar_class    = "danger" if is_mpox else "safe"
    text_class   = "mpox"  if is_mpox else "safe"
    fill_class   = "confidence-fill-danger" if is_mpox else "confidence-fill-safe"
    card_variant = "danger" if is_mpox else "safe"

    if not is_review:
        st.markdown(f"""
        <div class="result-card {card_variant}">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">
                <div>
                    <div class="result-label">Photo Analysis Result</div>
                    <p class="result-class {text_class}">{predicted_class if not is_inconclusive else "—"}</p>
                </div>
                <div style="text-align:right;">
                    <div class="result-label">Result Reliability</div>
                    <div class="confidence-text {bar_class}">{confidence*100:.1f}%</div>
                </div>
            </div>
            <div class="confidence-bar-bg">
                <div class="confidence-bar-fill {fill_class}" style="width:{confidence*100:.1f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------- BORDERLINE SKIN + EXTREME CONFIDENCE → review ----------
    if is_review:
        st.markdown(f"""
        <div class="final-card review">
            <div class="verdict review">🔎 Unusual Result — Review Recommended</div>
            <div class="verdict-sub">
                Only {skin_ratio*100:.1f}% of this image is skin-toned —
                lower than expected for a typical lesion close-up.
                This pattern often indicates a non-skin image (e.g. another type of medical scan,
                or an object with skin-like colors). Please verify this is a clear photo of a
                skin lesion, or upload a different image.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.get("saved_record_for") != st.session_state["scan_id"]:
            save_diagnosis_record(
                scan_id=st.session_state["scan_id"], source=source_label, img_rgb=img_rgb,
                skin_ratio=skin_ratio, image_verdict="Review Recommended",
                predicted_class=predicted_class, confidence=confidence, mpox_probability=mpox_prob,
                final_verdict="Review Recommended",
            )
            st.session_state["saved_record_for"] = st.session_state["scan_id"]

    # ---------- LOW CONFIDENCE → inconclusive ----------
    elif is_inconclusive:
        st.markdown(f"""
        <div class="final-card inconclusive">
            <div class="verdict inconclusive">⚠ Unable to Determine</div>
            <div class="verdict-sub">
                Model confidence is {confidence*100:.1f}%, below the required threshold of {CONFIDENCE_THRESHOLD*100:.0f}%.
                The uploaded image may not be a valid skin lesion, or the quality is insufficient for reliable diagnosis.
                Please upload a clear, close-up photo of the affected skin area.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.get("saved_record_for") != st.session_state["scan_id"]:
            save_diagnosis_record(
                scan_id=st.session_state["scan_id"], source=source_label, img_rgb=img_rgb,
                skin_ratio=skin_ratio, image_verdict="Inconclusive",
                predicted_class=predicted_class, confidence=confidence, mpox_probability=mpox_prob,
                final_verdict="Inconclusive",
            )
            st.session_state["saved_record_for"] = st.session_state["scan_id"]

    # ---------- NOT MPOX → done ----------
    elif not is_mpox:
        st.markdown(f"""
        <div class="final-card negative">
            <div class="verdict negative">✓ {predicted_class} Detected</div>
            <div class="verdict-sub">Image analysis does not indicate Mpox infection. No further screening required.</div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.get("saved_record_for") != st.session_state["scan_id"]:
            save_diagnosis_record(
                scan_id=st.session_state["scan_id"], source=source_label, img_rgb=img_rgb,
                skin_ratio=skin_ratio, image_verdict=predicted_class,
                predicted_class=predicted_class, confidence=confidence, mpox_probability=mpox_prob,
                final_verdict=predicted_class,
            )
            st.session_state["saved_record_for"] = st.session_state["scan_id"]

    # ---------- MPOX SUSPECTED ----------
    else:
        st.markdown("""
        <div class="symptoms-header">
            <p>⚠️ Possible Signs of Mpox Detected</p>
            <small>The image model has flagged this as a potential Mpox case.
            Please complete the clinical symptom questionnaire below to confirm the diagnosis.</small>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.get("saved_record_for") != st.session_state["scan_id"]:
            save_diagnosis_record(
                scan_id=st.session_state["scan_id"], source=source_label, img_rgb=img_rgb,
                skin_ratio=skin_ratio, image_verdict="Monkeypox (suspected)",
                predicted_class=predicted_class, confidence=confidence, mpox_probability=mpox_prob,
                final_verdict="Monkeypox (suspected — awaiting symptom check)",
            )
            st.session_state["saved_record_for"] = st.session_state["scan_id"]

        st.markdown('<div class="section-label">STEP 03 — Clinical Symptom Profile</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            rectal_pain    = st.selectbox("Rectal Pain",              ["No", "Yes"])
            sore_throat    = st.selectbox("Sore Throat",              ["No", "Yes"])
            penile_oedema  = st.selectbox("Penile Oedema",            ["No", "Yes"])
            oral_lesions   = st.selectbox("Oral Lesions",             ["No", "Yes"])
            solitary_lesion= st.selectbox("Solitary Lesion",          ["No", "Yes"])
        with col2:
            swollen_tonsils= st.selectbox("Swollen Tonsils",          ["No", "Yes"])
            hiv_infection  = st.selectbox("HIV Infection",            ["No", "Yes"])
            std            = st.selectbox("Sexually Transmitted Infection", ["No", "Yes"])
            muscle_pain    = st.selectbox("Muscle Aches & Pain",      ["No", "Yes"])
            swollen_lymph  = st.selectbox("Swollen Lymph Nodes",      ["No", "Yes"])

        if st.button("🔍 Check Symptoms"):
            features = np.array([[
                1 if rectal_pain     == "Yes" else 0,
                1 if sore_throat     == "Yes" else 0,
                1 if penile_oedema   == "Yes" else 0,
                1 if oral_lesions    == "Yes" else 0,
                1 if solitary_lesion == "Yes" else 0,
                1 if swollen_tonsils == "Yes" else 0,
                1 if hiv_infection   == "Yes" else 0,
                1 if std             == "Yes" else 0,
                1 if muscle_pain     == "Yes" else 0,
                1 if swollen_lymph   == "Yes" else 0,
            ]])

            with st.spinner("Running XGBoost symptom analysis…"):
                final_prediction = xgb_model.predict(features)

            result_label = str(final_prediction[0])
            is_positive  = "positive" in result_label.lower() or result_label == "1"

            if is_positive:
                st.markdown(f"""
                <div class="final-card positive">
                    <div class="verdict positive">⚠ High Risk Of Mpox</div>
                    <div class="verdict-sub">
                        Both image analysis and symptom profile indicate Mpox infection.
                        Immediate clinical referral is recommended.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.info("""
            What should you do next?

            • Avoid close physical contact if you have a rash
            • Wash your hands regularly
            • Cover affected skin areas
            • Visit a health facility if symptoms worsen
            • Follow Ministry of Health guidance
            """)

                symptom_summary = (
                    f"rectal_pain={rectal_pain}; sore_throat={sore_throat}; penile_oedema={penile_oedema}; "
                    f"oral_lesions={oral_lesions}; solitary_lesion={solitary_lesion}; swollen_tonsils={swollen_tonsils}; "
                    f"hiv_infection={hiv_infection}; std={std}; muscle_pain={muscle_pain}; swollen_lymph={swollen_lymph}"
                )
                save_diagnosis_record(
                    scan_id=st.session_state["scan_id"], source=source_label, img_rgb=img_rgb,
                    skin_ratio=skin_ratio, image_verdict="Monkeypox (suspected)",
                    predicted_class=predicted_class, confidence=confidence, mpox_probability=mpox_prob,
                    symptoms_checked=True, symptom_answers=symptom_summary,
                    final_verdict="High Risk Of Mpox",
                )
            else:
                st.markdown(f"""
                <div class="final-card negative">
                    <div class="verdict negative">✓ Low Risk Of Mpox</div>
                    <div class="verdict-sub">
                        Symptom profile does not support Mpox diagnosis despite image flag.
                        Clinical follow-up is advised.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                symptom_summary = (
                    f"rectal_pain={rectal_pain}; sore_throat={sore_throat}; penile_oedema={penile_oedema}; "
                    f"oral_lesions={oral_lesions}; solitary_lesion={solitary_lesion}; swollen_tonsils={swollen_tonsils}; "
                    f"hiv_infection={hiv_infection}; std={std}; muscle_pain={muscle_pain}; swollen_lymph={swollen_lymph}"
                )
                save_diagnosis_record(
                    scan_id=st.session_state["scan_id"], source=source_label, img_rgb=img_rgb,
                    skin_ratio=skin_ratio, image_verdict="Monkeypox (suspected)",
                    predicted_class=predicted_class, confidence=confidence, mpox_probability=mpox_prob,
                    symptoms_checked=True, symptom_answers=symptom_summary,
                    final_verdict="Low Risk Of Mpox",
                )


# ------------------ FOOTER ------------------
st.markdown("""
<hr class="styled-divider">
<div class="footer">
    <p>Mpox Health Checker · Multimodal AI Diagnostic System · Makerere University</p>
    <p style="margin-top:4px;">For research and academic evaluation only — not a substitute for clinical diagnosis</p>
</div>
""", unsafe_allow_html=True)
