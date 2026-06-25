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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── CSS variables ───────────────────────────────── */
:root {
    --bg-base:      #050A14;
    --bg-panel:     #0A1628;
    --bg-card:      #0D1E35;
    --bg-hover:     #112240;
    --border:       #1A3A5C;
    --border-glow:  #00C8B440;
    --teal:         #00C8B4;
    --teal-dim:     #00C8B420;
    --blue:         #3B82F6;
    --blue-dim:     #3B82F615;
    --green:        #10B981;
    --green-dim:    #10B98115;
    --red:          #EF4444;
    --red-dim:      #EF444415;
    --amber:        #F59E0B;
    --amber-dim:    #F59E0B15;
    --purple:       #8B5CF6;
    --text-primary: #F0F6FF;
    --text-secondary: #94A3B8;
    --text-muted:   #475569;
    --font-mono:    'JetBrains Mono', monospace;
}

/* ── Base reset ─────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stAppViewContainer"] .main .block-container {
    max-width: 920px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}
[data-testid="stHeader"]           { background: transparent !important; }
[data-testid="stDecoration"]       { display: none !important; }
[data-testid="stToolbar"]          { display: none !important; }

/* ── Sidebar ────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #06101E 0%, #050A14 100%) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: #94A3B8 !important; }

.sidebar-logo {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 3px;
    color: var(--teal) !important;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.sidebar-title {
    font-size: 18px;
    font-weight: 800;
    color: var(--text-primary) !important;
    margin: 0 0 20px;
    letter-spacing: -0.3px;
}
.sidebar-section {
    font-family: var(--font-mono);
    font-size: 9px;
    letter-spacing: 2.5px;
    color: var(--teal) !important;
    text-transform: uppercase;
    margin: 24px 0 12px;
    border-top: 1px solid var(--border);
    padding-top: 18px;
}
.sidebar-section:first-of-type { border-top: none; padding-top: 0; }

.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    font-size: 12px;
    border-bottom: 1px solid #0F2035;
}
.stat-row:last-child { border-bottom: none; }
.stat-row .stat-label { color: var(--text-muted) !important; font-size: 11px; }
.stat-row .stat-value {
    font-family: var(--font-mono);
    color: var(--text-primary) !important;
    font-size: 11px;
    font-weight: 600;
}
.stat-value.good  { color: var(--green)  !important; }
.stat-value.live  { color: var(--teal)   !important; }

.sidebar-p {
    font-size: 11px;
    line-height: 1.7;
    color: var(--text-secondary) !important;
}
.sidebar-disclaimer {
    background: linear-gradient(135deg, #0A1628, #081422);
    border: 1px solid var(--border);
    border-left: 3px solid var(--teal);
    border-radius: 8px;
    padding: 12px 14px;
    font-size: 11px;
    line-height: 1.6;
    color: var(--text-secondary) !important;
    margin-top: 12px;
}

/* ── Metric chips (model info row) ──────────────── */
.metric-chip-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin: 14px 0 0;
}
.metric-chip {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 12px;
    flex: 1;
    min-width: 70px;
    text-align: center;
}
.metric-chip .mc-val {
    font-family: var(--font-mono);
    font-size: 16px;
    font-weight: 700;
    color: var(--teal);
    display: block;
    line-height: 1;
}
.metric-chip .mc-lbl {
    font-size: 9px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
    display: block;
}

/* ── Hero banner ────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #071428 0%, #0A1E3A 40%, #050D1E 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 40px 44px 32px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,200,180,0.10) 0%, transparent 65%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 65%);
    border-radius: 50%;
    pointer-events: none;
}
/* animated top border glow */
.hero-banner-glow {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--teal), var(--blue), transparent);
    animation: glow-sweep 4s ease-in-out infinite;
}
@keyframes glow-sweep {
    0%, 100% { opacity: 0.3; }
    50%       { opacity: 1.0; }
}
.hero-eyebrow {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 3.5px;
    color: var(--teal);
    text-transform: uppercase;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.hero-eyebrow::before {
    content: '';
    width: 20px; height: 1px;
    background: var(--teal);
    display: inline-block;
}
.hero-title {
    font-size: 38px;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1.15;
    margin: 0 0 10px;
    letter-spacing: -0.5px;
}
.hero-title span { color: var(--teal); }
.hero-sub {
    font-size: 14px;
    color: var(--text-secondary);
    font-weight: 400;
    margin: 0;
    max-width: 520px;
    line-height: 1.6;
}
.badge-row {
    display: flex;
    gap: 8px;
    margin-top: 22px;
    flex-wrap: wrap;
}
.badge {
    font-family: var(--font-mono);
    font-size: 9px;
    letter-spacing: 1.2px;
    padding: 5px 14px;
    border-radius: 20px;
    border: 1px solid;
    text-transform: uppercase;
}
.badge-teal   { color: var(--teal);  border-color: #00C8B440; background: #00C8B410; }
.badge-blue   { color: var(--blue);  border-color: #3B82F640; background: #3B82F610; }
.badge-green  { color: var(--green); border-color: #10B98140; background: #10B98110; }
.badge-purple { color: var(--purple);border-color: #8B5CF640; background: #8B5CF610; }

/* ── Section labels ─────────────────────────────── */
.section-label {
    font-family: var(--font-mono);
    font-size: 9px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}

/* ── Upload / Camera zones ──────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 14px !important;
    padding: 28px !important;
    transition: all 0.25s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--teal) !important;
    background: var(--bg-hover) !important;
    box-shadow: 0 0 20px var(--teal-dim) !important;
}
[data-testid="stFileUploader"] label { color: var(--text-secondary) !important; }

[data-testid="stCameraInput"] {
    background: var(--bg-card) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 14px !important;
    padding: 18px !important;
}
[data-testid="stCameraInput"] label { color: var(--text-secondary) !important; }
[data-testid="stCameraInput"] button {
    background: linear-gradient(135deg, var(--teal), #0891B2) !important;
    color: #050A14 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
}

/* ── Tabs ───────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 5px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    color: var(--text-secondary) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    border-radius: 9px !important;
    padding: 8px 18px !important;
    transition: all 0.2s !important;
    letter-spacing: 0.3px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg, var(--teal), #0891B2) !important;
    color: #050A14 !important;
    box-shadow: 0 2px 12px rgba(0,200,180,0.30) !important;
}
[data-testid="stTabs"] [data-testid="stMarkdownContainer"] p { margin: 0; }

/* ── Scan metadata bar ──────────────────────────── */
.scan-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 20px;
    margin-top: 16px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-secondary);
}
.scan-meta-item {
    display: flex;
    align-items: center;
    gap: 6px;
}
.scan-live {
    color: var(--green);
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 2px;
    font-weight: 600;
}
.scan-live-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse-dot 1.8s infinite;
    display: inline-block;
}
@keyframes pulse-dot {
    0%   { box-shadow: 0 0 0 0   rgba(16,185,129,0.6); }
    70%  { box-shadow: 0 0 0 7px rgba(16,185,129,0);   }
    100% { box-shadow: 0 0 0 0   rgba(16,185,129,0);   }
}

/* ── Result card ────────────────────────────────── */
.result-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px 32px;
    margin-top: 24px;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 16px 16px 0 0;
}
.result-card.danger {
    border-color: #EF444430;
    background: linear-gradient(135deg, #120808, #180C0C);
}
.result-card.danger::before  { background: linear-gradient(90deg, var(--red), transparent); }
.result-card.safe {
    border-color: #10B98130;
    background: linear-gradient(135deg, #081812, #0A1F16);
}
.result-card.safe::before    { background: linear-gradient(90deg, var(--green), transparent); }

.result-label {
    font-family: var(--font-mono);
    font-size: 9px;
    letter-spacing: 2.5px;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-bottom: 6px;
}
.result-class {
    font-size: 30px;
    font-weight: 800;
    color: var(--text-primary);
    margin: 0;
    letter-spacing: -0.5px;
}
.result-class.mpox { color: var(--red); }
.result-class.safe { color: var(--green); }

.confidence-bar-bg {
    background: #0F2035;
    border-radius: 6px;
    height: 8px;
    margin-top: 16px;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}
.confidence-fill-safe   { background: linear-gradient(90deg, #10B981, #34D399); }
.confidence-fill-danger { background: linear-gradient(90deg, #EF4444, #F87171); }
.confidence-text {
    font-family: var(--font-mono);
    font-size: 26px;
    font-weight: 700;
    margin-top: 4px;
    line-height: 1;
}
.confidence-text.safe   { color: var(--green); }
.confidence-text.danger { color: var(--red);   }

/* ── Symptom form ───────────────────────────────── */
.symptoms-header {
    background: linear-gradient(135deg, #150808, #1C0C0C);
    border: 1px solid #EF444430;
    border-left: 3px solid var(--red);
    border-radius: 12px;
    padding: 18px 24px;
    margin: 24px 0 20px;
}
.symptoms-header p {
    color: #F87171;
    font-weight: 700;
    font-size: 15px;
    margin: 0 0 4px;
}
.symptoms-header small { color: var(--text-secondary); font-size: 12px; }

[data-testid="stSelectbox"] > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-size: 13px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 3px var(--teal-dim) !important;
}
[data-testid="stSelectbox"] label {
    color: var(--text-secondary) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
}

/* ── Buttons ────────────────────────────────────── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--teal) 0%, #0891B2 100%) !important;
    color: #050A14 !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 32px !important;
    width: 100% !important;
    margin-top: 12px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 16px rgba(0,200,180,0.25) !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(0,200,180,0.40) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── Final diagnosis card ────────────────────────── */
.final-card {
    border-radius: 16px;
    padding: 32px 36px;
    margin-top: 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.final-card::after {
    content: '';
    position: absolute;
    bottom: -40px; right: -40px;
    width: 120px; height: 120px;
    border-radius: 50%;
    opacity: 0.06;
}
.final-card.positive {
    background: linear-gradient(135deg, #150808 0%, #1C0E0E 100%);
    border: 1.5px solid #EF444440;
}
.final-card.positive::after { background: var(--red); }
.final-card.negative {
    background: linear-gradient(135deg, #071812 0%, #0A1F18 100%);
    border: 1.5px solid #10B98140;
}
.final-card.negative::after { background: var(--green); }
.final-card.inconclusive {
    background: linear-gradient(135deg, #14110A 0%, #1C1810 100%);
    border: 1.5px solid #F59E0B40;
}
.final-card.inconclusive::after { background: var(--amber); }
.final-card.rejected {
    background: linear-gradient(135deg, #110C1C 0%, #180F28 100%);
    border: 1.5px solid #8B5CF640;
}
.final-card.review {
    background: linear-gradient(135deg, #16120A 0%, #1E1810 100%);
    border: 1.5px solid #FB923C40;
}
.final-card .verdict {
    font-size: 28px;
    font-weight: 800;
    margin: 0 0 8px;
    letter-spacing: -0.3px;
}
.final-card .verdict.positive    { color: #F87171; }
.final-card .verdict.negative    { color: var(--green); }
.final-card .verdict.inconclusive{ color: var(--amber); }
.final-card .verdict.rejected    { color: var(--purple); }
.final-card .verdict.review      { color: #FB923C; }
.final-card .verdict-sub {
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.6;
    max-width: 480px;
    margin: 0 auto;
}

/* ── Pipeline steps ─────────────────────────────── */
.pipeline-row {
    display: flex;
    gap: 10px;
    margin: 24px 0 0;
    flex-wrap: wrap;
}
.pipeline-step {
    flex: 1;
    min-width: 110px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 14px;
    text-align: center;
    position: relative;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.pipeline-step:hover {
    border-color: var(--teal);
    box-shadow: 0 0 16px var(--teal-dim);
}
.pipeline-step .step-num {
    font-family: var(--font-mono);
    font-size: 9px;
    color: var(--teal);
    letter-spacing: 1.5px;
    font-weight: 600;
}
.pipeline-step .step-icon {
    font-size: 20px;
    margin: 6px 0 4px;
    display: block;
}
.pipeline-step .step-name {
    font-size: 12px;
    font-weight: 700;
    color: var(--text-primary);
}
.pipeline-step .step-desc {
    font-size: 10px;
    color: var(--text-muted);
    margin-top: 2px;
}

/* ── Model info panel ───────────────────────────── */
.model-panel {
    display: flex;
    gap: 10px;
    margin: 0 0 24px;
    flex-wrap: wrap;
}
.model-tag {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px 14px;
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--text-secondary);
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.model-tag .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    display: inline-block;
}

/* ── Rotating info banner ───────────────────────── */
.banner-carousel {
    position: relative;
    height: 56px;
    overflow: hidden;
    border-radius: 12px;
    margin-bottom: 20px;
}
.banner-slide {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 56px;
    display: flex;
    align-items: center;
    padding: 0 20px;
    gap: 12px;
    border-radius: 12px;
    opacity: 0;
    animation: banner-cycle 20s infinite;
    border: 1px solid var(--border);
}
.banner-slide:nth-child(1) { animation-delay: 0s;   background: linear-gradient(135deg,#071C38,#0A1E3A); border-color:#3B82F630; }
.banner-slide:nth-child(2) { animation-delay: 5s;   background: linear-gradient(135deg,#071C1A,#0A2220); border-color:#10B98130; }
.banner-slide:nth-child(3) { animation-delay: 10s;  background: linear-gradient(135deg,#1A1008,#201408); border-color:#F59E0B30; }
.banner-slide:nth-child(4) { animation-delay: 15s;  background: linear-gradient(135deg,#150A20,#1C1028); border-color:#8B5CF630; }
@keyframes banner-cycle {
    0%        { opacity:0; transform:translateY(8px);  }
    3%, 22%   { opacity:1; transform:translateY(0);    }
    25%, 100% { opacity:0; transform:translateY(-8px); }
}
.banner-icon { font-size: 22px; flex-shrink: 0; }
.banner-text { flex: 1; }
.banner-title {
    font-size: 12px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 2px;
}
.banner-sub {
    font-size: 11px;
    color: var(--text-secondary);
    line-height: 1.3;
}
.banner-tag {
    font-family: var(--font-mono);
    font-size: 9px;
    padding: 3px 8px;
    border-radius: 6px;
    letter-spacing: 1px;
    flex-shrink: 0;
}

/* ── Footer redesign ────────────────────────────── */
.site-footer {
    margin-top: 60px;
    border-top: 1px solid var(--border);
    padding-top: 36px;
    padding-bottom: 20px;
}
.footer-grid {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr;
    gap: 32px;
    margin-bottom: 28px;
}
.footer-brand .footer-logo {
    font-size: 22px;
    font-weight: 800;
    color: var(--teal);
    letter-spacing: -0.5px;
    margin-bottom: 8px;
}
.footer-brand p {
    font-size: 12px;
    color: var(--text-secondary);
    line-height: 1.7;
    margin: 0;
    max-width: 260px;
}
.footer-col-title {
    font-family: var(--font-mono);
    font-size: 9px;
    letter-spacing: 2px;
    color: var(--teal);
    text-transform: uppercase;
    margin-bottom: 12px;
}
.footer-link {
    display: block;
    font-size: 12px;
    color: var(--text-secondary);
    margin-bottom: 7px;
    text-decoration: none;
    transition: color 0.2s;
}
.footer-link:hover { color: var(--teal); }
.footer-bottom {
    border-top: 1px solid var(--border);
    padding-top: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
}
.footer-bottom p {
    font-size: 11px;
    color: var(--text-muted);
    margin: 0;
    font-family: var(--font-mono);
}
.footer-badge-row { display: flex; gap: 8px; flex-wrap: wrap; }
.footer-badge {
    font-family: var(--font-mono);
    font-size: 9px;
    padding: 3px 10px;
    border-radius: 6px;
    border: 1px solid var(--border);
    color: var(--text-muted);
    letter-spacing: 0.5px;
}

    border: none;
    border-top: 1px solid var(--border);
    margin: 32px 0;
}

/* ── Footer ─────────────────────────────────────── */
.footer {
    text-align: center;
    margin-top: 56px;
    padding: 24px 0 20px;
    border-top: 1px solid var(--border);
}
.footer p {
    font-size: 11px;
    color: var(--text-muted);
    margin: 0;
    font-family: var(--font-mono);
    letter-spacing: 0.5px;
}
.footer-logo {
    font-size: 18px;
    font-weight: 800;
    color: var(--teal);
    letter-spacing: -0.3px;
    margin-bottom: 6px;
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
    <div class="hero-banner-glow"></div>
    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
        <div class="hero-eyebrow">Multimodal AI Diagnostic Platform</div>
        <div class="scan-live"><span class="scan-live-dot"></span>SYSTEM ONLINE</div>
    </div>
    <h1 class="hero-title">Mpox<span> Health Checker</span></h1>
    <p class="hero-sub">
        Upload or capture a photo of the affected skin area and complete a short
        clinical questionnaire. The system fuses deep learning image analysis with
        symptom profiling to screen for Mpox and related conditions.
    </p>
    <div class="badge-row">
        <span class="badge badge-teal">VGG16 CNN</span>
        <span class="badge badge-blue">XGBoost Classifier</span>
        <span class="badge badge-green">Decision-Level Fusion</span>
        <span class="badge badge-purple">6-Class Detection</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="banner-carousel">
    <div class="banner-slide" style="background:rgba(0,200,180,0.07); border-color:#00C8B430;">
        <span class="banner-icon">🧬</span>
        <div class="banner-text">
            <div class="banner-title">VGG16 Deep Learning Engine</div>
            <div class="banner-sub">Pretrained on ImageNet · Fine-tuned on MSLD v2.0 · 537 clinical images · 6 classes</div>
        </div>
        <span class="banner-tag" style="background:#3B82F610;color:#3B82F6;border:1px solid #3B82F630;">CNN MODEL</span>
    </div>
    <div class="banner-slide" style="background:rgba(16,185,129,0.07); border-color:#10B98130;">
        <span class="banner-icon">🛡️</span>
        <div class="banner-text">
            <div class="banner-title">Early Detection Saves Lives</div>
            <div class="banner-sub">Mpox is treatable when caught early — this tool supports faster triage and clinical referral</div>
        </div>
        <span class="banner-tag" style="background:#10B98110;color:#10B981;border:1px solid #10B98130;">HEALTH TIP</span>
    </div>
    <div class="banner-slide" style="background:rgba(245,158,11,0.07); border-color:#F59E0B30;">
        <span class="banner-icon">⚠️</span>
        <div class="banner-text">
            <div class="banner-title">For Best Results</div>
            <div class="banner-sub">Use a clear, well-lit close-up photo of the affected skin area — avoid blurry or distant shots</div>
        </div>
        <span class="banner-tag" style="background:#F59E0B10;color:#F59E0B;border:1px solid #F59E0B30;">USAGE TIP</span>
    </div>
    <div class="banner-slide" style="background:rgba(139,92,246,0.07); border-color:#8B5CF630;">
        <span class="banner-icon">📊</span>
        <div class="banner-text">
            <div class="banner-title">Multimodal Fusion Architecture</div>
            <div class="banner-sub">Image CNN + XGBoost symptom classifier combined via decision-level fusion for higher accuracy</div>
        </div>
        <span class="banner-tag" style="background:#8B5CF610;color:#8B5CF6;border:1px solid #8B5CF630;">ARCHITECTURE</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="model-panel">
    <div class="model-tag"><span class="dot" style="background:#00C8B4;"></span>VGG16 · ImageNet pretrained</div>
    <div class="model-tag"><span class="dot" style="background:#3B82F6;"></span>XGBoost · 10 clinical features</div>
    <div class="model-tag"><span class="dot" style="background:#10B981;"></span>Dataset · MSLD v2.0 · 537 images</div>
    <div class="model-tag"><span class="dot" style="background:#F59E0B;"></span>Classes · 6 skin conditions</div>
</div>
""", unsafe_allow_html=True)


# ------------------ PIPELINE STEPS ------------------
st.markdown("""
<div class="pipeline-row">
    <div class="pipeline-step">
        <div class="step-num">STEP 01</div>
        <span class="step-icon">🖼️</span>
        <div class="step-name">Provide Image</div>
        <div class="step-desc">Upload or capture skin photo</div>
    </div>
    <div class="pipeline-step">
        <div class="step-num">STEP 02</div>
        <span class="step-icon">🧠</span>
        <div class="step-name">CNN Analysis</div>
        <div class="step-desc">VGG16 classifies lesion</div>
    </div>
    <div class="pipeline-step">
        <div class="step-num">STEP 03</div>
        <span class="step-icon">📋</span>
        <div class="step-name">Symptom Check</div>
        <div class="step-desc">XGBoost clinical screen</div>
    </div>
    <div class="pipeline-step">
        <div class="step-num">STEP 04</div>
        <span class="step-icon">📊</span>
        <div class="step-name">Fused Result</div>
        <div class="step-desc">Decision-level fusion</div>
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
    # Also store the SOURCE here, tied to the file_id, so it survives Streamlit
    # reruns (the `image_file is camera_file` identity check breaks on reruns
    # because Streamlit recreates widget objects each time the script runs).
    if "scan_id" not in st.session_state or st.session_state.get("last_file_id") != image_file.file_id:
        st.session_state["scan_id"] = "SCAN-" + "".join(random.choices(string.digits, k=6))
        st.session_state["last_file_id"] = image_file.file_id
        # Determine source now, at the moment we first see this file_id,
        # when widget identity checks are still valid.
        st.session_state["source_label"] = (
            "Camera Capture" if camera_file is not None and image_file.file_id == camera_file.file_id
            else "File Upload"
        )
        st.session_state["saved_record_for"] = None  # reset save-guard for new image

    source_label = st.session_state["source_label"]

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

    # ---------- PREPROCESS & PREDICT ----------
    # For camera captures: apply CLAHE (Contrast Limited Adaptive Histogram
    # Equalization) on the L-channel in LAB colour space to normalise uneven
    # webcam lighting before feeding into the model. Uploaded files are already
    # well-lit curated images so we skip this step for them to avoid altering
    # images that are already close to training-set conditions.
    img_for_model = img.copy()
    if source_label == "Camera Capture":
        lab        = cv2.cvtColor(img_for_model, cv2.COLOR_BGR2LAB)
        l, a, b    = cv2.split(lab)
        clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq       = clahe.apply(l)
        lab_eq     = cv2.merge([l_eq, a, b])
        img_for_model = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    img_resized = cv2.resize(img_for_model, (224, 224))
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

    # Build the per-class probability breakdown HTML.
    # Colours: Monkeypox → red, Healthy → green, all others → amber.
    CLASS_COLORS = {
        "Monkeypox":  "#F87171",   # red
        "Healthy":    "#34D399",   # green
        "Chickenpox": "#FBBF24",   # amber
        "Cowpox":     "#FBBF24",
        "HFMD":       "#FBBF24",
        "Measles":    "#FBBF24",
    }
    breakdown_rows = ""
    for cls, prob in sorted(zip(classes, probs), key=lambda x: x[1], reverse=True):
        pct   = prob * 100
        color = CLASS_COLORS.get(cls, "#FBBF24")
        bold  = "font-weight:700;" if cls == predicted_class else ""
        breakdown_rows += f"""
        <div style="margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="font-size:12px; color:#CBD5E1; {bold}">{cls}</span>
                <span style="font-family:'JetBrains Mono',monospace; font-size:12px;
                             color:{color}; {bold}">{pct:.1f}%</span>
            </div>
            <div style="background:#1E3A5F; border-radius:4px; height:6px; overflow:hidden;">
                <div style="width:{pct:.1f}%; height:100%; border-radius:4px;
                            background:{color}; transition:width 0.5s ease;"></div>
            </div>
        </div>"""

    if not is_review:
        st.markdown(f"""
        <div class="result-card {card_variant}">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">
                <div>
                    <div class="result-label">Photo Analysis Result</div>
                    <p class="result-class {text_class}">{predicted_class if not is_inconclusive else "—"}</p>
                </div>
                <div style="text-align:right;">
                    <div class="result-label">Top Confidence</div>
                    <div class="confidence-text {bar_class}">{confidence*100:.1f}%</div>
                </div>
            </div>
            <div class="confidence-bar-bg" style="margin-bottom:20px;">
                <div class="confidence-bar-fill {fill_class}" style="width:{confidence*100:.1f}%"></div>
            </div>
            <div class="result-label" style="margin-bottom:12px;">PROBABILITY BREAKDOWN — ALL CONDITIONS</div>
            {breakdown_rows}
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
<div class="site-footer">
    <div class="footer-grid">
        <div class="footer-brand">
            <div class="footer-logo">MpoxAI</div>
            <p>A multimodal AI diagnostic platform combining VGG16 deep learning with XGBoost clinical symptom classification. Built for academic research at Makerere University.</p>
        </div>
        <div>
            <div class="footer-col-title">Conditions Screened</div>
            <span class="footer-link">🔴 Monkeypox (Mpox)</span>
            <span class="footer-link">🟠 Chickenpox</span>
            <span class="footer-link">🟡 Cowpox</span>
            <span class="footer-link">🟢 HFMD</span>
            <span class="footer-link">🔵 Measles</span>
            <span class="footer-link">✅ Healthy Skin</span>
        </div>
        <div>
            <div class="footer-col-title">Tech Stack</div>
            <span class="footer-link">TensorFlow 2.18 · Keras</span>
            <span class="footer-link">VGG16 · ImageNet</span>
            <span class="footer-link">XGBoost Classifier</span>
            <span class="footer-link">OpenCV · NumPy</span>
            <span class="footer-link">Streamlit 1.56</span>
            <span class="footer-link">MSLD v2.0 Dataset</span>
        </div>
    </div>
    <div class="footer-bottom">
        <p>© 2026 MpoxAI · Makerere University · For Research & Academic Evaluation Only</p>
        <div class="footer-badge-row">
            <span class="footer-badge">RESEARCH ONLY</span>
            <span class="footer-badge">NOT CLINICAL ADVICE</span>
            <span class="footer-badge">v1.0.0</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
