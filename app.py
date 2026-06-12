import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import joblib

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="MpoxNet | AI Diagnostic System",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

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

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }

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
.final-card .verdict {
    font-size: 26px;
    font-weight: 700;
    margin: 0 0 6px;
}
.final-card .verdict.positive { color: #F87171; }
.final-card .verdict.negative { color: #34D399; }
.final-card .verdict.inconclusive { color: #FBBF24; }
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

classes = ["Chickenpox", "Cowpox", "Hand-Foot-Mouth", "Healthy", "Measles", "Mpox"]

# Minimum confidence to trust a prediction — below this = inconclusive
CONFIDENCE_THRESHOLD = 0.70


# ------------------ HERO ------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-eyebrow">🧬 Multimodal AI Diagnostic Platform</div>
    <h1 class="hero-title">Mpox<span>Net</span> Detection System</h1>
    <p class="hero-sub">
        CNN image classification fused with XGBoost symptom analysis
        for high-confidence differential diagnosis
    </p>
    <div class="badge-row">
        <span class="badge badge-teal">VGG16 · Image Model</span>
        <span class="badge badge-blue">XGBoost · Symptom Model</span>
        <span class="badge badge-green">Multimodal Fusion</span>
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
        <div class="step-name">CNN Screening</div>
        <div class="step-desc">VGG16 classification</div>
    </div>
    <div class="pipeline-step">
        <div class="step-num">STEP 03</div>
        <div class="step-name">Symptom Check</div>
        <div class="step-desc">If Mpox suspected</div>
    </div>
    <div class="pipeline-step">
        <div class="step-num">STEP 04</div>
        <div class="step-name">Final Verdict</div>
        <div class="step-desc">Fused diagnosis</div>
    </div>
</div>
<hr class="styled-divider">
""", unsafe_allow_html=True)


# ------------------ UPLOAD ------------------
st.markdown('<div class="section-label">STEP 01 — Upload Skin Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drag & drop or click to browse — JPG or PNG accepted",
    type=["jpg", "png"],
    label_visibility="visible"
)


# ------------------ MAIN LOGIC ------------------
if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    col_img, col_gap = st.columns([1, 0.05])
    with col_img:
        st.image(img_rgb, caption="Uploaded image", use_container_width=True)

    # Preprocess & predict
    img_resized = cv2.resize(img, (224, 224)) / 255.0
    img_input  = np.expand_dims(img_resized, axis=0)

    with st.spinner("Running CNN analysis…"):
        prediction     = vgg_model.predict(img_input)
        confidence     = float(np.max(prediction[0]))
        predicted_idx  = int(np.argmax(prediction[0]))
        predicted_class = classes[predicted_idx]

    is_mpox         = predicted_class == "Mpox"
    is_inconclusive = confidence < CONFIDENCE_THRESHOLD

    bar_class    = "danger" if is_mpox else "safe"
    text_class   = "mpox"  if is_mpox else "safe"
    fill_class   = "confidence-fill-danger" if is_mpox else "confidence-fill-safe"
    card_variant = "danger" if is_mpox else "safe"

    st.markdown(f"""
    <div class="result-card {card_variant}">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">
            <div>
                <div class="result-label">CNN Classification Result</div>
                <p class="result-class {text_class}">{predicted_class if not is_inconclusive else "—"}</p>
            </div>
            <div style="text-align:right;">
                <div class="result-label">Model Confidence</div>
                <div class="confidence-text {bar_class}">{confidence*100:.1f}%</div>
            </div>
        </div>
        <div class="confidence-bar-bg">
            <div class="confidence-bar-fill {fill_class}" style="width:{confidence*100:.1f}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------- LOW CONFIDENCE → inconclusive ----------
    if is_inconclusive:
        st.markdown(f"""
        <div class="final-card inconclusive">
            <div class="verdict inconclusive">⚡ Inconclusive Result</div>
            <div class="verdict-sub">
                Model confidence is {confidence*100:.1f}%, below the required threshold of {CONFIDENCE_THRESHOLD*100:.0f}%.
                The uploaded image may not be a valid skin lesion, or the quality is insufficient for reliable diagnosis.
                Please upload a clear, close-up photo of the affected skin area.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------- NOT MPOX → done ----------
    elif not is_mpox:
        st.markdown(f"""
        <div class="final-card negative">
            <div class="verdict negative">✓ {predicted_class} Detected</div>
            <div class="verdict-sub">Image analysis does not indicate Mpox infection. No further screening required.</div>
        </div>
        """, unsafe_allow_html=True)

    # ---------- MPOX SUSPECTED ----------
    else:
        st.markdown("""
        <div class="symptoms-header">
            <p>⚠️ Mpox Suspected — Symptom Verification Required</p>
            <small>The image model has flagged this as a potential Mpox case.
            Please complete the clinical symptom questionnaire below to confirm the diagnosis.</small>
        </div>
        """, unsafe_allow_html=True)

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

        if st.button("🔍 Run Symptom Analysis"):
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
                    <div class="verdict positive">⚠ Mpox CONFIRMED</div>
                    <div class="verdict-sub">
                        Both image analysis and symptom profile indicate Mpox infection.
                        Immediate clinical referral is recommended.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="final-card negative">
                    <div class="verdict negative">✓ Mpox Not Confirmed</div>
                    <div class="verdict-sub">
                        Symptom profile does not support Mpox diagnosis despite image flag.
                        Clinical follow-up is advised.
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ------------------ FOOTER ------------------
st.markdown("""
<hr class="styled-divider">
<div class="footer">
    <p>MpoxNet · Multimodal AI Diagnostic System · Final Year Project · COCIS, Makerere University</p>
    <p style="margin-top:4px;">For research and academic evaluation only — not a substitute for clinical diagnosis</p>
</div>
""", unsafe_allow_html=True)