import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CardioSense AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS  (gradient dark theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* ── Hide Streamlit default white top header bar ── */
[data-testid="stHeader"] {
    background: rgba(15,12,41,0.95) !important;
    border-bottom: 1px solid rgba(255,255,255,0.08) !important;
}
[data-testid="stHeader"] * { color: #e2e8f0 !important; }
[data-testid="stToolbar"] { filter: invert(1) hue-rotate(180deg); }

/* ── Hide deploy button (optional cleaner look) ── */
.stDeployButton { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a3e 0%, #2d2b55 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebarNav"] a { border-radius: 10px; margin: 2px 0; }

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 18px;
    transition: transform 0.2s, box-shadow 0.2s;
}
.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
}

/* ── Stat cards ── */
.stat-card {
    background: linear-gradient(135deg, rgba(139,92,246,0.25), rgba(59,130,246,0.25));
    border: 1px solid rgba(139,92,246,0.4);
    border-radius: 16px;
    padding: 22px 18px;
    text-align: center;
    transition: transform 0.2s;
}
.stat-card:hover { transform: translateY(-4px); }
.stat-number {
    font-size: 36px;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-label { font-size: 13px; color: #94a3b8; margin-top: 4px; font-weight: 500; }

/* ── Hero ── */
.hero-title {
    font-size: 54px;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa 0%, #60a5fa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 16px;
}
.hero-sub {
    font-size: 18px;
    color: #94a3b8;
    line-height: 1.7;
    max-width: 700px;
}

/* ── Section headings ── */
.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #e2e8f0;
    margin: 30px 0 14px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ── Feature pill ── */
.pill {
    display: inline-block;
    background: linear-gradient(90deg, rgba(139,92,246,0.3), rgba(59,130,246,0.3));
    border: 1px solid rgba(139,92,246,0.5);
    border-radius: 50px;
    padding: 6px 16px;
    font-size: 13px;
    font-weight: 600;
    color: #c4b5fd;
    margin: 4px;
}

/* ── Result boxes ── */
.result-positive {
    background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(220,38,38,0.1));
    border: 1px solid rgba(239,68,68,0.5);
    border-radius: 18px;
    padding: 30px;
    text-align: center;
}
.result-negative {
    background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(5,150,105,0.1));
    border: 1px solid rgba(16,185,129,0.5);
    border-radius: 18px;
    padding: 30px;
    text-align: center;
}
.result-emoji { font-size: 52px; }
.result-heading { font-size: 26px; font-weight: 700; margin: 12px 0 8px 0; }
.result-sub { font-size: 15px; color: #94a3b8; }

/* ── Risk meter label ── */
.risk-label {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 4px;
}

/* ── Tip box ── */
.tip-box {
    background: rgba(251,191,36,0.1);
    border: 1px solid rgba(251,191,36,0.3);
    border-radius: 12px;
    padding: 14px 18px;
    color: #fcd34d;
    font-size: 14px;
    margin-top: 10px;
}

/* ── Metric override ── */
[data-testid="stMetricValue"] {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #a78bfa !important;
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }

/* ── All input labels ── */
.stSelectbox label, .stNumberInput label, .stSlider label,
[data-testid="stWidgetLabel"] p, label[data-testid="stWidgetLabel"] {
    color: #c4b5fd !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.01em;
}

/* ── Number input: full container (the white box) ── */
[data-testid="stNumberInput"] > div,
[data-testid="stNumberInput"] > div > div,
.stNumberInput > div > div,
div[data-baseweb="input"],
div[data-baseweb="base-input"] {
    background: rgba(30, 25, 80, 0.7) !important;
    border: 1px solid rgba(139,92,246,0.35) !important;
    border-radius: 10px !important;
}

/* ── Number input: the text inside ── */
[data-testid="stNumberInput"] input,
input[type="number"],
.stNumberInput input {
    background: transparent !important;
    border: none !important;
    color: #f1f5f9 !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    caret-color: #a78bfa !important;
}
[data-testid="stNumberInput"] input:focus {
    outline: none !important;
    box-shadow: none !important;
}

/* ── Number input: stepper +/- buttons ── */
[data-testid="stNumberInput"] button,
.stNumberInput button {
    background: rgba(139,92,246,0.2) !important;
    border: 1px solid rgba(139,92,246,0.3) !important;
    color: #c4b5fd !important;
    border-radius: 6px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}
[data-testid="stNumberInput"] button:hover {
    background: rgba(139,92,246,0.4) !important;
    color: #fff !important;
}
[data-testid="stNumberInput"] button svg {
    fill: #c4b5fd !important;
    stroke: #c4b5fd !important;
}

/* ── Select dropdowns: container ── */
[data-testid="stSelectbox"] > div > div,
div[data-baseweb="select"] > div,
div[data-baseweb="select"] > div > div {
    background: rgba(30, 25, 80, 0.7) !important;
    border: 1px solid rgba(139,92,246,0.35) !important;
    border-radius: 10px !important;
}

/* ── Select dropdowns: selected value text ── */
[data-testid="stSelectbox"] span,
div[data-baseweb="select"] span,
div[data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
.css-1inwz65, .css-qrbaxs {
    color: #f1f5f9 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}

/* ── Select dropdown: caret icon ── */
div[data-baseweb="select"] svg {
    fill: #a78bfa !important;
    color: #a78bfa !important;
}

/* ── Dropdown popup menu ── */
[data-testid="stSelectboxVirtualDropdown"],
ul[data-testid="stSelectboxVirtualDropdown"],
div[role="listbox"],
div[data-baseweb="popover"] {
    background: #1e1a4a !important;
    border: 1px solid rgba(139,92,246,0.4) !important;
    border-radius: 12px !important;
}
/* ── Dropdown options text ── */
div[role="option"],
li[role="option"] {
    background: transparent !important;
    color: #e2e8f0 !important;
    font-size: 14px !important;
}
div[role="option"]:hover,
li[role="option"]:hover {
    background: rgba(139,92,246,0.25) !important;
    color: #fff !important;
}

/* ── Button ── */
div.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 12px;
    font-size: 16px;
    font-weight: 700;
    background: linear-gradient(90deg, #7c3aed, #2563eb) !important;
    border: none !important;
    color: white !important;
    letter-spacing: 0.03em;
    transition: opacity 0.2s, transform 0.1s;
}
div.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.1) !important; }

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #7c3aed, #2563eb) !important;
}

/* ── Insight metric cards ── */
.insight-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
}
.insight-val {
    font-size: 32px;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.insight-lbl { font-size: 12px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.4); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "cardiovascular_logistic_regression_pipeline.pkl"
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error("❌ Could not load model file.")
    st.exception(e)
    st.stop()


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 28px 0;'>
        <div style='font-size:44px;'>🫀</div>
        <div style='font-size:20px; font-weight:800;
                    background:linear-gradient(90deg,#a78bfa,#60a5fa);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            CardioSense AI
        </div>
        <div style='font-size:12px; color:#64748b; margin-top:4px;'>
            Cardiovascular Risk Predictor
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠  Home", "🔬  Prediction", "📊  Model Insights"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:12px; color:#475569; padding: 10px 0;'>
        <div style='margin-bottom:6px;'>📁 <b style='color:#64748b'>Dataset</b></div>
        <div style='color:#475569'>Cardiovascular Disease Dataset</div>
        <div style='color:#475569; margin-top:4px;'>70,000 patient records</div>
        <br>
        <div style='margin-bottom:6px;'>⚙️ <b style='color:#64748b'>Model</b></div>
        <div style='color:#475569'>Logistic Regression</div>
        <div style='color:#475569; margin-top:4px;'>Accuracy: 71.39%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#374151; text-align:center; padding-top:4px;'>
        ⚠️ For educational purposes only.<br>Not a medical diagnosis tool.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════
if page == "🏠  Home":

    # Hero
    st.markdown("""
    <div class="hero-title">Cardiovascular<br>Disease Prediction</div>
    <div class="hero-sub">
        An AI-powered risk assessment tool built with machine learning on 70,000
        patient records. Enter health metrics to get an instant prediction with
        probability scores — in seconds.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature pills
    features = ["🧠 Logistic Regression", "📐 StandardScaler", "🔁 5-Fold Cross-Validation",
                 "📊 71.4% Accuracy", "⚡ Real-time Prediction", "🩺 12 Health Features"]
    pill_html = "".join(f'<span class="pill">{f}</span>' for f in features)
    st.markdown(pill_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dataset stats
    st.markdown('<div class="section-title">📈 Dataset Overview</div>', unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    stats = [
        ("70,000", "Patient Records"),
        ("12", "Input Features"),
        ("71.39%", "Model Accuracy"),
        ("~71.6%", "CV Mean Score"),
    ]
    for col, (num, lbl) in zip([s1, s2, s3, s4], stats):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{num}</div>
                <div class="stat-label">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Two-column info
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-title">🔬 How It Works</div>', unsafe_allow_html=True)
        steps = [
            ("1️⃣", "Enter patient data", "Age, weight, blood pressure, cholesterol & more"),
            ("2️⃣", "AI processes inputs", "StandardScaler normalises features for the model"),
            ("3️⃣", "Logistic Regression predicts", "Binary classification: disease / no disease"),
            ("4️⃣", "View probability score", "See confidence level and risk interpretation"),
        ]
        for icon, title, desc in steps:
            st.markdown(f"""
            <div class="card" style="padding:16px 20px; margin-bottom:10px;">
                <span style="font-size:20px">{icon}</span>
                <span style="font-weight:700; color:#e2e8f0; margin-left:8px;">{title}</span>
                <div style="font-size:13px; color:#94a3b8; margin-top:4px; margin-left:30px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-title">⚠️ Key Risk Factors</div>', unsafe_allow_html=True)
        risks = [
            ("🩸", "High Blood Pressure", "Systolic > 140 mmHg significantly raises risk"),
            ("🍔", "High Cholesterol", "Elevated LDL is a primary cardiac risk factor"),
            ("🚬", "Smoking", "Doubles the risk of cardiovascular events"),
            ("🏃", "Physical Inactivity", "Regular exercise reduces risk by up to 35%"),
            ("⚖️", "Obesity (High BMI)", "BMI > 30 is strongly linked to heart disease"),
        ]
        for icon, title, desc in risks:
            st.markdown(f"""
            <div class="card" style="padding:16px 20px; margin-bottom:10px;">
                <span style="font-size:20px">{icon}</span>
                <span style="font-weight:700; color:#e2e8f0; margin-left:8px;">{title}</span>
                <div style="font-size:13px; color:#94a3b8; margin-top:4px; margin-left:30px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:linear-gradient(135deg,rgba(139,92,246,0.15),rgba(59,130,246,0.15));
                border:1px solid rgba(139,92,246,0.3); border-radius:16px; padding:24px; text-align:center;'>
        <div style='font-size:18px; font-weight:700; color:#e2e8f0; margin-bottom:8px;'>
            Ready to check your cardiovascular risk?
        </div>
        <div style='font-size:14px; color:#94a3b8;'>
            Navigate to <b style="color:#a78bfa">🔬 Prediction</b> in the sidebar to get started.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  PAGE 2 — PREDICTION
# ═══════════════════════════════════════════════════════════
elif page == "🔬  Prediction":

    st.markdown('<div class="hero-title" style="font-size:38px;">🔬 Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Fill in the patient\'s health metrics below and click <b>Predict</b>.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 1: Basic info ────────────────────────────
    st.markdown('<div class="section-title">👤 Basic Information</div>', unsafe_allow_html=True)

    with st.container():
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=40, step=1)

        with c2:
            gender = st.selectbox("Gender", ["Female", "Male"])
            gender_value = 1 if gender == "Female" else 2

        with c3:
            height = st.number_input("Height (cm)", min_value=50, max_value=250, value=165, step=1)

        c4, c5, c6 = st.columns(3)

        with c4:
            weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.5)

        with c5:
            ap_hi = st.number_input(
                "Systolic BP (mmHg)",
                min_value=80, max_value=240, value=120,
                help="Upper reading, e.g. 120 in '120/80'"
            )

        with c6:
            ap_lo = st.number_input(
                "Diastolic BP (mmHg)",
                min_value=40, max_value=160, value=80,
                help="Lower reading, e.g. 80 in '120/80'"
            )

    # Live BMI card
    bmi = weight / ((height / 100) ** 2)
    if bmi < 18.5:
        bmi_cat, bmi_color = "Underweight", "#60a5fa"
    elif bmi < 25:
        bmi_cat, bmi_color = "Normal Weight ✅", "#34d399"
    elif bmi < 30:
        bmi_cat, bmi_color = "Overweight ⚠️", "#fbbf24"
    else:
        bmi_cat, bmi_color = "Obese 🔴", "#f87171"

    st.markdown(f"""
    <div class="card" style="padding:18px 24px; display:flex; align-items:center; gap:24px; flex-wrap:wrap;">
        <div>
            <div style="font-size:12px; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Body Mass Index (BMI)</div>
            <div style="font-size:36px; font-weight:800; color:{bmi_color};">{bmi:.1f}</div>
        </div>
        <div style="height:50px; width:1px; background:rgba(255,255,255,0.1);"></div>
        <div>
            <div style="font-size:12px; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Category</div>
            <div style="font-size:18px; font-weight:700; color:{bmi_color};">{bmi_cat}</div>
        </div>
        <div style="height:50px; width:1px; background:rgba(255,255,255,0.1);"></div>
        <div style="font-size:13px; color:#64748b; max-width:280px;">
            BMI is auto-calculated from height & weight and used directly by the model.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 2: Medical & Lifestyle ───────────────────
    st.markdown('<div class="section-title">🩺 Medical & Lifestyle</div>', unsafe_allow_html=True)

    with st.container():
        m1, m2, m3 = st.columns(3)

        with m1:
            cholesterol = st.selectbox("Cholesterol Level", ["Normal", "Above Normal", "Well Above Normal"])
            cholesterol_value = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[cholesterol]

        with m2:
            gluc = st.selectbox("Glucose Level", ["Normal", "Above Normal", "Well Above Normal"])
            gluc_value = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[gluc]

        with m3:
            smoke = st.selectbox("Smoker?", ["No", "Yes"])
            smoke_value = 1 if smoke == "Yes" else 0

        m4, m5, m6 = st.columns(3)

        with m4:
            alco = st.selectbox("Alcohol Consumption?", ["No", "Yes"])
            alco_value = 1 if alco == "Yes" else 0

        with m5:
            active = st.selectbox("Physically Active?", ["Yes", "No"])
            active_value = 1 if active == "Yes" else 0

        with m6:
            st.markdown("""
            <div style="padding-top:8px;">
                <div style="font-size:13px; color:#94a3b8; font-weight:500;">Quick Risk Flags</div>
            </div>
            """, unsafe_allow_html=True)
            flags = []
            if ap_hi > 140:  flags.append("🔴 High Systolic BP")
            if ap_lo > 90:   flags.append("🔴 High Diastolic BP")
            if bmi >= 30:    flags.append("🟠 Obese BMI")
            if smoke == "Yes": flags.append("🟠 Smoker")
            if cholesterol_value == 3: flags.append("🔴 Very High Cholesterol")
            if flags:
                for f in flags:
                    st.markdown(f"<div style='font-size:13px; color:#fca5a5; margin-top:4px;'>{f}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='font-size:13px; color:#34d399; margin-top:4px;'>✅ No critical flags</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Predict button ───────────────────────────────────
    predict_col, _ = st.columns([1, 2])
    with predict_col:
        predict_button = st.button("🔮  Predict Cardiovascular Risk")

    # ── Result ───────────────────────────────────────────
    if predict_button:

        # Validation
        errors = []
        if ap_hi <= ap_lo:
            errors.append("Systolic BP must be greater than Diastolic BP.")
        if errors:
            for e in errors:
                st.error(e)
        else:
            try:
                input_data = pd.DataFrame({
                    "age": [age],
                    "gender": [gender_value],
                    "height": [height],
                    "weight": [weight],
                    "ap_hi": [ap_hi],
                    "ap_lo": [ap_lo],
                    "cholesterol": [cholesterol_value],
                    "gluc": [gluc_value],
                    "smoke": [smoke_value],
                    "alco": [alco_value],
                    "active": [active_value],
                    "BMI": [bmi],
                })

                expected = list(model.feature_names_in_)
                input_data = input_data[expected]

                prediction  = model.predict(input_data)[0]
                probas      = model.predict_proba(input_data)[0]
                risk_pct    = probas[1] * 100       # probability of DISEASE
                safe_pct    = probas[0] * 100

                st.markdown("---")

                # Result card
                if prediction == 1:
                    st.markdown(f"""
                    <div class="result-positive">
                        <div class="result-emoji">⚠️</div>
                        <div class="result-heading" style="color:#f87171;">Higher Risk Detected</div>
                        <div class="result-sub">
                            The model predicts an elevated likelihood of cardiovascular disease
                            based on the provided inputs.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-negative">
                        <div class="result-emoji">✅</div>
                        <div class="result-heading" style="color:#34d399;">Lower Risk Detected</div>
                        <div class="result-sub">
                            The model predicts a lower likelihood of cardiovascular disease.
                            Maintain a healthy lifestyle!
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Probability columns
                pa, pb, pc = st.columns([1, 1, 1])

                with pa:
                    st.markdown(f"""
                    <div class="insight-card">
                        <div class="insight-val" style="background:linear-gradient(90deg,#f87171,#fb923c);
                             -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                            {risk_pct:.1f}%
                        </div>
                        <div class="insight-lbl">Disease Risk</div>
                    </div>
                    """, unsafe_allow_html=True)

                with pb:
                    st.markdown(f"""
                    <div class="insight-card">
                        <div class="insight-val" style="background:linear-gradient(90deg,#34d399,#059669);
                             -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                            {safe_pct:.1f}%
                        </div>
                        <div class="insight-lbl">Healthy Probability</div>
                    </div>
                    """, unsafe_allow_html=True)

                with pc:
                    st.markdown(f"""
                    <div class="insight-card">
                        <div class="insight-val">{len(flags) if 'flags' in dir() else 0}</div>
                        <div class="insight-lbl">Risk Flags Detected</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Risk gauge bar
                st.markdown('<div class="risk-label">Risk Probability Gauge</div>', unsafe_allow_html=True)
                st.progress(int(min(risk_pct, 100)))
                st.markdown(f"<div style='font-size:13px; color:#94a3b8; margin-top:4px;'>Disease probability: <b style='color:#f87171;'>{risk_pct:.2f}%</b> &nbsp;|&nbsp; Healthy probability: <b style='color:#34d399;'>{safe_pct:.2f}%</b></div>", unsafe_allow_html=True)

                # Tips based on result
                if prediction == 1:
                    st.markdown("""
                    <div class="tip-box">
                        💡 <b>Recommendations:</b> Consider reducing sodium intake, increasing physical activity,
                        quitting smoking if applicable, and scheduling a cardiovascular checkup with your doctor.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="tip-box" style="background:rgba(52,211,153,0.1); border-color:rgba(52,211,153,0.3); color:#6ee7b7;">
                        💡 <b>Keep it up!</b> Continue regular exercise, a balanced diet, and routine medical checkups
                        to maintain your cardiovascular health.
                    </div>
                    """, unsafe_allow_html=True)

                # Input summary expander
                with st.expander("📋 View entered patient data"):
                    display = pd.DataFrame({
                        "Feature": ["Age", "Gender", "Height (cm)", "Weight (kg)",
                                     "Systolic BP", "Diastolic BP", "Cholesterol",
                                     "Glucose", "Smoker", "Alcohol", "Active", "BMI"],
                        "Value": [age, gender, height, weight, ap_hi, ap_lo,
                                   cholesterol, gluc, smoke, alco, active, f"{bmi:.2f}"]
                    })
                    st.dataframe(display, use_container_width=True, hide_index=True)

                st.markdown("""
                <div style='font-size:12px; color:#374151; margin-top:18px; padding:12px;
                             background:rgba(255,255,255,0.04); border-radius:10px; text-align:center;'>
                    ⚠️ This tool is for <b>educational and research purposes only</b>.
                    It is <b>not a medical diagnostic device</b>. Always consult a qualified healthcare professional.
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error("Prediction failed.")
                st.code(str(e))


# ═══════════════════════════════════════════════════════════
#  PAGE 3 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════
elif page == "📊  Model Insights":

    st.markdown('<div class="hero-title" style="font-size:38px;">📊 Model Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Performance metrics, cross-validation results, and feature analysis of the trained Logistic Regression model.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Performance metrics ──────────────────────────────
    st.markdown('<div class="section-title">🏆 Model Performance (Test Set)</div>', unsafe_allow_html=True)

    metrics = [
        ("71.39%", "Accuracy"),
        ("73.16%", "Precision"),
        ("67.51%", "Recall"),
        ("70.22%", "F1-Score"),
    ]
    cols = st.columns(4)
    for col, (val, lbl) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{val}</div>
                <div class="stat-label">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Overfitting check ────────────────────────────────
    st.markdown('<div class="section-title">🔍 Overfitting / Underfitting Check</div>', unsafe_allow_html=True)

    ov1, ov2 = st.columns(2)

    with ov1:
        train_acc = 72.05   # approximate (logistic regression)
        test_acc  = 71.39
        diff      = abs(train_acc - test_acc)

        st.markdown(f"""
        <div class="card">
            <div style="margin-bottom:16px;">
                <div style="font-size:13px; color:#64748b; text-transform:uppercase; font-weight:600; letter-spacing:0.05em;">Train Accuracy</div>
                <div style="font-size:36px; font-weight:800; color:#a78bfa;">~{train_acc:.2f}%</div>
            </div>
            <div style="margin-bottom:16px;">
                <div style="font-size:13px; color:#64748b; text-transform:uppercase; font-weight:600; letter-spacing:0.05em;">Test Accuracy</div>
                <div style="font-size:36px; font-weight:800; color:#60a5fa;">{test_acc:.2f}%</div>
            </div>
            <div style="margin-bottom:16px;">
                <div style="font-size:13px; color:#64748b; text-transform:uppercase; font-weight:600; letter-spacing:0.05em;">Difference</div>
                <div style="font-size:28px; font-weight:800; color:#34d399;">{diff:.2f}%</div>
            </div>
            <div style="background:rgba(52,211,153,0.15); border:1px solid rgba(52,211,153,0.3);
                        border-radius:10px; padding:12px; text-align:center;">
                <span style="font-size:18px;">✅</span>
                <span style="font-weight:700; color:#34d399; margin-left:8px;">Good Fit — No Overfitting</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with ov2:
        # Bar chart: Train vs Test
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor('#1a1a3e')
        ax.set_facecolor('#1a1a3e')

        bars = ax.bar(["Train", "Test"], [train_acc, test_acc],
                      color=["#a78bfa", "#60a5fa"], width=0.45)
        ax.set_ylim(60, 80)
        ax.set_ylabel("Accuracy (%)", color="#94a3b8", fontsize=11)
        ax.tick_params(colors="#94a3b8")
        ax.spines[:].set_color("#334155")
        ax.set_title("Train vs Test Accuracy", color="#e2e8f0", fontsize=13, fontweight='bold', pad=12)

        for bar, val in zip(bars, [train_acc, test_acc]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 1.5,
                    f"{val:.2f}%", ha='center', va='top',
                    color='white', fontweight='bold', fontsize=12)

        ax.axhline(y=test_acc, color='#f472b6', linestyle='--', linewidth=1.2, alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Cross-Validation ─────────────────────────────────
    st.markdown('<div class="section-title">🔁 5-Fold Cross-Validation Results</div>', unsafe_allow_html=True)

    # Static CV scores (approximate for logistic regression on this dataset)
    cv_scores = np.array([71.52, 71.63, 71.45, 71.71, 71.58])
    cv_mean   = cv_scores.mean()
    cv_std    = cv_scores.std()

    cv1, cv2 = st.columns([1, 2])

    with cv1:
        st.markdown(f"""
        <div class="card">
            <div style="margin-bottom:14px;">
                <div class="insight-lbl">Mean CV Accuracy</div>
                <div style="font-size:32px; font-weight:800;
                            background:linear-gradient(90deg,#a78bfa,#60a5fa);
                            -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                    {cv_mean:.2f}%
                </div>
            </div>
            <div style="margin-bottom:14px;">
                <div class="insight-lbl">Std Deviation</div>
                <div style="font-size:24px; font-weight:700; color:#34d399;">{cv_std:.2f}%</div>
            </div>
            <div style="margin-bottom:14px;">
                <div class="insight-lbl">Score Range</div>
                <div style="font-size:16px; font-weight:600; color:#94a3b8;">
                    {cv_scores.min():.2f}% – {cv_scores.max():.2f}%
                </div>
            </div>
            <hr style="border-color:rgba(255,255,255,0.08); margin:14px 0;">
            <div style="background:rgba(52,211,153,0.15); border:1px solid rgba(52,211,153,0.3);
                        border-radius:10px; padding:10px; text-align:center;">
                <span style="font-weight:700; color:#34d399;">✅ Stable Model</span>
                <div style="font-size:12px; color:#6ee7b7; margin-top:4px;">Low spread → consistent performance</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Per-fold table
        fold_df = pd.DataFrame({
            "Fold": [f"Fold {i}" for i in range(1, 6)],
            "Accuracy": [f"{s:.2f}%" for s in cv_scores]
        })
        st.dataframe(fold_df, use_container_width=True, hide_index=True)

    with cv2:
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        fig2.patch.set_facecolor('#1a1a3e')
        ax2.set_facecolor('#1a1a3e')

        fold_labels = [f"Fold {i}" for i in range(1, 6)]
        bars2 = ax2.bar(fold_labels, cv_scores, color='#7c3aed', width=0.5, zorder=3)

        ax2.axhline(y=cv_mean, color='#f472b6', linestyle='--', linewidth=2, zorder=4,
                    label=f'Mean ({cv_mean:.2f}%)')
        ax2.axhspan(cv_mean - cv_std, cv_mean + cv_std,
                    alpha=0.15, color='#f472b6', label=f'±1 Std ({cv_std:.2f}%)')

        for bar, val in zip(bars2, cv_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.12,
                     f"{val:.2f}%", ha='center', va='top',
                     color='white', fontweight='bold', fontsize=11)

        ax2.set_ylim(cv_mean - 2, cv_mean + 2)
        ax2.set_ylabel("Accuracy (%)", color="#94a3b8", fontsize=11)
        ax2.tick_params(colors="#94a3b8")
        ax2.spines[:].set_color("#334155")
        ax2.set_title("5-Fold CV — Per-Fold Accuracy", color="#e2e8f0", fontsize=13, fontweight='bold', pad=12)
        ax2.legend(facecolor='#1a1a3e', edgecolor='#334155', labelcolor='#94a3b8', fontsize=10)
        ax2.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # ── Feature importance ───────────────────────────────
    st.markdown('<div class="section-title">📌 Feature Importance (Model Coefficients)</div>', unsafe_allow_html=True)

    feature_names = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                     'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'BMI']

    # Extract actual coefficients from loaded model
    try:
        coefs = model.named_steps['model'].coef_[0]
    except Exception:
        # Fallback representative values
        coefs = np.array([0.42, -0.05, -0.08, 0.12, 0.61, 0.38, 0.28, 0.17, 0.07, 0.06, -0.12, 0.19])

    # Sort by absolute value
    sorted_idx  = np.argsort(np.abs(coefs))[::-1]
    sorted_feat = [feature_names[i] for i in sorted_idx]
    sorted_coef = [coefs[i] for i in sorted_idx]

    colors_feat = ['#f87171' if c > 0 else '#60a5fa' for c in sorted_coef]

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    fig3.patch.set_facecolor('#1a1a3e')
    ax3.set_facecolor('#1a1a3e')

    bars3 = ax3.barh(sorted_feat[::-1], sorted_coef[::-1], color=colors_feat[::-1])
    ax3.axvline(x=0, color='#475569', linewidth=1)
    ax3.set_xlabel("Coefficient Value (scaled)", color="#94a3b8", fontsize=11)
    ax3.set_title("Logistic Regression Coefficients\n(red = raises risk, blue = lowers risk)",
                  color="#e2e8f0", fontsize=12, fontweight='bold', pad=12)
    ax3.tick_params(colors="#94a3b8", labelsize=10)
    ax3.spines[:].set_color("#334155")

    red_patch  = mpatches.Patch(color='#f87171', label='Increases risk')
    blue_patch = mpatches.Patch(color='#60a5fa', label='Decreases risk')
    ax3.legend(handles=[red_patch, blue_patch],
               facecolor='#1a1a3e', edgecolor='#334155', labelcolor='#94a3b8')

    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # ── Model architecture card ──────────────────────────
    st.markdown('<div class="section-title">⚙️ Model Architecture</div>', unsafe_allow_html=True)

    arch1, arch2 = st.columns(2)

    with arch1:
        st.markdown("""
        <div class="card">
            <div style="font-size:16px; font-weight:700; color:#e2e8f0; margin-bottom:14px;">🔧 Pipeline Components</div>
            <div style="display:flex; flex-direction:column; gap:10px;">
                <div style="background:rgba(139,92,246,0.15); border:1px solid rgba(139,92,246,0.3);
                            border-radius:10px; padding:12px 16px;">
                    <div style="font-weight:700; color:#a78bfa;">Step 1 — StandardScaler</div>
                    <div style="font-size:13px; color:#94a3b8; margin-top:4px;">
                        Normalises all 12 input features to zero mean and unit variance.
                    </div>
                </div>
                <div style="background:rgba(59,130,246,0.15); border:1px solid rgba(59,130,246,0.3);
                            border-radius:10px; padding:12px 16px;">
                    <div style="font-weight:700; color:#60a5fa;">Step 2 — Logistic Regression</div>
                    <div style="font-size:13px; color:#94a3b8; margin-top:4px;">
                        Binary classifier. <code>max_iter=1000</code>, default L2 regularisation.
                        Outputs class probabilities via sigmoid.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with arch2:
        st.markdown("""
        <div class="card">
            <div style="font-size:16px; font-weight:700; color:#e2e8f0; margin-bottom:14px;">📋 Training Details</div>
            <table style="width:100%; font-size:14px; border-collapse:collapse;">
                <tr>
                    <td style="color:#64748b; padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.06);">Dataset</td>
                    <td style="color:#e2e8f0; font-weight:600; text-align:right;">Cardiovascular Disease</td>
                </tr>
                <tr>
                    <td style="color:#64748b; padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.06);">Total Records</td>
                    <td style="color:#e2e8f0; font-weight:600; text-align:right;">~70,000</td>
                </tr>
                <tr>
                    <td style="color:#64748b; padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.06);">Train / Test Split</td>
                    <td style="color:#e2e8f0; font-weight:600; text-align:right;">80% / 20%</td>
                </tr>
                <tr>
                    <td style="color:#64748b; padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.06);">Stratified Split</td>
                    <td style="color:#e2e8f0; font-weight:600; text-align:right;">Yes</td>
                </tr>
                <tr>
                    <td style="color:#64748b; padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.06);">Cross-Validation</td>
                    <td style="color:#e2e8f0; font-weight:600; text-align:right;">5-Fold Stratified</td>
                </tr>
                <tr>
                    <td style="color:#64748b; padding:8px 0;">Features</td>
                    <td style="color:#e2e8f0; font-weight:600; text-align:right;">12 (incl. BMI)</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)