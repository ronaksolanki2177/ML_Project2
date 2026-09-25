import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Cardio Disease Prediction",
    page_icon="❤️",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    .main {
        background-color: #f8fafc;
    }

    .title {
        font-size: 40px;
        font-weight: 700;
        color: inherit;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: inherit;
        opacity: 0.75;
        margin-bottom: 30px;
    }

    .result-positive {
        background-color: #fee2e2;
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid #dc2626;
        text-align: center;
    }

    .result-negative {
        background-color: #dcfce7;
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid #16a34a;
        text-align: center;
    }

    .result-title {
        font-size: 28px;
        font-weight: 700;
    }

    .result-text {
        font-size: 18px;
        margin-top: 10px;
    }

    div.stButton > button {
        width: 100%;
        height: 50px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: 600;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

from pathlib import Path

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "cardiovascular_decision_tree_19_08.pkl"
    model = joblib.load(model_path)
    return model


try:
    model = load_model()

except Exception as e:

    st.error("Unable to load model.pkl")
    st.exception(e)
    st.stop()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">❤️ Cardiovascular Disease Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Enter patient information to predict the possibility of cardiovascular disease.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# PATIENT INFORMATION
# =========================================================

st.subheader("👤 Patient Information")

col1, col2, col3 = st.columns(3)


# =========================================================
# AGE
# =========================================================

with col1:

    age = st.number_input(
        "Age (Years)",
        min_value=1,
        max_value=120,
        value=40,
        step=1
    )


# =========================================================
# GENDER
# =========================================================

with col2:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    gender_value = 1 if gender == "Female" else 2


# =========================================================
# HEIGHT
# =========================================================

with col3:

    height = st.number_input(
        "Height (cm)",
        min_value=50,
        max_value=250,
        value=165,
        step=1
    )


col4, col5, col6 = st.columns(3)


# =========================================================
# WEIGHT
# =========================================================

with col4:

    weight = st.number_input(
        "Weight (kg)",
        min_value=20.0,
        max_value=250.0,
        value=65.0,
        step=0.5
    )


# =========================================================
# SYSTOLIC BP
# =========================================================

with col5:

    ap_hi = st.number_input(
        "Systolic Blood Pressure",
        min_value=50,
        max_value=250,
        value=120,
        step=1
    )


# =========================================================
# DIASTOLIC BP
# =========================================================

with col6:

    ap_lo = st.number_input(
        "Diastolic Blood Pressure",
        min_value=30,
        max_value=200,
        value=80,
        step=1
    )


# =========================================================
# MEDICAL & LIFESTYLE INFORMATION
# =========================================================

st.subheader("🩺 Medical & Lifestyle Information")

col1, col2, col3 = st.columns(3)


# =========================================================
# CHOLESTEROL
# =========================================================

with col1:

    cholesterol = st.selectbox(
        "Cholesterol Level",
        [
            "Normal",
            "Above Normal",
            "Well Above Normal"
        ]
    )

    cholesterol_value = {
        "Normal": 1,
        "Above Normal": 2,
        "Well Above Normal": 3
    }[cholesterol]


# =========================================================
# GLUCOSE
# =========================================================

with col2:

    gluc = st.selectbox(
        "Glucose Level",
        [
            "Normal",
            "Above Normal",
            "Well Above Normal"
        ]
    )

    gluc_value = {
        "Normal": 1,
        "Above Normal": 2,
        "Well Above Normal": 3
    }[gluc]


# =========================================================
# SMOKING
# =========================================================

with col3:

    smoke = st.selectbox(
        "Smoking",
        ["No", "Yes"]
    )

    smoke_value = 1 if smoke == "Yes" else 0


col4, col5, col6 = st.columns(3)


# =========================================================
# ALCOHOL
# =========================================================

with col4:

    alco = st.selectbox(
        "Alcohol Consumption",
        ["No", "Yes"]
    )

    alco_value = 1 if alco == "Yes" else 0


# =========================================================
# PHYSICAL ACTIVITY
# =========================================================

with col5:

    active = st.selectbox(
        "Physically Active",
        ["No", "Yes"]
    )

    active_value = 1 if active == "Yes" else 0


# =========================================================
# BMI
# =========================================================

with col6:

    bmi = weight / ((height / 100) ** 2)

    st.metric(
        "BMI",
        f"{bmi:.2f}"
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.markdown("### 🔍 Prediction")

predict_button = st.button(
    "Predict Cardiovascular Disease"
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    try:

        # The training data converts age from days to years.
        # Therefore the model input must use age in years.

        age_years = age


        # =====================================================
        # INPUT DATA
        # =====================================================

        input_data = pd.DataFrame({

            "age": [age_years],

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

            # BMI is required by the trained model.
            "BMI": [bmi]

        })


        # =====================================================
        # CHECK MODEL FEATURES
        # =====================================================

        expected_features = list(model.feature_names_in_)

        missing_features = [
            feature
            for feature in expected_features
            if feature not in input_data.columns
        ]

        if missing_features:

            raise ValueError(
                f"Missing model features: {missing_features}"
            )


        # Put features in exactly the same order as training.
        input_data = input_data[expected_features]


        # =====================================================
        # PREDICTION
        # =====================================================

        prediction = model.predict(input_data)[0]


        # =====================================================
        # PROBABILITY
        # =====================================================

        probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            probability = probabilities[int(prediction)] * 100


        # =====================================================
        # RESULT
        # =====================================================

        st.markdown("---")

        if prediction == 1:

            st.markdown(
                """
<div class="result-positive">
    <div class="result-title">
        ⚠️ Cardiovascular Disease Detected
    </div>
    <div class="result-text">
        The model predicts a higher possibility
        of cardiovascular disease.
    </div>
</div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
<div class="result-negative">
    <div class="result-title">
        ✅ No Cardiovascular Disease
    </div>
    <div class="result-text">
        The model predicts a lower possibility
        of cardiovascular disease.
    </div>
</div>
                """,
                unsafe_allow_html=True
            )

        # =====================================================
        # PROBABILITY
        # =====================================================

        if probability is not None:

            st.markdown("### 📊 Prediction Probability")

            st.progress(
                int(probability)
            )

            st.write(
                f"Model confidence for predicted class: "
                f"**{probability:.2f}%**"
            )


        # =====================================================
        # SHOW INPUT DATA
        # =====================================================

        with st.expander("View Input Data"):

            display_data = input_data.copy()

            display_data["age"] = age

            st.dataframe(
                display_data,
                use_container_width=True
            )


    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.code(
            str(e)
        )

        st.warning(
            "Please make sure the model and UI use the same "
            "feature names and preprocessing."
        )