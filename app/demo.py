# Streamlit demo: predict next-visit cognitive decline from one visit's information
# Run from the repo folder with:  streamlit run app/demo.py

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from src import config
from sklearn.model_selection import PredefinedSplit, cross_val_predict
import altair as alt  # chart library that comes with Streamlit
from src.visualization.model_comparison import load_results


# Let this file find our project code in src/ (cuurently one folder up from app/)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Page title and short description
st.set_page_config(page_title="Cognitive Decline Risk", layout="centered")
st.title("Predicting Cognitive Decline")
st.write(
    "Enter information from a patient's visit to estimate the risk that their "
    "Clinical Dementia Rating (CDR) increases by the next visit."
)

# Required disclaimer: this is a class research project, not a critical medical tool
st.caption("Research demo for DSCI 4350 (Group 5). Not for clinical use.")

# Inputs the model uses: the 18 metadata features (asf dropped, it duplicates etiv)
FEATURES = config.MODEL_METADATA_FEATURES


@st.cache_resource  # train once when the app starts, not on every click
def train_model():
    """Train the best model from our tuning (gradient boosting, metadata-only)."""
    df = pd.read_csv(config.ML_READY_CSV)
    train = df[df["split"] == "train"]  # training patients only; test set stays untouched

    # Best settings found by Optuna (reports/tuning_results.csv, CV ROC-AUC 0.655)
    model = HistGradientBoostingClassifier(
        learning_rate=0.0111,
        max_iter=250,
        max_depth=4,
        min_samples_leaf=5,
        l2_regularization=0.0202,
        class_weight="balanced",          # same 6:1 weighting used in tuning
        random_state=config.RANDOM_STATE,
    )
    model.fit(train[FEATURES], train[config.TARGET_COLUMN])
    return model, train


model, train = train_model()

# Small note so the user knows which model is running
st.info(f"Model: tuned gradient boosting on metadata, trained on {len(train)} visit pairs "
        f"from {train['subject_id'].nunique()} patients.")


# ---------------------------------------------------------------- Input form
DAYS_PER_YEAR = 365.25

st.header("Patient information")

# Current visit, in two columns
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (years)", min_value=60, max_value=100, value=76)
    sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
    educ = st.number_input("Years of education", min_value=6, max_value=23, value=15)
    ses = st.select_slider("Socioeconomic status (1 = highest, 5 = lowest)",
                           options=[1, 2, 3, 4, 5], value=2)
with col2:
    mmse = st.slider("MMSE score (0-30, lower = worse)", min_value=0, max_value=30, value=29)
    cdr = st.selectbox("Current CDR", [0.0, 0.5, 1.0])
    etiv = st.number_input("eTIV (estimated total intracranial volume, cm³)",
                           min_value=1100, max_value=2000, value=1460)
    nwbv = st.number_input("nWBV (normalized whole-brain volume)",
                           min_value=0.640, max_value=0.850, value=0.734, step=0.001, format="%.3f")

days_to_next = st.slider("Days until the next visit (prediction horizon)",
                         min_value=180, max_value=1700, value=690, step=30)

# Previous visit (optional): used for the change-since-last-visit features
has_prev = st.checkbox("Patient has a previous visit")
if has_prev:
    p1, p2 = st.columns(2)
    with p1:
        days_since_prev = st.number_input("Days since the previous visit",
                                          min_value=180, max_value=1600, value=700)
        days_since_baseline = st.number_input("Days since the first visit",
                                              min_value=days_since_prev, max_value=2200,
                                              value=days_since_prev)
    with p2:
        prev_mmse = st.slider("MMSE at the previous visit (Mini-Mental State Examination)", 0, 30, value=mmse)
        prev_cdr = st.selectbox("CDR at the previous visit", [0.0, 0.5, 1.0])
        prev_nwbv = st.number_input("nWBV at the previous visit", min_value=0.640,
                                    max_value=0.850, value=nwbv, step=0.001, format="%.3f")
else:
    # First visit: no history, so all change features are 0 (same as in our dataset)
    days_since_prev = days_since_baseline = 0
    prev_mmse, prev_cdr, prev_nwbv = mmse, cdr, nwbv

# Turn the form into one row with the same 18 columns the model was trained on
years_since_prev = days_since_prev / DAYS_PER_YEAR if has_prev else None
patient = pd.DataFrame([{
    "age": age,
    "sex_male": 1 if sex == "Male" else 0,
    "educ": educ,
    "ses": ses,
    "ses_imputed": 0,                                   # entered by the user, not imputed
    "mmse": mmse,
    "cdr": cdr,
    "etiv": etiv,
    "nwbv": nwbv,
    "days_since_baseline": days_since_baseline,
    "days_to_next": days_to_next,
    "has_prev_visit": int(has_prev),
    "days_since_prev": days_since_prev,
    "mmse_change_prev": mmse - prev_mmse,
    "nwbv_change_prev": nwbv - prev_nwbv,
    "cdr_change_prev": cdr - prev_cdr,
    "mmse_rate_prev": (mmse - prev_mmse) / years_since_prev if has_prev else 0.0,
    "nwbv_rate_prev": (nwbv - prev_nwbv) / years_since_prev if has_prev else 0.0,
}])[FEATURES]

# Show the exact values going into the model (collapsed by default)
with st.expander("See the model inputs"):
    st.dataframe(patient.T.rename(columns={0: "value"}))

# ---------------------------------------------------------------- Prediction
st.header("Prediction")

# Model's risk score for this patient (0 to 1)
risk = model.predict_proba(patient)[0, 1]


@st.cache_resource  # computed once
def reference_risks():
    """Out-of-fold risk scores for the training pairs: each pair is scored by a model
    that never saw that patient, so the comparison below is fair."""
    cv = PredefinedSplit(train["cv_fold"])
    return cross_val_predict(model, train[FEATURES], train[config.TARGET_COLUMN],
                             cv=cv, method="predict_proba")[:, 1]


# Where this patient's score falls among the training pairs
percentile = (reference_risks() < risk).mean() * 100

# Same 0.5 cutoff used to compute recall and F1 in our evaluation
if risk >= 0.5:
    st.error("Flagged: higher risk of CDR increasing by the next visit")
else:
    st.success("Not flagged: lower risk of CDR increasing by the next visit")

c1, c2 = st.columns(2)
c1.metric("Risk score", f"{risk:.2f}")
c2.metric("Higher than this % of training visit pairs", f"{percentile:.0f}%")

st.caption(
    "Risk score from 0 to 1; 0.5 or above is flagged. About 14% of visit pairs in our data "
    "declined. The model's cross-validated ROC-AUC is 0.655, so predictions are uncertain "
    "and meant to show how the model works, not to guide care."
)

# ---------------------------------------------------------------- Model comparison
st.header("Why this model?")

st.write(
    "We compared MRI-only, metadata-only, and combined (multimodal) models using "
    "patient-grouped 5-fold cross-validation. The tuned metadata-only gradient boosting "
    "model scored highest (ROC-AUC 0.655). Adding MRI did not improve prediction with "
    "only 25 decline cases in training, and the first CNNs were near chance "
    "(ROC-AUC about 0.52)."
)

# Same results our team's comparison figure uses (baseline, tuned, and CNN runs)
# Loading in the image resulted in a blurry image; instead recreate it with the same data
comparison = load_results()

# One bar per model, best at the top, colored by stage (same colors as our report)
bars = alt.Chart(comparison).mark_bar(clip=True).encode(
    x=alt.X("roc_auc:Q", title="ROC-AUC (patient-grouped 5-fold CV)",
            scale=alt.Scale(domain=[0.45, 0.70])),
    y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
    color=alt.Color("stage:N", title="Stage",
                    scale=alt.Scale(domain=["baseline", "tuned", "CNN"],
                                    range=["#2a78d6", "#123c6e", "#eb6834"])),
    tooltip=["label", "roc_auc", "recall", "f1"],   # shown when you hover a bar
)

# Dashed line at 0.5 = random guessing
chance = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(
    strokeDash=[4, 4], color="gray").encode(x="x:Q")

st.altair_chart(bars + chance)
st.caption("Dashed line = chance (0.5). Hover over a bar to see recall and F1.")