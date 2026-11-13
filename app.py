from pathlib import Path
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from train_model import load_data, main as train_model, FEATURES
from database import init_db, save_prediction, read_predictions
from assistant import answer

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "heart_disease_pipeline.joblib"
METRICS_PATH = BASE / "models" / "metrics.json"

st.set_page_config(page_title="Healthcare Intelligence | AI Demo", page_icon="🏥", layout="wide")
st.title("🏥 Healthcare Intelligence & ML Research Dashboard")
st.caption("Academic prototype • UCI Heart Disease dataset • Model transparency")

st.error(
    "SAFETY NOTICE: This is an educational machine-learning demonstration, not a medical "
    "device or clinical decision-support tool. It is not validated for real patients. "
    "Do not use outputs for diagnosis, triage, treatment, or medical decisions."
)

@st.cache_resource
def load_model():
    import joblib
    if not MODEL_PATH.exists():
        train_model()
    return joblib.load(MODEL_PATH)

init_db()
if not MODEL_PATH.exists():
    with st.spinner("Training the demonstration model for the first time..."):
        train_model()
        st.cache_resource.clear()
bundle = load_model()
model = bundle["model"]
model_name = bundle["model_name"]

pages = ["Overview", "Dataset Explorer", "Model Evaluation", "Research Prediction Demo", "Feature Explanation", "Learning Assistant", "Saved Demo Runs"]
page = st.sidebar.radio("Navigate", pages)
st.sidebar.markdown("---")
st.sidebar.write(f"**Selected model:** {model_name}")
st.sidebar.caption("Research prototype — not for clinical use.")

X, y = load_data()

if page == "Overview":
    st.subheader("Project overview")
    a, b, c = st.columns(3)
    a.metric("Dataset records", len(X))
    b.metric("Input features", len(FEATURES))
    c.metric("Saved demo runs", len(read_predictions()))
    st.markdown("### Project modules")
    st.write("Data processing • Classification • Model evaluation • Feature explanation • API-ready pipeline • Educational assistant")
    st.markdown("### Target definition")
    st.write("The binary target is 0 when the original UCI `num` value is 0, and 1 when `num` is greater than 0.")
    st.markdown("### Data source")
    st.write("UCI Machine Learning Repository — Heart Disease (Dataset ID 45).")

elif page == "Dataset Explorer":
    st.subheader("Dataset explorer")
    df = pd.read_csv(BASE / "data" / "heart_disease.csv")
    st.write("Raw dataset preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.write("Missing values by feature")
    st.dataframe(df.isna().sum().rename("missing_count"))
    st.write("Numeric summary")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
    fig, ax = plt.subplots()
    ax.hist(X["age"].dropna(), bins=15)
    ax.set_title("Age distribution in the dataset")
    ax.set_xlabel("Age")
    ax.set_ylabel("Records")
    st.pyplot(fig)

elif page == "Model Evaluation":
    st.subheader("Model evaluation")
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
        st.write("Cross-validation comparison (training split)")
        comp = pd.DataFrame(metrics["cross_validation_comparison"]).T
        st.dataframe(comp, use_container_width=True)
        st.write("Holdout test metrics")
        st.json(metrics["holdout_test_metrics"])
        st.warning("These are small-dataset research estimates, not evidence of clinical performance.")
    else:
        st.info("Run `python train_model.py` to generate evaluation metrics.")

elif page == "Research Prediction Demo":
    st.subheader("Research prediction demonstration")
    st.warning("Use only fictional values for software testing. Never enter real patient information.")
    labels = {
        "age": ("Age (years)", 45, 18, 100, 1),
        "sex": ("Sex code (dataset encoding)", 1, 0, 1, 1),
        "cp": ("Chest pain type code", 2, 1, 4, 1),
        "trestbps": ("Resting blood pressure", 130, 80, 220, 1),
        "chol": ("Serum cholesterol", 240, 100, 600, 1),
        "fbs": ("Fasting blood sugar flag", 0, 0, 1, 1),
        "restecg": ("Resting ECG code", 0, 0, 2, 1),
        "thalach": ("Maximum heart rate achieved", 150, 60, 220, 1),
        "exang": ("Exercise-induced angina flag", 0, 0, 1, 1),
        "oldpeak": ("ST depression", 1.0, 0.0, 7.0, 0.1),
        "slope": ("ST segment slope code", 2, 1, 3, 1),
        "ca": ("Major vessels code (0–3)", 0, 0, 3, 1),
        "thal": ("Thal code", 3, 3, 7, 1)
    }
    with st.form("research_prediction"):
        record_label = st.text_input("Fictional record label", "Demo example")
        values = {}
        cols = st.columns(2)
        for i, (key, (label, default, low, high, step)) in enumerate(labels.items()):
            with cols[i % 2]:
                values[key] = st.number_input(label, min_value=float(low), max_value=float(high),
                                               value=float(default), step=float(step))
        submitted = st.form_submit_button("Run research demonstration")
    if submitted:
        row = pd.DataFrame([{k: values[k] for k in FEATURES}], columns=FEATURES)
        probability = float(model.predict_proba(row)[0, 1])
        prediction = int(model.predict(row)[0])
        st.metric("Dataset class predicted", str(prediction))
        st.metric("Model probability for class 1", f"{probability:.1%}")
        st.caption("This is a model output for a fictional input, not an individual's medical risk.")
        if st.button("Save fictional demo run"):
            rid = save_prediction(record_label or "Demo example", prediction, probability)
            st.success(f"Saved local demo run #{rid}.")

elif page == "Feature Explanation":
    st.subheader("Permutation feature importance")
    st.write("This estimates how much model score changes when one input feature is shuffled.")
    with st.spinner("Calculating feature importance..."):
        result = permutation_importance(model, X, y, n_repeats=8, random_state=42,
                                        scoring="roc_auc", n_jobs=-1)
    imp = pd.DataFrame({"feature": FEATURES, "importance_mean": result.importances_mean,
                        "importance_std": result.importances_std}).sort_values("importance_mean", ascending=False)
    st.bar_chart(imp.set_index("feature")["importance_mean"])
    st.dataframe(imp, use_container_width=True)
    st.caption("Importance is model-specific and does not establish causation or medical significance.")

elif page == "Learning Assistant":
    st.subheader("Healthcare and project learning assistant")
    st.info("This is a small local FAQ, not a connected generative AI model.")
    q = st.text_input("Ask about the dataset, model, heart disease, blood pressure, cholesterol or SHAP")
    if st.button("Ask") and q.strip():
        st.write(answer(q))

elif page == "Saved Demo Runs":
    st.subheader("Saved fictional demonstration runs")
    records = read_predictions()
    if records:
        st.dataframe(pd.DataFrame(records), use_container_width=True)
    else:
        st.info("No saved runs yet. Use the Research Prediction Demo page.")
