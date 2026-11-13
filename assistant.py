"""Static educational FAQ; deliberately not a medical advice chatbot."""
ANSWERS = {
    "blood pressure": "Blood pressure is the force of blood against artery walls. A healthcare professional interprets readings in context; this app cannot diagnose hypertension.",
    "cholesterol": "Cholesterol is a waxy substance in the blood. Clinicians interpret different cholesterol measurements together with a person's overall health context.",
    "heart disease": "Heart disease is a broad term for conditions affecting the heart or blood vessels. This project uses a small historical dataset for ML education only.",
    "dataset": "The UCI Heart Disease dataset has 303 instances and 13 commonly used input features. It is historical and small, so results may not generalize.",
    "model": "The model predicts a dataset label based on patterns in historical records. Its output is not an individual's medical risk or a diagnosis.",
    "shap": "SHAP is a method for explaining how input features contribute to a model output. It explains model behaviour, not medical causation."
}

def answer(question):
    q = question.lower()
    for key, value in ANSWERS.items():
        if key in q:
            return value
    return ("I can explain this project's dataset, model, heart disease terminology, blood pressure, "
            "cholesterol or SHAP. I cannot assess symptoms, diagnose conditions, or recommend treatment. "
            "For personal health concerns, speak with a qualified healthcare professional.")
