# AI-Powered Healthcare Intelligence — Final Academic Prototype

## Included
- UCI Heart Disease dataset downloader
- Preprocessing pipeline with missing-value imputation and categorical encoding
- Logistic Regression and Random Forest comparison
- Stratified 5-fold cross-validation on the training split
- Holdout test metrics
- Streamlit dashboard
- Permutation feature importance
- Local FAQ learning assistant
- FastAPI prediction endpoint
- SQLite storage for fictional demo runs

## Safety and scope
This is an academic machine-learning prototype, not a medical device or clinically validated
decision-support system. The UCI dataset is small (303 instances), historical, and not
representative of all populations. Outputs must not be used for diagnosis, triage, treatment,
or patient care. Do not enter real patient information. No LLM is connected in this version.

## Setup (Windows)
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Download data
```bash
python download_dataset.py
```
This downloads UCI dataset ID 45 and saves `data/heart_disease.csv`.
Internet access is required for the first download.

## Train and evaluate
```bash
python train_model.py
```
The script:
1. Converts non-numeric missing markers to NaN.
2. Imputes missing numeric/categorical values inside a pipeline.
3. One-hot encodes categorical variables.
4. Compares Logistic Regression and Random Forest using stratified 5-fold CV.
5. Selects the model using training-fold mean ROC-AUC.
6. Evaluates once on a held-out test split.
7. Saves the model and metrics under `models/`.

## Run dashboard
```bash
python -m streamlit run app.py
```
Open the local address shown in the terminal, usually `http://localhost:8501`.

## Run API (second terminal)
```bash
uvicorn api:app --reload
```
API docs: `http://127.0.0.1:8000/docs`

Example JSON for POST `/predict`:
```json
{
  "age": 45, "sex": 1, "cp": 2, "trestbps": 130, "chol": 240,
  "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
  "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 3
}
```

## Suggested final submission contents
- Problem statement and objectives
- Literature review and dataset description
- Data cleaning and preprocessing methodology
- Model comparison and evaluation methodology
- Dashboard screenshots
- Explainability discussion
- Limitations, ethics, privacy and future scope
- UCI dataset citation

## Dataset citation
Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease.
UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X
Dataset license: CC BY 4.0. Credit the dataset source in your report.

## Important limitations
- Only 303 records; small sample.
- Historical dataset, not contemporary clinical practice.
- Holdout/CV metrics are unstable with small data.
- No external validation, prospective evaluation, or clinical validation.
- The demo form accepts fictional inputs and does not represent a safe patient workflow.
- SQLite is for local demo records only; do not use it for identifiable health data.
# amnon-ai-healthcare-project
AI-powered healthcare analytics and patient risk prediction using machine learning

