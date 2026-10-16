# Student Performance Analytics & Prediction

A beginner-friendly Data Analytics + AI project built with Python, Pandas, Matplotlib, scikit-learn, and Streamlit.

## Project goal
Explore how study time, attendance, assignment completion, sleep, internet access, and previous exam scores relate to student outcomes. The project includes exploratory data analysis (EDA), visualizations, a regression model, and a small interactive dashboard.

## Important dataset note
`data/student_performance.csv` is a **synthetic sample dataset generated for this demo**, not real student records. The model demonstrates a workflow and should not be used to make real academic decisions.

## Features
- Dataset overview and summary statistics
- Correlation heatmap and score distribution chart
- Train/test split and Random Forest regression baseline
- MAE and R² evaluation metrics
- Streamlit interface for entering sample student characteristics and viewing a predicted score

## Setup (Python 3.10+ recommended)
```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Run analysis from terminal
```bash
python analysis.py
```
Charts and evaluation metrics are saved in `outputs/`.

## Files
- `app.py` — interactive dashboard and prediction form
- `analysis.py` — EDA, charts, model training/evaluation
- `data/student_performance.csv` — synthetic dataset
- `reports/project_report.pdf` — project report
- `requirements.txt` — dependencies

## Limitations and responsible use
- The data is synthetic and may not reflect real student populations.
- Correlation does not prove causation.
- Predictions are approximate and can be biased by the assumptions used to create the demo data.
- Do not enter personal or sensitive student data into a public deployment.
