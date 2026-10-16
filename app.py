from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "student_performance.csv")
FEATURES = [
    "study_hours_per_day", "attendance_percent",
    "assignments_completed_out_of_10", "sleep_hours",
    "previous_exam_score"
]
X = df[FEATURES].copy()
X["internet_access_yes"] = (df["internet_access"] == "Yes").astype(int)
y = df["final_score"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

st.set_page_config(page_title="Student Performance Analytics", page_icon="📊", layout="wide")
st.title("📊 Student Performance Analytics & Prediction")
st.caption("Educational demo using a synthetic dataset. Predictions are illustrative, not official grades.")
c1, c2, c3 = st.columns(3)
c1.metric("Sample students", f"{len(df):,}")
c2.metric("Average final score", f"{df['final_score'].mean():.1f}/100")
c3.metric("Model test MAE", f"{mae:.1f} points")

tab1, tab2, tab3 = st.tabs(["Explore data", "Visualizations", "Try a prediction"])
with tab1:
    st.subheader("Sample records")
    st.dataframe(df.head(20), use_container_width=True)
    st.subheader("Summary statistics")
    st.dataframe(df.select_dtypes(include="number").describe().round(2), use_container_width=True)
with tab2:
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots()
        sns.histplot(df["final_score"], bins=15, kde=True, ax=ax, color="steelblue")
        ax.set_title("Final score distribution")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation heatmap")
        st.pyplot(fig)
        plt.close(fig)
with tab3:
    st.subheader("Enter sample characteristics")
    col1, col2 = st.columns(2)
    with col1:
        study = st.slider("Study hours per day", 0.0, 8.0, 3.0, 0.5)
        attendance = st.slider("Attendance (%)", 0, 100, 80)
        assignments = st.slider("Assignments completed (out of 10)", 0, 10, 7)
    with col2:
        sleep = st.slider("Sleep hours", 0.0, 12.0, 7.0, 0.5)
        previous = st.slider("Previous exam score", 0, 100, 65)
        internet = st.selectbox("Internet access", ["Yes", "No"])
    if st.button("Estimate score", type="primary"):
        sample = pd.DataFrame([{
            "study_hours_per_day": study,
            "attendance_percent": attendance,
            "assignments_completed_out_of_10": assignments,
            "sleep_hours": sleep,
            "previous_exam_score": previous,
            "internet_access_yes": 1 if internet == "Yes" else 0
        }])
        estimate = float(model.predict(sample)[0])
        st.success(f"Illustrative predicted score: {estimate:.1f}/100")
        st.write("This estimate is produced by a model trained on synthetic demo data and should not be treated as a real grade prediction.")
st.divider()
st.caption("Built with Python, Pandas, Matplotlib, Seaborn, scikit-learn, and Streamlit.")
