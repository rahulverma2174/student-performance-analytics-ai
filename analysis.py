from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "student_performance.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
print("\nDATASET OVERVIEW")
print(df.head())
print("\nSUMMARY STATISTICS")
print(df.select_dtypes(include="number").describe().round(2))

numeric = df.select_dtypes(include="number")
plt.figure(figsize=(9, 6))
sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap: Student Performance")
plt.tight_layout()
plt.savefig(OUT / "correlation_heatmap.png", dpi=160)
plt.close()

plt.figure(figsize=(8, 5))
sns.histplot(df["final_score"], bins=15, kde=True, color="steelblue")
plt.title("Distribution of Final Scores")
plt.xlabel("Final score")
plt.tight_layout()
plt.savefig(OUT / "score_distribution.png", dpi=160)
plt.close()

features = [
    "study_hours_per_day", "attendance_percent",
    "assignments_completed_out_of_10", "sleep_hours",
    "previous_exam_score"
]
X = df[features].copy()
X["internet_access_yes"] = (df["internet_access"] == "Yes").astype(int)
y = df["final_score"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\nTest MAE: {mae:.2f} points")
print(f"Test R-squared: {r2:.3f}")
print("\nFeature importance:")
print(pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).round(3))

(OUT / "metrics.txt").write_text(
    f"Test MAE: {mae:.2f} points\nTest R-squared: {r2:.3f}\n",
    encoding="utf-8"
)
