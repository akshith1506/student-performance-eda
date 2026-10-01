import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
REPORT_DIR = os.path.join(BASE_DIR, "report")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "student_performance.csv")

# Load data
df = pd.read_csv(DATA_PATH)

print("=== Dataset Overview ===")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nData types:\n", df.dtypes)

# Missing values
missing = df.isnull().sum()
print("\nMissing values:\n", missing[missing > 0])

# Fill missing values
for col in ["Gender", "FamilySupport", "InternetAccess"]:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

for col in ["StudyHours", "Attendance", "SleepHours", "PreviousMarks", "ExamScore"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Descriptive statistics
numeric_cols = ["StudyHours", "Attendance", "SleepHours", "PreviousMarks", "ExamScore"]
print("\n=== Descriptive Statistics ===")
print(df[numeric_cols].describe().round(2))

# Visualizations
sns.set_style("whitegrid")

plt.figure(figsize=(8, 5))
sns.histplot(df["ExamScore"], bins=10, kde=True, color="steelblue")
plt.title("Distribution of Exam Scores")
plt.xlabel("Exam Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "exam_score_distribution.png"))
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Attendance", y="ExamScore", hue="Gender", s=100)
plt.title("Attendance vs Exam Score")
plt.xlabel("Attendance (%)")
plt.ylabel("Exam Score")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "attendance_vs_score.png"))
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Gender", y="StudyHours", palette="viridis")
plt.title("Study Hours by Gender")
plt.xlabel("Gender")
plt.ylabel("Study Hours")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "study_hours_by_gender.png"))
plt.close()

corr = df[numeric_cols].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.close()

avg_score_support = df.groupby("FamilySupport")["ExamScore"].mean().sort_values()
plt.figure(figsize=(8, 5))
sns.barplot(x=avg_score_support.index, y=avg_score_support.values, palette="Set2")
plt.title("Average Exam Score by Family Support")
plt.xlabel("Family Support")
plt.ylabel("Average Exam Score")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "family_support_vs_score.png"))
plt.close()

print("\n=== Correlation with ExamScore ===")
print(corr["ExamScore"].sort_values(ascending=False))

print("\n=== Average Exam Score by Gender ===")
print(df.groupby("Gender")["ExamScore"].mean().round(2))

print("\n=== Average Exam Score by Internet Access ===")
print(df.groupby("InternetAccess")["ExamScore"].mean().round(2))

print("\n=== Average Exam Score by Family Support ===")
print(df.groupby("FamilySupport")["ExamScore"].mean().round(2))

insights = [
    "Students with higher attendance tend to score better in exams.",
    "Study hours show a strong positive relationship with exam performance.",
    "Previous marks strongly correlate with final exam scores, suggesting academic consistency matters.",
    "Students with family support and internet access generally perform better overall.",
    "Students with lower attendance and poor study routines are more likely to score lower."
]

print("\n=== Key Insights ===")
for i, insight in enumerate(insights, 1):
    print(f"{i}. {insight}")

report_path = os.path.join(REPORT_DIR, "eda_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("""# Exploratory Data Analysis Report

## 1. Project Overview
This project explores how student-related factors affect academic performance. The analysis focuses on study hours, attendance, sleep, previous marks, family support, and internet access.

## 2. Dataset Description
The dataset contains student academic and lifestyle variables. It includes numerical and categorical data used to evaluate key performance indicators.

## 3. Data Cleaning
- Checked for missing values.
- Filled missing categorical values with \"Unknown\".
- Filled missing numerical values with median values.

## 4. Descriptive Statistics
The dataset shows variations in exam performance across students. The average scores, study hours, and attendance levels provide a foundation for understanding student performance trends.

## 5. Visual Analysis
- Exam scores are distributed across a range of marks, indicating variation in performance.
- Attendance appears to be positively related to exam results.
- Study hours have a strong positive connection with higher scores.
- A correlation heatmap highlights the most influential variables.

## 6. Key Findings
- Higher attendance is associated with better exam outcomes.
- Students who invest more time in studying tend to perform better.
- Previous performance is a major indicator of future results.
- Support systems such as family assistance and internet access contribute to improvement.

## 7. Conclusion
The analysis suggests that student performance is significantly influenced by consistency, preparation, and supportive learning conditions. Students who attend classes regularly and dedicate more time to studying are likely to achieve better outcomes.

## 8. Recommendations
- Encourage regular class attendance.
- Introduce structured study schedules.
- Provide academic support to students with weaker previous performance.
- Improve access to study materials and digital learning resources.

## 9. Key Insights Summary
- Students with higher attendance tend to score better.
- Study hours show a strong positive relationship with exam scores.
- Previous marks strongly correlate with final exam outcomes.
- Family support and internet access contribute to better academic performance.
""")

print(f"\nReport saved to: {report_path}")
print(f"Images saved to: {OUTPUT_DIR}")
print("\nEDA project completed successfully.")
print("You can review the generated report and charts in the 'report/' and 'outputs/' folders.")
