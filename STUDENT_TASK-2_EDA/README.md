# 📊 Student Performance EDA — AI & ML Internship Task 2

This project performs Exploratory Data Analysis (EDA) on the "Student Performance in Exams" dataset to discover trends, patterns, and relationships between student demographics and exam scores.

---

## 🎯 Objective

- Analyze exam performance using statistics and visualizations
- Understand how gender, parental education, lunch type, and test preparation affect math, reading, and writing scores
- Gain insights into score correlations and outliers

---

## 🛠 Tools & Libraries Used

- Python 🐍
- pandas
- seaborn
- matplotlib

---

## 📁 Dataset

**Name**: Students Performance in Exams  
**Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)  
**File**: `students.csv` in `data/` folder

---

## 🧪 Analysis Performed

- Dataset Overview and Info
- Summary Statistics (mean, std, min, max, etc.)
- Missing Value Check
- Histograms of all features
- Boxplots for math, reading, and writing scores
- Correlation matrix between scores
- Pairplots to visualize score relationships

---

## 📊 Key Visualizations

- `students_histograms.png` – Distribution of all features
- `students_boxplot.png` – Outlier detection in scores
- `students_correlation.png` – Correlation heatmap
- `students_pairplot.png` – Relationship between scores

---

## 🧠 Key Observations

- Students with test preparation generally score higher.
- Reading and writing scores are strongly correlated.
- Males tend to score slightly higher in math; females in reading and writing.
- Lunch type and parental education level show moderate impact on scores.

---

## 🚀 How to Run

1. Make sure your folder is structured like this:

STUDENT_TASK-2_EDA
├── eda_students.py
├── README.md
└── data
└── students.csv


2. Install required libraries (if not already installed):


pip install pandas seaborn matplotlib


3.python eda_students.py

python eda_students.py

