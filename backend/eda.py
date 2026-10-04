# =========================================================
# eda.py
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =========================================================
# CREATE GRAPH FOLDER
# =========================================================

os.makedirs("graphs", exist_ok=True)

# =========================================================
# HEART EDA
# =========================================================

heart_df = pd.read_csv(
    "datasets/cleaned/clean_heart.csv"
)

plt.figure(figsize=(12,8))

sns.heatmap(
    heart_df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title(
    "Heart Heatmap"
)

plt.savefig(
    "graphs/heart_heatmap.png"
)

plt.close()

# =========================================================
# DIABETES EDA
# =========================================================

diabetes_df = pd.read_csv(
    "datasets/cleaned/clean_diabetes.csv"
)

plt.figure(figsize=(12,8))

sns.heatmap(
    diabetes_df.corr(),
    annot=True,
    cmap='viridis'
)

plt.title(
    "Diabetes Heatmap"
)

plt.savefig(
    "graphs/diabetes_heatmap.png"
)

plt.close()

# =========================================================
# MENTAL HEALTH EDA
# =========================================================

mental_df = pd.read_csv(
    "datasets/cleaned/clean_mental.csv"
)

plt.figure(figsize=(12,8))

sns.heatmap(
    mental_df.corr(),
    annot=True,
    cmap='magma'
)

plt.title(
    "Mental Health Heatmap"
)

plt.savefig(
    "graphs/mental_heatmap.png"
)

plt.close()

print("\n🎉 EDA COMPLETED")