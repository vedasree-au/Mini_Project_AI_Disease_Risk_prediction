# =========================================================
# preprocessing.py
# =========================================================

import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder

# =========================================================
# CREATE FOLDERS
# =========================================================

os.makedirs("datasets/raw", exist_ok=True)

os.makedirs("datasets/cleaned", exist_ok=True)

# =========================================================
# HEART DATASET
# =========================================================

print("\n❤️ PREPROCESSING HEART DATASET...\n")

heart_df = pd.read_csv(
    "datasets/raw/heart.csv"
)

# SAVE RAW

heart_df.to_csv(

    "datasets/raw/raw_heart.csv",

    index=False
)

# FEATURE SELECTION

selected_features = [

    "age",
    "sex",
    "trestbps",
    "chol",
    "fbs",
    "thalach",
    "exang",
    "target"

]

clean_heart_df = heart_df[selected_features]

# SAVE CLEAN DATASET

clean_heart_df.to_csv(

    "datasets/cleaned/clean_heart.csv",

    index=False
)

print("✅ Clean Heart Dataset Saved")

# =========================================================
# DIABETES DATASET
# =========================================================

print("\n🩸 PREPROCESSING DIABETES DATASET...\n")

diabetes_df = pd.read_csv(
    "datasets/raw/diabetes_prediction_dataset.csv"
)

# SAVE RAW

diabetes_df.to_csv(

    "datasets/raw/raw_diabetes.csv",

    index=False
)

# LABEL ENCODING

gender_encoder = LabelEncoder()

smoking_encoder = LabelEncoder()

diabetes_df["gender"] = gender_encoder.fit_transform(
    diabetes_df["gender"]
)

diabetes_df["smoking_history"] = smoking_encoder.fit_transform(
    diabetes_df["smoking_history"]
)

# SAVE CLEAN DATASET

diabetes_df.to_csv(

    "datasets/cleaned/clean_diabetes.csv",

    index=False
)

print("✅ Clean Diabetes Dataset Saved")

# =========================================================
# MENTAL HEALTH DATASET
# =========================================================

print("\n🧠 PREPROCESSING MENTAL DATASET...\n")

mental_df = pd.read_csv(
    "datasets/raw/synthetic_mental_health_dataset.csv"
)

# SAVE RAW

mental_df.to_csv(

    "datasets/raw/raw_mental.csv",

    index=False
)

# LABEL ENCODING

for column in mental_df.columns:

    if mental_df[column].dtype == "object":

        le = LabelEncoder()

        mental_df[column] = le.fit_transform(
            mental_df[column]
        )

# SAVE CLEAN DATASET

mental_df.to_csv(

    "datasets/cleaned/clean_mental.csv",

    index=False
)

print("✅ Clean Mental Dataset Saved")

print("\n🎉 PREPROCESSING COMPLETED")