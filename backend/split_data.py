# =========================================================
# split_data.py
# =========================================================

import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

# =========================================================
# CREATE FOLDER
# =========================================================

os.makedirs("split_data", exist_ok=True)

# =========================================================
# HEART SPLIT
# =========================================================

heart_df = pd.read_csv(
    "datasets/cleaned/clean_heart.csv"
)

X_heart = heart_df.drop(
    "target",
    axis=1
)

y_heart = heart_df["target"]

heart_scaler = StandardScaler()

X_heart_scaled = heart_scaler.fit_transform(
    X_heart
)

X_train, X_test, y_train, y_test = train_test_split(

    X_heart_scaled,
    y_heart,

    test_size=0.2,

    random_state=42
)

joblib.dump(
    (X_train, X_test, y_train, y_test),
    "split_data/heart_split.pkl"
)

joblib.dump(
    heart_scaler,
    "models/heart_scaler.pkl"
)

print("✅ Heart Split Saved")

# =========================================================
# DIABETES SPLIT
# =========================================================

diabetes_df = pd.read_csv(
    "datasets/cleaned/clean_diabetes.csv"
)

X_diabetes = diabetes_df.drop(
    "diabetes",
    axis=1
)

y_diabetes = diabetes_df["diabetes"]

diabetes_scaler = StandardScaler()

X_diabetes_scaled = diabetes_scaler.fit_transform(
    X_diabetes
)

X_train, X_test, y_train, y_test = train_test_split(

    X_diabetes_scaled,
    y_diabetes,

    test_size=0.2,

    random_state=42
)

joblib.dump(
    (X_train, X_test, y_train, y_test),
    "split_data/diabetes_split.pkl"
)

joblib.dump(
    diabetes_scaler,
    "models/diabetes_scaler.pkl"
)

print("✅ Diabetes Split Saved")

# =========================================================
# MENTAL SPLIT
# =========================================================

mental_df = pd.read_csv(
    "datasets/cleaned/clean_mental.csv"
)

X_mental = mental_df.drop(
    "stress_level",
    axis=1
)

y_mental = mental_df["stress_level"]

mental_scaler = StandardScaler()

X_mental_scaled = mental_scaler.fit_transform(
    X_mental
)

X_train, X_test, y_train, y_test = train_test_split(

    X_mental_scaled,
    y_mental,

    test_size=0.2,

    random_state=42
)

joblib.dump(
    (X_train, X_test, y_train, y_test),
    "split_data/mental_split.pkl"
)

joblib.dump(
    mental_scaler,
    "models/mental_scaler.pkl"
)

print("✅ Mental Split Saved")

print("\n🎉 TRAIN TEST SPLIT COMPLETED")