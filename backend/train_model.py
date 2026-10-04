# =========================================================
# train_model.py
# FINAL MACHINE LEARNING TRAINING PIPELINE
# DISEASE RISK PREDICTION SYSTEM
# =========================================================

import os
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# LIBRARIES
# =========================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import joblib

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.linear_model import LogisticRegression

# =========================================================
# CREATE REQUIRED FOLDERS
# =========================================================

os.makedirs("models", exist_ok=True)

os.makedirs("graphs", exist_ok=True)

os.makedirs("datasets/raw", exist_ok=True)

os.makedirs("datasets/cleaned", exist_ok=True)

# =========================================================
# HEART DISEASE DATASET
# =========================================================

print("\n❤️ PROCESSING HEART DISEASE DATASET...\n")

heart_df = pd.read_csv(
    "datasets/heart.csv"
)

# =========================================================
# SAVE RAW DATASET
# =========================================================

heart_df.to_csv(

    "datasets/raw/raw_heart.csv",

    index=False
)

print(
    "✅ Raw Heart Dataset Saved"
)

# =========================================================
# EDA
# =========================================================

print(heart_df.head())

print("\nDataset Shape:")
print(heart_df.shape)

print("\nMissing Values:")
print(heart_df.isnull().sum())

print("\nDataset Information:")
print(heart_df.info())

print("\nStatistical Summary:")
print(heart_df.describe())

# =========================================================
# CORRELATION HEATMAP
# =========================================================

plt.figure(figsize=(12,8))

sns.heatmap(
    heart_df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title(
    "Heart Disease Correlation Heatmap"
)

plt.savefig(
    "graphs/heart_heatmap.png"
)

plt.close()

# =========================================================
# TARGET DISTRIBUTION
# =========================================================

plt.figure(figsize=(6,4))

sns.countplot(
    x='target',
    data=heart_df
)

plt.title(
    "Heart Disease Target Distribution"
)

plt.savefig(
    "graphs/heart_target_distribution.png"
)

plt.close()

# =========================================================
# FEATURE SELECTION
# =========================================================

selected_features = [

    "age",
    "sex",
    "trestbps",
    "chol",
    "fbs",
    "thalach",
    "exang"

]

X_heart = heart_df[selected_features]

y_heart = heart_df["target"]

# =========================================================
# SAVE CLEAN DATASET
# =========================================================

clean_heart_df = heart_df[
    selected_features + ["target"]
]

clean_heart_df.to_csv(

    "datasets/cleaned/clean_heart.csv",

    index=False
)

print(
    "✅ Clean Heart Dataset Saved"
)

# =========================================================
# FEATURE SCALING
# =========================================================

heart_scaler = StandardScaler()

X_heart_scaled = heart_scaler.fit_transform(
    X_heart
)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X_heart_scaled,
    y_heart,

    test_size=0.2,

    random_state=42

)

print("\nHeart Train Shape:", X_train.shape)

print("Heart Test Shape:", X_test.shape)

# =========================================================
# MODEL TRAINING
# =========================================================

heart_model = RandomForestClassifier(

    n_estimators=200,

    max_depth=10,

    random_state=42

)

heart_model.fit(
    X_train,
    y_train
)

# =========================================================
# EVALUATION
# =========================================================

y_pred = heart_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    f"\n✅ Heart Testing Accuracy: {accuracy*100:.2f}%"
)

print("\n📌 Classification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title(
    "Heart Disease Confusion Matrix"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig(
    "graphs/heart_confusion_matrix.png"
)

plt.close()

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance = heart_model.feature_importances_

feature_df = pd.DataFrame({

    "Feature": selected_features,

    "Importance": importance

})

feature_df = feature_df.sort_values(

    by="Importance",

    ascending=False

)

plt.figure(figsize=(10,6))

sns.barplot(

    x="Importance",

    y="Feature",

    data=feature_df

)

plt.title(
    "Heart Disease Feature Importance"
)

plt.savefig(
    "graphs/heart_feature_importance.png"
)

plt.close()

# =========================================================
# SAVE HEART MODEL
# =========================================================

joblib.dump(

    heart_model,

    "models/heart_model.pkl"

)

joblib.dump(

    heart_scaler,

    "models/heart_scaler.pkl"

)

print(
    "\n✅ Heart Disease Model Saved"
)

# =========================================================
# DIABETES DATASET
# =========================================================

print("\n🩸 PROCESSING DIABETES DATASET...\n")

diabetes_df = pd.read_csv(
    "datasets/diabetes_prediction_dataset.csv"
)

# =========================================================
# SAVE RAW DATASET
# =========================================================

diabetes_df.to_csv(

    "datasets/raw/raw_diabetes.csv",

    index=False
)

print(
    "✅ Raw Diabetes Dataset Saved"
)

# =========================================================
# EDA
# =========================================================

print(diabetes_df.head())

print("\nDataset Shape:")
print(diabetes_df.shape)

print("\nMissing Values:")
print(diabetes_df.isnull().sum())

print("\nDataset Information:")
print(diabetes_df.info())

print("\nStatistical Summary:")
print(diabetes_df.describe())

# =========================================================
# LABEL ENCODING
# =========================================================

gender_encoder = LabelEncoder()

smoking_encoder = LabelEncoder()

diabetes_df["gender"] = gender_encoder.fit_transform(
    diabetes_df["gender"]
)

diabetes_df["smoking_history"] = smoking_encoder.fit_transform(
    diabetes_df["smoking_history"]
)

# =========================================================
# SAVE CLEAN DATASET
# =========================================================

clean_diabetes_df = diabetes_df.copy()

clean_diabetes_df.to_csv(

    "datasets/cleaned/clean_diabetes.csv",

    index=False
)

print(
    "✅ Clean Diabetes Dataset Saved"
)

# =========================================================
# HEATMAP
# =========================================================

plt.figure(figsize=(12,8))

sns.heatmap(
    diabetes_df.corr(),
    annot=True,
    cmap='viridis'
)

plt.title(
    "Diabetes Correlation Heatmap"
)

plt.savefig(
    "graphs/diabetes_heatmap.png"
)

plt.close()

# =========================================================
# TARGET DISTRIBUTION
# =========================================================

plt.figure(figsize=(6,4))

sns.countplot(
    x='diabetes',
    data=diabetes_df
)

plt.title(
    "Diabetes Target Distribution"
)

plt.savefig(
    "graphs/diabetes_target_distribution.png"
)

plt.close()

# =========================================================
# FEATURES & TARGET
# =========================================================

X_diabetes = diabetes_df.drop(
    "diabetes",
    axis=1
)

y_diabetes = diabetes_df["diabetes"]

# =========================================================
# FEATURE SCALING
# =========================================================

diabetes_scaler = StandardScaler()

X_diabetes_scaled = diabetes_scaler.fit_transform(
    X_diabetes
)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X_diabetes_scaled,
    y_diabetes,

    test_size=0.2,

    random_state=42

)

print("\nDiabetes Train Shape:", X_train.shape)

print("Diabetes Test Shape:", X_test.shape)

# =========================================================
# MODEL TRAINING
# =========================================================

diabetes_model = LogisticRegression(
    max_iter=1000
)

diabetes_model.fit(
    X_train,
    y_train
)

# =========================================================
# EVALUATION
# =========================================================

y_pred = diabetes_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    f"\n✅ Diabetes Testing Accuracy: {accuracy*100:.2f}%"
)

print("\n📌 Classification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens'
)

plt.title(
    "Diabetes Confusion Matrix"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig(
    "graphs/diabetes_confusion_matrix.png"
)

plt.close()

# =========================================================
# SAVE DIABETES MODEL
# =========================================================

joblib.dump(
    diabetes_model,
    "models/diabetes_model.pkl"
)

joblib.dump(
    diabetes_scaler,
    "models/diabetes_scaler.pkl"
)

joblib.dump(
    gender_encoder,
    "models/gender_encoder.pkl"
)

joblib.dump(
    smoking_encoder,
    "models/smoking_encoder.pkl"
)

print(
    "\n✅ Diabetes Model Saved"
)

# =========================================================
# MENTAL HEALTH DATASET
# =========================================================

print("\n🧠 PROCESSING MENTAL HEALTH DATASET...\n")

mental_df = pd.read_csv(
    "datasets/synthetic_mental_health_dataset.csv"
)

# =========================================================
# SAVE RAW DATASET
# =========================================================

mental_df.to_csv(

    "datasets/raw/raw_mental.csv",

    index=False
)

print(
    "✅ Raw Mental Dataset Saved"
)

# =========================================================
# EDA
# =========================================================

print(mental_df.head())

print("\nDataset Shape:")
print(mental_df.shape)

print("\nMissing Values:")
print(mental_df.isnull().sum())

print("\nDataset Information:")
print(mental_df.info())

print("\nStatistical Summary:")
print(mental_df.describe())

# =========================================================
# LABEL ENCODING
# =========================================================

label_encoders = {}

for column in mental_df.columns:

    if mental_df[column].dtype == "object":

        le = LabelEncoder()

        mental_df[column] = le.fit_transform(
            mental_df[column]
        )

        label_encoders[column] = le

# =========================================================
# SAVE CLEAN DATASET
# =========================================================

clean_mental_df = mental_df.copy()

clean_mental_df.to_csv(

    "datasets/cleaned/clean_mental.csv",

    index=False
)

print(
    "✅ Clean Mental Dataset Saved"
)

# =========================================================
# HEATMAP
# =========================================================

plt.figure(figsize=(12,8))

sns.heatmap(
    mental_df.corr(),
    annot=True,
    cmap='magma'
)

plt.title(
    "Mental Health Correlation Heatmap"
)

plt.savefig(
    "graphs/mental_heatmap.png"
)

plt.close()

# =========================================================
# TARGET COLUMN
# =========================================================

target_column = "stress_level"

# =========================================================
# FEATURES & TARGET
# =========================================================

X_mental = mental_df.drop(
    target_column,
    axis=1
)

y_mental = mental_df[target_column]

# =========================================================
# FEATURE SCALING
# =========================================================

mental_scaler = StandardScaler()

X_mental_scaled = mental_scaler.fit_transform(
    X_mental
)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X_mental_scaled,
    y_mental,

    test_size=0.2,

    random_state=42

)

print("\nMental Train Shape:", X_train.shape)

print("Mental Test Shape:", X_test.shape)

# =========================================================
# MODEL TRAINING
# =========================================================

mental_model = RandomForestRegressor(

    n_estimators=200,

    max_depth=10,

    random_state=42

)

mental_model.fit(
    X_train,
    y_train
)

# =========================================================
# EVALUATION
# =========================================================

y_pred = mental_model.predict(X_test)

mse = mean_squared_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)

print(
    f"\n✅ Mental Health R2 Score: {r2:.2f}"
)

print(
    f"✅ Mean Squared Error: {mse:.2f}"
)

# =========================================================
# REGRESSION GRAPH
# =========================================================

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Values")

plt.ylabel("Predicted Values")

plt.title(
    "Mental Health Actual vs Predicted"
)

plt.savefig(
    "graphs/mental_regression_plot.png"
)

plt.close()

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance = mental_model.feature_importances_

feature_df = pd.DataFrame({

    "Feature": X_mental.columns,

    "Importance": importance

})

feature_df = feature_df.sort_values(

    by="Importance",

    ascending=False

)

plt.figure(figsize=(10,6))

sns.barplot(

    x="Importance",

    y="Feature",

    data=feature_df

)

plt.title(
    "Mental Health Feature Importance"
)

plt.savefig(
    "graphs/mental_feature_importance.png"
)

plt.close()

# =========================================================
# SAVE MENTAL MODEL
# =========================================================

joblib.dump(
    mental_model,
    "models/mental_model.pkl"
)

joblib.dump(
    mental_scaler,
    "models/mental_scaler.pkl"
)

joblib.dump(
    label_encoders,
    "models/mental_label_encoders.pkl"
)

print(
    "\n✅ Mental Health Model Saved"
)

# =========================================================
# FINAL SUCCESS MESSAGE
# =========================================================

print("\n" + "="*70)

print(
    "🎉 ALL RAW DATASETS SAVED"
)

print(
    "🎉 ALL CLEAN DATASETS SAVED"
)

print(
    "🎉 ALL MODELS TRAINED SUCCESSFULLY"
)

print(
    "🎉 ALL MODELS SAVED INSIDE 'models' FOLDER"
)

print(
    "🎉 ALL GRAPHS SAVED INSIDE 'graphs' FOLDER"
)

print(
    "🎉 DISEASE RISK PREDICTION PROJECT COMPLETED"
)

print("="*70)