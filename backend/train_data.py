# =========================================================
# train_model.py
# =========================================================

import joblib
import os

from sklearn.metrics import (
    accuracy_score,
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
# CREATE MODELS FOLDER
# =========================================================

os.makedirs("models", exist_ok=True)

# =========================================================
# HEART MODEL
# =========================================================

print("\n❤️ TRAINING HEART MODEL...\n")

X_train, X_test, y_train, y_test = joblib.load(
    "split_data/heart_split.pkl"
)

heart_model = RandomForestClassifier(

    n_estimators=200,

    max_depth=10,

    random_state=42
)

heart_model.fit(
    X_train,
    y_train
)

y_pred = heart_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    f"✅ Heart Accuracy: {accuracy*100:.2f}%"
)

print(
    classification_report(
        y_test,
        y_pred
    )
)

joblib.dump(
    heart_model,
    "models/heart_model.pkl"
)

print("✅ Heart Model Saved")

# =========================================================
# DIABETES MODEL
# =========================================================

print("\n🩸 TRAINING DIABETES MODEL...\n")

X_train, X_test, y_train, y_test = joblib.load(
    "split_data/diabetes_split.pkl"
)

diabetes_model = LogisticRegression(
    max_iter=1000
)

diabetes_model.fit(
    X_train,
    y_train
)

y_pred = diabetes_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    f"✅ Diabetes Accuracy: {accuracy*100:.2f}%"
)

print(
    classification_report(
        y_test,
        y_pred
    )
)

joblib.dump(
    diabetes_model,
    "models/diabetes_model.pkl"
)

print("✅ Diabetes Model Saved")

# =========================================================
# MENTAL HEALTH MODEL
# =========================================================

print("\n🧠 TRAINING MENTAL MODEL...\n")

X_train, X_test, y_train, y_test = joblib.load(
    "split_data/mental_split.pkl"
)

mental_model = RandomForestRegressor(

    n_estimators=200,

    max_depth=10,

    random_state=42
)

mental_model.fit(
    X_train,
    y_train
)

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
    f"✅ Mental Health R2 Score: {r2:.2f}"
)

print(
    f"✅ Mean Squared Error: {mse:.2f}"
)

joblib.dump(
    mental_model,
    "models/mental_model.pkl"
)

print("✅ Mental Model Saved")

print("\n🎉 ALL MODELS TRAINED SUCCESSFULLY")