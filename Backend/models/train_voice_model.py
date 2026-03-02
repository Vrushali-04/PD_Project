import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# ==============================
# CREATE MODELS FOLDER
# ==============================
os.makedirs("models", exist_ok=True)

# ==============================
# LOAD DATASET
# ==============================
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
data = pd.read_csv(url)

# ==============================
# SELECT 9 FEATURES (AS REQUIRED)
# ==============================
features = [
    'MDVP:Fo(Hz)',
    'MDVP:Jitter(%)',
    'MDVP:Shimmer',
    'HNR',
    'RPDE',
    'DFA',
    'spread1',
    'spread2',
    'PPE'
]

X = data[features]
y = data['status']

print("Class Distribution:\n", y.value_counts())

# ==============================
# TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# FEATURE SCALING
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# GRID SEARCH FOR BEST SVM
# ==============================
param_grid = {
    'C': [1, 5, 10, 20, 50, 100],
    'gamma': ['scale', 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

grid = GridSearchCV(
    SVC(class_weight='balanced', probability=True),
    param_grid,
    cv=5,
    n_jobs=-1
)

grid.fit(X_train_scaled, y_train)

print("\nBest Parameters Found:", grid.best_params_)

svm_model = grid.best_estimator_

# ==============================
# EVALUATE ON TEST SET
# ==============================
y_pred = svm_model.predict(X_test_scaled)
y_proba = svm_model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n✅ Voice Model Training Completed!")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==============================
# CROSS VALIDATION (10-FOLD)
# ==============================
X_scaled_full = scaler.fit_transform(X)
cv_scores = cross_val_score(
    svm_model,
    X_scaled_full,
    y,
    cv=10
)

print(f"\n10-Fold Cross Validation Accuracy: {cv_scores.mean() * 100:.2f}%")

# ==============================
# FINAL TRAINING ON FULL DATA
# ==============================
svm_model.fit(X_scaled_full, y)

# ==============================
# SAVE MODEL & SCALER
# ==============================
pickle.dump(svm_model, open("models/svm_voice_model.pkl", "wb"))
pickle.dump(scaler, open("models/voice_scaler.pkl", "wb"))

print("\n✅ Optimized Model and Scaler saved in /models folder.")