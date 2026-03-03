import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

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
# SELECT 9 FEATURES
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
y = data['status']  # 0 = Healthy, 1 = Parkinson

print("Original Class Distribution:\n", y.value_counts())

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
# PIPELINE WITH SMOTE + SCALER + SVM
# ==============================
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True, class_weight='balanced'))
])

# ==============================
# GRID SEARCH
# ==============================
param_grid = {
    'svm__C': [1, 10, 50, 100],
    'svm__gamma': ['scale', 0.1, 0.01],
    'svm__kernel': ['rbf']
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv_strategy,
    scoring='roc_auc',
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\nBest Parameters Found:", grid.best_params_)

best_model = grid.best_estimator_

# ==============================
# TEST SET EVALUATION
# ==============================
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n✅ Training Completed!")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==============================
# FINAL TRAINING ON FULL DATA
# ==============================
best_model.fit(X, y)

# Extract trained scaler and svm for backend
final_scaler = best_model.named_steps['scaler']
final_svm = best_model.named_steps['svm']

# ==============================
# SAVE MODEL & SCALER
# ==============================
pickle.dump(final_svm, open("models/svm_voice_model.pkl", "wb"))
pickle.dump(final_scaler, open("models/voice_scaler.pkl", "wb"))

print("\n✅ Optimized Model with SMOTE saved in /models folder.")