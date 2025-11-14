# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Step 1: Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
data = pd.read_csv(url)

# Step 2: Select required features
features = ['MDVP:Fo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Shimmer', 
            'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'PPE']
X = data[features]
y = data['status']  # 1 = Parkinson's, 0 = Healthy

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train SVM model
svm_model = SVC(kernel='rbf', C=2, gamma='auto', probability=True)
svm_model.fit(X_train_scaled, y_train)

# Step 6: Evaluate
y_pred = svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Voice Model Training Completed!")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Save the model and scaler
pickle.dump(svm_model, open("svm_voice_model.pkl", "wb"))
pickle.dump(scaler, open("voice_scaler.pkl", "wb"))

print("\nModel and scaler saved successfully in /models folder.")

