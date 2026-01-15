import os
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Ensure artifacts folder exists
os.makedirs("artifacts", exist_ok=True)

# Load data
df = pd.read_csv("heart-disease.csv")
X = df.drop("target", axis=1)
y = df["target"].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=100, C=1)
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Training done! Accuracy:", accuracy)

# Save artifacts
with open("artifacts/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("artifacts/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("artifacts/metrics.json", "w") as f:
    json.dump({"accuracy": accuracy}, f, indent=4)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.savefig("artifacts/confusion_matrix.png", bbox_inches="tight")
plt.close()
