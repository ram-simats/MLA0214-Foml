# Cell 4: Logistic Regression
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

data = [
    ['Old', 'High', 'Severe', 'Yes'],
    ['Middle', 'Medium', 'Moderate', 'Yes'],
    ['Young', 'Low', 'Mild', 'No'],
    ['Middle', 'Low', 'Mild', 'No']
]
columns = ['Age', 'PreviousVisits', 'ConditionSeverity', 'Readmission']
df = pd.DataFrame(data, columns=columns)

X = df.drop('Readmission', axis=1)
y = df['Readmission']

# One-hot encoding is standard for logistic regression with categorical data
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X)

# Train the model
lr_model = LogisticRegression()
lr_model.fit(X_encoded, y)

# Predict for a new patient case
new_patient = pd.DataFrame([['Old', 'Low', 'Mild']], columns=X.columns)
new_patient_encoded = encoder.transform(new_patient)

probabilities = lr_model.predict_proba(new_patient_encoded)[0]
prediction = lr_model.predict(new_patient_encoded)[0]

print("--- LOGISTIC REGRESSION OUTPUT ---")
print(f"Evaluating new patient: {new_patient.values[0]}")
print(f"Probability of 'No': {probabilities[0]:.4f}")
print(f"Probability of 'Yes': {probabilities[1]:.4f}")
print(f"Final class decision: {prediction}")
