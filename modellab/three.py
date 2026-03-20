# Cell 3: Naive Bayes Classifier
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
import warnings
warnings.filterwarnings('ignore') # Ignores standard sklearn format warnings

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

# Encode attributes numerically
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

# Train Classifier
nb_classifier = CategoricalNB()
nb_classifier.fit(X_encoded, y)

# Predict for a new patient case
new_patient = [['Old', 'Low', 'Mild']]
new_patient_encoded = encoder.transform(new_patient)

probabilities = nb_classifier.predict_proba(new_patient_encoded)[0]
prediction = nb_classifier.predict(new_patient_encoded)[0]

print("--- NAIVE BAYES OUTPUT ---")
print(f"Evaluating new patient: {new_patient[0]}")
print(f"Probability of 'No': {probabilities[0]:.4f}")
print(f"Probability of 'Yes': {probabilities[1]:.4f}")
print(f"Final Class Decision: {prediction}")
