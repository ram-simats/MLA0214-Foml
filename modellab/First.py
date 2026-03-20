# Cell 1: FIND-S Algorithm
data = [
    ['Old', 'High', 'Severe', 'Yes'],
    ['Middle', 'Medium', 'Moderate', 'Yes'],
    ['Young', 'Low', 'Mild', 'No'],
    ['Middle', 'Low', 'Mild', 'No']
]

# Separate features and target
X = [row[:-1] for row in data]
y = [row[-1] for row in data]

def train_find_s(features, targets):
    hypothesis = None
    for i, target in enumerate(targets):
        if target == 'Yes':
            if hypothesis is None:
                hypothesis = features[i].copy()
            else:
                for j in range(len(hypothesis)):
                    if hypothesis[j] != features[i][j]:
                        hypothesis[j] = '?'
    return hypothesis

final_hypothesis = train_find_s(X, y)
print("--- FIND-S OUTPUT ---")
print("Final Hypothesis representing strict readmission criteria:")
print(final_hypothesis)
