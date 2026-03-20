# Cell 2: Candidate Elimination Algorithm
data = [
    ['Old', 'High', 'Severe', 'Yes'],
    ['Middle', 'Medium', 'Moderate', 'Yes'],
    ['Young', 'Low', 'Mild', 'No'],
    ['Middle', 'Low', 'Mild', 'No']
]

X = [row[:-1] for row in data]
y = [row[-1] for row in data]

num_attributes = len(X[0])
S = ['0'] * num_attributes
G = [['?'] * num_attributes]

print("--- CANDIDATE ELIMINATION OUTPUT ---")
print(f"Initial S boundary: {S}")
print(f"Initial G boundary: {G}\n")

for i, (features, label) in enumerate(zip(X, y)):
    if label == 'Yes':
        if S == ['0'] * num_attributes:
            S = list(features)
        else:
            for j in range(len(S)):
                if S[j] != features[j]:
                    S[j] = '?'
        G = [g for g in G if all(g[k] == '?' or g[k] == S[k] for k in range(len(S)))]

    else: # label == 'No'
        temp_G = []
        for g in G:
            if all(g[k] == '?' or g[k] == features[k] for k in range(len(features))):
                for j in range(len(features)):
                    if g[j] == '?':
                        if S[j] != '?' and S[j] != features[j]:
                            new_g = list(g)
                            new_g[j] = S[j]
                            temp_G.append(new_g)
            else:
                temp_G.append(g)
        G = temp_G

    print(f"After Instance {i+1} ({features}, {label}):")
    print(f" S: {S}")
    print(f" G: {G}\n")
