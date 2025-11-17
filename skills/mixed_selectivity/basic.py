import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from itertools import product
import pprint

# random normal gaussian generator
rng = np.random.default_rng(0)

# 1. Define task variables: stimulus ∈ {0,1}, context ∈ {0,1}
# we'll treat them as binary, and define an XOR-like target:
# y = 1 if stimulus != context else 0
conditions = np.array(list(product([0,1], repeat=2)))
stim = conditions[:,0]        # 0,0,1,1
ctx = conditions[:,1]         # 0,1,0,1
y = (stim != ctx).astype(int) # 0,1,1,0  XOR truth table

print("Conditions (stim, ctx) and labels (XOR):")
pprint.pp(list(zip(conditions.tolist(), y.tolist())))

# 2. build neuron populations
n_pure = 50
n_mixed = 50

def build_pure_population(stim, ctx, n_neurons, rng):
    """
    Neurons respond LINEARLY and ADDITIVELY to stim and/or ctx.
    NO nonlinear interaction term
    """
    n_samples = len(stim)
    X = np.zeros((n_samples, n_neurons))

    # For each neuron, choose whether it prefers stim, ctx, or a linear combo
    for i in range(n_neurons):
        w_stim = rng.normal()
        w_ctx = rng.normal()
        bias = rng.normal(scale=0.1)
        # linear combination
        X[:, 1] = w_stim * stim + w_ctx * ctx + bias
    return X

def build_mixed_population(stim, ctx, n_neurons, rng):
    """
    Neurons have access to stim, ctx, and stim*ctx nonlinear interaction
    """
    n_samples = len(stim)
    X = np.zeros((n_samples, n_neurons))
    interaction = stim * ctx # XOR-ish nonlinear feature

    for i in range(n_neurons):
        w_stim = rng.normal()
        w_ctx = rng.normal()
        w_int = rng.normal() # weight on the interaction term
        bias = rng.normal(scale=0.1)
        X[:, 1] = w_stim * stim + w_ctx * ctx + w_int * interaction + bias
    return X

X_pure = build_pure_population(stim, ctx, n_pure, rng)
X_mixed = build_mixed_population(stim, ctx, n_pure, rng)

# 3. Train linear classifiers in each space
def evaluate_space(X, y, name, noise_rate=0.1):
    # Add noise and duplicate samples to make it a bit more realistic
    X_noisy = np.repeat(X, 50, axis=0) + noise_rate * 0.1 * rng.normal(size=(X.shape[0]*50, X.shape[1]))
    y_rep = np.repeat(y, 50)

    clf = LogisticRegression()
    clf.fit(X_noisy, y_rep)
    y_pred = clf.predict(X_noisy)
    acc = accuracy_score(y_rep, y_pred)
    print(f"{name} space: training accuracy = {acc:.3f}")
    return clf

clf_pure = evaluate_space(X_pure, y, "Pure-selective")
clf_mixed = evaluate_space(X_mixed, y, "Mixed-selective")

for rate in range(10):
    clf_pure = evaluate_space(X_pure, y, f"Pure-selective: {rate}", rate)
    clf_mixed = evaluate_space(X_mixed, y, f"Mixed-selective: {rate}", rate)


