import numpy as np


def softmax(scores):
    exp_s = np.exp(scores - scores.max())
    return exp_s / exp_s.sum()


# --- Scenario 1: Car vs Red Bus (equal utility → 50/50) ---

print("Two options — Car vs Red Bus:")
probs = softmax(np.array([1.0, 1.0]))
print(f"  Car:      {probs[0]:.0%}")
print(f"  Red Bus:  {probs[1]:.0%}")

# --- Scenario 2: Add a Blue Bus (identical to Red Bus) ---
# MNL / softmax treats every option as independent.

print("\nAdd a Blue Bus — MNL (softmax) predicts:")
probs_mnl = softmax(np.array([1.0, 1.0, 1.0]))
for name, p in zip(["Car", "Red Bus", "Blue Bus"], probs_mnl):
    print(f"  {name:10s} {p:.0%}")

print("\nBut intuitively, the blue bus should steal from the red bus, not the car:")
print(f"  {'Car':10s} 50%")
print(f"  {'Red Bus':10s} 25%")
print(f"  {'Blue Bus':10s} 25%")


# --- Fix: Nested Logit groups similar items ---
# Items in the same "nest" compete with each other more than with outsiders.
#
# mu (the "dissimilarity parameter") controls how substitutable items in a
# nest are. Think of it as: how much does adding a similar option eat into
# the existing options vs. expanding the category's overall appeal?
#
#   mu → 0: perfect substitutes — adding a blue bus doesn't change
#           the car's share at all (50/25/25)
#   mu = 1: no similarity structure — collapses back to MNL (33/33/33)
#
# In practice you don't choose mu — you estimate it from observed choices
# using maximum likelihood, the same way you'd fit any model parameter.
# At scale this is just an optimisation problem: find the mu that makes
# your users' actual choices most likely under the nested logit model.

def nested_logit_probs(utilities, nests, mu=0.5):
    utilities = np.asarray(utilities, dtype=float)
    probs = np.zeros(len(utilities))
    inclusive_values = []
    for nest in nests:
        v = utilities[nest] / mu
        iv = mu * (np.log(np.sum(np.exp(v - v.max()))) + v.max())
        inclusive_values.append(iv)
    nest_probs = softmax(np.array(inclusive_values))
    for k, nest in enumerate(nests):
        within = softmax(utilities[nest] / mu)
        for j, i in enumerate(nest):
            probs[i] = nest_probs[k] * within[j]
    return probs


nests = [[0], [1, 2]]  # car alone, buses grouped
utilities = np.array([1.0, 1.0, 1.0])

print("\nNested Logit — how mu changes the prediction:")
print(f"  {'mu':>4s}    {'Car':>4s}  {'Red Bus':>8s}  {'Blue Bus':>9s}")
for mu in [0.1, 0.3, 0.5, 0.7, 1.0]:
    p = nested_logit_probs(utilities, nests, mu=mu)
    tag = "  ← same as MNL (no nesting)" if mu == 1.0 else ""
    print(f"  {mu:4.1f}    {p[0]:4.0%}  {p[1]:8.0%}  {p[2]:9.0%}{tag}")

print("\nLower mu = buses are more substitutable = car keeps more share.")
print("mu is estimated from data, not chosen by hand.")
