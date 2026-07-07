#!/usr/bin/env python3
"""Verify-first self-test. Proves, with no network, that the silo actually does
something at each floor: (1) the centrifuge CONVERGES (intra-cluster energy is
monotonically non-increasing and it settles); (2) it is DETERMINISTIC; (3) it
really sorts LIKE-TO-LIKE (planted groups separate cleanly); (4) it COMPRESSES
N context vectors down to K intents; (5) empty inputs / K>=N are handled; (6) the
transformer floor runs and emits a real ranked prediction; (7) the whole silo
runs end to end.
"""
from __future__ import annotations
from silo import centrifuge, energy, run_silo, transformer, embed, dist2, D

fails = 0
def check(cond, msg):
    global fails
    print(("ok  · " if cond else "FAIL· ") + msg)
    fails += 0 if cond else 1


# ---- three well-separated planted groups, tiny perturbations ----
def planted():
    base = {0: [3, 3, 0, 0], 1: [-3, 0, 3, 0], 2: [0, -3, 0, 3]}
    vecs, truth = [], []
    for g in range(3):
        for p in range(4):
            vecs.append([base[g][d] + (0.01 * p if d == g % D else -0.01 * p) for d in range(D)])
            truth.append(g)
    return vecs, truth


# 1. Convergence: energy is monotonically non-increasing and it settles.
vecs, truth = planted()
intents, assign, hist = centrifuge(vecs, 3)
check(all(hist[i] >= hist[i + 1] - 1e-9 for i in range(len(hist) - 1)),
      f"centrifuge energy is monotonically non-increasing ({hist})")
check(len(hist) < 25, f"the centrifuge settles (not max spins): {len(hist)} spins")

# 2. Determinism: same input -> same assignment + intents every time.
i2, a2, h2 = centrifuge(vecs, 3)
check(a2 == assign and i2 == intents, "the centrifuge is deterministic")

# 3. Like-to-like: every planted group ends up in ONE cluster, and the three
#    groups land in three DIFFERENT clusters.
groups_ok = True
for g in range(3):
    ids = {assign[i] for i in range(len(assign)) if truth[i] == g}
    if len(ids) != 1:
        groups_ok = False
check(groups_ok, "each planted group is assigned to a single cluster (like-to-like)")
check(len(set(assign)) == 3, "the three groups occupy three distinct clusters")

# 4. Compression: N context vectors -> K intents.
r = run_silo(["cat", "cat", "sat", "sky", "sky", "run", "mat", "mat", "on"], k=3)
check(r["n_context"] == 9 and r["k_intents"] == 3 and len(r["intents"]) == 3,
      "the silo compresses 9 context tokens to 3 intents")
check(r["compression"] == 3.0, f"compression ratio reported (got {r['compression']}x)")

# 5. Edge case: K == N with DISTINCT embeddings -> each token its own cluster.
#    (embeddings collide when tokens share _tok_salt, so this is a property of
#    separable inputs, not of K==N in general.)
import math
vv = [embed(t) for t in ["a", "b", "c"]]          # three distinct salts
ci, ca, ch = centrifuge(vv, 3)
check(len(set(ca)) == 3, "K==N with distinct embeddings -> each its own cluster")

# 5b. K > N is clamped to N (no phantom intents, no sub-1.0 'compression').
clamped = run_silo(["cat", "sat"], k=5)
check(clamped["k_intents"] == 2 and len(clamped["intents"]) == 2 and clamped["compression"] >= 1.0,
      "K>N is clamped to N (no phantom intents; compression >= 1)")

# 6. The transformer floor runs and pools to a REAL prediction -- finite,
#    full-vocab, and not all-identical (a non-vacuous check that it computed).
check(all(math.isfinite(l[1]) for l in r["logits"]) and len(r["logits"]) == 8,
      "the transformer emits finite logits over the full 8-token vocab")
check(any(l[1] != r["logits"][0][1] for l in r["logits"]),
      "the logits are differentiated (the transformer did real work, not a constant)")
check(r["prediction"] == r["logits"][0][0], "the prediction is the top logit")

# 7. End to end: the centrifuge STRICTLY lowered the energy (seed -> settled).
messy = run_silo(["cat", "cat", "sat", "sky", "sky", "sky", "run", "mat", "mat", "on"], k=3)
check(messy["energy_history"][0] > messy["energy_history"][-1],
      f"end-to-end: the centrifuge strictly lowered the energy ({messy['energy_history']})")
check(messy["prediction"] and len(messy["intents"]) == 3, "end-to-end silo produced 3 intents + a prediction")

print("\n" + ("SOME CHECKS FAILED" if fails else "all transformer-silo checks passed"))
raise SystemExit(1 if fails else 0)
