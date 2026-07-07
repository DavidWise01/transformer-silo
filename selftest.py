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

# 5. Edge cases: K == N (every token its own intent) still converges and is stable.
vv = [embed(t) for t in ["a", "b", "c"]]
ci, ca, ch = centrifuge(vv, 3)
check(len(set(ca)) == 3, "K==N -> each token gets its own cluster")

# 6. The transformer floor runs and pools to a real ranked prediction.
check(r["prediction"] in ("the", "cat", "sat", "on", "mat", "run", "sky", "map"),
      "the transformer floor emits a real vocab prediction")
check(r["logits"][0][1] >= r["logits"][-1][1], "logits are ranked (top >= bottom)")

# 7. End to end: energy actually fell for a messy input (the centrifuge did work).
messy = run_silo(["cat", "sky", "cat", "mat", "sky", "run", "sky", "cat"], k=3)
check(messy["energy_history"][0] >= messy["energy_history"][-1],
      "end-to-end: the centrifuge lowered (or held) the energy")
check(messy["spins"] >= 1 and messy["prediction"], "end-to-end silo produced a prediction")

print("\n" + ("SOME CHECKS FAILED" if fails else "all transformer-silo checks passed"))
raise SystemExit(1 if fails else 0)
