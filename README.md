# transformer-silo v1 — an intent-engine centrifuge feeding a transformer

A two-floor **silo**. Downstairs, an **intent engine** spins the preloaded user
context in a *centrifuge* until like settles with like; upstairs, a **normal
transformer** runs over what comes out. It is a runnable, see-through sketch of an
architecture idea — and it really does each step.

```
  FLOOR 1 · THE INTENT ENGINE  (the centrifuge)
     ()\<   intake — the user's preloaded context (a bag of token vectors)
     <\o/>  spin  — assign each token to its nearest centroid, recompute
            centroids, repeat. "Centrifugal, like-to-like" = convergent
            similarity clustering: similar vectors settle into the same bin and
            the intra-cluster energy is non-increasing every spin until nothing moves.
     >>>>   out   — K intent summaries (the centroids): a real N→K compression.

  FLOOR 2 · A NORMAL TRANSFORMER
     a real (toy) forward pass over the K intents → a pooled prediction.
```

## What it actually accomplishes

- **It converges.** The centrifuge is Lloyd's algorithm; total intra-cluster
  energy is monotonically non-increasing and it settles (verified).
- **It sorts like-to-like.** Planted, well-separated groups end up each in a
  single cluster, in distinct clusters (verified).
- **It compresses.** N context vectors become K intents — a genuine `N/K`
  reduction of what the transformer must attend over.
- **It transforms.** A real toy transformer (RMSNorm → attention → MLP) runs over
  the K intents and pools to a ranked prediction.

## What it is not

It is **not** a trained model, and it makes **no** claim to beat a standard
transformer. It is an honest, deterministic **toy** that demonstrates the two-stage
"silo" shape David sketched — a clustering pre-pass that hands a tidier, compressed
intent representation to a transformer. Whether that *helps* on real data is an
empirical question this does not answer; it only shows the mechanism, runnably.

## Verify first

```bash
python selftest.py     # 12 checks: converges, deterministic, like-to-like, N→K, runs
python silo.py         # run the silo on a sample context
```

## Files

| File | Role |
|------|------|
| `silo.py` | the centrifuge (`centrifuge`) + the toy transformer + `run_silo` |
| `selftest.py` | proves convergence, determinism, like-to-like, compression, end-to-end |
| `index.html` | the two-floor silo — load context, spin the centrifuge, watch it feed the transformer |

Kin to [the-forward-pass](https://davidwise01.github.io/the-forward-pass/) (floor 2 is that
engine) and the [membrane map](https://davidwise01.github.io/membrane-map/) family.

---
David Lee Wise / ROOT0 / TriPod LLC · CC-BY-ND-4.0
