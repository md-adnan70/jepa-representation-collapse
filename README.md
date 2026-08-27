# Interpretability-Flavored JEPA on CIFAR-10

A small I-JEPA (Image Joint-Embedding Predictive Architecture) implementation
trained from scratch, followed by a mechanistic interpretability analysis of
the learned representations — in the same style as the Qwen3 attention/SAE
project already on the CV. Includes a diagnosed representation-collapse
failure mode and a VICReg-style regularization fix, validated across two
independent training runs.



## Headline finding: the fix is reproducible, the collapse is robust

Re-running the full pipeline a second time produced two notable results:

- The **regularized model reproduced almost exactly** — same effective-rank
  curve (peaking at ~115/128), same probe-accuracy peak (38.65% at layer 4),
  same SAE dead-feature fraction (~15%). This is a real signal that the
  VICReg-style fix converges to a stable, non-collapsed solution rather than
  getting lucky once.
- The **baseline collapsed even more severely on the second run** — effective
  rank dropped to single digits (3–6.6/128) rather than the ~11–14 seen on
  the first run. Collapse isn't a one-time fluke of a particular random seed;
  it's the default outcome of this architecture without explicit
  anti-collapse regularization.

Together, these make the before/after comparison considerably stronger than
a single run would: the failure mode is robust, and the fix is repeatable.

---

## Results summary

### Training: baseline vs. regularized

| Metric | Baseline (no reg) | Regularized (VICReg-style) | Change |
|---|---|---|---|
| Final `repr_std` | 0.0175 | **0.9749** | **55.5x** |
| Final prediction loss | 0.383 | 0.464 | higher (expected — see note below) |

![Baseline training curves — loss and repr_std collapsing](result_graphs/output.png)

![Baseline vs. regularized repr_std and loss comparison](<result_graphs/output (5).png>)

**Note on the loss increase:** prediction loss going *up* after regularization
is expected, not a regression. The baseline takes a shortcut — predicting a
near-constant vector — which is trivially easy and produces artificially low
loss. Once collapse is prevented, the model has to solve the actual
prediction task, which is harder and reflected honestly in a higher loss.
`repr_std`, effective rank, probe accuracy, and SAE health are the metrics
that show whether the representations are actually useful — and by all four,
the regularized model is far better.

### Effective rank by layer (out of 128 possible)

**Baseline (rerun):** collapsed further than the first run — effective rank
stays in the single digits at every layer, with no clear structure by depth.

![Effective rank by layer — baseline, collapsed](<result_graphs/output (1).png>)

**Regularized:**

| Layer | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Effective rank | 14.1 | 39.7 | 62.7 | 69.5 | 83.0 | 115.4 |

![Effective rank by layer — regularized model](<result_graphs/output (6).png>)

Rank climbs steadily with depth, reaching 90% of full capacity by the final
layer — the encoder builds up richer structure layer by layer instead of
staying flat and collapsed everywhere.

### Attention entropy by layer

**Baseline (rerun):**

![Attention entropy by layer — baseline](<result_graphs/output (2).png>)

**Regularized:**

| Layer | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Entropy (nats) | 1.29 | 1.32 | 0.59 | **2.00** | 0.34 | 0.26 |

![Attention entropy by layer — regularized model](<result_graphs/output (7).png>)

Non-monotonic in both runs, with a sharp spike at layer 4 in the regularized
model — notably the same layer where linear probe accuracy peaks. Reported
as an observation, not a firm conclusion.

### Linear probe accuracy by layer (frozen target encoder, CIFAR-10)

**Baseline (rerun):** stays low (~20–25%) with no clean trend by depth.

![Linear probe accuracy by layer — baseline](<result_graphs/output (3).png>)

**Regularized:**

| Layer | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Test accuracy | 35.1% | 36.2% | 37.0% | **38.65%** | 37.3% | 33.0% |

![Linear probe accuracy by layer — regularized model](<result_graphs/output (8).png>)

Peak sits at a middle layer (layer 4) rather than the input layer, and best
accuracy nearly doubles versus the collapsed baseline — consistent with the
pattern reported in the published I-JEPA/MAE literature, where semantic
information concentrates mid-to-late rather than at the very first layer.

### Sparse Autoencoder analysis

**Baseline (rerun):** reconstruction loss collapses to near-zero almost
immediately and the vast majority of features never fire — the SAE has
nothing to decompose because the underlying representations carry almost no
variance.

![SAE analysis — baseline, collapsed representations](<result_graphs/output (4).png>)

**Regularized:** a genuine reconstruction task, with a much broader spread
of feature firing frequencies and a large majority of features actually in
use (dead-feature fraction ~15%, vs. ~93% in the baseline).

![SAE analysis — regularized model](<result_graphs/output (9).png>)

---

## What's implemented

**`model/jepa.py`**
- `ViTEncoder`: small pre-norm transformer (patch embed + sin-cos position
  embeddings + transformer blocks), used for both context and target encoders
- `Predictor`: narrow transformer that predicts target-encoder representations
  at masked positions from context-encoder representations + positional mask
  tokens
- `block_mask`: I-JEPA-style block masking (multiple target blocks, one large
  context region, no overlap)
- `variance_loss` / `covariance_loss`: VICReg-style anti-collapse
  regularization terms, applied to the context encoder's pooled
  representations. Disabled by default (weight=0) so base behavior is
  unchanged unless explicitly turned on.
- `JEPA`: wraps everything — EMA target encoder update, stop-gradient, loss

**`analysis/probe.py`**
- `effective_rank`: entropy-of-eigenspectrum measure of how many independent
  directions the representations actually use (a soft collapse detector)
- `attention_entropy_per_layer`: same metric as the Qwen3 project, applied
  here to the JEPA target encoder
- `train_linear_probe`: per-layer frozen linear probe on CIFAR-10 labels

**`analysis/sae.py`**
- `TopKSAE`: top-k sparse autoencoder (same recipe as modern LLM
  interpretability SAEs), trained per-layer on target-encoder representations
- `feature_sparsity_stats`: per-feature firing frequency, to identify dead
  features and compare sparsity patterns across layers

**`train.py`** — CLI training script with a live collapse-metric warning.

**`analyze.py`** — runs all four interpretability lenses on a trained
checkpoint and saves plots + `results.json`.

**`jepa_interpretability_kaggle.ipynb`** — the notebook that produced the
results above. Self-contained: writes the module files, trains the baseline,
trains the VICReg-regularized comparison, runs all analysis, saves plots.



