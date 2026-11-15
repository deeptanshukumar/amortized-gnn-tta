# Conditioning version ablation

## Current final submission results

The authoritative final source is **Arvind's `AFML Research Project.pdf`, page 16**, as selected by the project owner. The tables below reproduce its Experiment 1 validation and test conditioning comparisons. These are PDF-reported values; the newer artifacts have not been independently replayed here. Gains and reported ± dispersion are **percentage points**; frozen accuracy is a percentage. The dispersion definition is not independently verified.

### Validation

| Model | Frozen accuracy % | IM-LoRA gain pp | A gain pp | B gain pp | C gain pp | D gain pp |
|---|---:|---:|---:|---:|---:|---:|
| GCN | 62.891 | −5.895 ± 0.460 | +0.118 ± 0.044 | +0.156 ± 0.042 | +0.157 ± 0.012 | +0.157 ± 0.025 |
| GraphSAGE | 62.112 | −6.351 ± 0.491 | +0.086 ± 0.004 | +0.076 ± 0.011 | +0.110 ± 0.032 | +0.089 ± 0.024 |
| GAT | 64.421 | −14.164 ± 1.324 | −0.057 ± 0.008 | −0.063 ± 0.048 | −0.062 ± 0.032 | −0.060 ± 0.014 |

### Final test

| Model | Frozen accuracy % | IM-LoRA gain pp | A gain pp | B gain pp | C gain pp | D gain pp |
|---|---:|---:|---:|---:|---:|---:|
| GCN | 60.198 | −8.393 ± 0.534 | +0.412 ± 0.105 | +0.411 ± 0.058 | +0.419 ± 0.059 | +0.452 ± 0.046 |
| GraphSAGE | 58.499 | −8.601 ± 0.267 | +0.210 ± 0.019 | +0.112 ± 0.127 | +0.198 ± 0.053 | +0.132 ± 0.105 |
| GAT | 60.845 | −14.820 ± 0.876 | −0.018 ± 0.035 | +0.016 ± 0.026 | −0.018 ± 0.014 | −0.010 ± 0.027 |

### What these results establish

There is no universal winning descriptor. Validation gains are small and architecture dependent: GCN C/D tie at the reported precision, GraphSAGE C has the largest mean, and all GAT means are negative. The test table is descriptive evidence, not permission to choose a different input after inspecting final labels. Small mean differences and the reported dispersion do not establish statistical significance without paired run-level analysis.

The older CSBM dimensions and descriptor details below belong to their original configuration, not a universal input schema for Arxiv or the newer runner. Verify the exact checkpoint/configuration before applying those descriptions to the final experiments. The final E4/E5 and E6 results are summarized in [README](README.md); E6 efficiency figures must remain separate from E1 input comparisons.

## Historical evidence retained below

Commit `c137e28` published the original scripts and JSON under `scratch/`. Arithmetic checks passed, including the roughly 45× screenshot. Those ratios measure components: adapted prediction is outside generator timers, and the compact comparison reuses conditioning outside diffusion timing. Exact replay requires matching checkpoints, seeds and hardware. These viewed CSBM scores are development evidence. Direct all-weight updates are not LoRA; the historical spectral ratio is not a condition number. Preserve the old measurements without treating them as final real-data conclusions.

---

# Historical CSBM conditioning ablation

**Date:** 2026-11-07  
**Benchmark:** Full Unconstrained CSBM Distribution-Shift Benchmark (612 evaluation pairs across GCN, GAT, and GraphSAGE)  
**Evaluated Approaches:**
1. **Regular Fine-Tuning (Test-Time Optimization TTA):** 30-step unsupervised gradient descent with information maximization on the shifted test graph.
2. **Hypernetwork Generator (Amortized TTA):** Single forward-pass prediction conditioned on versions A, B, C, and D.
3. **Diffusion Generator (Amortized TTA):** 10-step deterministic DDIM sampling conditioned on versions A, B, C, and D.

---

## 1. Conditioning Versions Hierarchy

The ablation investigates whether each additional information stream helps the generator adapt weights:

* **Version A (Target graph only, Dim 145):** Summaries of raw features $X$ (channel mean/std, global mean/std, norm, sparsity) and connections $E$ (degree statistics, density, raw Dirichlet energy, edge distance dispersion, center dispersion). **Computed strictly without any GNN model or labels.**
* **Version B (+ Hidden representation, Dim 246):** Version A + latent feature moments ($\mu_H, \sigma_H$) and hidden Dirichlet energy / dispersion from `model.encode(x, edge_index)`.
* **Version C (+ Predictive uncertainty, Dim 255):** Version B + label-free predictive Shannon entropy ($\mu_{\mathbb{H}}, \sigma_{\mathbb{H}}$, high-entropy ratio), prediction confidence, class margins, and empirical class prior distribution.
* **Version D (+ Pretrained weights, Dim 267):** Version C + compact parameter fingerprint (spectral SVD singular values $\sigma_1, \sigma_2$, condition ratio, Frobenius norm, distribution quantiles).

---

## 2. Full Benchmark Results Across Architectures

Evaluated across all **204 held-out test pairs per architecture** (6 models $\times$ 34 environments).

### GCN (Classification Head: 66 Parameters)
* **Baseline (Frozen Pretrained GCN):** 54.67%
* **Regular Fine-Tuning (30 steps IM):** 54.68% (**+0.00 pp**, Latency: 65.19 ms, Win Rate: 48.5%)

| Method | Conditioning Version | Input Dim | Adapted Acc | Net Gain (pp) | Win Rate | TTA Latency | Speedup vs. Fine-Tuning |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hypernetwork** | **Version A** *(Graph only)* | 145 | 54.17% | -0.50 pp | 35.3% | **0.77 ms** | **84.6× faster** |
| **Hypernetwork** | **Version B** *(+ Hidden)* | 246 | 54.19% | -0.48 pp | 34.3% | 1.22 ms | **53.3× faster** |
| **Hypernetwork** | **Version C** *(+ Uncertainty)* | 255 | 53.96% | -0.71 pp | 29.4% | 2.34 ms | **27.8× faster** |
| **Hypernetwork** | **Version D** *(+ Weights)* | 267 | 54.17% | -0.50 pp | 37.7% | 2.79 ms | **23.3× faster** |
| **Diffusion (DDIM)** | **Version A** *(Graph only)* | 145 | 54.91% | +0.24 pp | 60.8% | 4.60 ms | **14.2× faster** |
| **Diffusion (DDIM)** | **Version B** *(+ Hidden)* | 246 | **55.18%** | **+0.51 pp** | **71.1%** | **2.12 ms** | **30.7× faster** |
| **Diffusion (DDIM)** | **Version C** *(+ Uncertainty)* | 255 | 54.92% | +0.24 pp | 64.7% | 3.78 ms | **17.2× faster** |
| **Diffusion (DDIM)** | **Version D** *(+ Weights)* | 267 | 54.71% | +0.04 pp | 51.0% | 3.66 ms | **17.8× faster** |

---

### GAT (Classification Head: 70 Parameters)
* **Baseline (Frozen Pretrained GAT):** 54.49%
* **Regular Fine-Tuning (30 steps IM):** 54.31% (**-0.17 pp**, Latency: 47.62 ms, Win Rate: 40.2%)

| Method | Conditioning Version | Input Dim | Adapted Acc | Net Gain (pp) | Win Rate | TTA Latency | Speedup vs. Fine-Tuning |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hypernetwork** | **Version A** *(Graph only)* | 145 | 54.49% | +0.00 pp | 0.0% (identity) | **0.68 ms** | **69.9× faster** |
| **Hypernetwork** | **Version B** *(+ Hidden)* | 246 | 54.49% | +0.00 pp | 0.0% (identity) | 1.96 ms | **24.4× faster** |
| **Hypernetwork** | **Version C** *(+ Uncertainty)* | 255 | 54.49% | +0.00 pp | 0.0% (identity) | 3.31 ms | **14.4× faster** |
| **Hypernetwork** | **Version D** *(+ Weights)* | 267 | 54.09% | -0.39 pp | 46.6% | 3.71 ms | **12.8× faster** |
| **Diffusion (DDIM)** | **Version A** *(Graph only)* | 145 | 54.49% | +0.00 pp | 0.0% (identity) | 1.69 ms | **28.2× faster** |
| **Diffusion (DDIM)** | **Version B** *(+ Hidden)* | 246 | 54.49% | +0.00 pp | 0.0% (identity) | 3.02 ms | **15.8× faster** |
| **Diffusion (DDIM)** | **Version C** *(+ Uncertainty)* | 255 | **54.69%** | **+0.20 pp** | **56.4%** | 4.42 ms | **10.8× faster** |
| **Diffusion (DDIM)** | **Version D** *(+ Weights)* | 267 | 54.49% | +0.00 pp | 0.0% (identity) | 4.73 ms | **10.1× faster** |

---

### GraphSAGE (Classification Head: 130 Parameters)
* **Baseline (Frozen Pretrained GraphSAGE):** 51.77%
* **Regular Fine-Tuning (30 steps IM):** 52.53% (**+0.77 pp**, Latency: 29.68 ms, Win Rate: 74.5%)

| Method | Conditioning Version | Input Dim | Adapted Acc | Net Gain (pp) | Win Rate | TTA Latency | Speedup vs. Fine-Tuning |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hypernetwork** | **Version A** *(Graph only)* | 145 | 52.15% | +0.39 pp | 60.3% | **0.68 ms** | **44.0× faster** |
| **Hypernetwork** | **Version B** *(+ Hidden)* | 246 | **52.82%** | **+1.05 pp** | **69.6%** | 1.45 ms | **20.5× faster** |
| **Hypernetwork** | **Version C** *(+ Uncertainty)* | 255 | 52.65% | +0.88 pp | 68.1% | 2.12 ms | **14.0× faster** |
| **Hypernetwork** | **Version D** *(+ Weights)* | 267 | 52.63% | +0.86 pp | 67.6% | 2.42 ms | **12.3× faster** |
| **Diffusion (DDIM)** | **Version A** *(Graph only)* | 145 | 51.72% | -0.04 pp | 50.5% | 1.69 ms | **17.5× faster** |
| **Diffusion (DDIM)** | **Version B** *(+ Hidden)* | 246 | 50.97% | -0.80 pp | 40.2% | 2.45 ms | **12.1× faster** |
| **Diffusion (DDIM)** | **Version C** *(+ Uncertainty)* | 255 | **52.44%** | **+0.68 pp** | **65.7%** | 3.14 ms | **9.5× faster** |
| **Diffusion (DDIM)** | **Version D** *(+ Weights)* | 267 | 51.62% | -0.14 pp | 43.1% | 3.38 ms | **8.8× faster** |

---

## 3. Historical development observations and hypotheses

### 1. Does each additional input help in this order?
* **Version B performed well in some historical development comparisons:**
  * Going from Version A (target graph only) to Version B (+ hidden representation) produced substantial accuracy improvements:
    * GraphSAGE jumped from **+0.39 pp $\to$ +1.05 pp** (surpassing regular fine-tuning's +0.77 pp!).
    * On GCN Diffusion, Version B delivered the peak performance (**+0.51 pp** with a **71.1% win rate**).
  * *Why:* The raw graph (Version A) tells the generator what the data looks like, but the hidden representations (Version B) reveal how the *specific pretrained model* aggregates features under the shift.
* **Version C helped some historical diffusion comparisons:**
  * On GAT, Versions A, B, and D all triggered the "do-no-harm" fallback ($\Delta = 0$). **Version C (+ predictive uncertainty) was the only version that successfully adapted GAT (+0.20 pp, 56.4% win rate)** while Fine-Tuning degraded it (-0.17 pp).
  * On GraphSAGE Diffusion, Version C turned a negative gain (-0.80 pp) into a solid positive gain (**+0.68 pp**).
* **Compact Pretrained Weights (Version D) add Marginally Diminishing Returns:**
  * While compact weights provide structural information about parameter norms and spectral condition, they slightly increased input dimensionality (from 255 to 267) without improving performance over Version B or C.

### 2. TTA Latency & Speedup Analysis Across Versions
* **Version A is the Fastest (44× to 85× Faster than Fine-Tuning):**
  * Because Version A does not require passing the graph through the model to extract hidden states or predictions, the recorded component timer is **0.68 – 0.77 ms**.
* **Version B achieves 20× to 53× Speedups:**
  * Running `model.encode(x, edge_index)` adds ~0.5 ms, resulting in a recorded component time of **1.2 – 1.9 ms**.
* **Versions C & D achieve 9× to 28× Speedups:**
  * Computing softmax predictions, entropy, and SVD singular values adds ~1.0 – 1.5 ms, bringing recorded component time to **2.1 – 4.7 ms** (still **10× to 25× faster** than the 30–65 ms required for gradient fine-tuning).

```
TTA Inference Latency by Conditioning Version (GCN):
  Version A: [■       ] 0.77 ms  (84.6x faster than Fine-Tuning)
  Version B: [■■      ] 1.22 ms  (53.3x faster than Fine-Tuning)
  Version C: [■■■■    ] 2.34 ms  (27.8x faster than Fine-Tuning)
  Version D: [■■■■■   ] 2.79 ms  (23.3x faster than Fine-Tuning)
  Fine-Tune: [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■] 65.19 ms (1.0x Baseline)
```

---

## 4. Historical recommendations and their limits

The original report favored B for hypernetworks and C for diffusion in this CSBM development study. These are provisional observations, not optimality claims or evidence that predictive uncertainty prevents negative transfer. Component timings exclude required prediction work; they cannot establish complete adaptation speedups. The final real-data conditioning comparison above supersedes these recommendations for the submission.
