# Security Authentication — Biometrics

## Problem Formulation

1:1 verification (is this person who they claim?) and 1:N identification (who is this person?). Metric learning: learn embedding space where same-identity pairs have high cosine similarity, different-identity pairs have low similarity. Decision threshold on cosine distance.

Objective: Minimize FRR (False Reject Rate) subject to FAR (False Accept Rate) < 1/1,000,000.

---

## Data & Transformations

| Aspect | Detail |
|--------|--------|
| Training | MS1MV3 (cleaned): 5.2M images, 93K identities |
| Augmentation | Random crop, horizontal flip, color jitter, low-res degradation |
| Preprocessing | MTCNN detect → 5-point alignment → affine to 112×112 |
| Quality filter | Drop blur (Laplacian var < 100), extreme pose (>60°), occlusion |
| Bias mitigation | Balanced sampling across demographics (race, age, gender) |
| Liveness data | 500K attack samples (print, replay, 3D mask, deepfake) |

Key: Training data quality > quantity. Cleaning MS-Celeb-1M (removing label noise) gave +2% TAR@FAR=1e-6.

---

## Model Architecture

**Recognition**: IResNet-100 → 512-d L2-normalized embedding.

**ArcFace Loss** (Additive Angular Margin):

$$\mathcal{L} = -\log \frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cdot \cos\theta_j}}$$

Where $s=64$ (scale), $m=0.5$ (margin), $\theta$ = angle between feature and class center.

**Why angular margin works**: Forces intra-class compactness and inter-class separation in hyperspherical space. Geometric interpretation: pushes decision boundary by margin $m$ radians.

**Liveness**: CDCN (Central Difference Convolution Network) — captures fine-grained local patterns (moiré, color distortion) that spoof attacks exhibit.

**On-device (distilled)**: MobileFaceNet (1M params, 4MB INT8) — distilled from IResNet-100 with KL + ArcFace joint loss.

---

## Training & Optimization

| Config | Value |
|--------|-------|
| Optimizer | SGD (lr=0.1, momentum=0.9, wd=5e-4) |
| Schedule | Step decay at epochs [20, 28, 32, 36] by 0.1× |
| Batch | 512 (64 GPUs × 8) with Partial FC (sample 10% negatives) |
| Total epochs | 40 |
| Precision | FP16 (AMP) |
| Distributed | PyTorch DDP + Partial FC (memory efficient for 93K classes) |

**Partial FC** (InsightFace, 2022): Only compute softmax over random 10% of class centers per step. Enables training with millions of identities without OOM. Mathematically equivalent in expectation.

---

## Serving & Scale

**On-device pipeline** (50ms total):
```
Camera frame → Face detect (2ms) → Liveness (15ms) → Align (3ms) → 
MobileFaceNet embed (20ms) → Cosine sim with stored template → Decision
```

- **Secure Enclave**: Template stored encrypted, comparison happens inside TEE  
- **Cancelable biometrics**: Apply user-specific random projection to embedding before storage — revocable if compromised  
- **Server fallback**: For 1:N (large gallery), use FAISS IVF-PQ index, <10ms for 1M gallery

---

## Metrics & Evaluation

| Metric | Target | Benchmark |
|--------|--------|-----------|
| TAR @ FAR=1e-6 | >99.0% | LFW: 99.83% |
| EER | <0.1% | |
| APCER (liveness) | <1% | ISO 30107-3 |
| BPCER (liveness) | <5% | |
| Demographic parity | Δ<2% across groups | NIST FRVT |

---

## Alternatives & Tradeoffs

| Approach | Pro | Con |
|----------|-----|-----|
| ArcFace (IResNet-100) | SOTA accuracy | Too large for mobile |
| AdaFace (2022) | Quality-adaptive margin, better on low-quality | Marginal gain on HQ data |
| CosFace (additive cosine) | Simpler, stable training | Slightly worse than ArcFace |
| On-device only | Privacy-first, no network | Can't do 1:N, no updates |
| Server-side (cloud) | Large gallery, updatable | Privacy risk, latency |
| Multimodal fusion | Higher security | More sensors, cost |

---

## Recent Research (2024–2026)

1. **AdaFace** (CVPR 2022, widely adopted 2024+) — image quality adaptive margin  
2. **TransFace** (2023) — ViT-based face recognition, competitive with CNN  
3. **SynFace** (2024) — training with fully synthetic faces (no real data, privacy-safe)  
4. **FLIP** (Meta, 2024) — scaling face recognition with foundation models  
5. **DiffFace** (2025) — diffusion-based face synthesis for balanced training  
6. **FaceCoresetNet** (2025) — efficient training via coreset selection (3× speedup)

---

## Advanced Trick Questions & Answers

**Q1: FAR=1e-6 looks great, but your system has 1B users. What's the real-world false accept rate?**  
A: With 1B users, even FAR=1e-6 means 1000 false accepts per million attempts at scale. For 1:N identification (gallery=1B), effective FAR = 1 - (1 - FAR)^N ≈ N×FAR = 0.1%. Solution: (1) cascade with liveness, (2) multi-factor, (3) only use 1:1 verification (not 1:N).

**Q2: Your model is biased — 5% higher FRR for darker skin tones. How do you fix it without losing overall accuracy?**  
A: (1) Balanced training data (equal representation), (2) Group-specific thresholds (equalized odds), (3) Adversarial debiasing in embedding space, (4) Fairness-aware loss (min-max across groups). Trade-off: group-specific thresholds may lower security for majority group.

**Q3: An attacker generates a deepfake video of the enrolled user. How does your liveness detector respond?**  
A: Standard liveness (texture-based) may fail against high-quality deepfakes. Defenses: (1) IR depth sensor (can't be faked with 2D display), (2) Temporal consistency check (physiological signals: pulse, blink), (3) Adversarial perturbation detection, (4) Challenge-response (ask for random head movement).

**Q4: You need to update the enrolled template as the user ages. How?**  
A: Adaptive enrollment: (1) After each successful auth, blend new embedding with stored (EMA, α=0.01), (2) Keep last 5 templates, accept if any matches, (3) Re-enrollment prompt every 12 months. Risk: gradual template poisoning by impostor — mitigate with large auth streak requirement before update.

**Q5: Apple FaceID works with masks. How is that possible with only half the face visible?**  
A: (1) Periocular region is discriminative alone (TAR@FAR=1e-4 ~95%), (2) Fine-tune on masked-face augmented data, (3) Apple uses additional IR dot projector + depth map (structured light) which is less affected by occlusion, (4) Attention mechanism learns to weight visible regions.
