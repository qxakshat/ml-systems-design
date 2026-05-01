# Medical Diagnostics — Computer Vision

## Problem Formulation

Multi-task prediction on medical imaging: classification (14 pathologies), detection (lesion localization), segmentation (pixel-level delineation). Input: DICOM 512×512 (2D) or volumetric 3D (CT/MRI). Output: calibrated probability per pathology + spatial heatmap + segmentation mask.

Objective: Maximize sensitivity at fixed specificity operating point (clinical requirement).

---

## Data & Transformations

| Aspect | Detail |
|--------|--------|
| Volume | 2–5M labeled studies, 500M unlabeled (self-supervised pretraining) |
| Labeling | NLP-extracted from radiology reports (CheXpert labeler) + 3-radiologist consensus |
| Class imbalance | 50:1 healthy:pathology — use effective number reweighting (Cui et al.) |
| Preprocessing | Hounsfield windowing, CLAHE, 16-bit → normalized float |
| Augmentation | RandAugment-Med, elastic deformation, CutMix (α=1.0), Mixup (α=0.2) |
| 3D handling | Spacing normalization → isotropic 1mm³, sliding window with overlap |

Key transformation: **Self-supervised pretraining** (MAE / DINOv2 on unlabeled scans) before supervised fine-tuning yields +3–5% AUC lift over ImageNet init.

---

## Model Architecture

**Backbone**: ConvNeXt-V2-L (pre-trained DINOv2) or BiomedCLIP for multi-modal.

**Multi-task heads**:
- Classification: GAP → FC → 14-dim sigmoid  
- Detection: DINO-DETR (anchor-free, set prediction)  
- Segmentation: nnU-Net (auto-configured per dataset)

**Loss**:

$$\mathcal{L} = \lambda_1 \cdot \text{Focal}(\gamma=2) + \lambda_2 \cdot (\text{Dice} + \text{BCE}) + \lambda_3 \cdot (\text{GIoU} + \ell_1)$$

Where $\lambda$ values are learned via uncertainty weighting (Kendall et al., 2018):

$$\mathcal{L}_{total} = \sum_i \frac{1}{2\sigma_i^2}\mathcal{L}_i + \log \sigma_i$$

**Uncertainty**: MC-Dropout (T=30 forward passes) → predictive entropy for referral-to-human decisions.

---

## Training & Optimization

| Config | Value |
|--------|-------|
| Optimizer | AdamW (lr=1e-4, β₁=0.9, β₂=0.999, wd=0.05) |
| Schedule | Cosine with 5-epoch linear warmup |
| Batch | 64 (DDP across 8×A100, gradient accumulation=4) |
| Precision | BF16 mixed precision |
| EMA | decay=0.9999 (used for evaluation) |
| Regularization | Stochastic depth=0.2, DropPath, label smoothing=0.1 |
| Epochs | 100 (early stop patience=15 on val AUC) |

**Critical**: 5-fold stratified cross-validation → ensemble averaging → +1.5% AUC. TTA (8 augmentations) at inference adds +0.5% at 8× latency cost.

---

## Serving & Scale

```
DICOM → Preprocessing (CPU) → TensorRT FP16 Engine (GPU) → Post-processing → Report
                                    ↓
                          Triton Inference Server
                          Dynamic batching (max_batch=16, delay=50ms)
```

- **Latency**: 200ms/image (single), 50ms/image (batched)  
- **Throughput**: 500 img/sec/GPU (T4), 2000 img/sec/GPU (A100)  
- **Deployment**: Federated learning across hospitals (FedAvg with differential privacy ε=8)  
- **Monitoring**: CUSUM on confidence distribution for drift detection

---

## Metrics & Evaluation

| Metric | Target | Why |
|--------|--------|-----|
| AUC-ROC (macro) | >0.95 | Discrimination |
| Sensitivity @ Spec=95% | >0.90 | Clinical operating point |
| FROC (FP/image) | <0.5 at Sens=80% | Detection task |
| Dice | >0.85 | Segmentation |
| ECE | <0.03 | Calibration (critical for clinical trust) |

---

## Alternatives & Tradeoffs

| Approach | Pro | Con |
|----------|-----|-----|
| nnU-Net (auto-config) | SOTA segmentation, no manual tuning | Slow training search, not multi-task |
| BiomedCLIP (foundation) | Zero-shot, multi-modal | Lower fine-tuned performance vs. specialist |
| Federated Learning | Privacy-preserving | Communication overhead, heterogeneity |
| Ensemble (5-fold) | +1.5% AUC | 5× inference cost |
| Single model + TTA | Moderate quality-latency | 8× latency increase |

---

## Recent Research (2024–2026)

1. **BiomedCLIP** (Microsoft, 2024) — domain-specific CLIP on PubMed figure-caption pairs  
2. **LLaVA-Med** (Microsoft, 2024) — multi-modal LLM for medical visual QA  
3. **STU-Net** (2024) — scalable nnU-Net variant (1.4B params, SOTA on 14 datasets)  
4. **MedSAM** (2024) — Segment Anything adapted for medical imaging, zero-shot  
5. **MAIRA-2** (Microsoft, 2025) — grounded radiology report generation from CXR  
6. **CheXagent** (Stanford, 2025) — foundation model for CXR interpretation with tool use

---

## Advanced Trick Questions & Answers

**Q1: Your model achieves 0.98 AUC but radiologists reject it. Why?**  
A: AUC is threshold-independent. At the clinical operating point (e.g., Sens@Spec=95%), performance may be mediocre. Also: poor calibration (ECE), lack of explainability (no heatmaps), or failure on subgroups (age, device manufacturer). Always report threshold-specific metrics and stratified performance.

**Q2: You have 10 hospitals with data silos. How do you train without centralizing data?**  
A: Federated Learning (FedAvg/FedProx). Key issues: (1) non-IID data across sites → FedBN (batch-norm local), (2) privacy → secure aggregation + DP (ε=8), (3) communication → gradient compression. Alternative: split learning if compute-constrained at sites.

**Q3: A competitor claims 0.99 AUC on the same task. Should you be worried?**  
A: Check: (1) label leakage (patient-level split vs. study-level), (2) dataset selection bias (only clean cases), (3) evaluation on external data, (4) prevalence in test set. Most inflated numbers come from improper splitting — same patient in train+test inflates AUC by 5–10%.

**Q4: How do you handle the 30-day deployment delay before real labels arrive?**  
A: Monitor proxy metrics: (1) confidence distribution shift (KL divergence), (2) out-of-distribution detection (Mahalanobis distance in embedding space), (3) human override rate. Only retrain when drift is confirmed.

**Q5: Your model works on Siemens scanners but fails on GE. Root cause?**  
A: Domain shift from acquisition parameters (kVp, reconstruction kernel, slice thickness). Fix: (1) scanner-agnostic preprocessing (Hounsfield calibration), (2) domain randomization augmentation, (3) adversarial domain adaptation (DANN), (4) include scanner as a feature in batch-norm (conditional BN).
