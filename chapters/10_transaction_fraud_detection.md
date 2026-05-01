# Transaction Monitoring — Identify and Block Frauds

## Problem Formulation

Real-time binary classification: given a transaction $t$, predict P(fraud | transaction, user_history, merchant, device) within 50ms to approve/decline. Extreme class imbalance (fraud rate: 0.1%). Cost-sensitive: false negative ($500 avg loss) >> false positive ($5 customer friction cost).

Objective: Minimize expected cost:
$$\mathcal{L} = C_{FN} \cdot P(FN) + C_{FP} \cdot P(FP)$$

where $C_{FN}/C_{FP} \approx 100:1$.

---

## Data & Transformations

| Aspect | Detail |
|--------|--------|
| Volume | 10K–100K transactions per second (peak) |
| Fraud rate | 0.05–0.15% (highly imbalanced) |
| Labels | Confirmed fraud (chargeback, 30–60 day delay), reported fraud (immediate) |
| Latency | Label delay: 30–60 days (chargebacks). Creates train/serve skew |
| Feature count | 500+ features across transaction, user, merchant, device, network |
| Graph data | Transaction graph (user→merchant, user→device, device→IP) for ring detection |

**Feature engineering**:
- **Velocity features**: # transactions in last 1min / 5min / 1hr / 24hr (per user, per card, per device)
- **Deviation features**: amount vs. user's historical mean/std, merchant category deviation
- **Network features**: device fingerprint reuse count, IP geolocation vs. billing address distance
- **Sequence features**: time since last transaction, merchant category sequence anomaly
- **Graph features**: PageRank of user in transaction graph, connected component size

---

## Model Architecture

**Ensemble: XGBoost + GNN + Rules Engine**

**Layer 1 — Rules Engine** (deterministic, <1ms):
- Hard blocks: known compromised cards, sanctions list, velocity >$10K/min
- Whitelists: verified merchants, recurring payments
- Filters ~5% of transactions (pure fraud or pure legitimate)

**Layer 2 — XGBoost** (tabular features, <5ms):
- 500 features, 1000 trees, max_depth=8
- Trained on 90 days of data, refreshed daily
- Handles: amount anomalies, velocity spikes, merchant risk scores

**Layer 3 — GNN** (graph-based, <20ms for pre-computed embeddings):
$$h_v^{(l+1)} = \sigma\left(W^{(l)} \cdot \text{AGG}(\{h_u^{(l)} : u \in \mathcal{N}(v)\})\right)$$

- Detects fraud rings: shared devices, mule accounts, coordinated attacks
- GAT (Graph Attention Network) over transaction graph
- Node embeddings updated every 5 minutes (streaming)

**Fusion**:
$$\text{score} = w_1 \cdot P_{XGB}(\text{fraud}) + w_2 \cdot P_{GNN}(\text{fraud}) + \text{rule\_override}$$

With learned weights $w_1, w_2$ via stacking (logistic regression on held-out set).

---

## Training & Optimization

| Config | Value |
|--------|-------|
| XGBoost | scale_pos_weight=500, eta=0.05, subsample=0.8, daily retrain |
| GNN | 3-layer GAT, hidden=128, attention heads=4, batch=512 subgraphs |
| Label strategy | Confirmed fraud (30d delay) + early fraud reports (same-day) |
| Oversampling | SMOTE for XGBoost, focal loss for GNN |
| Threshold | Optimized for: precision@recall=0.90 (catch 90% fraud) |
| Adversarial training | Synthetic fraud patterns injected monthly (red team) |

**Focal Loss** (addresses class imbalance without oversampling):
$$FL(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)$$

With $\gamma$=2, $\alpha$=0.75: down-weights easy negatives, focuses on hard examples.

**Label delay problem**: Use "confirmed legitimate" (no chargeback after 60 days) as negative. Pre-60-day negatives are "presumed legitimate" — add label noise tolerance.

---

## Serving & Scale

```
Transaction arrives → Real-time feature computation (Flink, <10ms):
  velocity counters, device fingerprint, geo-distance →
Rules engine check (<1ms) → [block/allow/continue] →
XGBoost scoring (<5ms) → 
If score > 0.3: GNN scoring (<20ms, graph lookup) →
Final decision: approve / decline / challenge (3D Secure) →
Async: update graph, update velocity counters, log for training

Total: P50=15ms, P95=45ms, P99=80ms
```

- **Feature store**: Redis Cluster (velocity counters, device fingerprints) — 10M reads/sec  
- **Graph DB**: Neo4j/TigerGraph for fraud ring queries (batched, not real-time per txn)  
- **Cascade design**: Only ~5% of transactions hit the expensive GNN path  
- **Fallback**: If model service is down → rules-only mode (higher FP rate, acceptable)

---

## Metrics & Evaluation

| Metric | Purpose | Target |
|--------|---------|--------|
| Recall (TPR) | Catch fraud | >90% |
| Precision | Minimize false declines | >50% (given 100:1 cost ratio) |
| Dollar recall | % of fraud dollars caught | >95% |
| False decline rate | Customer friction | <0.5% of legitimate transactions |
| Detection latency | Time to block | <50ms (real-time), <1hr (batch for rings) |
| AUC-PR | Overall model quality | >0.85 (on imbalanced test set) |
| $ saved vs. $ friction | Net business value | Must be positive |

---

## Alternatives & Tradeoffs

| Approach | Pro | Con |
|----------|-----|-----|
| Rules-only | Interpretable, fast, no ML infra | Can't detect novel patterns, rigid |
| XGBoost (current L2) | Fast inference, strong tabular | Misses graph patterns |
| Deep network (MLP) | Flexible | No clear advantage over XGBoost for tabular |
| GNN (current L3) | Detects rings, collusion | Slow, graph maintenance is expensive |
| Autoencoder (anomaly) | Unsupervised, catches novel fraud | High FP rate, hard to threshold |
| Sequence model (LSTM) | Captures behavioral sequences | Latency, limited to single-user patterns |
| LLM-based (transaction narration) | Reasoning over patterns | Way too slow for real-time |

---

## Recent Research (2024–2026)

1. **BotRGCN** (2024) — Relational GCN for bot/fraud detection on transaction graphs  
2. **GTAN** (KDD 2024) — Graph Transformer for Anti-fraud with temporal edges  
3. **SUPP** (Visa, 2024) — Self-supervised pre-training on transaction sequences  
4. **Federated fraud detection** (2025) — Cross-bank fraud detection without sharing raw data  
5. **Concept drift adaptation** (2025) — Online model updates for rapidly evolving fraud tactics  
6. **Synthetic fraud generation** (2024) — GAN-based fraud simulation for training data augmentation

---

## Advanced Trick Questions & Answers

**Q1: Your model catches 92% of fraud. The fraud team says "great." But fraudsters are now testing with $1 transactions before making $5000 purchases. Your model ignores small amounts. How do you adapt?**  
A: (1) Don't use amount as a primary feature for the "testing" pattern — instead, detect the *behavioral signature*: rapid small transactions → large transaction from same card/device. (2) Velocity features: flag sequences of small txns from new device + new merchant categories. (3) Specific "card testing" model: detect burst of small auth attempts (often from same IP/device against multiple cards). (4) Treat card-testing as fraud *even if each individual txn is $1* — blocking early prevents the $5000 loss.

**Q2: Chargebacks arrive 30–60 days after fraud. Your model trained today uses labels from 60 days ago. A new fraud ring started 2 weeks ago using a novel technique. What happens?**  
A: Model is blind to it for 30–60 days (label delay). Mitigation: (1) Anomaly detection layer (unsupervised) — catches novel patterns without labels, (2) Early fraud reports (user-reported within hours) as "soft labels" — don't wait for chargeback, (3) Semi-supervised: if cluster of transactions shares device fingerprint with 2 confirmed frauds → label entire cluster as suspicious, (4) Human-in-the-loop: fraud analysts review top anomalies daily → fast label injection.

**Q3: You set a threshold at 0.5 (score > 0.5 → decline). A $10,000 transaction scores 0.48 and a $5 transaction scores 0.52. The $10,000 passes and the $5 is declined. Is this rational?**  
A: No. Cost-sensitive thresholding: decline if E[cost_of_approving] > E[cost_of_declining].
$$\text{Decline if: } P(\text{fraud}) \times \text{amount} > C_{FP}$$
For $10K txn: 0.48 × $10000 = $4800 > $5 friction cost → **should decline**.  
For $5 txn: 0.52 × $5 = $2.60 < $5 → **should approve**.  
Fix: amount-aware threshold: $\theta = C_{FP} / \text{amount}$. High-value transactions get lower thresholds.

**Q4: Your GNN detects fraud rings beautifully on historical data. In production, a fraud ring creates 1000 new accounts with no graph connections. The GNN sees isolated nodes. How do you handle it?**  
A: Cold-start fraud (no graph history). (1) Device/IP clustering: even if accounts are "new," they share device fingerprints, IP ranges, or registration patterns. Build *registration graph* not just *transaction graph*. (2) Behavioral biometrics: typing patterns, app navigation sequences (same bot → same behavior). (3) Rule: new account + high-value txn within 24hr of creation → auto-challenge (3D Secure). (4) The GNN is useless here — the XGBoost layer with velocity/device features must catch it.

**Q5: Regulators require you to explain why a transaction was declined (GDPR Article 22). Your GNN+XGBoost ensemble gives a score of 0.85 but no interpretable reason. What do you do?**  
A: (1) Post-hoc explanation: SHAP values on XGBoost features → "declined because: unusual merchant category + transaction from new device + 3 rapid transactions in 5 minutes." (2) For GNN: extract the subgraph that activated → "your card was used on a device linked to 5 other declined transactions." (3) Rule-based explanation layer: map model features to human-readable reasons (pre-defined templates). (4) Compliance: always provide top-3 reasons from a fixed taxonomy (amount anomaly, location anomaly, velocity, device risk, merchant risk). Even if GNN contributed, explain via proxy features.
