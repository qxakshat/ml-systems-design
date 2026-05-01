# Google Ad Click Prediction

## Problem Formulation

Predict P(click | query, ad, user, context) with extreme precision requirements. The predicted CTR directly determines ad ranking and pricing: $\text{AdRank} = \text{pCTR} \times \text{Bid}$. Miscalibration by 1% translates to $2B+ annual revenue impact.

Objective: Minimize calibrated log-loss (predicted probabilities must match observed click rates).

---

## Data & Transformations

| Aspect | Detail |
|--------|--------|
| Volume | 1T+ training examples (30 days rolling) |
| Features | Billions of sparse categorical features (query×ad×user crosses) |
| Label | Click (1) within session, no-click (0) |
| Negative downsampling | 100:1 → correct calibration: $p_{corrected} = \frac{p}{p + (1-p)/w}$ |
| Feature crosses | query_token × ad_keyword × device × hour (hash to 2^24 buckets) |
| Continuous training | Streaming updates — model never "stops" training |

**Feature engineering tiers**:
1. **Query**: text embed, intent, frequency, historical CTR
2. **Ad**: creative embed, landing page quality, advertiser trust, historical CTR (smoothed)
3. **User**: interest segments, demographics, past ad interactions
4. **Context**: device, time, location, position (for bias correction)
5. **Cross**: query×ad relevance, user×advertiser affinity, position×device

---

## Model Architecture

**DCN-v2** (Deep & Cross Network v2, Google 2021):

Cross layer (explicit feature interactions):
$$x_{l+1} = x_0 \odot (W_l \cdot x_l + b_l) + x_l$$

Deep network: 8 FC layers (2048→1024→512→256), ReLU + BatchNorm.

**Multi-Task (MMoE)** — Mixture of Experts:

$$y_k = h_k\left(\sum_{i=1}^{n} g_k^{(i)}(x) \cdot f_i(x)\right)$$

Where $g_k$ = gating network for task $k$, $f_i$ = expert network $i$.

Tasks: P(click), P(conversion|click), P(long_engagement), E[revenue].

**Embedding layer**: Billions of sparse IDs → trainable embeddings (dim 16–64). Total embedding table: 100s of TB, distributed across parameter servers.

**Calibration** (Platt scaling):
$$p_{calibrated} = \frac{1}{1 + e^{-(a \cdot \log(p/(1-p)) + b)}}$$

---

## Training & Optimization

| Config | Value |
|--------|-------|
| Optimizer | Adagrad (lr=0.01) for sparse, Adam for dense |
| Batch | 65536 (TPU v4 pods) |
| Training mode | Continuous/streaming (FTRL for online learning) |
| Model push | Every few hours (warm start from previous) |
| Negative downsample | 100:1 with calibration correction |
| Feature pruning | Weekly: remove features with <0.001% importance |
| Hardware | 10K+ TPU chips for training |

**FTRL-Proximal** (Follow The Regularized Leader) for online sparse learning:
$$w_{t+1} = \arg\min_w \left(g_{1:t} \cdot w + \frac{1}{2}\sum_{s=1}^t \sigma_s \|w - w_s\|^2 + \lambda_1\|w\|_1\right)$$

Enables per-coordinate learning rates and L1 sparsity (95% of weights are zero).

---

## Serving & Scale

```
Ad auction triggered → Candidate ads (1000) → Feature fetch (<2ms, cache) →
Model scoring (all candidates in single batch, <5ms) → 
Calibration → Ad Rank = pCTR × Bid × Quality → GSP Auction → Show ads
```

- **Latency budget**: <10ms total for scoring (ad auction is latency-critical)  
- **Serving hardware**: Custom TPU inference / quantized INT8 on CPU  
- **Cascade**: L0 (cheap features, filter 1000→200) → L1 (full model, score 200)  
- **Feature caching**: User features (1hr TTL), ad features (24hr TTL), cross features (computed real-time)

---

## Metrics & Evaluation

| Metric | Purpose | Target |
|--------|---------|--------|
| Log loss | Calibration quality | Lower is better |
| Normalized Entropy (NE) | Relative improvement | NE = logloss / entropy(background_CTR) |
| AUC | Discrimination | >0.80 |
| Calibration ratio | predicted/actual | 0.98–1.02 |
| Revenue per mille (RPM) | Business impact | Online A/B |
| Advertiser ROI | Ecosystem health | conversions/spend |

**Normalized Entropy** is the gold standard at Google/Meta — accounts for background CTR shift:
$$NE = \frac{-\frac{1}{N}\sum (y\log p + (1-y)\log(1-p))}{-(q\log q + (1-q)\log(1-q))}$$

---

## Alternatives & Tradeoffs

| Approach | Pro | Con |
|----------|-----|-----|
| Logistic Regression + FTRL | Interpretable, fast, proven | Limited feature interactions |
| GBDT (XGBoost) | Strong on tabular, fast | Can't handle sparse embeddings at scale |
| DCN-v2 (current) | Explicit + implicit interactions | Complex, expensive |
| Transformer (AutoInt) | Attention-based feature interaction | Latency at serving |
| DLRMs (Meta) | Proven at scale | Huge embedding tables |
| LLM-based (ad text understanding) | Better semantic matching | Cost, latency constraints |

---

## Recent Research (2024–2026)

1. **GDCN** (Google, 2024) — Gated Deep Cross Network, gating improves DCN-v2 by filtering noise  
2. **FinalMLP** (Huawei, 2024) — two-stream MLP matching/exceeding DCN-v2  
3. **HSTU** (Meta, 2024) — Hierarchical Sequential Transduction Units for ads  
4. **DCNM** (Google, 2025) — Deep Cross Network with Mixture of experts  
5. **Privacy Sandbox (Topics API)** — Google's cookieless targeting (fundamentally changes feature space)  
6. **Gemini Ads** (Google, 2025) — LLM-generated ad creatives with predicted CTR optimization

---

## Advanced Trick Questions & Answers

**Q1: Your model's AUC is 0.82 (excellent) but calibration is off by 5% (predicts 5% when true rate is 4.75%). Revenue drops. Why does calibration matter more than AUC?**  
A: In ad auctions, ranking is $\text{pCTR} \times \text{Bid}$. Over-prediction means ads win auctions they shouldn't → advertisers pay more per click → ROI drops → advertisers reduce budget → revenue drops long-term. Under-prediction means high-quality ads lose → worse user experience. Calibration directly affects pricing. A perfectly discriminating but miscalibrated model destroys the auction equilibrium.

**Q2: Position 1 has 10× higher CTR than position 5. If you train on click data directly, what happens?**  
A: Position bias: the model learns P(click | position) not P(relevance). Ads that historically appeared in high positions get inflated pCTR. Solutions: (1) Position as a feature at training, removed at inference ("counterfactual" position), (2) Inverse Propensity Weighting: weight each example by 1/P(examine|position), (3) Joint model: P(click) = P(examine|pos) × P(click|examine,relevant).

**Q3: Chrome removes third-party cookies in 2025. Your CTR model loses 20% of user features. What's your strategy?**  
A: (1) First-party data becomes critical (Google logged-in users, search history), (2) Privacy Sandbox Topics API (coarse interest categories), (3) Contextual targeting revival (page content, query intent), (4) On-device learning (federated/private), (5) Cohort-level modeling instead of individual. Expect: 5–15% revenue impact initially, recoverable over 12–18 months with new signal development.

**Q4: An advertiser bids $100/click on "insurance" but your model predicts 0.1% CTR. Should you show this ad?**  
A: AdRank = 0.001 × $100 = $0.10. Compare with competing ads. If another ad has pCTR=5% × $3 bid = $0.15, the $100 bid loses. This is correct — the auction selects for user-relevant ads, not just high bids. But: (1) Check if low pCTR is due to cold start (new ad), (2) Consider exploration: new ads deserve traffic to learn true CTR, (3) Quality score floor: if ad quality is too low, reject regardless of bid.

**Q5: Your training pipeline has a 6-hour delay (data collection → feature computation → model update). A breaking news event causes a spike in query "earthquake insurance." Your model's predictions are stale. How do you handle this?**  
A: (1) Real-time feature layer: query trending score (computed in Flink, <1min delay) as an input feature, (2) Fallback rules: if query is trending + no recent click data → use broader category CTR, (3) Hierarchical smoothing: query-level CTR → keyword-level → category-level → global, (4) Never rely solely on historical query CTR for tail/trending queries.
