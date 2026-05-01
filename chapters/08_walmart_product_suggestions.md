# Walmart Product Suggestions — Retrieval + Ranking

## Problem Formulation

Multi-stage ranking system for e-commerce: given query (or context), return products maximizing purchase probability while satisfying business constraints (inventory, margins, sponsored slots). Omnichannel: same user may search on web, app, or in-store kiosk.

Objective: Maximize GMV (Gross Merchandise Value) per search session = Σ P(purchase_i) × price_i.

---

## Data & Transformations

| Aspect | Detail |
|--------|--------|
| Catalog | 500M+ products (1P + 3P marketplace) |
| Queries | 1B+ searches/day |
| Labels | Click, add-to-cart, purchase (within 7-day attribution window) |
| Cold items | 50K new products/day (no behavioral signals) |
| Features | Product: title, image, price, rating, attributes. User: purchase history, browse patterns |
| Omnichannel | Unified user profile across web + app + store |
| Position bias | Significant — position 1 gets 10× more clicks than position 10 |

**Feature engineering highlights**:
- **Query-product match**: BM25 score, semantic similarity, attribute match (e.g., "red dress" → color=red)  
- **User-product affinity**: Collaborative filtering score, brand affinity, price range match  
- **Contextual**: Time of day, device, location (nearest store availability), season  
- **Behavioral velocity**: CTR/ATC/CVR with temporal decay (recent more important)

---

## Model Architecture

**Stage 1 — Retrieval** (Two-Tower):

$$\text{score}(q, p) = E_q(q)^T \cdot E_p(p)$$

Query tower: BERT-base (title understanding) + user embedding → 256-d.  
Product tower: title + image (CLIP) + attributes → 256-d.

Trained with in-batch negatives + hard negatives (top-BM25 non-purchased).

**Stage 2 — Ranking** (Multi-Task MMoE):

$$\text{score} = w_1 P(\text{click}) + w_2 P(\text{ATC}) + w_3 P(\text{purchase}) + w_4 E[\text{revenue}]$$

Architecture: shared embedding layer → 6 expert networks → task-specific gating → task towers.

**ESMM** (Entire Space Multi-Task Model) for CVR:

$$P(\text{purchase}) = P(\text{click}) \times P(\text{purchase}|\text{click})$$

Avoids sample selection bias: CVR model trained on all impressions, not just clicks.

**Stage 3 — Re-ranking** (business logic + diversity):
- Category dedup (max 3 products per brand in top-10)
- Sponsored product injection (position 1, 4, 8)
- Inventory filter (real-time stock check)
- Margin-weighted boosting (private label +10% score boost)

---

## Training & Optimization

| Config | Value |
|--------|-------|
| Retrieval | Adam (lr=5e-5), batch=4096, contrastive loss (τ=0.05) |
| Ranking | AdamW (lr=1e-3), batch=8192, BF16 |
| Labels | Click (1d window), ATC (3d), Purchase (7d) |
| Position bias | IPW: weight = 1/P(examine|position), learned from randomized data |
| Refresh | Retrieval: weekly. Ranking: daily. Embeddings: nightly |
| Negative sampling | Impressed-not-clicked (hard), random catalog (easy), ratio 7:3 |

**Debiasing trick**: Run 1% random traffic (show random products regardless of model score) to collect unbiased data for position bias estimation and model evaluation.

---

## Serving & Scale

```
Query → Understanding (tokenize, spell-correct, entity extract) [5ms] →
Parallel retrieval: BM25 (Vespa) + ANN (HNSW) + Behavioral (Redis) [50ms] →
Merge & dedup → 1000 candidates →
Feature assembly (Redis + Bigtable) [15ms] →
Ranking model (Triton, GPU batched) [20ms] →
Re-ranking + business rules [5ms] →
Top-48 products returned

Total: P50=80ms, P99=150ms
```

- **ANN index**: HNSW in Vespa (ef_search=200, 500M vectors, 128-d)  
- **Real-time inventory**: Redis pub-sub from warehouse management system  
- **A/B testing**: per-query randomization with sticky user assignment  
- **Geo-routing**: query routed to nearest datacenter for local inventory awareness

---

## Metrics & Evaluation

| Metric | Stage | Target |
|--------|-------|--------|
| Recall@1000 | Retrieval | >95% of purchasable items |
| NDCG@10 | Ranking | +X% vs. control (relative) |
| CTR | Short-term engagement | Monitor, not optimize |
| ATC rate | Mid-funnel | Key leading indicator |
| Conversion rate | Purchase | Primary online metric |
| Revenue per search (RPS) | Business | North Star |
| Null result rate | UX | <2% |

---

## Alternatives & Tradeoffs

| Approach | Pro | Con |
|----------|-----|-----|
| BM25 + simple ranker | Fast, interpretable, reliable | Misses semantic intent |
| Dense retrieval only | Semantic understanding | Fails on exact SKU searches |
| LLM-based query rewriting | Better understanding | Latency, cost |
| Multi-tower (separate query types) | Specialized models per intent | Complexity |
| Learning-to-rank (LambdaMART) | Strong tabular baseline | Can't use embeddings directly |
| End-to-end differentiable retrieval | Theoretically optimal | Doesn't scale to 500M items |

---

## Recent Research (2024–2026)

1. **HSTU** (Meta, 2024) — sequential rec Transformer handling trillion-scale  
2. **TIGER** (Google, 2024) — generative retrieval with semantic IDs for recommendations  
3. **EBR at Walmart** (2024) — embedding-based retrieval at scale (industry report)  
4. **DSI++** (Google, 2025) — differentiable search index with incremental updates  
5. **RecSys-LLM** (Amazon, 2025) — LLM as ranker for product search  
6. **Vespa bm25+ANN fusion** (2024) — hybrid retrieval at production scale

---

## Advanced Trick Questions & Answers

**Q1: Your search system returns great results for "iPhone 15 case" but terrible results for "something to keep my phone safe." How do you bridge this gap?**  
A: This is the vocabulary gap between user intent and product catalog. Solutions: (1) Dense retrieval (semantic embedding matches intent to product descriptions), (2) Query rewriting via LLM (expand "keep phone safe" → "phone case, screen protector"), (3) Behavioral signals: users who searched "keep phone safe" eventually bought phone cases → learn this mapping, (4) Entity resolution: map user intent → product category → expand to category-level retrieval.

**Q2: Sponsored products team wants their ads in position 1, 4, and 8. Your ranking model puts the best organic result in position 1. How do you measure the impact of ad insertion on organic metrics?**  
A: (1) Run A/B: control (no ads in top-3) vs. treatment (ads in position 1). Measure: organic CTR shift, session conversion rate, return visit rate, (2) Causal analysis: if ad displaces a highly relevant organic result → user friction increases, (3) Auction-based insertion: sponsored product must meet minimum relevance threshold (don't show irrelevant ads in top positions), (4) Track "organic NDCG excluding ad slots" as guardrail.

**Q3: A product has 4.8 stars and 10,000 reviews vs. another with 4.9 stars and 5 reviews. Which should rank higher?**  
A: Wilson score interval lower bound (not raw average):

$$\text{score} = \frac{\hat{p} + \frac{z^2}{2n} - z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

With $n$=10000 and $\hat{p}$=0.96: lower bound ≈ 0.955. With $n$=5 and $\hat{p}$=0.98: lower bound ≈ 0.70. The high-volume product wins. Alternatively: Bayesian (Beta prior) or simply require min 30 reviews before rating affects ranking.

**Q4: Your model is optimized for conversion (purchases), but the PM says customers are adding cheap items to cart and not buying expensive ones. Revenue per search is flat. What's wrong?**  
A: Optimizing P(purchase) equally for all items favors cheap, impulse-buy products (consumables, accessories) which have inherently higher conversion rates. Fix: (1) Optimize E[revenue] = P(purchase) × price, (2) Stratified calibration per price tier (separate models or price-aware features), (3) Multi-objective: include E[margin] as a task head, (4) Don't just maximize conversion — maximize *basket value* per session.

**Q5: Black Friday traffic is 10× normal. Your retrieval latency spikes from 50ms to 500ms. What's your degradation strategy?**  
A: Graceful degradation plan: (1) Reduce ANN search ef from 200→50 (faster, lower recall but acceptable), (2) Cache hot queries (top-1000 queries cover 30% of traffic — pre-compute results), (3) Kill behavioral retrieval path (Redis under load) — fall back to BM25+ANN only, (4) Reduce candidate set from 1000→200 (ranker scores fewer items), (5) Pre-warm caches in the week before Black Friday with anticipated query patterns.
