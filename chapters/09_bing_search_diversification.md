# Bing Search Diversification

## Problem Formulation

Given query $q$ with ambiguous/multi-faceted intent, re-rank results to maximize coverage of diverse subtopics while maintaining relevance. Trade off between showing the single-best result vs. covering multiple plausible user intents.

Objective: Maximize α-NDCG — a diversity-aware ranking metric that rewards covering distinct subtopics.

$$\alpha\text{-NDCG@k} = \frac{\sum_{i=1}^k \frac{\text{gain}_i}{log_2(i+1)}}{\text{ideal}}$$

where gain is discounted if the subtopic was already covered by higher-ranked results.

---

## Data & Transformations

| Aspect | Detail |
|--------|--------|
| Queries | 10B+ searches/day |
| Ambiguity | ~30% of queries have multiple valid intents |
| Subtopic taxonomy | Query → intents → sub-intents (mined from click logs, QA datasets) |
| Labels | SERP annotations (relevance + subtopic coverage) via crowd-sourcing |
| Intent distribution | P(intent_i | query) estimated from click distribution over result types |
| Features | Query: entity type, category, frequency, reformulation patterns. Doc: topic, entity, freshness |

**Query classification by diversity need**:
- **Navigational**: single intent (e.g., "facebook login") → no diversification needed  
- **Informational ambiguous**: multiple intents (e.g., "jaguar" → car, animal, Mac OS)  
- **Faceted**: single entity, multiple aspects (e.g., "iPhone 15" → price, specs, reviews, buy)

---

## Model Architecture

**Phase 1 — Intent Mining** (offline):

- Click-through inversions: for query $q$, cluster clicked documents into subtopics using doc embeddings + LDA
- Query reformulation graph: "jaguar car" and "jaguar animal" are subtopics of "jaguar"
- P(intent_i | q) = frequency-weighted from click logs + smoothed

**Phase 2 — Relevance Ranking** (neural):

Standard transformer-based ranker scores: $s(q, d)$ = cross-encoder relevance score.

**Phase 3 — Diversification** (MMR / xQuAD / PM-2):

**xQuAD** (eXplicit Query Aspect Diversification):
$$d^* = \arg\max_{d \in R} \left[(1-\lambda) P(d|q) + \lambda \sum_{i} P(q_i|q) \cdot P(d|q_i) \prod_{d_j \in S}(1 - P(d_j|q_i))\right]$$

Where:
- $P(d|q)$ = base relevance
- $P(q_i|q)$ = intent probability  
- $P(d|q_i)$ = doc relevance to specific intent
- Product term = novelty (discount if intent already covered)
- $\lambda$ = diversity-relevance tradeoff (0.3–0.7 depending on query type)

**PM-2** (Proportionality-based):
Allocate slots proportionally to intent probabilities, fill greedily by marginal gain.

**MMR** (Maximal Marginal Relevance):
$$\text{MMR} = \arg\max_{d} [\lambda \cdot \text{Sim}(d, q) - (1-\lambda) \cdot \max_{d_j \in S} \text{Sim}(d, d_j)]$$

---

## Training & Optimization

| Config | Value |
|--------|-------|
| Intent classifier | DistilBERT fine-tuned on query→intent, 8 macro-categories |
| Relevance model | Cross-encoder (DeBERTa-v3), trained with listwise loss |
| Intent probability | Softmax over click-cluster proportions, Laplace smoothed |
| λ tuning | Grid search per query-type bucket, optimized on α-NDCG |
| Subtopic extraction | LLM-based (GPT-4 class) for tail queries, cached for head |
| Evaluation | Weekly SERP audits (5K queries, 3 judges per query) |

**Key insight**: λ should be query-dependent. Navigational queries → λ≈0 (pure relevance). Ambiguous queries → λ≈0.7 (strong diversification).

---

## Serving & Scale

```
Query → Intent classifier (5ms) → 
Parallel: Relevance ranking (existing pipeline, returns top-200) +
          Intent probability estimation (2ms) +
          Document-intent scoring (batch, 10ms) →
Diversification re-ranking (xQuAD greedy, top-200→top-10) [3ms] →
SERP blending (web + images + news + videos) →
Result page rendered

Total diversification overhead: <20ms on top of base ranking
```

- **Caching**: Head queries (top-100K) have pre-computed intent distributions  
- **Fallback**: If intent model fails → fall back to MMR (doc-doc similarity only, no intent needed)  
- **Vertical blending**: Diversification also applies across verticals (web, image, video, news, shopping)

---

## Metrics & Evaluation

| Metric | Purpose | Target |
|--------|---------|--------|
| α-NDCG@10 | Diversity + relevance combined | Primary offline metric |
| Intent recall@5 | Fraction of intents covered in top-5 | >0.85 for ambiguous queries |
| Subtopic recall@10 | All identified subtopics covered | >0.90 |
| ERR-IA | Intent-aware expected reciprocal rank | Higher is better |
| NDCG@10 (standard) | Relevance guardrail | Must not drop >1% |
| Abandonment rate | User satisfaction | Online A/B, must not increase |

---

## Alternatives & Tradeoffs

| Approach | Pro | Con |
|----------|-----|-----|
| MMR | Simple, no intent taxonomy needed | Only doc-doc dissimilarity, no intent model |
| xQuAD | Explicit intent coverage, tunable | Requires intent taxonomy mining |
| PM-2 | Proportional fairness | Ignores doc-doc redundancy |
| DPP (Determinantal Point Process) | Elegant submodularity | Cubic complexity O(n³), slow |
| RL-based (bandit) | Learns from user feedback | Cold start, exploration cost |
| LLM re-ranking | Can reason about coverage | Latency prohibitive at scale |

---

## Recent Research (2024–2026)

1. **GDR** (WSDM 2024) — Graph-based Diversified Retrieval with GNN for document relationships  
2. **DIVA** (SIGIR 2024) — Diversification via LLM-based intent extraction (zero-shot subtopics)  
3. **MixDiversity** (Bing, 2024) — mixture of diversification strategies conditioned on query type  
4. **PersonalDiv** (Google, 2025) — user-personalized λ (users who explore want more diversity)  
5. **Fairness-aware search** (2025) — diversity across source providers, not just topical  
6. **RAG Diversity** (Microsoft, 2025) — applying diversification to retrieval for LLM context

---

## Advanced Trick Questions & Answers

**Q1: User searches "Apple." Your system shows results for Apple Inc (80% intent), apple fruit (15%), and Apple Records (5%). The user clicks apple fruit. Next time they search "Apple," should you remember?**  
A: Personalization vs. diversification tension. (1) Short-term: yes, boost apple-fruit in this session (session context). (2) Long-term: don't over-personalize — next week they may want Apple Inc. (3) Strategy: keep diversified slate but re-weight intent probabilities: P(fruit|"apple", this_user) = 0.4 (boosted from 0.15). Still show Apple Inc in position 1 (majority intent) but promote fruit to position 2. (4) Never fully collapse diversity — user intents change.

**Q2: Your α-NDCG improved by 5% but online A/B shows no click-through improvement. Why?**  
A: (1) Users don't click diverse results — they satisfy with position 1 if it's relevant (position bias dominates). α-NDCG measures ideal coverage, but users have one intent per session. (2) Diversity helps the *wrong-intent* users — they're the minority (20%), so average CTR barely moves. (3) Better online metric: "session success rate" — did the user find what they needed without reformulation? Track: (a) reformulation rate (should decrease), (b) return-to-SERP rate (should decrease), (c) dwell time on clicked result (should increase).

**Q3: You have 10 blue links on page 1. Position 1 costs you nothing extra. But diversifying means putting a less-relevant (to majority intent) result in position 3. Quantify the tradeoff.**  
A: Expected utility model: Position 3 under pure-relevance: P(click) = P(exam|pos3) × P(click|exam, relevant) = 0.7 × 0.3 = 0.21. Under diversification: different-intent doc → P(click|exam, relevant_to_minority) = 0.7 × P(minority_intent) × P(click|relevant) = 0.7 × 0.2 × 0.4 = 0.056. Net loss for majority users = 0.21 - 0.056 = 0.154. But for minority users (20% of traffic), going from 0 coverage to coverage at position 3 → their session success rate jumps from ~30% to ~70%. Acceptable if: 0.2 × Δ(success_minority) > 0.8 × Δ(loss_majority).

**Q4: News breaks: "Jaguar announces new electric SUV." Your intent distribution for "jaguar" was 70% car, 25% animal, 5% other. How quickly should the system adapt?**  
A: (1) Real-time intent shift detection: monitor click distribution on "jaguar" SERP — if car-clicks spike from 70%→90% in 1 hour, dynamically adjust P(car|"jaguar"). (2) Freshness boost: inject trending news articles into SERP regardless of static diversification. (3) Temporal decay: recent click signals weighted exponentially more. (4) Separate "trending" overlay: breaking news gets dedicated SERP module (above organic results), not competing with diversification logic.

**Q5: Your diversity system works great for English queries. You're launching in Japan. What breaks?**  
A: (1) Intent taxonomy is language/culture-dependent: "Apple" in Japan → Apple Inc dominates even more (less apple-fruit intent). (2) Subtopic mining from click logs requires sufficient data — tail queries in new market have sparse clicks. (3) Word segmentation issues: Japanese has no spaces → ambiguity is at character level. (4) Vertical preferences differ: Japanese users expect more image/video results. (5) Solution: start with LLM-based intent extraction (zero-shot, multilingual) → bootstrap intent taxonomy → refine with local click data over 3–6 months.
