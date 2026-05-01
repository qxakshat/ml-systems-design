# Social Media Friend Recommendations — Graph-Based

## Problem Formulation

Link prediction on a heterogeneous social graph: given user $u$, rank candidate users by $P(\text{friendship}_{u,v} | \mathcal{G})$. Two-stage: (1) candidate generation from graph structure, (2) ranking with learned features. Optimize for accepted friend requests, not just sends.

Objective: Maximize connections formed per recommendation impression.

---

## Data & Transformations

| Aspect | Detail |
|--------|--------|
| Graph | 3B+ nodes, 400B+ edges (Meta scale) |
| Node types | User, Group, Page, Event, Post |
| Edge types | friendship, follows, likes, comments, co-membership, co-attendance |
| Features | Profile (age, location), activity level, interest embeddings, accept history |
| Label | Friendship formed within 7 days of recommendation (positive) |
| Negative | Impressed but not actioned in 30 days |
| Temporal split | Train on t<T, validate on T<t<T+7d, test on t>T+7d |

**Key**: Mutual friends count is the single most predictive feature (~60% of signal). Everything else is incremental.

---

## Model Architecture

**GNN (Embedding Generation)**: PinSage / GraphSAGE with neighbor sampling.

$$h_v^{(l)} = \sigma\left(W^{(l)} \cdot \text{AGG}\left(\{h_u^{(l-1)} : u \in \mathcal{N}(v)\}\right)\right)$$

Sampling: [25, 10] neighbors per layer (2-hop). AGG = mean or attention-weighted.

**Link Prediction Score**:

$$\text{score}(u, v) = \text{MLP}\left([h_u \| h_v \| h_u \odot h_v \| |h_u - h_v| \| f_{uv}]\right)$$

Where $f_{uv}$ = handcrafted edge features (mutual friends, Jaccard, Adamic-Adar).

**Ranking Model**: DCN-v2 (Deep & Cross Network) with GNN embeddings + 50 engineered features.

**Adamic-Adar Index** (strong baseline):

$$AA(u,v) = \sum_{w \in \mathcal{N}(u) \cap \mathcal{N}(v)} \frac{1}{\log |\mathcal{N}(w)|}$$

---

## Training & Optimization

| Config | Value |
|--------|-------|
| GNN Optimizer | Adam (lr=1e-3), 100 epochs |
| Mini-batch | 1024 target edges, NeighborLoader |
| Negatives | 5:1 ratio (random + hard: 2-hop non-friends) |
| Loss | Binary CE with label smoothing=0.05 |
| Graph partition | METIS into 1000 shards (cross-partition edges via RPC) |
| Ranker | AdamW (lr=5e-4), batch=8192, daily retrain |

**Scaling trick**: Pre-compute GNN embeddings offline (nightly), use as static features in online ranker. Avoids GNN inference at serving time.

---

## Serving & Scale

```
User request → FoF candidates (precomputed, Redis) → 
Feature assembly (real-time + cached) → DCN-v2 scoring (10ms) → 
Re-ranking (diversity, freshness) → Top-20
```

- **Candidate generation** (offline, hourly): FoF (2-hop), contact matching, co-location, community detection  
- **Candidates per user**: ~1000 (stored in Redis)  
- **Serving latency**: <100ms P99  
- **Real-time triggers**: new friendship → Kafka event → update FoF candidates for affected users

---

## Metrics & Evaluation

| Metric | What it measures |
|--------|-----------------|
| Send rate | User engagement with recommendations |
| Accept rate | Precision of recommendations |
| Recall@20 | Coverage of actual future friends |
| New connections/week | Business growth metric |
| Time-to-10-friends | Cold-start effectiveness |
| Long-term retention | Do connections drive DAU? |

---

## Alternatives & Tradeoffs

| Approach | Pro | Con |
|----------|-----|-----|
| FoF + heuristics (Adamic-Adar) | Simple, fast, 80% of value | No personalization |
| Node2Vec (random walks) | Captures structural equivalence | Not inductive (new users fail) |
| GraphSAGE | Inductive, scalable | Neighbor explosion at scale |
| PinSage (importance sampling) | Proven at Pinterest (3B nodes) | Complex engineering |
| Full GNN at serving | Freshest embeddings | Latency prohibitive (>1s) |
| Pre-computed embeddings + ranker | Fast serving | Staleness (1hr–1day) |

---

## Recent Research (2024–2026)

1. **GraphSAGE v2** (Meta, 2024) — improved sampling with importance weights  
2. **SEAL** (link prediction via subgraph) — extracts enclosing subgraph, applies GNN  
3. **LinkFormer** (2024) — Transformer-based link prediction (no message passing)  
4. **UltraGNN** (Meta, 2025) — trillion-edge GNN training on single cluster  
5. **GraphMAE2** (2024) — self-supervised pre-training for graph transformers  
6. **TGL** (Temporal Graph Learning, 2025) — real-time graph updates with continuous-time modeling

---

## Advanced Trick Questions & Answers

**Q1: You observe that recommendations are creating echo chambers — users only connect within their existing community. How do you address this?**  
A: (1) Granovetter's weak ties theory — explicitly boost cross-community connections (bridging score), (2) Diversity constraint in re-ranking (MMR with community membership as redundancy signal), (3) Track "community diversity index" as guardrail metric, (4) Exploration budget: 10% of recommendations from different clusters.

**Q2: A new user with zero connections joins. Your GNN has no neighbors to aggregate. What do you do?**  
A: Cold-start cascade: (1) Contact list upload → instant FoF from phone contacts, (2) Profile similarity (school, employer, location) → content-based fallback, (3) Popularity-based (people-you-may-know from same city), (4) After first 3 connections → GNN becomes effective. Key: reduce time-to-first-friend to <24h.

**Q3: Your accept rate is 15% but the PM wants 30%. What's your response?**  
A: Accept rate and volume are inversely correlated (precision-recall tradeoff). Showing only highest-confidence recs (mutual friends ≥ 5) gives 40% accept rate but 90% fewer impressions. Proper metric: total accepted connections/week (rate × volume). Propose composite metric: connections formed per DAU.

**Q4: You notice that removing the GNN and using only mutual friend count + logistic regression gets 95% of the GNN performance. Should you keep the GNN?**  
A: Depends on marginal gain vs. engineering cost. GNN helps most for (1) users with few mutual friends (cold-ish start), (2) cross-community connections, (3) emerging social structures. If +5% lift on these segments → keep. But simplify: use GNN embeddings only as additional features to the simple model, not as the primary system.

**Q5: Privacy regulations require you to stop using contact lists. How much does this hurt, and what's your mitigation?**  
A: Contact matching is ~30% of total connections formed (especially cold start). Mitigation: (1) Strengthen co-location signals (same WiFi, Bluetooth proximity — with consent), (2) Interest-based matching from group/page activity, (3) Incentivize profile completion (school, employer), (4) Accept short-term metric drop while alternative signals ramp.
