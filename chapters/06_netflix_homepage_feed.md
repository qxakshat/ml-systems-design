# Netflix Home Page Content Feed

## Problem Formulation

Full-page optimization: select which rows to show (row selection), which items per row (within-row ranking), row ordering, and which artwork variant. Maximize long-term member retention via streaming hours and content satisfaction.

Objective: Maximize $E[\text{quality\_play}]$ = P(play) × P(complete | play) × P(enjoy | complete).

---

## Data & Transformations

| Aspect | Detail |
|--------|--------|
| Users | 260M+ subscribers, 500M+ profiles |
| Catalog | ~15K titles per region |
| Interactions | 100B+ implicit events/day (play, browse, scroll, hover, pause, rewind) |
| Features | User: taste embeddings, watch history, time patterns. Item: genre tags, cast, popularity curve |
| Label | Quality play: completion > 70% AND not followed by immediate churn |
| Cold start | New member: 5 onboarding selections → bootstrap taste profile |

**Implicit feedback** signals ranked by value:
1. Completed viewing (strongest positive)
2. Watched > 50% (positive)
3. Started but quit < 10% (weak negative)
4. Scrolled past (implicit negative)
5. Actively removed from "continue watching" (strong negative)

---

## Model Architecture

**Stage 1 — Candidate Generation** (offline, hourly):

- **Collaborative Filtering**: Matrix factorization via ALS on implicit feedback matrix $R_{u,i}$
  - $\min_{U,V} \sum_{(u,i) \in \mathcal{S}} c_{ui}(r_{ui} - u_i^T v_i)^2 + \lambda(\|U\|^2 + \|V\|^2)$
  - Confidence $c_{ui} = 1 + \alpha \cdot \text{watch\_time}_{ui}$
- **Sequential**: SASRec (Self-Attentive Sequential Rec) — Transformer on watch history
- **Content-based**: Item-item similarity via BERT embeddings of descriptions

**Stage 2 — Ranking** (online):

DCN-v2 with multi-objective heads:

$$\text{final\_score} = 0.4 \cdot P(\text{play}) + 0.4 \cdot P(\text{complete}|\text{play}) + 0.2 \cdot P(\text{enjoy})$$

**Stage 3 — Page Assembly**:

Row selection via contextual bandit (Thompson Sampling):

$$\theta_r \sim \text{Beta}(\alpha_r, \beta_r), \quad \text{select rows with highest } \theta_r$$

---

## Training & Optimization

| Config | Value |
|--------|-------|
| CF (ALS) | Spark, 200M users × 15K items, rank=128, λ=0.01, 15 iterations |
| Sequential | 12-layer Transformer, d=256, 200 item history, causal mask |
| Ranker | PyTorch, AdamW (lr=1e-3), batch=8192, A100 GPUs |
| Refresh | CF: weekly. Sequential: daily. Ranker: daily (incremental) |
| Offline eval | Replay evaluation with IPS correction |
| Online eval | Interleaving (team-draft) → A/B test → long-term holdout |

**Critical pipeline**: Offline metrics (Recall@K, NDCG) are weakly correlated with online metrics (streaming hours). Always validate offline gains with online A/B. Netflix reports ~30% of offline wins translate to online wins.

---

## Serving & Scale

```
Request → EVCache (user features) → Candidate retrieval (pre-computed) →
Feature assembly → Ranker inference (GPU) → Row optimizer → 
Artwork selection (bandit) → Response JSON
```

- **Pre-computation**: User×item scores computed hourly for top-2000 candidates  
- **Real-time signals**: Last session activity, time-of-day, device  
- **Artwork personalization**: Per (user, title) pair, select from 5–20 artwork variants via Thompson Sampling. Reported +20% engagement.  
- **Latency**: 250ms total budget (100ms for ranking, 50ms for page assembly)

---

## Metrics & Evaluation

| Metric | Level | Purpose |
|--------|-------|---------|
| Take rate | Short-term | Immediate engagement |
| Streaming hours | Medium-term | Core business KPI |
| Quality plays (>70% completion) | Medium-term | Satisfaction proxy |
| Monthly retention | Long-term | Business health |
| Effective catalog size | Ecosystem | Content diversity/ROI |
| Time-to-first-play | UX | Session start quality |

---

## Alternatives & Tradeoffs

| Approach | Pro | Con |
|----------|-----|-----|
| Collaborative Filtering (ALS) | Simple, proven, fast | Cold start, popularity bias |
| Sequential (SASRec/BERT4Rec) | Captures temporal patterns | Needs sufficient history |
| LLM-based rec (P5, InstructRec) | Natural language reasoning | Latency, cost, limited catalog |
| Bandit exploration | Handles cold items | Lower short-term performance |
| Full page optimization (RL) | Globally optimal layout | Reward attribution difficulty |
| Simple popularity-based | No personalization cost | 50% worse engagement |

---

## Recent Research (2024–2026)

1. **HSTU** (Meta, 2024) — Hierarchical Sequential Transduction for trillion-scale recommendations  
2. **Actions Speak Louder Than Words** (Meta, 2024) — generative retrieval for recommendations  
3. **RecFormer** (2024) — language model as recommender (text-based item representation)  
4. **KuaiRand** (2024) — unbiased recommendation dataset with randomized exposure  
5. **DreamRec** (2025) — diffusion model for sequential recommendation  
6. **Netflix's Artwork Personalization** (RecSys 2025) — contextual bandit at scale

---

## Advanced Trick Questions & Answers

**Q1: Your recommendation system achieves great engagement metrics but Netflix's content team complains that 80% of their catalog never gets recommended. How do you fix this?**  
A: This is the popularity bias / long-tail problem. Solutions: (1) Exploration budget: 5% of recommendations from underexposed content (ε-greedy or Thompson Sampling), (2) Inverse propensity weighting in training to upweight rare items, (3) "Effective catalog size" as a guardrail metric, (4) Causal reasoning: some content is genuinely unwanted. Report: "85% of catalog was recommended to at least N users" vs "all catalog gets equal exposure."

**Q2: A user watches 3 horror movies in one evening. Next morning, their homepage is all horror. Is this correct behavior?**  
A: No — this is "session bias" overfitting. Solutions: (1) Separate session preferences from long-term taste (dual-encoder: session context + user profile), (2) Time-of-day context (horror at night ≠ morning preference), (3) Diversity constraint in page assembly (genre entropy), (4) "Taste diversification" — hedge across known user interests, don't collapse to last session.

**Q3: You run an A/B test. Treatment improves streaming hours by +0.5% but reduces monthly retention by -0.1%. What do you do?**  
A: This is the exploitation trap — maximizing short-term engagement (autoplay, cliffhangers) can hurt long-term satisfaction. Netflix has observed this. Decision: (1) Retention is the North Star — don't ship, (2) Investigate mechanism: are we recommending addictive but low-quality content? (3) Run longer holdout (3 months) to confirm retention signal, (4) Build composite metric: $\text{score} = \text{hours} - \lambda \cdot \text{churn\_risk}$.

**Q4: Two team members debate: "We should use implicit feedback only" vs. "We need explicit ratings." Who's right?**  
A: Both have merit. Implicit (watches, skips) is abundant but noisy (background playing ≠ enjoyment). Explicit (thumbs up/down) is sparse but high-signal. Netflix moved from 5-star ratings to thumbs because: (1) higher response rate, (2) less social desirability bias, (3) easier to calibrate. Best: use implicit for engagement prediction, explicit for satisfaction/quality prediction. Separate models for each.

**Q5: Your model recommends "Emily in Paris" to a user who rated it 1 star last year. Bug or feature?**  
A: Bug — negative signals must be hard constraints, not soft signals. But nuance: (1) If it's a new season, the user might reconsider → show with "New Season" badge, (2) Check if the profile is shared (different household member), (3) Never re-recommend explicitly disliked content unless there's a strong contextual signal. Implement negative feedback as a hard filter in post-ranking, not a feature in the model.
