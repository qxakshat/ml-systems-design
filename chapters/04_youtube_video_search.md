# YouTube Video Search System

## Problem Formulation

Multi-modal retrieval + ranking: given text query $q$, return ranked list of videos maximizing user satisfaction (measured by engaged watch time, not just clicks). Two-tower retrieval (offline embeddings) + cross-encoder ranking (online scoring).

Objective: Maximize satisfied clicks (watch > 50% of video) while maintaining result diversity.

---

## Data & Transformations

| Aspect | Detail |
|--------|--------|
| Corpus | 800M+ videos, growing 500h/minute |
| Signals per video | Title, description, tags, ASR transcript, visual frames, audio, engagement stats |
| Query logs | 10B+ query-click pairs (90 days), with dwell time |
| Labels | Satisfied click (watch>30s for short, >50% for long), long-click, share |
| Negative sampling | Impressed but not clicked (hard negatives from top-100 BM25) |
| Freshness | New video must be searchable within 5 minutes of upload |

**Video representation** (offline computed):
- **Text**: title + first 200 chars description + ASR (first 5 min) → BERT → 256-d
- **Visual**: 1fps keyframes → ViT-L/14 → temporal mean pooling → 256-d  
- **Audio**: AST (Audio Spectrogram Transformer) → 128-d
- **Fused**: cross-attention Transformer over modalities → 256-d final embedding

---

## Model Architecture

**Retrieval (Two-Tower)**:

$$\text{sim}(q, v) = \cos(E_q(q), E_v(v))$$

Query tower: BERT-base (110M) → projection → 256-d. Video tower: multi-modal fusion → 256-d.

**Training loss** (sampled softmax with temperature):

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(q, v^+) / \tau)}{\sum_{v \in \mathcal{B}} \exp(\text{sim}(q, v) / \tau)}$$

$\tau = 0.07$ (learned), in-batch negatives (batch=4096) + hard negatives from BM25.

**Ranking (Cross-Encoder)**: Query + video features jointly scored.

$$\text{rank\_score} = w_1 \cdot P(\text{click}) + w_2 \cdot E[\text{watch\_time}] + w_3 \cdot P(\text{satisfied})$$

Multi-task with shared Transformer backbone, task-specific heads.

---

## Training & Optimization

| Config | Value |
|--------|-------|
| Retrieval | Adafactor, lr=1e-3, 500K steps, TPU v4 pod (256 chips) |
| Ranking | AdamW, lr=2e-5, daily incremental training |
| Position bias | Inverse propensity weighting (IPW) from randomization experiments |
| Freshness | Time-decay feature: $\text{fresh\_score} = e^{-\lambda \cdot \text{age\_hours}}$ |
| Hard negatives | Top BM25 results with no clicks (ratio 1:7 hard:random) |

**Key insight**: Mixing 30% hard negatives during retrieval training improves Recall@1000 by 8% but hurts easy queries. Solution: curriculum — start with random negatives (20 epochs), then add hard negatives.

---

## Serving & Scale

```
Query → Spell correct + Tokenize (5ms) → 
Parallel: [BM25 inverted index (20ms)] + [Dense ANN via ScaNN (30ms)] →
Merge + dedup (1000 candidates) → Feature fetch (20ms) →
Cross-encoder ranking (50ms, GPU batch) → 
Re-rank (diversity, policy) → Top-20
```

- **Index**: ScaNN with 4-bit PQ, 800M vectors, served across 1000 shards  
- **Freshness**: streaming index for videos <1h old (separate small index, merged into main hourly)  
- **Total latency**: P50=150ms, P99=300ms  
- **Personalization**: user embedding (last 50 watches via lightweight Transformer) concatenated to query

---

## Metrics & Evaluation

| Metric | Measures | Target |
|--------|----------|--------|
| NDCG@5 | Ranking quality | Online A/B |
| Recall@1000 | Retrieval coverage | >90% of relevant |
| Satisfied click rate | User satisfaction | +X% vs control |
| Abandonment rate | Failed searches | <15% |
| Query reformulation | Search friction | Lower = better |
| Long-click rate (>50%) | Content quality | Key North Star |

---

## Alternatives & Tradeoffs

| Approach | Pro | Con |
|----------|-----|-----|
| BM25 only (lexical) | Fast, interpretable, no training | Misses semantic matches |
| Dense only (ANN) | Semantic understanding | Misses exact keyword matches |
| Hybrid (BM25 + Dense) | Best of both, +15% recall | Fusion tuning, 2× index cost |
| Single-stage (no ranker) | Simple, fast | Lower quality (no cross-attention) |
| Multi-modal embedding | Finds visually similar content | High compute (video processing) |
| LLM-based search (Gemini) | Natural language understanding | Latency (>1s), cost, hallucination |

---

## Recent Research (2024–2026)

1. **SPLADE++ (2024)** — learned sparse retrieval matching BM25 efficiency with neural quality  
2. **GTE-Qwen2 (2025)** — 7B-param embedding model with SOTA retrieval  
3. **ColBERT v2 (2024)** — late-interaction for efficient cross-encoder-like quality at bi-encoder cost  
4. **Video-LLaMA2 (2025)** — video understanding via LLM for rich content features  
5. **Matryoshka Embeddings (2024)** — train once, truncate for speed/quality tradeoff  
6. **InternVideo2 (2025)** — 6B-param video foundation model, SOTA on video retrieval

---

## Advanced Trick Questions & Answers

**Q1: Your retrieval model ranks a clickbait video #1 for "learn python". The watch time is 5 seconds. How do you solve this systematically?**  
A: (1) Label = satisfied engagement, not clicks. Train on dwell time > threshold, not CTR. (2) Multi-objective: penalize high-CTR + low-completion signals. (3) Post-ranking quality filter: suppress if historical completion < 20%. (4) Creator-side: demonetize/suppress channels with consistently low completion.

**Q2: A brand-new video has zero engagement signals. How do you decide where to rank it?**  
A: Cold-start strategy: (1) Content-based quality score (production value, thumbnail quality, title clarity), (2) Channel authority (historical performance), (3) Exploration budget: show new videos to small traffic slice, observe real engagement, then promote/demote. (4) Predicted engagement from similar videos (transfer from title/topic embeddings).

**Q3: You notice dense retrieval misses the video "Python Tutorial for Beginners" for query "python tutorial" because the embedding focuses on semantics over keywords. How do you fix this?**  
A: This is the "keyword matching" failure of dense retrieval. Solutions: (1) Hybrid retrieval (BM25 handles exact match), (2) Lexically-aware embeddings (SPLADE, ColBERT), (3) Negation/exact-match aware training: include hard negatives where title contains query but is irrelevant. (4) Query-document term matching as an explicit feature in ranking.

**Q4: How do you handle multi-lingual search (query in Hindi, relevant video in English)?**  
A: (1) Multilingual embedding model (mE5, multilingual-E5-large), (2) Query translation (but adds latency), (3) Cross-lingual contrastive training: align query-video pairs across languages, (4) Language preference as personalization signal — some users prefer dubbed content.

**Q5: The search team ships a model that improves NDCG@5 by +2% but increases serving cost by 50%. Should you ship it?**  
A: Calculate ROI: if +2% NDCG → +X% watch time → +$Y ad revenue. Compare $Y against +50% serving cost ($Z). At YouTube's scale, even 0.1% watch time improvement = $100M+/year. But also consider: (1) model distillation to reduce cost, (2) cascade — use expensive model only for top-100 re-ranking, (3) latency impact on user experience.
