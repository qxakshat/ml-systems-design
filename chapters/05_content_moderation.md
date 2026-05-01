# Content Moderation — Harmful Content Detection

## Problem Formulation

Multi-label, multi-modal classification with cascaded architecture. Given content (text, image, video), predict violation probability across 8+ policy categories. Cascaded: fast filters handle 90% of traffic; heavy models score remaining 10%.

Objective: Maximize proactive detection rate (before user reports) while maintaining false positive rate < 1%.

---

## Data & Transformations

| Aspect | Detail |
|--------|--------|
| Scale | 500M+ posts/day, ~0.5% violation rate |
| Labels | Human review (5 annotators, majority vote), inter-rater κ > 0.7 |
| Categories | Hate speech, violence, nudity, CSAM, terrorism, self-harm, misinformation, spam |
| Hierarchy | Binary (safe/unsafe) → category → sub-category → severity (1–5) |
| Multi-modal | 40% text-only, 30% image, 20% video, 10% mixed (memes) |
| Challenge | Adversarial obfuscation: l33tsp34k, Unicode tricks, image text overlay |

**Data pipeline**: Active learning loop — model-uncertain samples routed to human review → labels fed back into training weekly.

---

## Model Architecture

**Cascaded system** (efficiency by design):

| Layer | Model | Coverage | Latency | Catches |
|-------|-------|----------|---------|---------|
| L0 | Hash matching (PDQ/PhotoDNA) | 100% | <1ms | Known bad (CSAM) |
| L1 | Regex + keyword + blocklist | 100% | <2ms | Obvious violations |
| L2 | Distilled BERT-small (66M) | 100% | 10ms | 70% of violations |
| L3 | XLM-RoBERTa-XL (3.5B) + ViT-G | 10% (uncertain) | 200ms | 25% more |
| L4 | Human review | 1% (edge cases) | hours | Final 5% |

**Text model** (L3): XLM-RoBERTa-XL with adapter layers per policy region.

**Multi-modal fusion** (hateful memes):

$$\text{score} = \text{MLP}([t_{cls}; v_{cls}; t_{cls} \odot v_{cls}])$$

Where $t_{cls}$ = text CLS token, $v_{cls}$ = image CLS token from cross-attention fusion (VisualBERT/FLAVA).

**Key**: Hateful memes require joint understanding — image and text individually benign but harmful together. Cross-modal attention is critical.

---

## Training & Optimization

| Config | Value |
|--------|-------|
| Optimizer | AdamW (lr=2e-5, wd=0.01) |
| Batch | 2048 across 128 GPUs |
| Loss | Focal (γ=2) with class-frequency inverse weighting |
| Multi-task | Uncertainty-weighted loss across 8 categories |
| Refresh | Weekly full retrain, daily incremental (new patterns) |
| Adversarial | 10% training batch = adversarial augmentations |

**Continual learning**: Elastic Weight Consolidation (EWC) prevents catastrophic forgetting when adding new policy categories. Replay buffer: 10% historical examples mixed into every batch.

---

## Serving & Scale

```
Content posted → Async queue (Kafka) → L0 hash check → L1 regex →
L2 lightweight model → [if uncertain] → L3 heavy model → Decision

Decision: {auto-remove, reduce-distribution, human-review, approve}
```

- **Pre-publish blocking**: Only for highest-confidence violations (score > 0.98, precision > 99.5%)  
- **Async path**: Most content scored within 30s of posting  
- **Priority**: Viral content (reshares > threshold) gets expedited scoring  
- **Scale**: L2 processes 500M items/day, L3 processes 50M items/day

---

## Metrics & Evaluation

| Metric | Target | Notes |
|--------|--------|-------|
| Proactive rate | >95% | Caught before any user reports |
| Prevalence | <0.05% | Violating views / total views |
| Precision (auto-remove) | >99.5% | Free speech requirement |
| Recall (overall) | >95% | Safety requirement |
| Time-to-action | <60s | For high-severity (CSAM, terrorism) |
| Appeal overturn rate | <5% | Model error proxy |

---

## Alternatives & Tradeoffs

| Approach | Pro | Con |
|----------|-----|-----|
| LLM-based (GPT-4 judge) | Flexible, few-shot adaptation | Cost ($), latency, hallucination |
| Cascade (current) | Cost-efficient, fast for clear cases | Complex engineering |
| Single large model (all traffic) | Simpler architecture | 100× compute cost at scale |
| Community-driven (user reports) | No ML needed for detection | Reactive (harm already done) |
| CLIP zero-shot | Instant policy updates (text prompt) | Lower accuracy than fine-tuned |
| Federated moderation | Cultural context per region | Inconsistent enforcement |

---

## Recent Research (2024–2026)

1. **LLM-as-Judge for moderation** (2024) — GPT-4 for policy-adherence scoring with chain-of-thought  
2. **Llama Guard 3** (Meta, 2025) — open-source safety classifier (8B params, multi-lingual)  
3. **WildGuard** (Allen AI, 2024) — unified safety moderation with refusal detection  
4. **Hateful Memes Challenge** (Meta) — multi-modal hate detection benchmark  
5. **VIGIL** (2025) — video grounded integrity and labeling (temporal policy violation detection)  
6. **Constitutional AI for moderation** (Anthropic, 2025) — self-supervised safety via principles

---

## Advanced Trick Questions & Answers

**Q1: Your model catches hate speech in English at 97% recall but only 60% in Arabic. The PM says "ship it globally." What do you do?**  
A: Do NOT ship with known disparity. Options: (1) Cross-lingual transfer (XLM-R is multilingual, but fine-tuning data is English-heavy), (2) Translate-then-classify (adds latency + errors), (3) Language-specific adapters with active learning in Arabic, (4) Hire Arabic-speaking annotators (data bottleneck, not model). Set per-language quality gates before launch.

**Q2: Adversaries are using Unicode homoglyphs (Cyrillic "а" instead of Latin "a") to bypass your text classifier. How do you handle this at scale?**  
A: (1) Unicode normalization (NFKC) + confusable mapping (icu4c) in preprocessing, (2) Character-level CNN that's robust to substitutions, (3) Adversarial training with augmented homoglyph variants, (4) Tokenizer-agnostic: byte-level BPE (like mGPT) handles mixed scripts naturally. Most important: keep a living adversarial playbook updated weekly.

**Q3: A comedian posts satire about violence. Your model flags it. How do you systematically reduce false positives on satire/news/education?**  
A: (1) Context features: account type (verified comedian, news org), historical content patterns, audience reaction (laughs vs. reports), (2) Intent classifier as auxiliary head, (3) Genre detection (satire, news, educational) with reduced sensitivity, (4) Appeal fast-track for verified accounts. Tradeoff: lower sensitivity on public figures may miss real violations.

**Q4: You achieve 99% precision on auto-removal. At 500M posts/day and 0.5% violation rate, how many legitimate posts do you incorrectly remove daily?**  
A: 500M × 0.5% = 2.5M violations. But auto-removal applies to high-confidence subset (say 50% of violations = 1.25M). At 99% precision, FP = 1% × 1.25M = 12,500 legitimate posts removed daily. At scale, even 99% precision means thousands of errors. Solution: (1) Increase threshold for auto-remove, (2) Easy appeal flow, (3) Monitor by category.

**Q5: A new type of harmful content emerges (e.g., AI-generated CSAM). You have zero training examples. How do you respond in <48 hours?**  
A: (1) CLIP zero-shot: describe the violation in text, use as detector (low recall but fast), (2) Few-shot: collect 50–100 examples from reports → fine-tune adapter layers (not full model), (3) Rule-based: if synthetically generated (AI detector) + minor features → flag for human review, (4) Hash the confirmed cases immediately (PDQ/PhotoDNA for instant future blocking). Speed > perfection in crisis response.
