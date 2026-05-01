# Content Synthesis & RAG

## Problem Formulation

Given a user query $q$, retrieve relevant passages from a knowledge corpus $\mathcal{C}$, then generate a grounded, faithful answer $a$ that cites sources. Must minimize hallucination while maximizing answer quality and coverage.

Objective: Maximize answer quality $Q(a|q)$ subject to faithfulness constraint:
$$a^* = \arg\max_a P(a|q, R) \quad \text{s.t.} \quad \text{every claim in } a \text{ is supported by } R$$

where $R = \text{retrieve}(q, \mathcal{C}, k)$ = top-k retrieved passages.

---

## Data & Transformations

| Aspect | Detail |
|--------|--------|
| Corpus | 10M–10B documents (enterprise docs, web, PDFs, code) |
| Chunk strategy | Semantic chunking (paragraph-level, 256–512 tokens) with overlap |
| Embedding model | E5-large / BGE / Cohere embed-v3 (1024-d) |
| Index | HNSW (Faiss/Pinecone/Weaviate) — sub-linear search |
| Query types | Factual, analytical, multi-hop, comparison, summarization |
| Ground truth | Human-annotated (query, relevant_passages, gold_answer) triplets |

**Chunking strategies** (critical design choice):
- **Fixed-size**: 512 tokens, 50-token overlap — simple, uniform  
- **Semantic**: split on paragraph/section boundaries — preserves meaning  
- **Hierarchical**: document → section → paragraph (retrieve at multiple granularities)  
- **Proposition-level**: decompose into atomic facts — best for precision, expensive  
- **Late chunking**: embed full document, chunk embeddings post-hoc (retains global context)

---

## Model Architecture

**Stage 1 — Retrieval** (Hybrid):

Dense: $\text{score}_{dense}(q, d) = E_q(q)^T \cdot E_d(d)$  
Sparse: $\text{score}_{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{tf(t,d) \cdot (k_1+1)}{tf(t,d) + k_1(1-b+b\cdot|d|/\text{avgdl})}$

Fusion: $\text{score} = \alpha \cdot \text{score}_{dense} + (1-\alpha) \cdot \text{score}_{BM25}$ (RRF or linear combo, α=0.6 typical).

**Stage 2 — Re-ranking** (Cross-encoder):

$$\text{relevance}(q, d) = \text{CrossEncoder}([q; d]) \in [0, 1]$$

Rerank top-100 → select top-5. Model: fine-tuned DeBERTa or BGE-reranker.

**Stage 3 — Generation** (LLM):

$$a = \text{LLM}(\text{system\_prompt} + q + \text{concat}(R_{1:k}))$$

With citation instruction: "Answer based ONLY on provided context. Cite sources as [1], [2]..."

**Advanced patterns**:
- **Corrective RAG (CRAG)**: if retrieval confidence < threshold → web search fallback  
- **Self-RAG**: model decides when to retrieve, evaluates its own generations  
- **Adaptive RAG**: route query to (no retrieval | single-step | multi-step) based on complexity  
- **GraphRAG**: build knowledge graph from corpus → traverse for multi-hop queries

---

## Training & Optimization

| Config | Value |
|--------|-------|
| Embedding model | Contrastive learning (InfoNCE), batch=4096, hard negatives |
| Re-ranker | Cross-encoder fine-tuned on MS MARCO + domain data |
| Generator | SFT on (query, context, answer) triplets + RLHF for faithfulness |
| Embedding refresh | Incremental — new documents added, stale removed (CDC pipeline) |
| Negative mining | BM25 top-100 not in gold set → hard negatives for dense retrieval |
| Faithfulness training | DPO with preference: faithful answer > hallucinated answer |

**Hallucination mitigation training**:
$$\mathcal{L}_{faithful} = -\log P(\text{supported\_answer}|q, R) + \beta \cdot \log P(\text{hallucinated\_answer}|q, R)$$

Train model to prefer generating only claims verifiable against retrieved context.

---

## Serving & Scale

```
User query → Query understanding (intent, entities) [10ms] →
Query expansion (HyDE: hypothetical document embedding) [100ms, optional] →
Parallel retrieval: Dense ANN + BM25 + Knowledge Graph [50ms] →
Fusion + dedup → top-100 candidates →
Cross-encoder re-ranking → top-5 passages [100ms] →
LLM generation with context (streaming) [1-3s] →
Citation verification (NLI model) [200ms, async] →
Answer with inline citations displayed

Total: P50=2s (dominated by generation), P95=4s
```

- **Index updates**: Near real-time (Kafka → embed → index in <5 min)  
- **Caching**: Query embedding cache + common query result cache (30% hit rate)  
- **Context window management**: if top-5 passages exceed context limit → summarize older passages  
- **Streaming**: First tokens appear in <500ms (retrieval completes before generation starts)

---

## Metrics & Evaluation

| Metric | Purpose | Target |
|--------|---------|--------|
| Context precision | Retrieved docs are relevant | >0.80 |
| Context recall | All needed info is retrieved | >0.85 |
| Faithfulness (RAGAS) | Answer supported by context | >0.90 |
| Answer relevance | Answer addresses the query | >0.85 |
| Hallucination rate | Claims not in context | <5% |
| Citation accuracy | Citations match claims | >95% |
| Latency (time-to-first-token) | UX | <500ms |

**RAGAS framework** (Retrieval Augmented Generation Assessment):
- Faithfulness = fraction of claims in answer supported by retrieved context
- Answer relevance = semantic similarity between answer and query intent
- Context precision = fraction of retrieved chunks that are relevant
- Context recall = fraction of ground-truth info present in retrieved context

---

## Alternatives & Tradeoffs

| Approach | Pro | Con |
|----------|-----|-----|
| Naive RAG (retrieve + generate) | Simple, works for basic QA | Fails on multi-hop, hallucination |
| Advanced RAG (re-rank + filter) | Better precision | Added latency, complexity |
| GraphRAG | Multi-hop reasoning | Expensive graph construction |
| Long-context LLM (no retrieval) | Simpler architecture | Cost at scale (all docs in context), stale |
| Fine-tuned LLM (no retrieval) | Fast inference | Can't update knowledge, hallucination |
| Agentic RAG | Iterative refinement | Slow, expensive, non-deterministic |
| ColBERT (late interaction) | Token-level matching | Larger index, more complex |

---

## Recent Research (2024–2026)

1. **CRAG** (Meta, 2024) — Corrective RAG with retrieval quality self-assessment  
2. **Self-RAG** (2024) — Model learns when to retrieve and self-reflects on generation  
3. **GraphRAG** (Microsoft, 2024) — Community summaries from knowledge graphs for global queries  
4. **ColPali** (2024) — Vision-language model for document retrieval (bypass OCR)  
5. **Contextual Retrieval** (Anthropic, 2025) — Prepend chunk-level context to improve retrieval  
6. **RAPTOR** (Stanford, 2024) — Recursive abstractive processing for tree-organized retrieval

---

## Advanced Trick Questions & Answers

**Q1: Your RAG system retrieves 5 relevant passages but 2 of them contradict each other (e.g., one says "product launched in 2023" another says "2024"). How does the LLM handle this?**  
A: Without explicit handling, the LLM may arbitrarily pick one or hallucinate a merge. Solutions: (1) Conflict detection: NLI model checks pairwise consistency of retrieved passages *before* generation. Flag contradictions. (2) Recency bias: prefer more recent source (add timestamp as metadata, instruct LLM to prefer latest). (3) Source authority: rank by document trustworthiness (official docs > blog posts). (4) Transparent output: "Source A states X [1], however Source B states Y [2]" — let user decide.

**Q2: A user asks "What is our company's parental leave policy?" Your embedding search returns the 2022 policy document (most similar text). But the policy was updated in 2024 and the new document uses different wording. The old doc ranks higher. How do you fix this?**  
A: Classic freshness-relevance tradeoff. (1) Metadata filtering: always prefer documents with latest `last_modified` date when query is about "current policy." (2) Document versioning: mark superseded documents as deprecated → exclude from retrieval. (3) Hybrid score: $\text{final} = \text{relevance} + \gamma \cdot \text{recency\_boost}$ where $\gamma$ tuned per document type. (4) Document lifecycle management: when new policy is uploaded, explicitly link to (and optionally remove) old version.

**Q3: Your RAG system works great for English documents. A user asks a question in English but the answer is in a French PDF that was never translated. Retrieval fails. What's your approach?**  
A: Multilingual/cross-lingual retrieval. (1) Use multilingual embedding model (e5-multilingual, mGTE) — embeds English query and French doc in same space. (2) Query translation: translate query to French, retrieve, translate answer back. (3) Document translation at index time: translate all non-English docs to English (expensive but simplest). (4) Best: multilingual embeddings + multilingual re-ranker + instruct LLM to "answer in the user's language regardless of source language."

**Q4: Your enterprise RAG indexes 10M documents. A user with "Engineering" role asks about HR salary bands. The retrieved documents contain confidential HR data. How do you enforce access control?**  
A: (1) **Pre-retrieval filtering**: every document has ACL metadata (allowed roles/users). At query time, filter index to only documents user has access to. Never retrieve what you can't show. (2) **Chunk-level ACLs**: if a document has mixed sensitivity, chunk-level permissions. (3) **Post-retrieval check**: even if retrieval somehow returns unauthorized doc, generation pipeline has a security filter. (4) **Never rely on the LLM** to enforce access — it will leak information through reasoning. Access control must be at the retrieval/infrastructure layer.

**Q5: You have a 128K context window LLM. The CEO says "just put all documents in the context, no need for retrieval." When does this actually make sense vs. when does RAG win?**  
A: Long-context wins when: (1) corpus is small (<100K tokens total), (2) questions require holistic understanding (summarize all docs), (3) latency is acceptable (long-context = expensive). RAG wins when: (1) corpus is large (>1M tokens — can't fit), (2) questions are specific (find needle in haystack — retrieval is O(1), attention is O(n²)), (3) corpus changes frequently (re-indexing is cheaper than re-processing), (4) cost matters (processing 128K tokens per query at $15/M tokens vs. retrieving 5 chunks). At 10M documents, long-context is physically impossible and economically insane. RAG is mandatory.
