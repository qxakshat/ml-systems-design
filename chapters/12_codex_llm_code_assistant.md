# Codex — LLM-Based Code Assistant

## Problem Formulation

Given context (code prefix, suffix, file structure, repository), generate correct, idiomatic code completions. Supports: inline completion (FIM), chat-based generation, multi-file edits, and agentic coding (plan + execute). Measured by acceptance rate and functional correctness.

Objective: Maximize $P(\text{correct\_code} | \text{prefix}, \text{suffix}, \text{repo\_context})$ subject to latency < 300ms for inline completions.

---

## Data & Transformations

| Aspect | Detail |
|--------|--------|
| Pre-training corpus | 1T+ tokens from permissively-licensed code (GitHub, StackOverflow, docs) |
| Languages | 50+ (Python, JS/TS, Java, C/C++, Go, Rust — weighted by usage) |
| Context window | 128K tokens (repo-level context) |
| FIM training | Fill-in-the-Middle: (prefix, suffix) → middle (50% of training) |
| Alignment data | Human preference pairs: (prompt, good_completion, bad_completion) |
| Filtering | Remove PII, secrets, license-violating code, low-quality (lint score < threshold) |

**Data pipeline**:
1. Deduplicate at file level (MinHash, Jaccard > 0.8)  
2. Quality filter: parseable, passes basic lint, has docstrings  
3. Repository-level: keep repo structure (imports, cross-file references)  
4. FIM transformation: randomly split files into (prefix, middle, suffix)  
5. Instruction data: (natural language instruction, code solution) pairs from commits, PRs, issues

---

## Model Architecture

**Base: GPT-4 class decoder-only Transformer**

$$P(x_t | x_{<t}) = \text{softmax}(W_o \cdot \text{TransformerBlock}^L(E(x_{<t})))$$

Key architectural choices:
- **Mixture of Experts (MoE)**: 8 experts, top-2 routing per token → 4× parameter count at same compute
- **Grouped Query Attention (GQA)**: 8 KV heads shared across 64 query heads (saves KV cache)
- **RoPE**: Rotary positional embeddings (extrapolates to longer sequences)
- **SwiGLU activation**: $\text{SwiGLU}(x) = (xW_1) \odot \sigma(xW_g) \cdot W_2$

**Fill-in-the-Middle (FIM)**:
Training format: `<PRE> prefix <SUF> suffix <MID> middle`  
At inference: given prefix + suffix → generate middle.

**Speculative Decoding** (for latency):
- Draft model (1B params) generates $k$ candidate tokens
- Target model (100B+) verifies all $k$ in single forward pass
- Accept matching prefix, reject + resample from target distribution
- Speedup: 2–3× for code (high predictability)

---

## Training & Optimization

| Config | Value |
|--------|-------|
| Pre-training | 10K+ GPUs, 3–6 months, cosine schedule |
| Optimizer | AdamW (lr=3e-4 → 3e-5, warmup 2K steps) |
| Context | 128K tokens (with RoPE scaling) |
| FIM ratio | 50% FIM, 50% left-to-right |
| Post-training | SFT on instruction data → RLHF/DPO alignment |
| DPO loss | $\mathcal{L}_{DPO} = -\log\sigma(\beta(\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}))$ |
| Reward model | Trained on human preference data (correct > compiles > wrong) |

**RLHF pipeline**:
1. SFT on curated (instruction, code) pairs  
2. Reward model training on preference rankings  
3. PPO optimization against reward (with KL constraint to SFT policy)  
4. Alternative: DPO (Direct Preference Optimization) — skips reward model

---

## Serving & Scale

```
User types code → Debounce (300ms pause) → 
Context assembly: current file (prefix/suffix) + open tabs + repo structure [5ms] →
Speculative draft model (1B, on-device or edge) [50ms] →
Target model verification (100B+, GPU cluster) [200ms] →
Streaming tokens to IDE → User sees completion

For chat/agentic: full target model generation with longer context [1-5s]
```

- **KV cache optimization**: PagedAttention (vLLM) — 3× throughput vs. naive  
- **Batching**: Continuous batching (Orca) — new requests join in-flight batch  
- **Quantization**: INT4 (GPTQ/AWQ) for inference — 4× memory reduction, ~5% quality loss  
- **Multi-tenant GPU**: A100/H100 clusters, 100K+ concurrent users per cluster  
- **Caching**: Common completions cached (imports, boilerplate) — cache hit rate ~15%

---

## Metrics & Evaluation

| Metric | Purpose | Target |
|--------|---------|--------|
| HumanEval pass@1 | Functional correctness | >90% |
| Acceptance rate | User satisfaction proxy | >30% of shown completions |
| Completion persistence | Code kept after 5 min | >70% of accepted |
| Characters saved | Productivity | 40%+ keystrokes saved |
| Latency (P50/P95) | UX | <300ms / <800ms |
| SWE-bench | Real-world bug fixing | >50% |
| MBPP+ | Broader correctness | >85% |

---

## Alternatives & Tradeoffs

| Approach | Pro | Con |
|----------|-----|-----|
| GPT-4 class (current) | Best quality, reasoning | Expensive, latency |
| Smaller model (7B) on-device | Fast, private, no network | Lower quality, limited context |
| RAG over codebase | Leverages repo knowledge | Retrieval errors, stale index |
| Fine-tuned per org | Knows internal APIs/style | Expensive per customer, data privacy |
| Tree-search (AlphaCode) | Explores solution space | Very expensive (generate 1M samples) |
| Agent-based (multi-step) | Can plan, test, iterate | Slow, expensive, non-deterministic |
| Retrieval-augmented generation | Best of both worlds | Complexity, retrieval latency |

---

## Recent Research (2024–2026)

1. **DeepSeek-Coder-V2** (2024) — MoE code model rivaling GPT-4 at fraction of cost  
2. **StarCoder2** (BigCode, 2024) — 15B open model trained on Software Heritage  
3. **SWE-Agent** (Princeton, 2024) — Agent framework for real GitHub issue resolution  
4. **CodeAct** (2025) — Code-as-action paradigm for LLM agents  
5. **AlphaEvolve** (DeepMind, 2025) — Evolutionary code generation with LLM-guided mutations  
6. **Cursor/Windsurf** (2025) — IDE-native agents with multi-file edit capabilities

---

## Advanced Trick Questions & Answers

**Q1: Your code completion model suggests `eval(user_input)` in a Python web app. The user accepts it. How do you prevent this?**  
A: (1) Static analysis guardrail: post-generation, run security lint (Semgrep rules) before showing to user. Block known dangerous patterns (eval, exec, SQL injection, path traversal). (2) RLHF training: include security-violating completions as "rejected" in preference data. (3) Context-aware: if file imports Flask/Django → apply web-security rules. (4) Caveat: can't block everything — provide inline warning ("⚠️ potential security issue") rather than silent censorship for ambiguous cases.

**Q2: Your model was trained on code up to October 2024. A user is writing code using a library released in January 2025. Completions are wrong. How do you handle knowledge cutoff?**  
A: (1) RAG: index the library's documentation + source code → inject into context at inference time. (2) LSP integration: use language server to validate completions against actual installed package API. (3) Retrieval from user's project: if `node_modules/` or `venv/` contains the library, read type stubs/definitions. (4) Graceful degradation: detect when confidence is low (high entropy on API call tokens) → don't show completion rather than show wrong one.

**Q3: Two users accept the exact same completion. User A's code works, User B's code breaks. Same completion, different outcomes. Why?**  
A: Context beyond the visible prefix/suffix matters. (1) Different import versions (numpy 1.x vs 2.x), (2) Different variable types flowing into the function (type mismatch not visible in local context), (3) Different runtime environments (OS-specific behavior), (4) The model sees *local* context but not *global* state. Fix: expand context window to include relevant imports, type information (from LSP), and test results from previous runs.

**Q4: You want to fine-tune on a company's private codebase (10M lines). But the base model already knows "public best practices." The fine-tuned model starts generating internal API calls even when the user is writing open-source code. How do you control this?**  
A: (1) Context-conditioned generation: add a special token or system prompt indicating "internal repo" vs. "open source" mode. (2) LoRA adapters: keep base model frozen, train lightweight adapter for internal code → activate only when user is in internal repo. (3) Retrieval gating: only retrieve from internal codebase when file path is within company workspace. (4) Never fine-tune the base model directly — use adapters/prefix-tuning to keep capabilities separable.

**Q5: Your inline completion shows a 35% acceptance rate. PM wants 50%. You can make completions more conservative (shorter, more obvious) to increase acceptance. Should you?**  
A: Goodhart's Law — optimizing acceptance rate directly hurts value. (1) Short/obvious completions (closing brackets, variable names) inflate acceptance rate but save zero keystrokes of meaningful typing. (2) Better metric: "characters of novel code accepted" or "time saved per session." (3) A bold 10-line completion with 20% acceptance but high value when accepted > a trivial 3-character completion with 80% acceptance. (4) Solution: optimize E[value] = P(accept) × complexity_of_completion. Weight longer, more complex accepted completions higher.
