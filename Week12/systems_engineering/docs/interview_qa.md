# Week 12 Interview Q&A — Systems & Senior Engineering

These are the highest-probability questions at senior ML engineer interviews
covering inference optimization, distributed training, and production reliability.

---

## Memory & Profiling

**Q: Why does memory scale with sequence length in transformers?**

A: Two mechanisms. First, the self-attention score matrix is O(B × heads × S²):
each layer materializes a full `[B, heads, S, S]` tensor for backprop. At
S=2048 with 32 heads on a 32-layer model, this alone is ~8 GB of activation
memory. Second, the KV cache at inference grows as O(2 × L × B × S × H) —
linearly in S. Both grow with context length; the attention matrix grows
faster (quadratically) during training.

---

**Q: What dominates memory at inference?**

A: The KV cache, at long context. The model weights are fixed. The KV cache
grows as `2 × L × B × S × H × dtype_bytes` — linearly with sequence length.
For a 70B model at seq=4096, batch=1, FP16, this is ~16 GB, comparable in
size to the model's FP16 weights. At batch=32 it's 512 GB — far beyond a
single GPU. This is why continuous batching (vLLM's PagedAttention) and
context compression matter at scale.

---

**Q: How does KV cache work? What does it trade?**

A: During autoregressive generation, each decode step would normally recompute
attention over the full prefix — O(S) work per step, O(S²) total. KV cache
stores the key and value tensors for every past token, per layer. Each new
decode step only computes Q for the new token and attends over the cached K/V
— reducing per-step work from O(S) to O(1) in attention projections.

Trade: memory grows as O(L × B × S × H). At long context or large batch,
this can exhaust GPU memory. Solutions: sliding-window attention (cache only
last W tokens), quantized KV cache (INT8/INT4), and PagedAttention (virtual
memory paging for KV blocks).

---

**Q: What is gradient checkpointing and when should you use it?**

A: Gradient checkpointing (activation checkpointing) recomputes intermediate
activations during the backward pass instead of saving them. Only one checkpoint
per N layers is stored. This reduces activation memory from O(L × B × S × H)
to O(√L × B × S × H) — roughly a 70% reduction for typical depths.

Cost: ~33% more FLOPs (one extra forward pass per layer during backward).

Use it whenever you're activation-memory-bound during training, which is
almost always when fine-tuning large models with moderate batch sizes.

```python
model.gradient_checkpointing_enable()
```

---

**Q: For a 7B model training with Adam in FP16 mixed precision, how much GPU memory do you need?**

A: Static memory (params + gradients + optimizer states):
- FP16 weights: 7B × 2 bytes = 14 GB
- FP16 gradients: 7B × 2 bytes = 14 GB
- FP32 Adam m+v+master: 7B × 12 bytes = 84 GB
- **Static total: ~112 GB**

Plus activations: depends heavily on batch size and sequence length.
For batch=8, seq=2048, without gradient checkpointing: add ~30-50 GB.

Conclusion: needs at least 2× A100 (80 GB each) just for static state.
With gradient checkpointing: activations drop ~70%, fitting more comfortably
on 2 GPUs.

---

## Throughput Optimization

**Q: Why does batching improve throughput?**

A: Transformer inference is memory-bandwidth-bound at small batch sizes. The
GPU loads weight tensors from HBM (high-bandwidth memory) regardless of
whether it applies them to 1 or 32 sequences. Batching amortizes this fixed
weight-load cost across N sequences, giving near-linear throughput scaling
until you hit either: (a) compute saturation, (b) KV cache fills GPU memory.

The crossover point is the "batch size knee" — typically batch=8-32 for
typical models on modern GPUs.

---

**Q: What's the tradeoff between latency and throughput?**

A: They are fundamentally opposed. Latency is minimized by processing each
request immediately with batch=1. Throughput is maximized by accumulating
large batches. Dynamic batching (used by vLLM, TGI, Triton) is the production
compromise: accumulate requests for up to T milliseconds or until batch reaches
size N, whichever comes first. This gives the first request in a batch latency
close to batch=1, while subsequent requests in the same batch get near-zero
marginal latency overhead.

---

**Q: How would you serve 1M requests/day efficiently?**

A: 1M / 86,400 ≈ 11.6 req/s. Assume 128 output tokens per request:
- Required throughput: ~1,500 tok/s sustained, ~4,500 tok/s at peak (3× burst)

Steps:
1. **Quantize** to INT8 — halves memory, increases throughput ~2×
2. **Dynamic batching** — batch=16 gives ~10× throughput over batch=1
3. **KV cache** — ensure it's enabled (default in most frameworks)
4. **Horizontal scaling** — 2-3 A10G GPUs (each ≈5k tok/s INT8) handles peak
5. **Autoscaling** — spin up extra replicas during traffic spikes
6. **Speculative decoding** — if the model is large, use a small draft model
   to generate 4-token candidates, large model verifies in one pass → 2-3×
   speedup for free-form generation

---

**Q: What is speculative decoding?**

A: A technique to speed up autoregressive generation using two models: a small
"draft" model and a large "target" model.

1. Draft model generates K tokens cheaply in K sequential steps
2. Target model verifies all K tokens in ONE forward pass (parallel, fast)
3. Accepted tokens are kept; on first rejection, roll back and resample

Speedup: if K=4 tokens are accepted on average, you get 4 target-model tokens
per 1.5 model calls (1 draft + 0.5 target on average). That's ~2-3×.

Requirements: draft model must be much smaller (10-100×); same vocabulary.
Works best for text with predictable patterns (code, templates).

---

## FSDP and ZeRO

**Q: What is the difference between FSDP and ZeRO?**

A: Both implement the same core idea — shard model state across GPUs to
eliminate redundancy. The differences are:

| | DeepSpeed ZeRO | PyTorch FSDP |
|--|----------------|--------------|
| Origin | Microsoft DeepSpeed | PyTorch native (1.11+) |
| Integration | External library | Built into PyTorch |
| ZeRO-3 equivalent | ZeRO-3 | `FULL_SHARD` strategy |
| CPU offloading | ZeRO-Infinity (mature) | Limited |
| Composability | Less composable | Better with torch.compile, etc. |
| Ease of use | More config options | Simpler API for standard cases |

For most PyTorch projects, FSDP is preferred for its native integration.
For very large models with CPU offloading needs, DeepSpeed ZeRO-Infinity
is more mature.

---

**Q: When does ZeRO-3 help vs when is it overkill?**

ZeRO-3 helps when:
- ✓ The model doesn't fit on a single GPU even at batch=1
- ✓ You have fast intra-node interconnects (NVLink, at least 600 GB/s)
- ✓ You want to maximize model size per GPU budget

ZeRO-3 is overkill when:
- ✗ Model fits on one GPU — just use DDP (ZeRO-0), it's simpler
- ✗ You have slow inter-node networking (10 GbE) — comm becomes the bottleneck
- ✗ Batch sizes are very small — communication overhead dominates

Rule of thumb: ZeRO-2 is often the sweet spot. It shards gradients and
optimizer states (saves 75% of the 12 bytes/param optimizer overhead) while
keeping communication cheaper than ZeRO-3 (no all-gather on forward pass).

---

**Q: Why does communication become the bottleneck in ZeRO-3?**

A: ZeRO-3 requires two collective operations per transformer layer per step:
- **All-gather (forward):** reconstruct full parameter tensor from shards
- **Reduce-scatter (backward):** shard gradient contributions across GPUs

Total communication volume ≈ 3× model parameter bytes per training step.
For a 7B FP16 model: 3 × 14 GB = 42 GB of data moved per step.

On **NVLink** (600 GB/s): 42 GB / 600 GB/s ≈ 70 ms — fast, not a bottleneck.
On **InfiniBand** (400 Gb/s ≈ 50 GB/s): 42 GB / 50 GB/s ≈ 840 ms — significant.
On **10 GbE** (1.25 GB/s): 42 GB / 1.25 GB/s ≈ 33 seconds — completely unusable.

Lesson: ZeRO-3 is only viable with high-speed interconnects. Multi-node
ZeRO-3 on Ethernet is not a good idea.

---

## Production Reliability

**Q: What breaks first in agent systems?**

A: Tool reliability, consistently. The model itself is usually not the
failure point — it's the external systems the agent calls. A tool with 95%
reliability used 10 times has a 60% chain success rate. Failures compound.

After tool reliability, the next failure is usually retry loops — the agent
keeps retrying a broken tool, burning tokens and time, until it hits the
context limit and produces a confused final response.

The third most common failure is context overflow — the agent's history grows
until early context is silently truncated, causing it to "forget" its original
objective.

---

**Q: How do you monitor agent performance?**

A: At minimum, emit a structured log entry for every tool call and every LLM
call. Include: session_id, step, event type, tool name, latency_ms, success,
tokens_in, tokens_out, schema_valid, failure_type.

From this, derive:
1. **Error rate by failure type** — which failure dominates?
2. **P99 latency per tool** — which tool is your tail-latency risk?
3. **Token growth per session** — context bloat indicator
4. **Schema violation rate over time** — model quality drift signal
5. **Session length distribution** — are sessions getting stuck in loops?

Alert thresholds:
- Error rate > 2× baseline → incident
- Schema violation rate > 5% → model regression check
- Session exceeds N steps → potential retry loop

---

**Q: How do you detect silent failures?**

A: Silent failures require proactive detection:

1. **Schema validation on every output** — validate before acting, not after
2. **Factual consistency checks** — run a secondary LLM judge on a 1% sample
3. **Intermediate state assertions** — after a "write file" tool call, assert
   the file actually exists and has expected content
4. **Eval regression testing** — run golden test suite before every deploy
5. **Schema violation rate monitoring** — this is the canary; it spikes
   before users notice wrong outputs

The key insight: silent failures have a *signature* in your structured logs
even when they don't throw errors. Schema violations, unexpected token counts,
session length outliers — these are all detectable patterns.

---

**Q: What is abstention? Why does it matter?**

A: Abstention is the model returning "I don't know" or declining to answer
when its confidence is low. A well-calibrated model has:
- High abstention rate on hard/ambiguous questions
- High accuracy on the questions it does answer

Why it matters: a model that always answers — even when wrong — is dangerous
in production because its hallucinations look identical to correct answers.
A model that abstains on low-confidence inputs gives users a meaningful signal.

You can increase abstention rate via:
- RLHF rewards for correct refusal on out-of-distribution inputs
- Confidence thresholding (if model's log-prob on answer < threshold, abstain)
- Verbalized uncertainty ("I'm not certain, but..." → flag for human review)

---

**Q: How would you evaluate a coding agent?**

A: Don't evaluate on BLEU or surface similarity — evaluate on functional
correctness:

1. **pass@k** — run generated code against a test suite. Does it pass?
   `pass@k` = probability at least 1 of k samples passes all tests.
2. **Test coverage** — does the code test the right things, or trivially pass?
3. **Security scan** — does the code contain known vulnerability patterns?
4. **Regression tests** — does the new code break existing functionality?
5. **Human review rate** — what fraction requires human correction?

The "coding agent" problem is really an *evaluation* problem: you need a
reliable test harness that the agent can't trivially game (e.g., by deleting
the test file).

---

**Q: How would you reduce inference cost by 50%?**

A: Multiple levers, pick based on your constraints:

1. **INT8 quantization** (~2× throughput, <1% quality loss) — fastest win
2. **Larger batch sizes** with dynamic batching — amortizes weight loading
3. **KV cache quantization** (INT8 KV) — halves KV cache memory → longer
   seqs or larger batches
4. **Speculative decoding** — 2-3× speedup for generative tasks
5. **Model distillation** — smaller model, same task quality (1-3 months work)
6. **Prompt compression** — compress context with a small encoder model
7. **Caching common responses** — exact-match or semantic cache for
   frequently-asked questions

Most practical 50% reduction: INT8 quantization + batch size optimization.
These can often be done in a week and compound multiplicatively.

---

**Q: What breaks when you increase the context window?**

A: Multiple things:

1. **Attention compute:** O(S²) per layer → quadratic increase in FLOPS
2. **KV cache memory:** O(L × B × S × H) → linear increase, but can OOM at scale
3. **Effective recall degrades:** models struggle to retrieve information from
   the middle of very long contexts ("lost in the middle" problem)
4. **Throughput drops:** longer sequences mean fewer sequences fit in a batch
5. **Cost increases:** most APIs charge per token; 10× context = 10× cost
6. **Fine-tuning becomes harder:** training on long sequences requires more
   memory and takes longer per step

Practical limit: 32K-128K context is "long" for today's models. Beyond that,
you typically need Flash Attention (fuses the S² attention kernel to avoid
materializing the full matrix), ring attention (for extreme lengths), or
retrieval augmentation instead of fitting everything in context.

---

## High-Level System Design

**Q: How would you design a system to detect reward hacking?**

A: Reward hacking (Goodhart's Law: when a measure becomes a target, it ceases
to be a good measure) requires out-of-distribution detection:

1. **Hold-out hard test set** — a set of examples the reward model never
   trained on. If reward improves but performance on this set drops, you're
   hacking.
2. **Multiple orthogonal reward signals** — gaming all simultaneously is harder.
   E.g., for a coding agent: test pass rate + code review score + human eval.
3. **Distribution shift detection** — compare the model's output distribution
   to human outputs. If they diverge (via KL or classifier), flag for review.
4. **Adversarial probing** — regularly probe with examples designed to trigger
   known shortcut strategies.
5. **Random human audit** — 0.1% of production outputs, truly random
   (not best-of-N), reviewed by humans unfamiliar with the model.

---

**Q: How do you design safe tool use for an LLM agent?**

A: Defense in depth:

1. **Whitelist, not blacklist** — define exactly which tools the agent can call.
   Any unrecognized tool name is rejected immediately.
2. **Schema validation** — every tool call is validated against a JSON schema
   before execution. Type errors and missing fields are caught.
3. **Path/resource allowlists** — if the tool takes a file path, validate
   against an allowlist of directories (`/data/`, `/tmp/`).
4. **Per-tool permissions** — read-only tools vs read-write tools vs
   network tools have different approval thresholds.
5. **Human-in-the-loop for irreversible actions** — any tool that deletes,
   sends email, or executes code requires explicit human confirmation.
6. **Audit log** — log every tool execution (not just calls) for forensic review.
7. **Sandboxing** — code execution happens in an isolated container with
   no network access and read-only filesystem except `/tmp`.
