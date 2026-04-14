# Week 12 Study Guide — Systems & Senior Engineering

## Overview

This week you move from *model quality* to *system quality*. The goal is to
internalize how transformers actually consume memory, how to optimize
inference throughput, how to distribute training across GPUs, and how
production agent systems fail. These topics are what separate senior
ML engineers from those who only know how to train models.

---

## Day 1 — Profiling Memory

### The Four Memory Buckets

Every byte of GPU memory during training goes into one of these buckets:

| Bucket | Bytes per Param | Notes |
|--------|-----------------|-------|
| Parameters | 2 (FP16) | The model weights |
| Gradients | 2 (FP16) | One per parameter |
| Optimizer states | 12 (FP32) | Adam: m + v + master copy |
| **Static total** | **16** | **~16× weight size** |
| Activations | varies | O(B × S × H) per layer |
| KV cache | varies | O(L × B × S × H), inference only |

**The 16× rule:** A 7B parameter model (14 GB in FP16) needs ~112 GB of GPU
memory for static training state alone. That's why you need multiple A100s.

### Activation Memory

Activations are the intermediate tensors kept alive for backprop. Each
transformer layer saves roughly:

```
acts_per_layer ≈ B × S × (3H + heads×S + H + 4H) bytes × dtype_size
```

The dangerous term is `heads × S`: the attention score matrix is **O(S²)**.
At S=2048 with 32 heads, this adds ~32 × 2048 × 2048 × 2 ≈ 268 MB per layer.
With 32 layers, that's 8.6 GB just for attention scores.

### Gradient Checkpointing

Gradient checkpointing recomputes activations during the backward pass instead
of saving them. The trade-off:

- **Memory:** saves ~70% of activation memory (stores only √L checkpoints)
- **Compute:** costs ~33% extra FLOPs (recomputation)
- **When to use:** always during fine-tuning of large models

```python
model.gradient_checkpointing_enable()
```

### KV Cache

During autoregressive generation, the model accumulates key/value pairs for
every token:

```
KV cache = 2 × L × B × S × H × dtype_bytes
```

For a 7B model at seq=2048, batch=1, FP16: ≈ **1.07 GB**.
This grows **linearly** with sequence length — the primary memory constraint
at long context inference.

### Mental Model Diagram

```
┌─────────────────────────────────────────┐
│          GPU Memory Budget               │
├──────────────┬──────────────────────────┤
│ Static (16×) │ Parameters (2B/param FP16)│
│              │ Gradients  (2B/param FP16)│
│              │ Optimizer  (12B/param FP32│
├──────────────┼──────────────────────────┤
│ Dynamic      │ Activations (O(B×S²×L))  │
│              │ KV cache   (O(B×S×L×H))  │
└──────────────┴──────────────────────────┘
```

---

## Day 2 — Throughput Optimization

### Latency vs Throughput

| | Definition | When to optimize |
|---|---|---|
| **Latency** | Time to complete one request | User-facing, real-time |
| **Throughput** | Requests (or tokens) per second | Batch processing, cost |

These goals **conflict**. Low latency requires prioritizing the current
request. High throughput requires batching many requests together.

### Why Batching Works

Transformer inference is **memory-bandwidth-bound** at batch=1. The GPU
spends most of its time loading weights from HBM (high-bandwidth memory)
into compute units — but only uses them for one sequence at a time.

With batch=N, you load weights once but apply them to N sequences
simultaneously. The arithmetic intensity increases, making better use of
the GPU's compute capacity.

**Throughput scaling:**
- Small batches: near-linear in batch size
- Large batches: flattens when KV cache fills memory or attention compute dominates

### KV Cache

Without KV cache, each decode step processes the entire prefix → O(S²) work per step.
With KV cache, only the new token is processed → O(S) amortized.

Typical speedup: **5–20×** depending on sequence length.

```python
# Enable (default True) or disable:
model.generate(..., use_cache=True)
```

### Quantization

| Format | Bits | Memory | Throughput | Quality Loss |
|--------|------|--------|------------|-------------|
| FP32 | 32 | baseline | 1× | 0% |
| FP16/BF16 | 16 | 0.5× | ~1.5× | <0.1% |
| INT8 | 8 | 0.25× | ~2× | <1% |
| INT4 | 4 | 0.125× | ~3× | 1-3% |

### Serving at Scale

For 1M requests/day:
- 1M req / 86,400s ≈ **11.6 req/s**
- At 64 tokens/req output: 743 tok/s minimum
- With INT8 quantization on A10G (≈5k tok/s): 1 GPU handles it comfortably
- Add 2× headroom for traffic bursts → 2 GPUs + autoscaling

---

## Day 3 — FSDP and ZeRO

### The Core Problem

Training a 7B model needs ~112 GB of static memory. A single A100 has 80 GB.
Solution: shard the memory across N GPUs.

### ZeRO Stages

```
Stage 0 (DDP):   each GPU has FULL copy of everything
Stage 1:         optimizer states sharded (÷N)
Stage 2:         optimizer + gradients sharded (÷N)
Stage 3:         params + gradients + optimizer sharded (÷N)
```

**Memory per GPU (bytes/param, FP16, Adam):**

| Stage | Formula | 8 GPUs, 7B model |
|-------|---------|-----------------|
| 0 | 16 | 112 GB |
| 1 | 4 + 12/N | ~5.5 GB static |
| 2 | 2 + 14/N | ~3.75 GB static |
| 3 | 16/N | ~14 GB total |

### FSDP (PyTorch Native ZeRO-3)

FSDP is PyTorch's built-in implementation of ZeRO-3-style sharding.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=mp_policy,
    device_id=device,
)
```

**What happens during training:**
1. **Forward:** `all_gather` params for each layer just before use
2. **Backward:** `reduce_scatter` gradients after each layer
3. **After backward:** discard gathered params (only keep shard)

**Communication cost:** 3× model size per training step (2× all-gather + 1× reduce-scatter)

### FSDP Sharding Strategies

| Strategy | Equivalent | When to use |
|----------|-----------|-------------|
| `NO_SHARD` | DDP | Model fits on 1 GPU |
| `SHARD_GRAD_OP` | ZeRO-2 | Reduce grad memory |
| `FULL_SHARD` | ZeRO-3 | Model doesn't fit on 1 GPU |
| `HYBRID_SHARD` | ZeRO-3 + DDP | Multi-node, fast intra-node network |

### ZeRO vs FSDP: Key Differences

| | DeepSpeed ZeRO | PyTorch FSDP |
|--|----------------|--------------|
| Integration | External library | PyTorch native |
| Compatibility | Most frameworks | Best with PyTorch |
| Stage 3 equivalent | ZeRO-3 | FULL_SHARD |
| Offloading (CPU) | ZeRO-Infinity | Limited support |
| Communication backend | NCCL | NCCL / Gloo |

---

## Day 4 — Failure Modes in Production

### The Fundamental Insight

> "Agents are distributed systems with a language model core."

Every failure mode from distributed systems engineering applies — *plus* a
new class of failures unique to probabilistic orchestrators.

### Failure Mode Taxonomy

#### 1. Context Window Overflow
- **Cause:** Cumulative message history exceeds `max_context_len`
- **Symptom:** Silent truncation → agent "forgets" its task framing
- **Detection:** Track `token_count` per session; alert at 80% capacity
- **Fix:** Sliding window + summarize dropped messages

#### 2. Tool Deadlocks
- **Cause:** Tool call blocks indefinitely (hung API, network issue)
- **Symptom:** Agent session freezes, no output, no error
- **Detection:** Per-call latency timeout alert
- **Fix:** `asyncio.wait_for(tool_call(), timeout=N)` + circuit breaker

#### 3. Hallucinated Function Calls
- **Cause:** Model generates plausible-but-wrong tool names or arguments
- **Symptom:** `KeyError`, wrong data written, unexpected API calls
- **Detection:** JSON schema validation + tool name whitelist **before** execution
- **Fix:** Validate every tool call; never execute unvalidated model output

#### 4. Retry Loops
- **Cause:** Agent retries failing tool indefinitely without budget
- **Symptom:** Runaway token usage, infinite loops, cost explosion
- **Detection:** Step counter alert when session exceeds N steps
- **Fix:** `max_attempts` cap + exponential backoff with jitter

#### 5. Latency Cascades
- **Cause:** Sequential pipeline where one slow tool delays all downstream
- **Symptom:** P99 latency grows super-linearly with pipeline depth
- **Detection:** Per-tool P99 monitoring; pipeline-level P99 vs sum-of-P50
- **Fix:** Parallel execution, request hedging, per-hop timeouts

#### 6. Memory Leaks
- **Cause:** Python objects (embeddings, history) accumulate across sessions
- **Symptom:** Gradual memory growth → OOM after hours of operation
- **Detection:** Process RSS monitoring over time
- **Fix:** Explicit session teardown + bounded history buffers

### Compound Failure Rate

If each tool has reliability `r` and the pipeline has `n` tools:

```
P(chain success) = r^n
```

| r | n=5 | n=10 |
|---|-----|------|
| 99% | 95.1% | 90.4% |
| 95% | 77.4% | 59.9% |
| 90% | 59.0% | 34.9% |

Design for failure. A 95%-reliable tool is not reliable in a 10-step chain.

---

## Day 5 — Observability and Logging

### The Observability Stack for LLM Systems

| Signal | What it tells you |
|--------|------------------|
| Structured logs | Per-call latency, success/fail, tokens |
| Latency histograms | P50/P95/P99 per tool |
| Token usage | Cost, context bloat detection |
| Schema violation rate | Model quality drift |
| Eval regression tests | Overall model quality vs baseline |
| Prompt traces | Exact input that caused a failure |

### Minimum Required Log Fields

```json
{
  "timestamp":   "2026-01-15T12:34:56.789Z",
  "session_id":  "sess_0042",
  "step":        7,
  "event":       "tool_call",
  "tool":        "search_web",
  "latency_ms":  234.5,
  "success":     true,
  "tokens_in":   1200,
  "tokens_out":  48,
  "schema_valid": true,
  "error":       null,
  "failure_type": null
}
```

### Schema Violation Rate as a Health Signal

Schema violation rate is the **earliest detectable signal** of model
regression. When the model starts generating invalid JSON or wrong field
types, it's a sign of:
- Distribution shift in the model's outputs
- A new fine-tuning run that degraded instruction-following
- Context window overflow causing the model to lose its output format

Alert threshold: violation rate > 2× baseline.

### Canary Deployment Strategy

1. Deploy new model to 5% of traffic
2. Compare in parallel: error rate, latency P99, schema violations, eval scores
3. Auto-rollback if canary error rate > stable + 2σ after N=1000 requests
4. Promote to 100% only after 24h stable canary window

---

## Days 6-7 — The Blog Post

The blog post ["What Breaks When LLMs Become Agents"](../results/day67/blog_post.md)
synthesizes all five days into a senior-engineer-level artifact.

**Key claims (back these up in interviews):**
1. Tool reliability compounds — 95% × 10 steps = 60% success
2. Schema violation rate is your earliest regression signal
3. KV cache grows linearly; attention is O(S²) — both bound long-context perf
4. ZeRO-3 / FSDP FULL_SHARD scales static memory as O(1/N)
5. Reward hacking is Goodhart's Law applied to self-improving loops

---

## Key Formulas to Memorize

```
Training static memory (bytes) = num_params × 16
  (2 weights + 2 grads + 12 optimizer, FP16/FP32 mixed)

KV cache (bytes) = 2 × L × B × S × H × 2
  (2=K+V, L=layers, B=batch, S=seq, H=hidden, 2=FP16)

ZeRO-3 per-GPU = 16 × num_params / N
  (N = number of GPUs)

Attention complexity = O(B × heads × S²)
  (the S² term is why long context is expensive)

Chain reliability = r^n
  (r = per-tool reliability, n = chain length)
```
