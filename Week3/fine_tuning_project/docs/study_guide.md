# Study Guide: Fine-Tuning LLMs (Week 3)

A practical reference for the design decisions, concepts, and interview
questions covered in this project.

---

## 1. SFT — Supervised Fine-Tuning

### What it is
SFT is **behaviour cloning**.  You collect demonstration pairs (instruction,
response) and train the model with cross-entropy loss to predict the response
given the instruction.  The model learns to *mimic* the style, format, and
content of the demonstrations.

### Why masking matters
Without assistant-token masking, the cross-entropy loss is computed over the
*entire* sequence — including the user prompt.  But the prompt is deterministic
(it was provided as input); there is nothing to learn from predicting it.

With `completion_only_loss=True` (TRL), only assistant tokens contribute to the
loss.  Every gradient step is a signal about what a good assistant response
looks like, not wasted on memorising the question.

```
With masking:    loss = CE(assistant tokens only)    ← informative
Without masking: loss = CE(all tokens)               ← diluted, wasteful
```

### What SFT is good at
- **Format compliance** — the model mimics bullet lists, section headers,
  JSON structure, refusals, etc. from demonstrations very effectively.
- **Style imitation** — polite tone, conciseness, vocabulary.
- **Safety** — pairing harmful questions with refusal responses teaches the
  model to decline gracefully.

### What SFT cannot do
- **Preference trade-offs** — SFT averages over all demonstrations.  If a
  prompt has multiple valid continuations, the model learns a blend rather
  than choosing the *best* one.
- **Out-of-distribution generalisation** — it interpolates between training
  examples, not reasoning from principles.
- **Reward maximisation** — for that you need RLHF, DPO, or a reward model.

---

## 2. LoRA — Low-Rank Adaptation

### Core idea
Instead of updating all *W* (large and expensive), inject trainable rank-*r*
matrices *A* and *B* (small and cheap):

```
W_new = W_frozen + (B @ A) * (alpha / r)
```

Where:
- `r` = rank (typically 4–64).  Controls adapter capacity.
- `alpha` = scaling factor (typically = `r` or `2r`).  Scales the adapter
  output magnitude relative to the frozen weights.
- Only `A` and `B` are trained — < 1% of total parameters.

### Rank intuition
Low rank means the adapter can only represent *r* linearly independent
directions in weight space.  This acts as an implicit regulariser — the model
can't overfit to arbitrary noisy patterns in small datasets.

Higher rank = more capacity = closer to full fine-tuning = more risk of
overfitting on tiny datasets.

### Why LoRA works
Pre-trained language models occupy a small "intrinsic dimension" in weight
space — most fine-tuning directions live in a low-dimensional subspace.
Low-rank updates are sufficient to capture task-specific adaptation.

### LoRA vs. full FT trade-offs

| Dimension | Full FT | LoRA |
|---|---|---|
| Trainable params (GPT-2) | 124 M (100%) | ~300 K (0.24%) |
| Checkpoint size | ~500 MB | ~3 MB |
| Training speed | 1× (baseline) | ~3–5× faster |
| Quality (small datasets) | Risk overfitting | Regularised → often better |
| Quality (large datasets) | Slightly higher ceiling | Nearly equal |
| Deployment | Swap full model | Load base + tiny adapter |

---

## 3. QLoRA — Quantized LoRA

### Core idea
QLoRA = 4-bit base model + 16-bit LoRA adapters.

The frozen base model is loaded in NF4 (NormalFloat4) format, which cuts VRAM
by ~4×.  The LoRA adapter matrices (A and B) are kept in bfloat16 for precise
gradient computation.  On the forward pass, NF4 weights are dequantised to
bf16 for the matrix multiplication, then requantised.

### NF4 vs. INT4
- INT4 assigns 16 equally-spaced bins to the value range.
- NF4 assigns bins according to the Gaussian distribution of LLM weights:
  more bins near zero, fewer at the extremes.  This matches the empirical
  distribution of pretrained weights and reduces quantisation error.

### When QLoRA matters
- For GPT-2 (124 M params), 4-bit saves ~120 MB — modest.
- For LLaMA-7B (7B params), 4-bit brings 28 GB → ~5 GB — fits on a 8 GB GPU.
- For LLaMA-70B, 4-bit brings 280 GB → ~40 GB — fits on 2× A100 80 GB.

See `docs/qlora_notes.md` for failure modes.

---

## 4. Full Fine-Tuning

### When to use it
- Small base models (GPT-2, GPT-Neo, < 1 B params) where VRAM is not a concern.
- Datasets large enough that overfitting is unlikely (> 100 K examples).
- Tasks where the distribution shift is so large that low-rank adapters are
  insufficient (e.g., switching from English to a new language).

### Regularisation tips
Because all parameters are updated, small datasets risk overfitting:
- Use a low learning rate (5e-5 vs. LoRA's 5e-4).
- Add weight decay (L2, `weight_decay=0.01`).
- Add LR warmup (`warmup_ratio=0.03`) to prevent early instability.
- Monitor validation loss; stop if it diverges from training loss.

### Catastrophic forgetting
Full FT on a narrow dataset can cause the model to "forget" general
capabilities.  Strategies:
- **EWC (Elastic Weight Consolidation)**: penalise large changes to weights
  that were important for previous tasks (not implemented here).
- **Replay**: mix original pretraining data with fine-tuning data.
- **LoRA**: because it only changes a tiny fraction of weights, LoRA is
  naturally resistant to catastrophic forgetting.

---

## 5. Evaluation Design

### Why three metrics?

**Format compliance** (eval_format.py) measures surface-level instruction
following.  A high score means the model reproduces the correct structure
(bullets, sections, titles) from training demonstrations.  It does NOT measure
factual accuracy or helpfulness.

**Grounding metrics** (eval_grounding.py) measure whether the model uses the
provided context rather than generating from memory.  Citation presence (a
verbatim phrase from context appearing in the response) is a strong signal that
the model is reading the context passage.

**Combining both** (eval_summary.py) exposes the trade-off: a model can score
100% on format compliance by memorising templates, while completely ignoring
context.  A model with high citation scores is actually grounded.

### Why compare base vs. fine-tuned?
The *delta* between base GPT-2 and fine-tuned GPT-2 is the measurable effect
of fine-tuning.  A base model mostly generates fluent but irrelevant continuations
of the prompt.  A fine-tuned model should follow the instruction format and
extract answers from context.

---

## 6. Dataset Design

### Three-source mixture rationale

| Dataset | Purpose | What the model learns |
|---|---|---|
| ifeval-like | Format compliance | Bullet lists, sections, placeholders, titles |
| do-not-answer | Safety | Politely decline harmful questions |
| trivia_qa | Grounded QA | Extract answers from context passages |

Mixing datasets prevents the model from over-specialising on any single task
pattern.  The `--max_per_source` flag lets you balance the mixture.

### Chat template choice
```
<|user|>
{instruction}
<|assistant|>
{response}<eos>
```
This template is:
- Simple — only two roles, no system prompt confusion.
- Unambiguous — `<|assistant|>` is not a natural English phrase, so it won't
  appear accidentally in content, preventing false-positive masking.
- GPT-2 compatible — injected at runtime; not baked into the tokenizer.

---

## 7. Interview Q&A

**Q: What is the difference between SFT and RLHF?**
A: SFT is behaviour cloning — the model learns to reproduce demonstrations.
RLHF adds a reward model that scores outputs, then uses RL to maximise reward.
SFT is great for format and style; RLHF is needed for preference trade-offs
(e.g., "more helpful" vs. "more concise").

**Q: Why does LoRA work so well despite updating < 1% of parameters?**
A: Pre-trained LLMs occupy a low intrinsic dimension — the task-specific
adaptation directions live in a small subspace of weight space.  Low-rank
updates are sufficient to capture those directions.  Additionally, the
implicit regularisation from the low-rank constraint helps on small datasets.

**Q: What is the purpose of the `alpha / r` scaling in LoRA?**
A: It normalises the adapter output so that changing `r` doesn't change the
effective learning rate.  Setting `alpha = r` keeps the scale constant; setting
`alpha = 2r` amplifies the adapter's effect.  Higher `alpha` → stronger
adaptation → more risk of overfitting.

**Q: Why is bf16 preferred over fp16 for QLoRA?**
A: bf16 has the same exponent range as fp32 (8-bit exponent) but only 7 bits
of mantissa vs. fp32's 23.  This means it can represent very large and very
small values without overflow or underflow.  fp16 has only a 5-bit exponent
and is prone to NaN/inf spikes when activations or gradients are large.

**Q: What is `completion_only_loss` doing at the code level?**
A: TRL tokenises the full conversation, then scans for the assistant header
token(s) (`<|assistant|>\n`).  All token positions *before* the first assistant
header get `labels = -100`, which PyTorch's CrossEntropyLoss ignores.  Only
assistant tokens contribute to the gradient.

**Q: What is catastrophic forgetting and how does LoRA mitigate it?**
A: Catastrophic forgetting is when fine-tuning on a narrow task degrades
general capabilities (e.g., the model forgets how to speak English after
fine-tuning on Python code).  LoRA mitigates this because the frozen base
weights are never modified — only the tiny adapter matrices change.  The base
model's general knowledge is preserved exactly.

**Q: How would you scale this to LLaMA-7B?**
A: Use QLoRA: load the 7B model in 4-bit NF4 (fits in ~5 GB VRAM), add 16-bit
LoRA adapters targeting `q_proj` and `v_proj` (LLaMA attention modules), and
use `paged_adamw_8bit` to keep optimizer states in paged CPU memory.  The
training code is identical to our GPT-2 QLoRA script — only the model name and
`target_modules` need changing.

**Q: What does citation presence measure and why is it a proxy?**
A: Citation presence checks whether a verbatim ≥10-char phrase from the context
appears in the response.  It's a proxy for context-grounded reasoning because
models that read and copy from the context will naturally reproduce phrases from
it, while models that generate from parametric memory won't.  It's imperfect —
a model could accidentally copy phrases without understanding them.

**Q: What is the intrinsic dimension hypothesis?**
A: Aghajanyan et al. (2020) showed that LLMs can be fine-tuned effectively by
updating only a very small number of parameters (the "intrinsic dimension" of
the task).  For most NLP tasks, this is well below 1% of model parameters.
LoRA operationalises this insight by explicitly constraining updates to a
low-rank subspace.

---

## 8. Key Commands

```bash
# Activate environment
conda activate llm-learning
cd Week3/fine_tuning_project

# Smoke test all variants (~2 min)
bash scripts/run_all_sft.sh --smoke

# Full run
bash scripts/run_all_sft.sh

# Individual training
python src/train_sft.py --data_dir data --output_dir results/sft-gpt2 --epochs 1
python src/train_lora_sft.py --config configs/lora_r8.yaml
python src/train_qlora_sft.py --compute_dtype bf16 --gradient_checkpointing
python src/train_fullft.py --data_dir data --output_dir results/fullft-gpt2

# Individual evals
python src/eval_format.py --model_dir results/sft-gpt2 --data_dir data
python src/eval_grounding.py --ft_model_dir results/sft-gpt2 --data_dir data
python src/eval_summary.py --model_dirs results/sft-gpt2,results/lora-r8-alpha16
```

---

## References

- [QLoRA paper](https://arxiv.org/abs/2305.14314) — Dettmers et al. 2023
- [LoRA paper](https://arxiv.org/abs/2106.09685) — Hu et al. 2022
- [Intrinsic dimension](https://arxiv.org/abs/2012.13255) — Aghajanyan et al. 2021
- [TRL documentation](https://huggingface.co/docs/trl) — SFTTrainer, SFTConfig
- [PEFT documentation](https://huggingface.co/docs/peft) — LoraConfig, get_peft_model
