# Fine-Tuning Project — Week 3

A hands-on exploration of LLM fine-tuning methods: SFT, LoRA, QLoRA, and full
fine-tuning, all applied to GPT-2 with identical datasets, chat templates, and
assistant-token masking for fair comparison.

---

## Project Structure

```
fine_tuning_project/
├── configs/
│   ├── base.yaml              # Base experiment configuration
│   ├── lora_r4.yaml           # LoRA rank=4, alpha=8
│   ├── lora_r8.yaml           # LoRA rank=8, alpha=16
│   └── lora_r16.yaml          # LoRA rank=16, alpha=32
├── data/
│   ├── ifeval-like-data-subset/   # Format-constrained instructions
│   ├── do-not-answer/             # Safety refusal dataset
│   └── trivia_qa-subset/          # Grounded QA pairs
├── docs/
│   ├── qlora_notes.md         # QLoRA failure modes & debugging
│   └── study_guide.md         # Design decisions, interview Q&A
├── scripts/
│   └── run_all_sft.sh         # Run all training + eval in sequence
├── src/
│   ├── train_sft.py           # Day 2: Full SFT trainer
│   ├── train_lora_sft.py      # Day 3: LoRA SFT trainer
│   ├── train_qlora_sft.py     # Day 4: QLoRA SFT trainer
│   ├── train_fullft.py        # Day 5: Full fine-tune (short run)
│   ├── eval_format.py         # Day 6: Format constraint checker
│   ├── eval_grounding.py      # Day 6: Grounded QA evaluation
│   └── eval_summary.py        # Day 6: Combined eval summary
└── results/                   # Outputs, checkpoints (gitignored)
```

---

## SFT vs LoRA vs QLoRA: Comparison

| Aspect            | Full SFT        | LoRA             | QLoRA            | Full FT (short)  |
|-------------------|-----------------|------------------|------------------|-------------------|
| **Trainable %**   | 100%            | ~0.5–2%          | ~0.5–2%          | 100%              |
| **VRAM (GPT-2)**  | ~1.5 GB         | ~0.8 GB          | ~0.5 GB          | ~1.5 GB           |
| **Speed**         | Baseline        | 2–3× faster      | 2–4× faster      | Baseline          |
| **Quality**       | Best (if enough data) | Very close  | Slightly lower   | Limited (few steps) |
| **Checkpoint**    | Full model      | Adapters only    | Adapters only    | Full model        |
| **Deployment**    | Replace model   | Base + adapter   | Base + adapter   | Replace model     |
| **Risk**          | Catastrophic forgetting | Low     | NaN/divergence   | Minimal (few steps) |

### When to use what

- **Full SFT**: You have enough data and compute, want maximum quality
- **LoRA**: Production default — 95% of quality at 5% of cost
- **QLoRA**: When GPU memory is the bottleneck (e.g., 7B+ models on consumer GPUs)
- **Full FT (short)**: Baseline comparison, quick sanity checks

---

## What Changes in the Model

### Full SFT / Full FT
Every weight in the model is updated. The model learns new behaviors but risks
**catastrophic forgetting** — losing capabilities it had before fine-tuning.
The entire model must be saved and deployed.

### LoRA (Low-Rank Adaptation)
The pretrained weights are **frozen**. Small trainable matrices (A, B) are
injected into attention layers:

```
W_new = W_frozen + (B @ A) * (alpha / r)
```

- `r` (rank): Controls adapter capacity (4, 8, 16...)
- `alpha`: Scaling factor (typically 2×r)
- Only A and B are trained (< 1% of total parameters)
- At inference, adapters can be merged into the base model (zero overhead)

### QLoRA (Quantized LoRA)
Same as LoRA, but the **base model is loaded in 4-bit NF4 format**:
- Base weights: 4-bit quantized (NormalFloat4)
- LoRA adapters: 16-bit (full precision gradients)
- Activations: dequantized on-the-fly to compute dtype (bf16/fp16)
- Cuts VRAM by ~4× compared to full precision

---

## How to Reproduce

### Prerequisites

```bash
conda activate llm-learning
# Required: transformers, trl, peft, bitsandbytes, datasets, pandas, pyyaml
```

### Day 2: SFT Training

```bash
cd src/
python train_sft.py \
    --data_dir ../data \
    --output_dir ../results/sft-gpt2 \
    --max_samples 200 \
    --epochs 1 \
    --batch_size 4
```

### Day 3: LoRA Training (sweep)

```bash
# Using YAML configs (recommended):
python src/train_lora_sft.py --config configs/lora_r4.yaml
python src/train_lora_sft.py --config configs/lora_r8.yaml
python src/train_lora_sft.py --config configs/lora_r16.yaml

# Or direct CLI:
python src/train_lora_sft.py \
    --data_dir data --lora_r 8 --lora_alpha 16 \
    --output_dir results/lora-r8 --max_samples 200
```

### Day 4: QLoRA Training

```bash
python src/train_qlora_sft.py \
    --data_dir data \
    --output_dir results/qlora-gpt2 \
    --max_samples 200 \
    --compute_dtype bf16 \
    --gradient_checkpointing \
    --use_paged_optimizer
```

### Day 5: Full Fine-Tune (short)

```bash
python src/train_fullft.py \
    --data_dir data \
    --output_dir results/fullft-gpt2 \
    --max_steps 50 \
    --max_samples 200
```

### Day 6: Evaluation

```bash
# Format evaluation (ifeval-like constraints):
python src/eval_format.py --model_dir results/sft-gpt2 --data_dir data

# Gold response sanity check (should score ~100%):
python src/eval_format.py --use_gold_responses --data_dir data

# Grounded QA evaluation (trivia_qa):
python src/eval_grounding.py --model_dir results/sft-gpt2 --data_dir data

# Compare base vs fine-tuned:
python src/eval_grounding.py --model_dir gpt2 --data_dir data

# Combined summary across all methods:
python src/eval_summary.py --data_dir data \
    --model_dirs results/sft-gpt2 results/lora-r8 results/qlora-gpt2 results/fullft-gpt2
```

### Day 7: Run Everything

```bash
bash scripts/run_all_sft.sh
```

---

## Interview Q&A

### Day 2: SFT Basics

**Q: What is SFT and why is it called "behavior cloning"?**
A: SFT trains a model to imitate demonstration data. Like behavior cloning in
robotics, it copies the *surface patterns* of expert behavior without
understanding underlying preferences. Great for format/style, weak for
preference trade-offs.

**Q: Why mask loss on user/system tokens?**
A: The user's question is *given* — there's nothing to learn from predicting
deterministic prompt text. Masking focuses every gradient step on what a good
assistant response looks like, improving sample efficiency.

### Day 3: LoRA

**Q: How does LoRA reduce trainable parameters?**
A: Instead of updating the full weight matrix W (d×d), LoRA trains two small
matrices A (d×r) and B (r×d) where r << d. The update is W + BA*(alpha/r).
For GPT-2 with r=8, this is < 1% of total parameters.

**Q: What happens when you increase rank r?**
A: Higher rank = more adapter capacity = closer to full fine-tuning quality,
but more parameters, slower training, larger checkpoints. Diminishing returns
above r=16 for most tasks.

### Day 4: QLoRA

**Q: How does 4-bit quantization work with training?**
A: Base weights are stored in NF4 (4-bit NormalFloat) but dequantized to
bf16/fp16 for computation. Only the LoRA adapters receive gradients in 16-bit.
This gives ~4× memory savings with minimal quality loss.

**Q: What are common QLoRA failure modes?**
A: NaN loss (bf16 overflow on non-Ampere GPUs — use fp16), loss divergence
(lower the learning rate), degraded long-context quality (quantization
introduces small errors that accumulate). See `docs/qlora_notes.md`.

### Day 5: Full FT

**Q: Why run a short full fine-tune?**
A: Fair comparison. By using the *exact same* dataset, template, and masking,
we isolate the effect of the training method. Short runs show how quickly each
method begins to learn.

### Day 6: Evaluation

**Q: What does "grounding" mean in QA evaluation?**
A: A grounded model answers from provided context (like RAG), not from
parametric memory. We measure this with exact match, substring match, and
citation presence — whether the answer appears in the given context.

**Q: Why separate format eval from grounding eval?**
A: They measure different capabilities. A model can follow format constraints
perfectly (SFT strength) while still hallucinating answers (SFT weakness).
Separating them reveals the true capability profile.

---

## Expected Results

With GPT-2 and 200 training examples:

- **Base GPT-2**: ~5–15% format accuracy, ~2–5% exact match on QA
- **SFT (1 epoch)**: ~30–50% format accuracy, ~10–20% exact match
- **LoRA r=8**: ~25–45% format accuracy, similar to SFT
- **QLoRA**: ~20–40% format accuracy (slightly lower due to quantization noise)
- **Full FT (50 steps)**: ~15–25% format accuracy (not enough steps to converge)

Note: GPT-2 is a small model — these numbers illustrate *relative* differences
between methods, not production-quality results.

---

## Key Intuitions

1. **SFT = behavior cloning**: Copies surface patterns, not preferences
2. **LoRA = low-rank updates**: The "intrinsic dimension" of fine-tuning is small
3. **QLoRA = quantized base + precise adapters**: Memory savings with minimal quality cost
4. **Masking matters**: Training on the right tokens is as important as training on the right data
5. **Fair comparison requires identical setup**: Same data, template, masking, tokenizer

---

## Further Reading

- [LoRA paper](https://arxiv.org/abs/2106.09685) — Hu et al. 2022
- [QLoRA paper](https://arxiv.org/abs/2305.14314) — Dettmers et al. 2023
- [TRL documentation](https://huggingface.co/docs/trl)
- [PEFT documentation](https://huggingface.co/docs/peft)
- `docs/study_guide.md` — Detailed study guide with interview prep
- `docs/qlora_notes.md` — QLoRA failure modes and debugging
