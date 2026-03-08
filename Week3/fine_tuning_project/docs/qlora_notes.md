# QLoRA: Failure Modes, Debugging, and Design Decisions

QLoRA (Quantized LoRA) is remarkably robust in practice, but specific
configurations can trigger silent or loud failures.  This document catalogs
the most common failure modes and their fixes.

---

## 1. Loss Divergence

**Symptom:** Training loss suddenly spikes to `inf` or `nan` and never recovers,
typically within the first 50–200 steps.

**Root cause:** fp16 arithmetic has a narrow dynamic range (~6 × 10⁻⁵ to ~6.5 × 10⁴).
The dequantization step in 4-bit forward passes can produce intermediate values
outside this range, especially when activations are large.

**Fixes (in order of preference):**
1. Switch to `--compute_dtype bf16`.  bfloat16 has the same exponent range as
   fp32 (8 bits) and is far less likely to overflow.  Requires Ampere+ GPU
   (RTX 30xx, A100, TITAN RTX).
2. Lower the learning rate.  QLoRA recommends `2e-4` vs. LoRA's `5e-4`.
   Large gradient steps amplify quantisation noise.
3. Add a gradient clipping cap: `max_grad_norm=1.0` in SFTConfig (default in TRL).
4. Use `bnb_4bit_use_double_quant=True` (already on by default in our config).
   Double quantisation reduces systematic error in the quantisation constants.

---

## 2. NaN Losses (Silent Failure)

**Symptom:** Loss prints `nan` from step 1; model never learns.

**Root cause #1:** Layer norms in the base model are in fp16 and accumulate NaN.
`prepare_model_for_kbit_training()` casts layer norms to fp32 automatically.
Skipping this call causes silent NaN.

**Root cause #2:** The tokenizer's `pad_token` is not set.  GPT-2 has no `<pad>`;
if `pad_token` is `None` the attention mask is wrong and loss is computed on
garbage positions.  Always set `tokenizer.pad_token = tokenizer.eos_token`.

**Fix:**
```python
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
```

---

## 3. Degraded Long-Context Quality

**Symptom:** Short outputs look reasonable; outputs > ~200 tokens drift off-topic
or become incoherent faster than fp16 models.

**Root cause:** NF4 quantisation introduces small systematic errors in every
attention weight.  These errors are negligible for short sequences but compound
over long autoregressive chains.  The error accumulation is proportional to
sequence length × model depth.

**Mitigations:**
- Enable `bnb_4bit_use_double_quant=True` (reduces per-weight error by ~0.5 bits).
- Use `nf4` rather than `int4` quantisation type; NF4 assumes Gaussian weight
  distribution (valid for most LLMs) and assigns more bins near zero.
- Keep context length ≤ 512 tokens for GPT-2; for large models (7B+) QLoRA
  handles 2048+ tokens well because the quantisation error per layer is
  smaller relative to the signal.
- At inference, decode with temperature 0 (greedy) or low temperature — sampling
  amplifies the noise from quantisation errors.

---

## 4. Gradient Instability with Gradient Checkpointing

**Symptom:** Training hangs or crashes with:
```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

**Root cause:** Gradient checkpointing recomputes activations on the backward pass.
For this to work, the input to each layer must have `requires_grad=True`.  With
4-bit frozen base weights, inputs don't naturally require grad — the gradient
must be "injected" via `model.enable_input_require_grads()`.

**Fix:**  Always call `prepare_model_for_kbit_training(use_gradient_checkpointing=True)`
before applying the LoRA config.  This function calls `enable_input_require_grads()`
internally.

Also disable the KV-cache when using gradient checkpointing (they are incompatible):
```python
model.config.use_cache = False
```

---

## 5. OOM with Paged Optimizer

**Symptom:** Training crashes with CUDA OOM even though VRAM seemed sufficient.

**Root cause (A — optimizer states):** The default AdamW stores two fp32
momentum tensors per parameter.  For 7B models this alone requires ~56 GB.
`paged_adamw_8bit` stores optimizer states in 8-bit and pages them to CPU
memory, dramatically reducing the VRAM footprint.

**Root cause (B — activation peaks):** Even with paged optimizer, activation
peaks during the forward pass can OOM.  Gradient checkpointing trades VRAM for
compute by recomputing activations.  Use both together for the lowest VRAM:
```bash
python src/train_qlora_sft.py \
    --gradient_checkpointing \
    --use_paged_optimizer \
    --batch_size 1 \
    --grad_accum 16
```

**Note for GPT-2:** The paged optimizer has negligible benefit for a 124 M
parameter model (optimizer states are ~500 MB).  It matters for 7B+ models.

---

## 6. LoRA Adapter Not Applied to All Layers

**Symptom:** Training loss is unexpectedly high and doesn't decrease; VRAM usage
is suspiciously low; `trainable_params` printed by `get_nb_trainable_parameters()`
is near zero.

**Root cause:** `target_modules` in the LoraConfig must exactly match the module
names in the model.  GPT-2 uses `c_attn` and `c_proj`; other models use
`q_proj`, `v_proj`, `k_proj`, `o_proj` (LLaMA family).

**Fix:**
```python
# Find the correct module names for your model:
for name, module in model.named_modules():
    print(name)
# Then set target_modules accordingly.
```

For GPT-2: `target_modules=["c_attn", "c_proj"]`
For LLaMA: `target_modules=["q_proj", "v_proj"]` (minimum) or all attention projections.

---

## Summary Table

| Failure Mode | Symptom | Quick Fix |
|---|---|---|
| Loss divergence | Loss → inf/nan, late in training | Use bf16 compute dtype; lower LR |
| NaN from step 1 | Loss = nan immediately | Call `prepare_model_for_kbit_training`; set pad token |
| Long-context drift | Good short outputs, bad long outputs | Enable double_quant; reduce temperature |
| Gradient checkpointing crash | RuntimeError no grad_fn | `prepare_model_for_kbit_training(use_gradient_checkpointing=True)` |
| OOM with large model | CUDA OOM even with 4-bit | Add `--use_paged_optimizer --gradient_checkpointing` |
| No learning | Trainable params ≈ 0 | Fix `target_modules` to match model architecture |

---

## QLoRA vs. LoRA vs. Full FT Decision Guide

```
Need lowest VRAM?         → QLoRA  (4-bit base; fits 7B on 16 GB)
Need fastest training?    → LoRA   (16-bit; no dequant overhead)
Need highest quality?     → Full FT (all params; needs most VRAM)
Production / deployment?  → LoRA adapters are tiny (< 50 MB vs. full model)
Experimenting / research? → QLoRA lets you run bigger base models on consumer GPUs
```

---

## References

- Dettmers et al. (2023) "QLoRA: Efficient Finetuning of Quantized LLMs"
  https://arxiv.org/abs/2305.14314
- Hu et al. (2022) "LoRA: Low-Rank Adaptation of Large Language Models"
  https://arxiv.org/abs/2106.09685
- bitsandbytes documentation: https://huggingface.co/docs/bitsandbytes
