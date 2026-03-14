# VLM Project — Week 5

Hands-on exploration of Vision-Language Models: contrastive learning,
ViT attention mechanics, VLM fusion strategies, grounding failures, and
frozen vs fine-tuned encoder comparison.

---

## Project Structure

```
vlm_project/
├── configs/
│   ├── clip.yaml              # Day 1: Mini-CLIP hyperparameters
│   └── vlm.yaml               # Day 3: VLM training hyperparameters
├── docs/
│   ├── study_guide.md         # Full concept walkthrough + 20 interview Q&As
│   └── grounding_failures_analysis.md  # Deep analysis of VLM hallucination
├── scripts/
│   └── run_all.sh             # End-to-end pipeline (add --smoke for quick test)
├── src/
│   ├── utils.py               # Shared: models, datasets, checkpoints, helpers
│   ├── day1_clip.py           # Day 1: Mini-CLIP contrastive training
│   ├── day2_vit.py            # Day 2: ViT attention maps + patch shuffle
│   ├── day3_fusion.py         # Day 3: Prefix-concat vs cross-attention VLM
│   ├── day4_failures.py       # Day 4: 30-prompt grounding failure analysis
│   ├── day5_compare.py        # Day 5: Frozen vs fine-tuned vision encoder
│   └── day67_vlm.py           # Days 6-7: Uncertainty + full adversarial suite
├── results/                   # Auto-created; holds plots, checkpoints, logs
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
cd Week5/vlm_project

# Smoke-test: all scripts with minimal data (~5–10 min total)
bash scripts/run_all.sh --smoke

# Full pipeline (~1–2 hours on TITAN RTX 24GB)
bash scripts/run_all.sh
```

Or run individual days:

```bash
# Day 1: Mini-CLIP (infonce loss, projections, similarity matrix)
python src/day1_clip.py --max_samples 500 --epochs 2

# Day 2: ViT attention visualisation
python src/day2_vit.py

# Day 3: VLM training (both fusion modes)
python src/day3_fusion.py --max_samples 500 --max_steps 100

# Day 4: Grounding failure analysis (requires Day 3 checkpoint)
python src/day4_failures.py --checkpoint results/vlm-prefix/checkpoint.pt

# Day 5: Frozen vs fine-tuned comparison
python src/day5_compare.py --max_steps 150

# Days 6-7: Full eval with uncertainty (requires Day 3 or Day 5 checkpoint)
python src/day67_vlm.py --n_mc_samples 4
```

---

## Architecture Summary

### Mini-CLIP (Day 1)

```
Image  → CLIP ViT-B/32 (frozen) → 512-dim → Linear(512, 256) → L2-norm
Text   → CLIP text encoder (frozen) → 512-dim → Linear(512, 256) → L2-norm
Loss   = symmetric InfoNCE with learnable temperature τ
```

Training only the projection heads and temperature. The CLIP backbone
learns nothing new; we learn the best projection of its features into a
256-dim alignment space.

### MiniVLM (Days 3-7)

**Prefix-Concat mode:**
```
Image → CLIP ViT-B/32 → last_hidden_state[:, :4] → Linear(768, 768)
     → 4 prefix tokens → cat([vis_prefix, text_embeds]) → GPT-2
```

**Cross-Attention mode (Flamingo-inspired):**
```
Image → CLIP ViT-B/32 → 4 patch tokens → Linear(768, 768)
     → encoder_hidden_states → GPT-2 (add_cross_attention=True)
```

Both modes are trained on image captioning:
```
Input:  [vis_tokens] "Question: What is in this image? Answer:"
Target: "a photo of a {class_name}"
```

---

## Datasets

| Dataset | Description | How used |
|---------|-------------|----------|
| `zh-plus/tiny-imagenet` | 64×64 images, 200 ImageNet classes | All training |
| `cifar10` | 32×32 images, 10 classes | Fallback if tiny-imagenet unavailable |
| Synthetic PIL images | Generated on-the-fly | Spatial / counting tests (Days 4, 6-7) |

Caption template: `"a photo of a {class_name}"`

---

## Results Summary

After full training you'll find:

| Script | Key outputs |
|--------|-------------|
| Day 1 | `results/day1/training_curves.png`, `similarity_matrix.png`, `mini_clip.pt` |
| Day 2 | `results/day2/image_*/attention_layers.png`, `patch_shuffle_similarity.png` |
| Day 3 | `results/vlm-prefix/checkpoint.pt`, `vlm-cross_attention/checkpoint.pt`, `fusion_comparison.png` |
| Day 4 | `results/day4/failures.json`, `failure_rate.png`, `analysis.txt` |
| Day 5 | `results/comparison.png`, `vlm-frozen/checkpoint.pt`, `vlm-finetuned/checkpoint.pt` |
| Days 6-7 | `results/day67/eval_results.json`, `summary.png`, `attention_maps/` |

---

## Key Takeaways

### CLIP Alignment ≠ Spatial Grounding
CLIP matches whole images to whole captions. It does not know *where* objects
are, *how many* there are, or their spatial relationships. This is the root
cause of most VLM grounding failures.

### Patch Tokens ≠ Word Tokens
Word tokens carry discrete symbolic meaning. Patch tokens carry local texture
whose meaning depends on spatial position. Shuffling patches destroys CLIP's
performance because layout is implicit in CLIP ViT but never explicitly
supervised.

### Frozen Encoder Is Usually Better (for small data)
Fine-tuning the vision encoder risks catastrophic forgetting of CLIP's
broad visual representations. Frozen + trained projection + trained LLM is
the production-proven approach (LLaVA, InstructBLIP).

### Hallucination Is Structural, Not Accidental
Multimodal hallucination emerges naturally from: (a) LLM's strong language
prior, (b) weak spatial visual signal, (c) no negative-object training signal.
Fixing it requires better visual representations and explicit alignment
training, not just scaling.

---

## Dependencies

```bash
pip install -r requirements.txt
```

All packages are already in the `llm-learning` conda environment.
