# Week 6 Study Guide: Deeper Multimodal Reasoning

## Overview

This week goes from "it works" to "I understand why — and why it breaks."
Each day builds on a specific architectural or training concept, backed by
experiments that give real, interpretable numbers.

---

## Day 1: Video-Language Modeling

### Core Concept: Temporal Tokens

A standard ViT processes one image at a time. For video, you need to process
**sequences of frames** and understand how they relate in time.

**VideoMAE** (Video Masked Autoencoders) extends image MAE to video:

```
Single image ViT:
  Image (H×W×3) → patches → [CLS, p1, p2, ..., p196] → embedding

VideoMAE:
  Video (T×H×W×3) → tubelets (2×16×16 cubes) → [t1p1, t1p2, ..., t8p196]
  Each token encodes WHERE (h,w) AND WHEN (t) the patch is.
  Position embedding: (t, h, w) jointly.
  Masking: 90% of tubes masked during training → forces temporal reasoning.
```

### Frame Sampling Tradeoffs

| Strategy | Cost | Benefit | Use case |
|----------|------|---------|----------|
| Uniform (1 of N) | 1× | Simple | Static scenes |
| Dense (all N) | N× | Most info | Fast action |
| Sparse keyframes | low | Efficient | Slow events |
| Learned sampling | variable | Adaptive | General |

### Why Naive Frame Stacking Fails

```
Naive: [frame1_patches | frame2_patches | ... | frame16_patches]
          ↓196 tokens       ↓196 tokens           ↓196 tokens
  Total: 3136 tokens → exceeds GPT-2's 1024-token window after just 5 frames
  + No temporal order: shuffling frames gives identical attention
```

**Fix**: VideoMAE's tubelet positional embedding encodes (t, h, w) jointly.
**Alternative**: Temporal attention (Q-Former style) → compress T frames to Q tokens.

### Experiment: What You Learned

CLIP mean pooling of N frames = test-time ensembling.
Each augmented frame adds noise; averaging over N frames reduces variance by √N.
This gives a free accuracy boost without training.

---

## Day 2: Token Budget & Representation Collapse

### The Token Budget Problem

```
Attention cost per layer: O((V + T)²)
  V = visual tokens, T = text tokens

Model          V     T      (V+T)²      relative cost
─────────────────────────────────────────────────────
CLIP+GPT-2     4     256    ≈65K        1×
BLIP-2         32    256    ≈83K        1.3×
LLaVA-1.5     576    256    ≈692K       10.6×
Naive 16-frame 784   256    ≈1.1M       17×
```

**Key insight**: image tokens grow with resolution², but the LLM context is fixed.

### BLIP-2's Q-Former Solution

```
Input image (224×224) → ViT-g/14 → 257 tokens (B, 257, 1408)  ← 1 CLS + 16×16 patches
                                         ↓
                                   Q-Former (32 learned queries)
                                   Self-attn: queries communicate
                                   Cross-attn: queries read patches
                                         ↓
                           Output: (B, 32, 768)  ← ALWAYS 32 tokens
                                         ↓
                               Linear projection → LLM
```

**Why Q-Former is resolution-invariant**: no matter if input is 224×224 or 1024×1024,
Q-Former outputs exactly 32 query tokens. Only the ViT forward pass gets more expensive.

### Representation Collapse

When you prune too aggressively (e.g., keep 4 of 49 patches):
- You retain **global semantics** ("dog is present")
- You lose **spatial structure** (dog is in the top-right corner)
- You lose **fine-grained detail** (breed, exact pose)

The inflection point (from our experiments): losing below ~16 patches drops accuracy sharply.

---

## Day 3: Instruction Tuning for VLMs

### LLaVA Architecture

```
         ┌──────────────┐
Image ─→ │ CLIP ViT-L/14 │ (frozen)
         └──────┬───────┘
                │ 14×14 = 196 patch embeddings
                ↓
         ┌──────────────┐
         │  MLP (2-layer)│ (trainable) ← the "connector"
         └──────┬───────┘
                │ 576 visual tokens (at 336px)
                ↓
         ┌──────────────┐
Text ──→ │   LLaMA-2    │ (LoRA fine-tuned)
         └──────────────┘
```

Total trainable params: ~7M (connector) + LoRA on LLaMA-2.

### GPT-4 Generated Supervision

LLaVA authors fed image captions + bounding box metadata to GPT-4 text-only:

```
Input to GPT-4:
  "The image contains: a dog running on grass, a red ball (top-left corner)."

GPT-4 generates:
  Conversation: "Where is the ball? → In the top-left corner, the dog is chasing a red ball."
  Detail: "This image shows a golden retriever mid-stride in a lush green field..."
  Reasoning: "Why might the dog be running? → Dogs instinctively chase thrown objects..."
```

150K training pairs generated this way for near-zero cost.

### What Changes With Instruction Tuning

| Without (BLIP-2 base) | With (InstructBLIP) |
|------------------------|---------------------|
| "a dog" | "This image shows a golden retriever running in a field." |
| "2" | "I can count 2 dogs in the image; they appear to be playing." |
| Ignores instruction format | Follows instruction format |
| Doesn't do multi-turn | Can handle follow-up questions |

**Important caveat**: instruction tuning improves FORMAT, not factual accuracy.
Hallucination rates may even increase (model says more → more opportunities to be wrong).

### InstructBLIP vs LLaVA: Where the Instruction Goes

```
LLaVA:          instruction → LLM only
  Visual tokens from CLIP → connector → LLM ← instruction

InstructBLIP:   instruction → Q-Former (visual extraction stage)
  Image patches ─cross-attn─→ Q-Former ← instruction (self-attn)
  The Q-Former extracts INSTRUCTION-RELEVANT visual features.
  → Better grounding for task-specific queries.
```

---

## Day 4: Why VLMs Hallucinate Spatial Facts

### Root Cause 1: CLIP Alignment Is Global

CLIP loss: `similarity([CLS]_image, [EOS]_text)` for matching pairs.

```
Image CLS embedding ──→ encodes "dog is present" but NOT "dog is at x=0.3, y=0.7"
Caption EOS embedding ─→ encodes "golden retriever" but NOT spatial position

InfoNCE says: make these closer than the 511 non-matching captions.
Nothing in this loss requires spatial localization.
```

### Root Cause 2: No Hard Spatial Constraints on Patches

ViT patches have sinusoidal position embeddings. After 12 attention layers,
each patch embedding is a MIXTURE of all patches via global attention.

When Q-Former cross-attends to these mixed embeddings, queries learn
"red and blue shapes are present" but not "red is LEFT of blue."

```
Spatial test: [RED LEFT | BLUE RIGHT]
Q-Former query for "left color": attends to... all patches globally.
No mechanism forces it to weight left patches more than right patches.
→ Falls back to LLM prior: "What color is commonly described as 'left'?"
```

### Root Cause 3: LLM Prior Dominates

Experiment: ask Flan-T5 (NO image) the same questions.
The LLM has strong priors from training text:
- "How many circles?" → LLM expects "1", "2", or "3" (most common in web text)
- "Is there a unicorn?" → "no" (fantastical creatures don't exist in normal photos)

When visual evidence is weak (and it's always weak in VLMs), these priors win.

### Root Cause 4: Visual Signal Enters LLM Only Once

```
LLM (40 layers):
  Layer 0:  [vis_token_1, ..., vis_token_32, text_token_1, ...]
             ↑ visual information enters here, once
  Layer 1:  [mixed representations]
  Layer 2:  [even more mixed, more LLM-dominated]
  ...
  Layer 39: [output tokens, heavily language-prior-dominated]
```

By the output layer, 39 layers of causal language modeling have diluted the
32 visual tokens with text context. Deep LLMs are effectively text models that
receive a short visual "hint."

### Autonomous Driving Analogy

| VLM failure | AV equivalent |
|-------------|---------------|
| Global CLIP alignment | Camera + LiDAR late fusion without calibration |
| No spatial constraints on patches | Uncalibrated extrinsic parameters |
| LLM prior overrides visual | Prediction head ignoring sensor disagreement |
| Shallow visual entry into LLM | Single-frame perception without temporal context |

---

## Day 5: Architecture Comparison

### Q-Former Deep Dive

```python
# Q-Former forward pass (conceptually):
class QFormerLayer:
    def forward(queries, image_patches):
        # Step 1: Queries learn from each other
        queries = self_attention(queries, queries, queries)
        queries = layernorm(queries)

        # Step 2: Queries extract info from image patches
        queries = cross_attention(queries, image_patches, image_patches)
        queries = layernorm(queries)

        # Step 3: FFN
        queries = ffn(queries)
        queries = layernorm(queries)
        return queries

# After 6 layers:
# Each query has attended to ALL patches 6 times,
# and to all OTHER queries 6 times.
# Queries specialize: some focus on object categories, some on spatial layout.
```

### Why Q-Former Needs 2-Stage Training

**Stage 1 (representation learning)**: Train Q-Former + ViT, LLM frozen.
- Goal: make Q-Former output meaningful visual representations
- Without this stage: Q-Former outputs random noise → LLM gets garbage → no gradient signal

**Stage 2 (generative pre-training)**: Add LLM, train Q-Former projection + LoRA.
- Goal: align Q-Former's visual representations with LLM's token space
- This stage requires Stage 1 to have succeeded first

Skipping Stage 1 is why naive "CLIP features → linear → LLM" needs more data to converge.

### Architecture Table Summary

| Model | Vis tokens | LLM frozen | Key strength | Key weakness |
|-------|-----------|-----------|-------------|--------------|
| CLIP+GPT-2 prefix | 4 | No | Simplest, fastest | Very low capacity |
| CLIP+GPT-2 cross-attn | 49 | No | More spatial info | Context pressure |
| Mini Q-Former (ours) | 32 | No | Adaptive compression | Harder to optimize |
| BLIP-2 | 32 | Yes | Modular, efficient | No end-to-end optimization |
| LLaVA-1.5 | 576 | No | Best spatial detail | High compute, needs large context |

---

## Days 6–7: Final Project v2

### Uncertainty Estimation via Temperature Sampling

```python
# MC Dropout alternative for VLMs (which don't use dropout at inference):
# Run N stochastic forward passes with temperature T > 1

for _ in range(N_SAMPLES):
    output = model.generate(
        image + question,
        do_sample=True,
        temperature=0.8,
        output_scores=True,   # get per-token logits
    )
    entropy = -sum(p * log(p) for p in softmax(output.scores))

mean_entropy = mean(entropy over samples and token positions)
# High mean_entropy → model is uncertain → abstain
```

### Confidence Thresholding Tradeoffs

```
τ = 0.0 (no threshold): 100% coverage, baseline accuracy
τ = 0.5:               ~80% coverage, higher accuracy (abstain on confused answers)
τ = 1.0:               ~60% coverage, highest accuracy on answered subset
τ = ∞:                  0% coverage (abstain on everything)
```

The optimal τ depends on the cost of:
- False positive (wrong answer accepted): HIGH cost in safety-critical settings
- False negative (correct answer rejected = unnecessary abstention): lower cost

### Heuristic Failure Detection

| Heuristic | Detects | Precision | Recall |
|-----------|---------|-----------|--------|
| Repetition ratio > 0.4 | Model degeneration | Medium | Low |
| Content words < 3 | Uninformative answer | Low | Medium |
| Contains "yes" AND "no" | Model contradiction | High | Very Low |
| Entropy > τ | General uncertainty | Tunable | Tunable |

The heuristics have low F1 individually but can be combined as an ensemble.
Entropy-based thresholding is generally the best single signal.

---

## Interview Q&A Pairs

**Q1: What is a tubelet embedding in VideoMAE, and why is it better than stacking frame patches?**

A: A tubelet is a spatiotemporal patch that spans 2 consecutive frames and a 16×16 spatial region. The tubelet embedding encodes (time, height, width) position jointly, giving the model explicit temporal context. Naive frame stacking treats patches from different frames as independent, losing temporal order. VideoMAE's tubelet embeddings let the model learn that "a patch at position (t=3, h=5, w=7)" and "(t=4, h=5, w=7)" are temporally adjacent — enabling motion understanding.

---

**Q2: Why does BLIP-2's Q-Former have a fixed 32-query output regardless of image resolution?**

A: Q-Former uses 32 learnable query tokens as input. These queries cross-attend to image patch features from ViT, then output 32 refined embeddings. The output count = input query count = 32, regardless of how many image patches the ViT produced. The ViT cost scales with resolution (more patches), but the Q-Former output is always 32 tokens. This gives BLIP-2 a constant LLM context budget — a key design choice for efficiency.

---

**Q3: What's the difference between BLIP-2 and InstructBLIP architecturally?**

A: Nearly identical. Both use the same ViT + Q-Former + LLM structure. The key difference: in InstructBLIP, the instruction text is injected INTO the Q-Former during cross-attention, not only into the LLM. The Q-Former therefore extracts instruction-relevant visual features (e.g., for "what color is on the left?", the Q-Former can focus on spatial/color features). In BLIP-2, the Q-Former extracts generic visual features, and the instruction arrives at the LLM only. This makes InstructBLIP better at instruction-specific visual grounding.

---

**Q4: Why doesn't CLIP's contrastive training objective learn spatial grounding?**

A: CLIP's InfoNCE loss measures similarity between the image's CLS token and the caption's EOS token — both are global, whole-sequence representations. There's no spatial supervision: the loss doesn't ask "where is the dog?" or "what's in the top-left corner?" CLIP learns that an image of a red ball is more similar to "a red ball" than "a blue car," but not that the red ball is at any particular location. To learn spatial grounding, you need grounding datasets with bounding boxes (e.g., RefCOCO) or region-level supervision.

---

**Q5: How would you estimate uncertainty for a VLM in a production system?**

A: Several approaches, from cheapest to most expensive:

1. Temperature sampling entropy: run N forward passes with temperature > 1, compute token entropy from output distributions. High entropy → uncertain.
2. MC Dropout: apply dropout at inference time (requires model to have dropout layers), compute variance across N runs.
3. Ensemble: run M different VLMs (or M checkpoints), measure disagreement.
4. Conformal prediction: calibrate on a held-out set to produce statistically valid confidence intervals.

For production at inference speed: entropy-based thresholding with N=5-10 samples is a practical tradeoff.

---

**Q6: Why is the LLM prior so dominant in VLMs? How do you reduce this effect?**

A: LLMs are trained on trillions of tokens and have strong statistical priors about question-answer distributions. Visual information enters the LLM as ~32 soft tokens, but is processed through 30-40 LLM layers of causal attention where the LLM's prior is entrenched. By the output layer, the visual signal is heavily diluted. To reduce prior dominance: (1) use more visual tokens (LLaVA: 576 vs BLIP-2: 32); (2) use high-resolution vision encoders (more spatial detail); (3) train with grounding supervision (bounding boxes) to force visual evidence to matter; (4) use RLHF/DPO to penalize hallucinated answers.

---

**Q7: What is the "Q-Former cold start problem" and how does BLIP-2 solve it?**

A: If you initialize Q-Former randomly and immediately connect it to a frozen LLM, the Q-Former outputs random noise. The LLM, receiving garbage visual tokens, produces garbage text. The LM loss gradient flows back through the projection layer into Q-Former, but the signal is dominated by LLM confusion, not visual content. BLIP-2 solves this with 2-stage training: Stage 1 trains Q-Former against an image-text matching objective (without LLM), so Q-Former first learns to produce meaningful visual summaries. Stage 2 then connects to the LLM, which now receives sensible visual tokens.

---

**Q8: Why do VLMs fail on counting tasks?**

A: Counting requires discrete enumeration — you must sequentially attend to each object, mark it as counted, and not re-count it. This is an iterative process. Attention mechanisms produce weighted sums over all tokens simultaneously. A CLIP embedding of "5 circles" is similar to a CLIP embedding of "4 circles" because global image statistics are similar. There's no explicit mechanism in the attention-pooled embedding that encodes "I saw exactly 5 distinct non-overlapping round objects." The LLM then guesses based on its prior over how often each count appears in training data.

---

**Q9: Compare Q-Former, cross-attention, and prefix-concat as VLM connectors. When would you choose each?**

A:
- **Prefix-concat** (linear projection, 4-16 tokens): Simplest to implement and train. Best for prototyping or when LLM context is generous. Loses most spatial/detail information due to aggressive compression.
- **Cross-attention** (GPT-2 attends to image patches at every layer): Richer — the LLM can query image features at each depth. But passes ALL patches, creating context pressure. Best when spatial detail matters and you have a large context window.
- **Q-Former** (32 learned queries, 6 cross-attention layers): Best compute/quality tradeoff for production. Resolution-independent output budget. Requires careful 2-stage training. Best choice when you need to scale to variable-resolution inputs.

---

**Q10: How does instruction tuning change VLM behavior? What doesn't it change?**

A: Instruction tuning changes: (1) output FORMAT — answers are natural language, not short captions; (2) instruction-following ability — model responds to the specific question asked; (3) multi-turn capability — model can handle follow-up questions. It does NOT change: (1) fundamental factual accuracy — hallucination rates may even increase; (2) spatial grounding ability — still fails on left/right questions; (3) counting ability — still guesses. Instruction tuning is about communication style, not perceptual capability.

---

**Q11: What makes VLMs brittle in autonomous driving scenarios specifically?**

A: Three properties compound in AV:

1. **Real-time constraint**: you can't run N=10 uncertainty samples at 10Hz. You must trust the single greedy output.
2. **Tail risk matters**: average-case performance (benchmarks) is irrelevant. One spatial hallucination at the wrong moment is catastrophic.
3. **Distribution shift**: AV cameras see rain, night, fog, unusual angles. VLMs trained on clean internet images fail on these edge cases with unknown failure modes.

Current best practice: VLMs are used as an offline analysis tool or for non-safety metadata, never as primary perception in the safety path.

---

**Q12: What is the difference between BLIP-2's "vision frozen" approach and LLaVA's "LLM tuned" approach? Which is better?**

A: BLIP-2 freezes both the ViT and the LLM; only Q-Former and its projection are trained. This is extremely sample-efficient and preserves the pre-trained LLM's general capabilities. But the visual representations are limited to what the pre-trained LLM can use, and the LLM doesn't adapt to the visual domain. LLaVA fine-tunes the LLM (via LoRA), allowing joint visual-language optimization. This generally achieves better instruction-following accuracy but requires more training data and compute. There's no universal winner: frozen LLMs are better for low-data regimes; tuned LLMs are better when you have large instruction datasets.

---

**Q13: Why does patch pruning work (top-K patches by attention score)?**

A: CLIP's CLS token focuses its attention on the most discriminative patches — those containing edges, textures, and objects, not blank background. When you keep only the top-K patches by CLS attention weight, you retain the patches that contribute most to the image classification decision. Background patches (sky, empty walls) contribute little; foreground object patches contribute a lot. The key insight: natural images are sparse in terms of discriminative information. The top ~16 patches (out of 49) often contain 90%+ of the classification-relevant signal.

---

**Q14: What is representation collapse in the context of VLMs?**

A: Representation collapse in VLMs occurs when visual information is compressed too aggressively before reaching the LLM. At extreme compression (e.g., 1-4 tokens), the visual representation collapses to a coarse semantic tag — "dog is present" — losing all spatial structure, fine-grained attributes, and relative positions. This makes the VLM answer "dog" to any question about the image, regardless of what's actually asked. Collapse can also happen during training when the connector learns to ignore visual input and the LLM learns to answer from language priors alone.

---

**Q15: How would you build a production-ready uncertainty-aware VLM for a robotics application?**

A:
1. **Model**: Use InstructBLIP or LLaVA-1.5 fine-tuned on your robot's visual domain.
2. **Uncertainty**: Temperature sampling (N=5 samples at T=0.8) → token entropy.
3. **Threshold**: Calibrate τ on a held-out set to achieve target false positive rate.
4. **Abstention**: When entropy > τ, return "uncertain" and trigger a human-in-the-loop or a classical fallback (e.g., YOLO + structured spatial query).
5. **Cross-validation**: Cross-check spatial claims with depth sensor or LiDAR detections.
6. **Monitoring**: Log all VLM queries + uncertainty scores. Alert when entropy distribution shifts (indicates distribution shift / degraded visual input).
7. **Never in safety path**: Use VLM for scene description metadata, never as primary control signal for collision-avoidance decisions.
