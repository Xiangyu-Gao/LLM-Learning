# Study Guide: VLM & Multimodal Alignment (Week 5)

A practical walkthrough of every day's concepts, what each script does,
and a full set of interview Q&A pairs you should be able to answer cold.

---

## Day 1 — Contrastive Learning & CLIP

### Core Idea: InfoNCE / Symmetric Contrastive Loss

Given a batch of B image–text pairs, CLIP computes a B×B similarity matrix:

```
sim[i,j] = cosine_sim(img_i, txt_j)
```

The loss pushes diagonal entries (matching pairs) high and off-diagonal
entries (mismatches) low:

```
L = 0.5 × (CE(sim/τ, labels) + CE(sim.T/τ, labels))
```

where `τ` (temperature) is learnable and `labels = [0, 1, …, B-1]`.

**Why temperature matters:**
- Small τ → sharper distribution → harder negatives → faster but unstable
- Large τ → flatter distribution → softer matching → more forgiving
- Learnable τ lets the model find the right sharpness for the data

**What "alignment" means vs "grounding":**
- **Alignment**: the image embedding and caption embedding are close in a
  shared vector space. "cat on a mat" is globally similar to cat images.
- **Grounding**: knowing *where* the cat is, *how many* cats there are,
  the cat's spatial relationship to the mat. CLIP does NOT do this.

### Optimizer Pitfalls with Identity-Initialized Projections

When projection heads are initialized as the identity (or near-identity), using
`weight_decay > 0` with AdamW is actively harmful. Weight decay adds a penalty
proportional to the L2 norm of weights, pushing them toward 0. The identity
matrix has weights = 1, so every gradient step partially erases the CLIP
alignment you started with. Always set `weight_decay=0` when fine-tuning on
top of a strong pretrained initialization unless you have abundant data.

### What the script does (`day1_clip.py`)
- Loads tiny-imagenet deduplicated to **one image per class** (200 unique pairs)
- Shuffles records before train/val split so both splits see all classes
- Freezes the CLIP backbone; trains only two `Linear(512→512)` projection heads
- Measures a zero-shot baseline (identity projection) before any training
- InfoNCE loss with learnable temperature, `weight_decay=0`
- Plots: training loss, Recall@1, 16×16 similarity matrix heatmap
- Key observation: the diagonal of the similarity matrix should be brighter
  than off-diagonal after training; trained recall should beat zero-shot baseline

### Interview Q&A

**Q: What is InfoNCE loss and why does it work?**
A: InfoNCE treats every other item in the batch as a negative sample. For
image i, the correct text is txt_i; all other B-1 texts are negatives. The
cross-entropy over the similarity row is equivalent to maximising a lower
bound on the mutual information between images and text. It works because
(a) large batches provide many diverse negatives cheaply, (b) the temperature
controls the penalty magnitude for near-miss negatives.

**Q: Why is temperature τ important in contrastive loss?**
A: Temperature scales the logits before softmax. Low τ amplifies differences
between positive and negative similarities, making the gradient signal sharper.
CLIP found τ ≈ 0.07 works well in practice; it is often made learnable as
`log_τ` to keep it positive.

**Q: What is the difference between CLIP alignment and spatial grounding?**
A: CLIP alignment is *global semantic* matching — the entire image representation
matches the entire caption representation. Spatial grounding requires knowing
*where* objects are in the image. CLIP's training signal never asks "where is
the cat?" — only "does this image contain a cat?". This is why CLIP fails at
left/right questions and object counting.

---

## Day 2 — Patch Tokens vs Text Tokens

### Key question: why are image patches NOT equivalent to word tokens?

| Dimension         | Word tokens                     | Patch tokens                       |
|-------------------|---------------------------------|------------------------------------|
| Discreteness      | Discrete vocabulary (~50k)      | Continuous 768-dim embeddings      |
| Independence      | Carry meaning alone             | Meaning depends on spatial context |
| Position encoding | Learned or sinusoidal           | Learned 2D grid (7×7 for ViT-B/32) |
| Inductive bias    | Permutation-invariant by design | Spatial layout matters             |

Word tokens carry independent semantic meaning: "cat" means cat regardless
of its position in the sentence. A 32×32 patch of sky means very different
things depending on whether it is at the top or centre of the image.

### Attention Rollout

Rollout propagates attention through all transformer layers:

```python
result = I   (identity)
for each layer:
    attn = 0.5 * attn_avg + 0.5 * I    # residual connection
    attn = attn / attn.sum(-1)          # normalise
    result = attn @ result
```

The CLS row of `result` shows which patches the model *effectively* attended
to across all 12 layers — capturing indirect attention paths.

### Patch Shuffle Experiment

Shuffling 32×32 patches degrades CLIP similarity because:
1. Global layout cues (sky at top, ground at bottom) are destroyed
2. Object boundaries are split across non-adjacent patches
3. ViT's positional embeddings are misaligned with content

**Why ViT has weaker spatial bias than CNN:**
- CNN: translation-equivariant by construction (shared convolutional filters)
- ViT: positional embeddings are *learned*, not hard-coded → can be bypassed
- ViT compensates with scale: larger models learn spatial structure implicitly

### Interview Q&A

**Q: Why are image patches not equivalent to word tokens?**
A: Word tokens have discrete symbolic identity — "cat" is always "cat".
Patch tokens are dense floating-point vectors encoding local texture and
colour; their semantic content is entangled with their spatial position. A
32×32 patch of brown fur means "ear" near the top of a face but "paw" near
the bottom. ViT handles this via positional embeddings, but these are learned
rather than inherent — the model has no built-in guarantee that position 7
is always "top-right."

**Q: What is attention rollout and why is it useful?**
A: Raw attention at layer L only shows direct attention at that layer. But
information flows through skip connections and multiple hops of attention.
Rollout multiplies attention matrices across all layers (with 0.5 identity
for residual connections) to estimate the *total* attention flow from CLS
to each patch. It is more faithful than looking at a single layer.

**Q: Why does patch shuffling hurt CLIP but less than expected?**
A: CLIP's image encoder aggregates spatial information into a single CLS
token. For simple class-level retrieval (e.g. "photo of a dog"), the texture
and colour content of individual patches is often sufficient — the layout
is less critical. The degradation is real but moderate, which reveals that
CLIP relies more on *what* (texture, colour) than *where*.

---

## Day 3 — Vision Encoder + LLM Fusion

### Fusion Strategies

**1. Prefix-Concat (ClipCap / LLaVA style)**
```
CLIP ViT → last_hidden_state[:, :V] → Linear(768, 768)
→ V prefix tokens → [prefix | text_embeds] → GPT-2
```
Simple, effective. V=4 means 4 image tokens prepended to the text.
The projection learns to map CLIP's space → GPT-2's space.

**2. Cross-Attention (Flamingo style)**
```
CLIP ViT → patch tokens → Linear(768, 768) → encoder_hidden_states
GPT-2 (with add_cross_attention=True) → attends to image at every layer
```
More expressive: image features are injected at *every transformer layer*
rather than once at the input. Requires modifying GPT-2 config at init time.

### Why the cross-attention init matters
GPT-2's `add_cross_attention` flag must be set in `GPT2Config` *before* the
model is constructed — the layers are added in `__init__`. Setting the flag
after `from_pretrained` does nothing. We work around this by initialising
a fresh GPT-2 with the flag, then copying pretrained weights by key name.

### Key insight
Most VLMs "translate" vision into text-like embeddings that the LLM
pattern-matches against training distribution. They do not "see" in any
perceptual sense. The image prefix tells GPT-2 roughly what *category* of
image it is; the LLM generates the most likely response for that category
from its language priors.

### Interview Q&A

**Q: How does LLaVA connect the vision encoder to the LLM?**
A: LLaVA uses a simple MLP to project CLIP's visual tokens into the LLM's
embedding space, then prepends them as prefix tokens. The LLM is fine-tuned
on instruction-following data that includes visual context. This is the
"prefix-concat" or "visual prefix tuning" approach.

**Q: What is Flamingo's fusion mechanism?**
A: Flamingo inserts "gated cross-attention" layers between frozen LLM layers.
The new layers attend from LLM activations (queries) to a "Perceiver Resampler"
output of the visual features (keys/values). A learnable gate controls how
much visual information is mixed in. The LLM backbone remains frozen;
only the new cross-attention parameters are trained.

**Q: When would you use cross-attention fusion vs prefix concatenation?**
A: Prefix concatenation is simpler, trains faster, and works well when the
visual context is compact (single image, few tokens). Cross-attention is
better for long visual sequences (video, multiple images) because cross-
attention cost is O(text_len × image_tokens) whereas prefix concatenation
is O((text_len + image_tokens)²). Cross-attention also allows the vision
signal to influence every LLM layer rather than just the input.

---

## Day 4 — Grounding Failures

### Three Root Causes of Multimodal Hallucination

**1. LLM prior dominates visual signal**
The LLM has been pretrained on billions of tokens and has strong priors:
"How many circles?" → plausible answers are 1–5 based on corpus statistics.
A small visual prefix (4 × 768 floats) cannot easily override these priors.
The model defaults to the most likely textual answer for the question type.

**2. Contrastive alignment ≠ pixel-level grounding**
CLIP's training objective says "match this image to this caption globally."
No gradient ever asked: "where is the circle?", "is the object on the left?"
The CLS token is a bag-of-visual-concepts, not a spatial map.

**3. No spatial supervision in training**
Our training objective is captioning: "a photo of a {class}". This gives
zero information about object location, count, or relative position.
The projection layer learns to associate the visual embedding with a class
word — nothing more. Spatial/counting questions are entirely out-of-distribution.

### Interview Q&A

**Q: Why do VLMs hallucinate objects that aren't in the image?**
A: Three interacting factors: (1) The LLM's language prior assigns high
probability to plausible-sounding object mentions. (2) The visual encoder
gives only a coarse semantic signal — "this looks like an outdoor scene" —
which is consistent with many objects the model might hallucinate. (3) The
alignment training (CLIP / image-text contrastive) doesn't explicitly penalise
mentioning absent objects; the gradient only pushes the global embedding
toward the correct caption.

**Q: What is object hallucination and how do you measure it?**
A: Object hallucination is when a VLM mentions objects in its output that
are not present in the image. It is measured by comparing the set of objects
in the generated text against a ground-truth object list (from human
annotation or object detection). POPE benchmark (Polling-based Object Probing
Evaluation) is the standard: ask yes/no questions about present and absent
objects, compute F1.

**Q: How does instruction tuning reduce hallucination?**
A: Instruction tuning trains the model on explicit Q&A pairs where correct
answers include "No, there is no X" for absent objects. The model learns
that it is acceptable (and rewarded) to deny the presence of an object.
Without this training, the model's prior is to give helpful-sounding answers
that assume all plausible objects are present.

---

## Day 5 — Frozen vs Fine-tuned Vision Encoder

### Trade-offs

| Property               | Frozen                   | Fine-tuned                  |
|------------------------|--------------------------|-----------------------------|
| CLIP zero-shot ability | Preserved                | Degraded (catastrophic forgetting) |
| Domain adaptation      | Limited (CLIP's features) | Can adapt                  |
| Training stability     | High                     | Lower (can diverge)         |
| Hallucination risk     | Moderate                 | Can increase with overfitting |
| Compute                | Lower                    | Higher                      |
| Data requirement       | Lower                    | Higher (needs diversity)    |

### Catastrophic forgetting probe
After fine-tuning, compute cosine similarity between the original CLIP features
and the fine-tuned features on held-out images. Score = 1.0 means no change
(frozen case). Scores < 1.0 reveal how much the encoder has drifted from its
pretrained representation.

### Interview Q&A

**Q: Should you freeze or fine-tune the vision encoder in a VLM?**
A: It depends on the data size and domain. Frozen is safer for small datasets
(< 100k examples) and when you need to preserve CLIP's zero-shot breadth.
Fine-tuning is better for large in-domain datasets where CLIP's features are
suboptimal (e.g. medical imaging, satellite imagery). Most production systems
(LLaVA-1.5, InstructBLIP) freeze the vision encoder and only train the
projection + LLM.

**Q: What is catastrophic forgetting in vision encoders?**
A: When a pretrained vision encoder is fine-tuned on a narrow task, its
weights shift toward that task's distribution. Features that were useful for
other tasks (zero-shot classification, retrieval) are overwritten. This is
catastrophic forgetting — the model becomes better at the fine-tune task
but loses its generalisation ability.

---

## Days 6–7 — Full Mini-VLM

### MC Dropout Uncertainty

During inference, if we set dropout layers to `train()` mode, each forward
pass produces a *different* output due to random neuron dropping. Running N
stochastic samples and measuring their disagreement (entropy of the token
distribution at each position) estimates model uncertainty:

- Low entropy → model consistently generates the same tokens → high confidence
- High entropy → model generates many different tokens → low confidence

This is a lightweight proxy for Bayesian uncertainty that requires no
architecture changes.

### Attention Map Extraction

Teacher-forced forward: pass the question + correct answer as input, enable
`output_attentions=True` in GPT-2. The returned attention tensors have shape
`(batch, heads, seq_len, seq_len)`. Averaging over heads gives a heat map
showing which tokens each output token attended to most. Visual prefix tokens
(vis0..vis3) should ideally receive attention when answering visual questions.

---

## Full Interview Q&A — 20 Questions

1. **What is CLIP and what is it trained to do?**
   A: CLIP (Contrastive Language-Image Pre-training) trains a vision encoder
   and text encoder jointly using symmetric InfoNCE loss on 400M image–text
   pairs from the web. The model learns a shared embedding space where matching
   image–text pairs are close and non-matching pairs are far apart.

2. **What is the CLIP training objective mathematically?**
   A: For a batch of B pairs, compute B×B cosine similarity matrix. Apply CE
   loss on rows (image→text) and columns (text→image), average both. Temperature
   τ scales logits before softmax. Labels are diagonal (identity).

3. **Why is CLIP good for zero-shot classification?**
   A: Encode the image and N class-name prompts ("a photo of a {class}").
   Find the prompt with highest cosine similarity to the image. Because CLIP
   was trained on diverse web captions, it generalises to unseen classes.

4. **What is ViT (Vision Transformer)?**
   A: ViT splits an image into P×P non-overlapping patches, linearly embeds
   each patch, adds positional embeddings, and feeds the sequence through a
   standard transformer encoder. The CLS token's output is used as the image
   representation. ViT-B/32 uses 32×32 patches, giving 7×7=49 patches on a
   224×224 image.

5. **What positional encoding does ViT use?**
   A: Learned 1D positional embeddings added to patch embeddings. Unlike
   sinusoidal (fixed), these are trained end-to-end and can be interpolated
   for different resolutions. They do not enforce hard spatial structure —
   the model can in principle ignore them.

6. **How does prefix-concat VLM fusion work?**
   A: CLIP visual tokens are projected into the LLM's embedding dimension,
   then concatenated to the beginning of the tokenised prompt. The LLM sees
   a sequence of [visual tokens | text tokens] and generates the output
   autoregressively.

7. **How does cross-attention VLM fusion (Flamingo) work?**
   A: The LLM receives visual features as `encoder_hidden_states`. At each
   transformer layer, a cross-attention module allows LLM activations to
   attend to visual tokens. The LLM backbone is kept frozen; only the new
   cross-attention parameters are trained.

8. **What is the difference between ViT and CNN inductive biases?**
   A: CNNs encode translation equivariance (shared weights across positions)
   and locality (small receptive fields grow hierarchically). ViT has neither
   by default: attention is global from layer 1, and positional structure is
   purely learned. ViT outperforms CNNs at scale because its global attention
   captures long-range dependencies that CNNs miss.

9. **Why does attention rollout work better than single-layer attention?**
   A: Information flows through residual connections and multiple attention
   hops. A single layer's attention shows *direct* attention, but the CLS
   token might only directly attend to a few tokens, which in turn attended
   to the relevant patches. Rollout propagates these indirect paths.

10. **What is object hallucination in VLMs?**
    A: When a VLM generates text describing objects that are not present in
    the image. The model's language prior overrides the weak visual signal,
    leading it to confabulate plausible objects for the given scene context.

11. **Name three causes of VLM hallucination.**
    A: (1) Strong LLM prior overrides weak visual signal. (2) CLIP alignment
    is global/semantic, not spatial/object-level — no grounding. (3) Training
    on caption data provides no explicit "absent object" supervision.

12. **What is the POPE benchmark?**
    A: Polling-based Object Probing Evaluation. Ask yes/no questions about
    objects that are or are not present. Measures F1 score. Adversarial split
    uses objects that co-occur frequently with present objects (hard negatives).

13. **Should you freeze the vision encoder when building a VLM? Why?**
    A: Usually yes, especially for small datasets. Frozen preserves CLIP's
    broad zero-shot features. Fine-tuning risks catastrophic forgetting and
    needs much more data to be beneficial. Production VLMs (LLaVA-1.5,
    InstructBLIP) freeze the encoder and train only projection + LLM.

14. **What is catastrophic forgetting?**
    A: When a neural network is fine-tuned on a new task, gradient updates
    overwrite weights needed for previous tasks. The model loses its
    generalisation ability while gaining narrow task-specific performance.

15. **How does MC Dropout estimate uncertainty?**
    A: Keep dropout active at inference. Sample N stochastic forward passes.
    Measure disagreement across samples (e.g. entropy of token distribution
    at each position). High disagreement = high uncertainty. Cheap proxy for
    Bayesian inference without architecture changes.

16. **What is temperature scaling in generation?**
    A: Divide logits by temperature T before softmax. T < 1 → sharper,
    more deterministic. T > 1 → flatter, more random. T = 1 → unchanged.
    Used to control diversity vs quality in text generation.

17. **How does LLaVA train the projection layer?**
    A: Two stages: (1) Pretrain only the MLP projection on image–caption
    pairs (frozen vision encoder, frozen LLM). (2) Fine-tune the projection
    + LLM on instruction-following data. This separates visual alignment
    from instruction following.

18. **What is the Perceiver Resampler in Flamingo?**
    A: A cross-attention module that maps a variable number of visual tokens
    (from high-resolution images) to a fixed number of learned query vectors.
    The output is a compact fixed-size visual context used as keys/values
    in the LLM cross-attention. Allows processing arbitrary-resolution images.

19. **Why can't CLIP count objects?**
    A: CLIP's training objective never asked it to count. InfoNCE matches
    "three cats playing" to images with multiple cats, but the global embedding
    does not track the number. The CLS token aggregates all spatial information
    into one vector — count information is not explicitly encoded.

20. **What is the key difference between VLM grounding and VLM alignment?**
    A: Alignment means the image embedding is near the correct caption in
    shared space — *semantically* related. Grounding means the model can
    localise, count, and reason about specific objects in specific spatial
    positions. CLIP achieves alignment but not grounding. Grounded VLMs require
    additional supervision (bounding boxes, region features, dense captions).
