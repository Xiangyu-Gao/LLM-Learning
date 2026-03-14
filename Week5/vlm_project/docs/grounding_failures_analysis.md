# Why Multimodal Hallucination Happens — Deep Analysis

*Written as a senior-engineer-level explanation for interview use.*

---

## The Core Problem

Vision-language models hallucinate because they are, fundamentally, language
models that have been shown an image. The architecture creates a massive
asymmetry: GPT-2 (or any large LLM) has billions of parameters encoding
statistical patterns from trillions of text tokens. The visual signal — even
after CLIP processing — arrives as a handful of 768-dimensional vectors.
When these conflict, the language model almost always wins.

---

## Root Cause 1: LLM Prior Dominates the Visual Signal

A language model develops strong conditional priors during pretraining.
Given the prompt "Question: How many circles are in the image? Answer:", GPT-2
will generate a number based on the distribution of such answers in its training
corpus — not by parsing the image. This is rational from the model's perspective:
most of its training signal came from text with no visual input, so text-only
reasoning is what it has learned to do best.

The visual prefix (4 × 768 = 3,072 floats in our model) has to compete with
the LLM's billions of parameters to influence the output. In a shallow
adaptation regime (projection layer only), the projection learns to associate
the CLIP embedding with a *category label* — not spatial details. The LLM
receives something like "this is a {class}" and generates accordingly.

**Quantitative signal**: In our experiments, the hallucination failure rate is
consistently high across all categories, even when the "correct" answer is
clearly encoded in the image. This confirms that the model falls back to its
language prior rather than extracting the answer from the visual input.

---

## Root Cause 2: Contrastive Alignment Is Not Pixel-Level Grounding

CLIP's training objective is:

> "Make the global embedding of image i closer to the global embedding of caption i than to any other caption in the batch."

This objective rewards *semantic category matching* ("dog image ↔ dog caption")
but provides zero gradient signal for spatial relationships ("the red shape is
on the LEFT"), object counts ("there are THREE circles"), or object presence
("there is NO unicorn in this image").

The CLS token — which we use as our visual prefix — aggregates all spatial
information from 49 patches into a single 768-dim vector. Even if the
individual patches encode rich local information, the CLS pooling discards
spatial layout. A CLIP CLS vector for an image with "red circle on the left,
blue square on the right" is indistinguishable from one with "red circle on
the right, blue square on the left" for a CLIP model trained only on class-level
captions.

**The distinction that matters:**
- **Alignment**: image embedding ≈ caption embedding in L2 distance
- **Grounding**: model can answer "where?", "how many?", "is there a?"

Production VLMs that achieve grounding use additional supervision:
- **Dense captioning** (GRiT, KOSMOS-2): describe objects with bounding boxes
- **Referring expression comprehension** (RefCOCO): "the cat on the left"
- **Visual grounding datasets** (Flickr30k Entities, VisualGenome)

---

## Root Cause 3: Training Distribution Mismatch

Our training objective is image captioning: given the visual prefix, generate
"a photo of a {class}". This trains the model on:

- *What* is in the image (class category)
- Absolute descriptions, not relative ("a cat", not "the cat on the left")
- Single-entity descriptions (class labels are not spatial)

At inference, we switch to question-answering format:
"Question: Is there a unicorn? Answer:"

This is a *different prompt distribution* from training. The model has never
seen "Question: … Answer:" during training, so it is immediately operating
out-of-distribution. The VLM cannot reliably answer questions it was never
trained to answer, so it falls back to generating the most common caption-like
continuation: the class name.

---

## Root Cause 4: No Negative Object Supervision

Nothing in CLIP training or our captioning objective ever penalises the model
for mentioning an absent object. Captions are positive descriptions of what
*is* present. A model that generates "there is a unicorn" for any image is
not penalised during caption training.

Human annotators and RLHF-based alignment are what teach production VLMs to
say "I don't see a unicorn in this image." Without explicit negative examples
or preference training that rewards correct refusals, the model defaults to
asserting plausible-sounding content.

---

## The Path to Better Grounding

Understanding why hallucination happens directly points to the fixes:

| Root Cause | Mitigation |
|---|---|
| LLM prior too strong | More visual tokens (LLaVA-1.5: 576 tokens), higher-res encoder |
| No spatial alignment signal | Add dense/regional supervision during VLM training |
| Prompt distribution mismatch | Instruction tuning in Q&A format from the start |
| No negative supervision | RLHF/DPO using POPE negatives; SFT on "No, there is no X" examples |
| Weak vision signal | Fine-tune vision encoder on task-specific data (carefully) |

The senior insight: hallucination is not a bug to be patched — it is the
natural consequence of mismatched inductive biases between how vision is
encoded (globally, semantically, without spatial grounding) and how language
models are expected to use that signal (spatially, precisely, with negation).
Fixing it requires rethinking the visual representation itself, not just
the fusion mechanism.
