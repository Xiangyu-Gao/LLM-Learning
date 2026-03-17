"""
day3_instruct.py — Day 3: Instruction Tuning for VLMs

LLaVA's core insight: the same frozen vision encoder + frozen LLM can become a
powerful instruction-following assistant if you fine-tune the connector on
GPT-4-generated Q&A data.  Architecture barely changes; DATA changes everything.

Two experiments:

Experiment A: BLIP-2 base vs InstructBLIP side-by-side
────────────────────────────────────────────────────────
Load both models and ask the same 12 prompts.

  BLIP-2 (blip2-opt-2.7b):   trained on image-text pairs, VQA datasets
                               → short, captioning-style answers
  InstructBLIP (flan-t5-xl): same Q-Former + ViT, but Q-Former was instruction-tuned
                               on 26 datasets in instruction format
                               → detailed, natural, instruction-following answers

  The dramatic OUTPUT DIFFERENCE with nearly identical ARCHITECTURE is the lesson.
  It proves instruction tuning is about data format, not model size.

  InstructBLIP key difference: the instruction text is injected INTO the Q-Former
  as soft prompts during cross-attention.  The Q-Former learns to extract
  instruction-relevant visual features — not just generic image features.

Experiment B: MiniVLM prompt format sensitivity
────────────────────────────────────────────────
Load our Week 5 VLM checkpoint and probe it with different prompt formats.
No retraining — just show how the same model responds differently to:

  Format 1 (captioning):     "a photo of a"
  Format 2 (question):       "Question: What is in this image? Answer:"
  Format 3 (instruction):    "Describe the main object in this image."
  Format 4 (chain-of-thought): "Step by step, what do I see in this image? First,"

  Expected: the model was trained on Format 2 (W5 Day 3) so it responds best
  to that format.  Format 1 produces acceptable output.  Format 3+4 fail
  because the model has never seen that template.

  This demonstrates WHY LLaVA's instruction tuning matters even for small models.

Outputs:
  results/day3/blip2_vs_instruct.txt  — side-by-side model comparison
  results/day3/prompt_sensitivity.txt — MiniVLM with different prompt formats
  results/day3/instruction_analysis.txt — LLaVA architecture explanation
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
WEEK5_SRC = Path(__file__).parent.parent.parent.parent / "Week5" / "vlm_project" / "src"
sys.path.insert(0, str(WEEK5_SRC))

import torch

from utils import (
    load_clip_vision,
    load_gpt2,
    load_clip_full,
    load_vlm_checkpoint,
    load_image_caption_dataset,
)


# ── Experiment A: BLIP-2 vs InstructBLIP ──────────────────────────────────────

# 12 prompts designed to reveal instruction-following differences
PROMPTS = [
    # Basic description
    ("Describe what you see in this image.",
     "description"),
    # BLIP-2 native format — the only format BLIP-2 base was trained on.
    # "Question: X Answer:" triggers completion mode; all other prompts echo back.
    ("Question: What animal is shown in this image? Answer:",
     "blip2_native"),
    # Q&A style
    ("What is the main subject of this image?",
     "qa"),
    # Instruction following
    ("List three characteristics you observe in this image.",
     "instruction"),
    # Comparative
    ("Is this image depicting a living thing or an object? Explain.",
     "reasoning"),
    # Spatial (will reveal failures)
    ("Where is the main object positioned in the image?",
     "spatial"),
    # Counting (will reveal failures)
    ("How many distinct objects can you count in this image?",
     "counting"),
    # Attribute
    ("What color is the main object in this image?",
     "attribute"),
    # Context / scene
    ("What type of environment is shown in this image?",
     "scene"),
    # Negation (adversarial)
    ("Is there a car in this image? Answer yes or no and briefly explain.",
     "negation"),
    # Chain-of-thought
    ("Think step by step. What category does this image belong to?",
     "chain_of_thought"),
    # Factual extension
    ("What is an interesting fact about the main subject in this image?",
     "factual"),
    # Safety (hallucination risk)
    ("Does this image show any danger or risk to humans?",
     "safety"),
]


def load_blip2_base(device, model_id="Salesforce/blip2-opt-2.7b"):
    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        print(f"  Loading BLIP-2 base ({model_id}) …")
        proc  = Blip2Processor.from_pretrained(model_id, use_safetensors=True)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            device_map="auto",
        )
        model.eval()
        return model, proc
    except Exception as e:
        print(f"  [Warning] BLIP-2 base failed: {e}")
        return None, None


def load_instructblip(device, model_id="Salesforce/instructblip-flan-t5-xl"):
    try:
        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        print(f"  Loading InstructBLIP ({model_id}) …")
        proc  = InstructBlipProcessor.from_pretrained(model_id, use_safetensors=True)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            device_map="auto",
        )
        model.eval()
        return model, proc
    except Exception as e:
        print(f"  [Warning] InstructBLIP failed: {e}")
        return None, None


def generate_blip2(model, proc, image, prompt, max_new_tokens=80):
    """Generate with BLIP-2 base (no instruction conditioning in Q-Former)."""
    inputs = proc(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    if hasattr(inputs.get("pixel_values", None), "to"):
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return proc.decode(ids[0], skip_special_tokens=True).strip()


def generate_instructblip(model, proc, image, prompt, max_new_tokens=80):
    """Generate with InstructBLIP (instruction injected into Q-Former)."""
    inputs = proc(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return proc.decode(ids[0], skip_special_tokens=True).strip()


def experiment_a(blip2, blip2_proc, iblip, iblip_proc,
                 sample_records, results_dir):
    """Side-by-side comparison of BLIP-2 base vs InstructBLIP."""
    print("\n── Experiment A: BLIP-2 base vs InstructBLIP ──")
    print("  NOTE: InstructBLIP injects instruction into Q-Former → task-specific features")

    if blip2 is None and iblip is None:
        print("  [Skipping: neither model loaded]")
        return

    # Use 3 sample images with diverse labels
    test_imgs = [r["image"].convert("RGB").resize((224, 224))
                 for r in sample_records[:3]]
    captions  = [r["caption"] for r in sample_records[:3]]

    lines = ["BLIP-2 base vs InstructBLIP: Side-by-Side Comparison",
             "=" * 70]

    for img_idx, (img, cap) in enumerate(zip(test_imgs, captions)):
        lines.append(f"\n{'─'*70}")
        lines.append(f"Image {img_idx+1}: {cap}")
        lines.append(f"{'─'*70}")

        for prompt, category in PROMPTS[:8]:    # First 8 prompts
            lines.append(f"\n[{category.upper()}] {prompt}")

            if blip2 is not None:
                b2_ans = generate_blip2(blip2, blip2_proc, img, prompt)
                lines.append(f"  BLIP-2 base:   {b2_ans}")
            if iblip is not None:
                ib_ans = generate_instructblip(iblip, iblip_proc, img, prompt)
                lines.append(f"  InstructBLIP:  {ib_ans}")

    output = "\n".join(lines)
    print(output[:3000] + "\n  [... truncated, see file for full output]")
    (results_dir / "blip2_vs_instruct.txt").write_text(output)
    print(f"\n  Full comparison saved → {results_dir}/blip2_vs_instruct.txt")


# ── Experiment B: MiniVLM prompt format sensitivity ───────────────────────────

MINIVLM_PROMPTS = [
    # Format the model was trained on
    ("Question: What is in this image? Answer:",  "trained_format"),
    # Slight variation
    ("Q: Describe the image. A:",                 "qa_variation"),
    # Captioning style (different from training)
    ("This image shows",                          "captioning"),
    # Instruction style (very different)
    ("Describe the main object in this image.",   "instruction"),
    # No prompt (cold start)
    ("",                                          "no_prompt"),
]


def find_week5_checkpoint():
    """Search for Week 5 VLM checkpoint."""
    candidates = [
        Path(__file__).parent.parent.parent.parent / "Week5" / "vlm_project" /
        "results" / "vlm-prefix" / "checkpoint.pt",
        Path(__file__).parent.parent / "results" / "vlm-prefix" / "checkpoint.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


@torch.no_grad()
def ask_minivlm(vlm, tok, clip_proc, image, prompt, device, max_new=30):
    pv = clip_proc(images=image, return_tensors="pt")["pixel_values"].to(device)
    vlm.gpt2.generation_config.max_length = None

    if prompt.strip():
        enc   = tok(prompt, return_tensors="pt").to(device)
        ids   = enc["input_ids"]
        msk   = enc["attention_mask"]
        gen   = vlm.generate(pv, prompt_ids=ids, prompt_mask=msk,
                             max_new_tokens=max_new, do_sample=False)
    else:
        # No prompt: just visual prefix
        gen = vlm.generate(pv, max_new_tokens=max_new, do_sample=False)

    return tok.decode(gen[0], skip_special_tokens=True)


def experiment_b(sample_records, results_dir, device):
    """Show how prompt format affects our Week 5 MiniVLM output."""
    print("\n── Experiment B: MiniVLM Prompt Format Sensitivity ──")

    ckpt_path = find_week5_checkpoint()
    if ckpt_path is None:
        print("  [Week 5 checkpoint not found — skipping Exp B]")
        print("  (Run Week 5 Day 3 first to generate a checkpoint)")
        note = ("MiniVLM checkpoint not found.\n"
                "Run Week5/vlm_project/src/day3_fusion.py first.\n")
        (results_dir / "prompt_sensitivity.txt").write_text(note)
        return

    print(f"  Loaded checkpoint: {ckpt_path}")
    vlm, tok, cfg = load_vlm_checkpoint(ckpt_path, device)
    _, clip_proc = load_clip_full(device)
    vlm.eval()

    lines = [
        "MiniVLM Prompt Format Sensitivity",
        "=" * 60,
        f"Checkpoint: {ckpt_path}",
        f"Config: {cfg}",
        "",
        "Key: model was trained on 'Question: ... Answer:' format.",
        "Other formats produce out-of-distribution behaviour.",
        "",
    ]

    for rec in sample_records[:3]:
        img = rec["image"].convert("RGB").resize((224, 224))
        lines.append(f"\n{'─'*60}")
        lines.append(f"Image: {rec['caption']}")
        lines.append("─"*60)

        for prompt, fmt_name in MINIVLM_PROMPTS:
            gen = ask_minivlm(vlm, tok, clip_proc, img, prompt, device)
            prompt_display = prompt if prompt else "[empty]"
            lines.append(f"\n  [{fmt_name}]")
            lines.append(f"  Prompt: {prompt_display}")
            lines.append(f"  Output: {gen}")
            in_dist = "✓ trained format" if fmt_name == "trained_format" else ""
            if in_dist:
                lines.append(f"         {in_dist}")
            print(f"  [{fmt_name}] → '{gen[:60]}'")

    output = "\n".join(lines)
    (results_dir / "prompt_sensitivity.txt").write_text(output)
    print(f"\n  Results saved → {results_dir}/prompt_sensitivity.txt")


# ── LLaVA architecture explanation ───────────────────────────────────────────

LLAVA_ANALYSIS = """\
Instruction Tuning for VLMs: How LLaVA Changed Everything
==========================================================

The Setup (2023)
-----------------
LLaVA (Large Language and Vision Assistant) used:
  Vision encoder:  CLIP ViT-L/14  (frozen)
  Connector:       Simple linear projection (W ∈ ℝ^{1024×4096})
  Language model:  LLaMA-7B (fine-tuned with LoRA)

Nothing exotic.  What was new: the TRAINING DATA.

GPT-4 Generated Supervision
------------------------------
LLaVA's authors couldn't label 100K image-instruction pairs manually.
So they fed image captions + bounding box metadata to GPT-4 text-only and asked:
  "Generate instruction-following Q&A pairs for an image with this description."

GPT-4 produced three types of data:
  1. Conversation: multi-turn Q&A about image content
  2. Detailed description: long, paragraph-level image descriptions
  3. Complex reasoning: "What would happen if..." / "Why does..."

Result: 150K instruction-following training examples at near-zero cost.

What Instruction Tuning Does to the VLM
-----------------------------------------
Without instruction tuning (BLIP-2 base):
  - Model answers are short and captioning-style: "a dog on a couch"
  - Model ignores the specific question being asked
  - Multi-turn conversation not supported

After instruction tuning (InstructBLIP / LLaVA):
  - Model follows the FORM of the instruction
  - Answers are appropriately length and detail for the question
  - Can handle comparisons, reasoning, spatial queries
  - Does NOT necessarily improve factual accuracy
    (hallucination rates remain similar or even increase)

InstructBLIP vs LLaVA: Key Architectural Difference
-----------------------------------------------------
LLaVA: instruction goes to LLM only
  [image patches] → CLIP → Linear → [16×16 vis tokens]
  [instruction]  ─────────────────────────────────────→ [LLM]
  The LLM must figure out which visual features are relevant.

InstructBLIP: instruction injects INTO Q-Former
  [image patches] ──cross-attn──→ [Q-Former]
  [instruction]  ──self-attn──→↗
  The Q-Former EXTRACTS instruction-relevant visual features.
  → Better grounding, more task-specific visual representation.

Why This Matters for Safety-Critical Deployment
------------------------------------------------
Instruction tuning improves USABILITY but not RELIABILITY.
  - A VLM will say "There are 3 cars" confidently whether or not it's true.
  - Instruction format makes it more human-readable but not more truthful.
  - For AV/robotics: never rely on a VLM answer without a fallback mechanism.

Takeaway
---------
LLaVA showed that a simple linear connector + instruction-tuned data
can match or beat much more complex architectures.
Architecture choice (prefix vs Q-Former vs cross-attention) matters less
than training data quality and format.
This is the fundamental lesson of the LLaVA paper.
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Loading data]")
    records = load_image_caption_dataset(max_samples=500)
    sample_records = records[:5]
    print(f"  Sample images: {len(sample_records)}")

    # ── Experiment A: BLIP-2 vs InstructBLIP ──────────────────────────────────
    if not args.skip_blip2:
        blip2, blip2_proc = load_blip2_base(device, args.blip2_model)
        if not args.skip_instructblip:
            iblip, iblip_proc = load_instructblip(device, args.instructblip_model)
        else:
            iblip, iblip_proc = None, None

        experiment_a(blip2, blip2_proc, iblip, iblip_proc,
                     sample_records, results_dir)

        # Free GPU memory between models
        if blip2 is not None:
            del blip2; torch.cuda.empty_cache()
        if iblip is not None:
            del iblip; torch.cuda.empty_cache()
    else:
        print("  [Skipping Experiment A (--skip_blip2)]")

    # ── Experiment B: MiniVLM prompt sensitivity ───────────────────────────────
    experiment_b(sample_records, results_dir, device)

    # ── Write analysis ────────────────────────────────────────────────────────
    (results_dir / "instruction_analysis.txt").write_text(LLAVA_ANALYSIS)
    print(f"\nLLaVA analysis written → {results_dir}/instruction_analysis.txt")
    print(LLAVA_ANALYSIS[:1000] + "\n  [... see file for full analysis]")
    print(f"\nAll outputs in {results_dir}/")


def parse_args():
    p = argparse.ArgumentParser(description="Day 3: Instruction Tuning for VLMs")
    p.add_argument("--blip2_model",       default="Salesforce/blip2-opt-2.7b")
    p.add_argument("--instructblip_model",default="Salesforce/instructblip-flan-t5-xl")
    p.add_argument("--skip_blip2",        action="store_true")
    p.add_argument("--skip_instructblip", action="store_true")
    p.add_argument("--results_dir",       default="results/day3")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("Day 3 — Instruction Tuning for VLMs (LLaVA / InstructBLIP)")
    print("=" * 60)
    main(args)
