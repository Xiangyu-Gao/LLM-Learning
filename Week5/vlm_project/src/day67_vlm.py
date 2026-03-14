"""
day67_vlm.py — Days 6–7: Full Mini-VLM Project

Deliverables:
  1. Load best VLM checkpoint (Day 5 finetuned or Day 3 prefix fallback).
  2. Run 30 adversarial test prompts across three categories:
       - 10 object-hallucination  (HALLUCINATION_QUESTIONS from day4)
       - 10 spatial reasoning     (SPATIAL_TESTS from day4)
       - 10 counting              (COUNTING_TESTS from day4)
  3. Per-prompt uncertainty estimation via MC Dropout:
       - Set model.train() to activate dropout
       - Sample 8 stochastic completions
       - Measure token-entropy mean and std as the uncertainty score
  4. Attention map extraction via teacher-forced forward pass:
       - output_attentions=True on GPT-2
       - Save per-query attention heatmap PNGs
  5. Final failure-rate table + summary plot
  6. Structured JSON log of every prediction

Outputs (all in results/day67/):
  eval_results.json
  summary.png
  attention_maps/  (one PNG per test prompt)
"""

import argparse
import json
import sys
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from utils import (
    load_clip_vision,
    load_clip_full,
    load_gpt2,
    load_image_caption_dataset,
    load_vlm_checkpoint,
    make_spatial_image,
    make_counting_image,
    COLOR_MAP,
    overlay_attention_on_image,
)
from day3_fusion import MiniVLM, make_dataloader
from day4_failures import (
    HALLUCINATION_QUESTIONS,
    SPATIAL_TESTS,
    COUNTING_TESTS,
    is_failure,
    ask,
)


# ── MC Dropout uncertainty ─────────────────────────────────────────────────────

def _set_dropout_train(model):
    """Set only Dropout modules to train mode (activates dropout at inference)."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_uncertainty(vlm, tok, clip_processor, image, question,
                            device, n_samples=8, max_new_tokens=20):
    """
    Estimate uncertainty via MC Dropout.

    Steps:
      1. Freeze all parameters, but call _set_dropout_train to activate dropout.
      2. Generate n_samples completions stochastically.
      3. For each position, compute the entropy of the generated token
         distribution across samples (approximated by voting distribution).
      4. Return (mean_entropy, std_entropy, generated_texts).

    A high mean_entropy indicates the model is uncertain about its answer.
    """
    vlm.eval()
    _set_dropout_train(vlm.gpt2)

    pv = clip_processor(
        images=image.resize((224, 224)).convert("RGB"),
        return_tensors="pt"
    )["pixel_values"].to(device)

    prompt = f"Question: {question} Answer:"
    enc    = tok(prompt, return_tensors="pt", padding=True).to(device)
    ids    = enc["input_ids"]
    msk    = enc["attention_mask"]

    vlm.gpt2.generation_config.max_length = None

    completions = []
    for _ in range(n_samples):
        gen_ids = vlm.generate(
            pv, prompt_ids=ids, prompt_mask=msk,
            max_new_tokens=max_new_tokens, do_sample=True, temperature=1.0,
        )
        text = tok.decode(gen_ids[0], skip_special_tokens=True)
        completions.append(text)

    # Token-level diversity: count unique tokens per position across samples
    # Tokenise all completions to get token sequences
    token_seqs = [
        tok(c, return_tensors="pt")["input_ids"].squeeze(0).tolist()
        for c in completions
    ]
    max_len = max(len(s) for s in token_seqs) if token_seqs else 0

    entropies = []
    for pos in range(max_len):
        tokens_at_pos = [
            s[pos] if pos < len(s) else tok.eos_token_id
            for s in token_seqs
        ]
        # Empirical distribution over vocabulary at this position
        unique, counts = np.unique(tokens_at_pos, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append(entropy)

    vlm.eval()   # restore eval mode
    return (
        float(np.mean(entropies)) if entropies else 0.0,
        float(np.std(entropies))  if entropies else 0.0,
        completions,
    )


# ── Attention map extraction ──────────────────────────────────────────────────

@torch.no_grad()
def extract_gpt2_attention(vlm, tok, clip_processor, image, question,
                            answer, device):
    """
    Teacher-forced forward pass through GPT-2 with output_attentions=True.
    Returns the last-layer attention matrix of shape (heads, seq_len, seq_len).

    We visualise it as a query-key heatmap where rows=query tokens, cols=key tokens.
    """
    vlm.eval()
    pv = clip_processor(
        images=image.resize((224, 224)).convert("RGB"),
        return_tensors="pt"
    )["pixel_values"].to(device)

    full_text = f"Question: {question} Answer: {answer}"
    enc = tok(full_text, return_tensors="pt", padding=True,
              max_length=64, truncation=True).to(device)
    ids = enc["input_ids"]
    msk = enc["attention_mask"]

    B = 1
    vis_prefix = vlm.encode_image(pv)                         # (1, V, 768)
    text_embs  = vlm.gpt2.transformer.wte(ids)               # (1, T, 768)
    combined   = torch.cat([vis_prefix, text_embs], dim=1)    # (1, V+T, 768)
    V = vlm.num_vis_tokens
    full_mask  = torch.cat([
        torch.ones(B, V, dtype=msk.dtype, device=device),
        msk
    ], dim=1)

    out = vlm.gpt2(
        inputs_embeds=combined,
        attention_mask=full_mask,
        output_attentions=True,
    )

    # sdpa backend silently returns None for attention tensors.
    # Fall back to eager attention if needed.
    if out.attentions is None or out.attentions[-1] is None:
        from transformers import GPT2LMHeadModel
        eager_gpt2 = GPT2LMHeadModel.from_pretrained(
            "gpt2", attn_implementation="eager", use_safetensors=True
        ).to(device)
        eager_gpt2.load_state_dict(vlm.gpt2.state_dict())
        eager_gpt2.eval()
        out = eager_gpt2(
            inputs_embeds=combined,
            attention_mask=full_mask,
            output_attentions=True,
        )

    # Last layer attention: (1, heads, V+T, V+T)
    last_attn = out.attentions[-1].squeeze(0)   # (heads, V+T, V+T)
    # Average over heads → (V+T, V+T)
    avg_attn = last_attn.float().mean(0).cpu().numpy()

    # Token labels: V visual tokens + text tokens
    text_tokens = [tok.decode([t]) for t in ids[0].cpu().tolist()]
    labels = [f"vis{i}" for i in range(V)] + text_tokens

    return avg_attn, labels


def save_attention_map(attn, labels, title, path):
    """Save attention heatmap as PNG."""
    N = attn.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, N * 0.4), max(5, N * 0.4)))
    im = ax.imshow(attn, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03)
    if N <= 24:
        short = [l[:8] for l in labels]
        ax.set_xticks(range(N)); ax.set_xticklabels(short, rotation=90, fontsize=6)
        ax.set_yticks(range(N)); ax.set_yticklabels(short, fontsize=6)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Key (attended to)"); ax.set_ylabel("Query (attending)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# ── Evaluation loop ───────────────────────────────────────────────────────────

def run_full_eval(vlm, tok, clip_processor, records, device, results_dir,
                  n_mc_samples=8):
    attn_dir = results_dir / "attention_maps"
    attn_dir.mkdir(exist_ok=True)

    all_results = []

    # ── Build test cases ──────────────────────────────────────────────────────
    test_cases = []

    # 10 hallucination (real images)
    for i, (question, expected) in enumerate(HALLUCINATION_QUESTIONS):
        rec = records[i % len(records)]
        test_cases.append({
            "category": "hallucination",
            "image":     rec["image"].resize((224, 224)).convert("RGB"),
            "question":  question,
            "expected":  expected,
            "label":     f"hall_{i+1:02d}",
        })

    # 10 spatial (synthetic)
    for i, (lc, rc, question, expected) in enumerate(SPATIAL_TESTS):
        image = make_spatial_image(
            COLOR_MAP.get(lc, (220,50,50)), COLOR_MAP.get(rc, (50,100,220))
        )
        test_cases.append({
            "category": "spatial",
            "image":    image,
            "question": question,
            "expected": expected,
            "label":    f"spatial_{i+1:02d}",
        })

    # 10 counting (synthetic)
    for i, (n, question, expected) in enumerate(COUNTING_TESTS):
        image = make_counting_image(n)
        test_cases.append({
            "category": "counting",
            "image":    image,
            "question": question,
            "expected": expected,
            "label":    f"count_{i+1:02d}",
        })

    # ── Run each test case ────────────────────────────────────────────────────
    print(f"\n[Running {len(test_cases)} test cases with MC Dropout (n={n_mc_samples})]")
    for tc in test_cases:
        image    = tc["image"]
        question = tc["question"]
        expected = tc["expected"]
        label    = tc["label"]

        # Greedy generation
        gen = ask(vlm, tok, clip_processor, image, question, device)
        failed = is_failure(gen, expected)

        # MC Dropout uncertainty
        mean_ent, std_ent, samples = mc_dropout_uncertainty(
            vlm, tok, clip_processor, image, question, device,
            n_samples=n_mc_samples, max_new_tokens=15,
        )

        # Attention map (teacher-forced with greedy answer)
        try:
            attn, attn_labels = extract_gpt2_attention(
                vlm, tok, clip_processor, image, question, gen, device
            )
            attn_path = attn_dir / f"{label}.png"
            save_attention_map(
                attn, attn_labels,
                title=f"{label} | Q: {question[:40]}",
                path=attn_path,
            )
        except Exception as e:
            print(f"    (attention map skipped: {e})")
            attn_path = None

        flag = "FAIL" if failed else "PASS"
        print(f"  [{flag}] {label:12s} | gen='{gen[:35]:35s}' "
              f"| unc={mean_ent:.3f}±{std_ent:.3f}")

        all_results.append({
            "label":           label,
            "category":        tc["category"],
            "question":        question,
            "expected":        expected,
            "generated":       gen,
            "failed":          failed,
            "uncertainty_mean": mean_ent,
            "uncertainty_std":  std_ent,
            "mc_samples":      samples,
        })

    return all_results


# ── Summary ───────────────────────────────────────────────────────────────────

def summarise(results, results_dir):
    categories = ["hallucination", "spatial", "counting"]
    rates = {}
    unc_means = {}
    for cat in categories:
        cat_r = [r for r in results if r["category"] == cat]
        if cat_r:
            rates[cat]     = sum(r["failed"] for r in cat_r) / len(cat_r)
            unc_means[cat] = np.mean([r["uncertainty_mean"] for r in cat_r])
        else:
            rates[cat] = 0.0; unc_means[cat] = 0.0

    print("\n" + "=" * 60)
    print("FINAL EVALUATION SUMMARY — Days 6–7")
    print("=" * 60)
    print(f"  {'Category':<20} {'Fail Rate':>10} {'Uncertainty':>13}")
    print("  " + "-" * 45)
    for cat in categories:
        print(f"  {cat:<20} {rates[cat]:>9.0%}  {unc_means[cat]:>12.4f}")
    print("=" * 60)

    # Summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    colors = {"hallucination": "tab:red", "spatial": "tab:orange", "counting": "tab:blue"}
    x = np.arange(len(categories))
    w = 0.6

    bars = ax1.bar(x, [rates[c] for c in categories], width=w,
                   color=[colors[c] for c in categories], edgecolor="black")
    for bar, cat in zip(bars, categories):
        ax1.text(bar.get_x() + w/2, bar.get_height() + 0.02,
                 f"{rates[cat]:.0%}", ha="center", fontsize=12)
    ax1.set_xticks(x); ax1.set_xticklabels(categories)
    ax1.set_ylim(0, 1.2); ax1.set_ylabel("Failure Rate")
    ax1.set_title("Grounding Failure Rate by Category"); ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, [unc_means[c] for c in categories], width=w,
            color=[colors[c] for c in categories], edgecolor="black")
    for i, cat in enumerate(categories):
        ax2.text(i, unc_means[cat] + 0.005, f"{unc_means[cat]:.3f}",
                 ha="center", fontsize=11)
    ax2.set_xticks(x); ax2.set_xticklabels(categories)
    ax2.set_ylabel("Mean MC-Dropout Entropy")
    ax2.set_title("Uncertainty (MC Dropout) by Category"); ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("Days 6–7: Full Mini-VLM Evaluation", fontsize=13)
    plt.tight_layout()
    out = results_dir / "summary.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSummary plot saved → {out}")

    return rates, unc_means


# ── Main ──────────────────────────────────────────────────────────────────────

def get_vlm(args, device):
    """Load checkpoint (tries finetuned → prefix → quick train fallback)."""
    paths_to_try = [
        Path(args.checkpoint),
        Path("results/vlm-finetuned/checkpoint.pt"),
        Path("results/vlm-prefix/checkpoint.pt"),
    ]
    for path in paths_to_try:
        if path.exists():
            print(f"Loading checkpoint: {path}")
            vlm, tok, cfg = load_vlm_checkpoint(path, device)
            _, proc = load_clip_full(device)
            return vlm, tok, proc

    print("No checkpoint found. Training a quick model (200 steps) …")
    from day3_fusion import main as train_main, parse_args as pa
    fa = pa()
    fa.max_samples  = 500
    fa.max_steps    = 200
    fa.epochs       = 1
    fa.fusion_modes = ["prefix"]
    fa.results_dir  = "results"
    train_main(fa)

    vlm, tok, cfg = load_vlm_checkpoint(
        Path("results/vlm-prefix/checkpoint.pt"), device
    )
    _, proc = load_clip_full(device)
    return vlm, tok, proc


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    vlm, tok, clip_processor = get_vlm(args, device)

    records = load_image_caption_dataset(max_samples=20, split="train")

    all_results = run_full_eval(
        vlm, tok, clip_processor, records, device, results_dir,
        n_mc_samples=args.n_mc_samples,
    )

    rates, unc_means = summarise(all_results, results_dir)

    # Save JSON log
    out_json = results_dir / "eval_results.json"
    with open(out_json, "w") as f:
        json.dump({
            "failure_rates":   rates,
            "uncertainty":     unc_means,
            "results":         all_results,
        }, f, indent=2)
    print(f"\nFull log saved → {out_json}")
    print(f"Attention maps → {results_dir}/attention_maps/")
    print(f"\nAll outputs in {results_dir}/")


def parse_args():
    p = argparse.ArgumentParser(description="Days 6-7: Full Mini-VLM evaluation")
    p.add_argument("--checkpoint",    default="results/vlm-finetuned/checkpoint.pt")
    p.add_argument("--results_dir",   default="results/day67")
    p.add_argument("--n_mc_samples",  type=int, default=8,
                   help="Number of MC dropout samples for uncertainty")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("Days 6–7 — Full Mini-VLM: Uncertainty + Adversarial Suite")
    print("=" * 60)
    main(args)
