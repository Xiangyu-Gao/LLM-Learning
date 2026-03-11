"""DPO (Direct Preference Optimisation) trainer.

Key insight
-----------
DPO eliminates the explicit reward model step in RLHF by re-parameterising the
constrained RL optimisation directly as a classification loss over preference
pairs.  The key identity is:

    r*(x, y) = β · log[π*(y|x) / π_ref(y|x)] + β · log Z(x)

Because log Z(x) cancels in the Bradley-Terry pairwise objective, the optimal
reward is a function of the *ratio* of log-probs between the policy and a frozen
reference model.  This means we can train the policy *directly* on preference
data without ever materialising a reward model.

DPO loss:
    L_DPO = -E[ log σ( β·(log π(y_w|x)/π_ref(y_w|x)
                          - log π(y_l|x)/π_ref(y_l|x)) ) ]

where y_w = chosen ("winner"), y_l = rejected ("loser"), β controls how far the
policy is allowed to deviate from the reference.

Data format expected by TRL DPOTrainer (columns):
  "prompt"   : the shared prompt string
  "chosen"   : the preferred completion
  "rejected" : the dispreferred completion

Usage
-----
  python src/train_dpo.py \\
      --prefs_dir data/preferences \\
      --output_dir results/dpo-gpt2 \\
      --sft_model_path ../Week3/fine_tuning_project/results/sft-gpt2 \\
      --epochs 1 \\
      --max_samples 200

Interview Q&A
-------------
Q: What is DPO optimising?
A: DPO optimises the same objective as RLHF-PPO — a KL-constrained reward
   maximisation — but in closed form.  Specifically, it maximises
   E[r*(x,y)] − β·KL[π(·|x) || π_ref(·|x)], where the optimal policy under
   this objective has the analytic form π* ∝ π_ref · exp(r*/β).
   DPO plugs in the Bradley-Terry preference model for r* and derives a loss
   that depends only on the policy log-probs and the reference log-probs.
   No reward model, no sampling, no value function — just two forward passes
   (policy + frozen reference) per batch.

Q: What is β in DPO?
A: β (default 0.1) is the KL regularisation coefficient.  Higher β → policy
   stays closer to the SFT reference.  Lower β → stronger preference optimisation
   but risk of reward hacking or degeneration.  In practice 0.01–0.5 are common.

Q: What are DPO's failure modes?
A: (1) Chosen and rejected responses are too similar → gradient vanishes.
   (2) Bad SFT checkpoint → reference is already poor, DPO amplifies noise.
   (3) Distribution mismatch: DPO is *offline* (uses a fixed preference dataset).
   If the base model is far from the data distribution, log-probs are poorly
   calibrated.  (4) Mode collapse: very low β can cause the policy to collapse
   to a single high-reward completion pattern.
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

# ---------------------------------------------------------------------------
# Chat template (identical to Week 3 to keep tokenisation consistent)
# ---------------------------------------------------------------------------
CHAT_TEMPLATE = (
    "{% for msg in messages %}"
    "{% if msg['role'] == 'user' %}"
    "<|user|>\n{{ msg['content'] }}\n"
    "{% endif %}"
    "{% if msg['role'] == 'assistant' %}"
    "<|assistant|>\n{{ msg['content'] }}{{ eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_preferences(prefs_dir: Path, split: str = "train") -> Dataset:
    """Load preference pairs from JSONL files.

    Each line: {"prompt": ..., "chosen": ..., "rejected": ..., "source": ...}
    TRL DPOTrainer needs exactly: prompt, chosen, rejected columns.
    """
    path = prefs_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Preference file not found: {path}\n"
            "Run `python data/make_preferences.py` first to generate it."
        )
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                # DPOTrainer expects plain strings for prompt/chosen/rejected
                records.append({
                    "prompt": rec["prompt"],
                    "chosen": rec["chosen"],
                    "rejected": rec["rejected"],
                })
    print(f"  {split:5s} preferences: {len(records):>6,}")
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# YAML config loader
# ---------------------------------------------------------------------------

def load_yaml_config(path: str | None) -> dict:
    """Load a YAML config file and return as dict.  Returns {} if path is None."""
    if path is None:
        return {}
    try:
        import yaml
    except ImportError:
        print("Warning: PyYAML not installed; ignoring --config flag.")
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Win-rate evaluation helper
# ---------------------------------------------------------------------------

def compute_win_rate(trainer: DPOTrainer, eval_dataset: Dataset, n: int = 50) -> float:
    """Estimate win-rate: fraction of eval pairs where chosen log-prob > rejected.

    This is a proxy metric computed on the *trained* policy (not the reference).
    A win-rate > 0.5 means the policy assigns higher probability to the preferred
    response than the rejected one — a necessary (but not sufficient) condition
    for good alignment.
    """
    import torch

    model = trainer.model
    tokenizer = trainer.processing_class
    model.eval()
    device = next(model.parameters()).device

    wins = 0
    total = min(n, len(eval_dataset))
    with torch.no_grad():
        for i in range(total):
            row = eval_dataset[i]

            def _logprob(text: str) -> float:
                full = row["prompt"] + " " + text
                ids = tokenizer(full, return_tensors="pt", truncation=True, max_length=512)
                ids = {k: v.to(device) for k, v in ids.items()}
                out = model(**ids, labels=ids["input_ids"])
                # out.loss is mean NLL; negate to get log-prob
                return -out.loss.item()

            lp_chosen = _logprob(row["chosen"])
            lp_rejected = _logprob(row["rejected"])
            if lp_chosen > lp_rejected:
                wins += 1

    rate = wins / total if total > 0 else 0.0
    print(f"  Win-rate on {total} eval pairs: {rate:.3f}")
    return rate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DPO training on preference pairs (TRL DPOTrainer)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None,
                        help="Path to YAML config (values override argparse defaults)")
    parser.add_argument("--model_name", default="gpt2",
                        help="Base model or SFT checkpoint path")
    parser.add_argument("--sft_model_path",
                        default="../Week3/fine_tuning_project/results/sft-gpt2",
                        help="Path to SFT checkpoint used as reference model")
    parser.add_argument("--prefs_dir", default="data/preferences",
                        help="Directory containing train.jsonl and eval.jsonl")
    parser.add_argument("--output_dir", default="results/dpo-gpt2")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO β: KL regularisation strength (higher = stay closer to ref)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap dataset size for quick smoke-tests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_win_rate", action="store_true",
                        help="Compute win-rate metric on eval set after training")
    parser.add_argument("--swanlab", action="store_true",
                        help="Enable SwanLab logging")
    parser.add_argument("--run_name", default=None,
                        help="SwanLab run name (defaults to output_dir basename)")
    args = parser.parse_args()

    # Override with YAML config if provided
    cfg = load_yaml_config(args.config)
    for k, v in cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)

    prefs_dir = Path(args.prefs_dir)

    # ── datasets ─────────────────────────────────────────────────────────────
    print("Loading preference datasets...")
    train_ds = load_preferences(prefs_dir, split="train")
    eval_ds  = load_preferences(prefs_dir, split="eval")

    if args.max_samples is not None:
        train_ds = train_ds.select(range(min(args.max_samples, len(train_ds))))
        eval_ds  = eval_ds.select(range(min(args.max_samples // 10 + 1, len(eval_ds))))

    print(f"Total train: {len(train_ds):,}  eval: {len(eval_ds):,}")

    # ── tokenizer ─────────────────────────────────────────────────────────────
    # Load from SFT checkpoint if it exists so we get the same chat template.
    # Fall back to base model if checkpoint not found.
    ref_model_path = args.sft_model_path
    if not Path(ref_model_path).exists():
        print(f"Warning: SFT checkpoint not found at {ref_model_path}. Using {args.model_name}.")
        ref_model_path = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(ref_model_path)
    tokenizer.chat_template = CHAT_TEMPLATE  # always inject to be safe
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── models ────────────────────────────────────────────────────────────────
    # policy  : the model we are training
    # ref_model: frozen copy of the SFT checkpoint used as the KL anchor
    #
    # Why do we need a reference model?
    #   Without the KL term the policy collapses to degenerate high-reward
    #   strings (reward hacking).  The reference model acts as a "language prior"
    #   that keeps the policy grounded in coherent text.
    print(f"Loading policy model from: {ref_model_path}")
    policy = AutoModelForCausalLM.from_pretrained(ref_model_path)

    print(f"Loading reference model from: {ref_model_path}")
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path)
    # Freeze reference model: it never receives gradients
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ── DPO config ────────────────────────────────────────────────────────────
    # build the DPO config; note that `max_prompt_length` was removed in
    # recent TRL versions, so we only pass `max_length` here.  The trainer will
    # automatically truncate both prompt and response to stay within this limit.
    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,                    # KL regularisation coefficient
        max_length=args.max_length,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        # precompute_ref_log_probs=True saves memory at the cost of RAM for cached logprobs
        precompute_ref_log_probs=False,
    )

    # ── SwanLab callback ──────────────────────────────────────────────────────
    callbacks = []
    if args.swanlab:
        from swanlab.integration.transformers import SwanLabCallback
        import swanlab
        run_name = args.run_name or Path(args.output_dir).name
        swanlab.init(project="alignment", experiment_name=run_name,
                     config=vars(args))
        callbacks.append(SwanLabCallback())

    # ── trainer ───────────────────────────────────────────────────────────────
    trainer = DPOTrainer(
        model=policy,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=callbacks or None,
    )

    # ── training ──────────────────────────────────────────────────────────────
    print("\nStarting DPO training...")
    print(f"  Policy / ref  : {ref_model_path}")
    print(f"  Train pairs   : {len(train_ds):,}")
    print(f"  Eval pairs    : {len(eval_ds):,}")
    print(f"  Beta (KL coef): {args.beta}")
    print(f"  Epochs        : {args.epochs}")
    print(f"  Batch size    : {args.batch_size} (×{args.grad_accum} grad accum)")
    print()
    print("Logged metrics (watch these during training):")
    print("  rewards/chosen   : mean log-prob ratio for chosen responses")
    print("  rewards/rejected : mean log-prob ratio for rejected responses")
    print("  rewards/margins  : chosen − rejected (should be positive and growing)")
    print("  logits/kl        : KL divergence from reference (should stay bounded)")
    print()

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved DPO model → {args.output_dir}")

    # ── win-rate ──────────────────────────────────────────────────────────────
    if args.eval_win_rate:
        print("\nComputing win-rate on eval set...")
        compute_win_rate(trainer, eval_ds)


if __name__ == "__main__":
    main()
