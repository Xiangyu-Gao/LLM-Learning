"""PPO-style RL training using TRL RLOOTrainer.

Key insight
-----------
PPO (Proximal Policy Optimisation) is the workhorse RL algorithm for RLHF.
It trains a policy by generating completions, scoring them with a reward model,
and then updating the policy to increase expected reward — while staying close
to a reference policy via a KL penalty.

Note on TRL 0.29.0
-------------------
TRL 0.29.0 ships **RLOOTrainer** (REINFORCE Leave-One-Out) instead of the older
PPOTrainer.  RLOO is a simpler RL baseline that shares PPO's key ideas:

  1. Online sampling: the policy generates completions from real prompts.
  2. KL penalty: log π(y|x) − log π_ref(y|x) is subtracted from the reward to
     prevent reward hacking.
  3. Policy gradient update: the policy is nudged toward completions with
     positive advantage (above-average reward).

RLOO vs PPO differences:
  - RLOO has no value network: advantage is estimated as reward − mean(rewards
    for same prompt), so K completions per prompt are needed (default K=2).
  - PPO has a separate value head; requires on-policy rollouts + clipping.
  - RLOO is lower variance than vanilla REINFORCE but higher variance than PPO.
  - For small models on single GPU, RLOO is faster and simpler.

Reward function
---------------
We use a *proxy* reward that combines:
  1. FORMAT SCORE (0–1): does the response end with a complete sentence?
  2. LENGTH SCORE (0–1): is the response between 10 and 200 chars (a proxy for
     "not degenerate and not rambling")?
  3. GROUNDEDNESS SCORE (0–1): does the response contain at least one word from
     the prompt (a very weak grounding check)?

Combined: reward = 0.4 * format + 0.3 * length + 0.3 * groundedness − KL_penalty

Usage
-----
  python src/train_ppo.py \\
      --prefs_dir data/preferences \\
      --output_dir results/ppo-gpt2 \\
      --sft_model_path ../Week3/fine_tuning_project/results/sft-gpt2 \\
      --max_samples 100

Interview Q&A
-------------
Q: Why is PPO hard in LLMs?
A: Several compounding challenges:
   (1) Credit assignment over long sequences — a 200-token response gets one
       reward signal; distributing credit to individual tokens is hard.
   (2) Value function training — the value head must estimate expected return
       at each token position, but LM hidden states are not designed for this.
       A poorly trained value head makes advantage estimates noisy.
   (3) On-policy requirement — PPO needs fresh rollouts from the current policy
       at every update step.  For large models this means expensive generation
       (O(N) forward passes per token) before every gradient update.
   (4) Hyperparameter sensitivity — KL coefficient, clip ratio, entropy bonus,
       learning rates for policy and value networks all interact.  Getting all of
       them right simultaneously is empirically difficult.
   (5) Distribution shift — as the policy improves, the reward model sees OOD
       completions and may give unreliable scores.  This causes "reward hacking".

Q: What is the KL penalty and why do we need it?
A: The KL penalty is: β · KL[π(·|x) || π_ref(·|x)] = β · Σ_t [log π(y_t|x,y<t)
   − log π_ref(y_t|x,y<t)].  We subtract it from the reward before the policy
   gradient update.  Without it, the policy quickly diverges from sensible text
   by exploiting reward model blindspots ("reward hacking").  The SFT checkpoint
   acts as a "language prior" — the KL penalty keeps the policy from forgetting
   how to generate coherent text while it learns to score higher.
"""

import argparse
import json
import torch
from pathlib import Path

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import RLOOConfig, RLOOTrainer

# ---------------------------------------------------------------------------
# Chat template (identical to Week 3)
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
# Reward functions (split into components so SwanLab logs each separately)
#
# RLOOTrainer accepts a list of callables for reward_funcs and logs each one:
#   rewards/reward_format    — is the response grammatically complete?
#   rewards/reward_length    — is the response an appropriate length?
#   rewards/reward_grounding — does the response relate to the prompt?
#
# Splitting makes it immediately visible which aspect is improving/degrading.
# ---------------------------------------------------------------------------

def reward_format(completions: list[str], **kwargs) -> list[float]:
    """1.0 if completion ends with sentence-final punctuation, else 0.0.

    Scaled to [-0.5, 1.0] to give a clear positive/negative signal.
    """
    rewards = []
    for comp in completions:
        comp = comp.strip()
        score = 1.0 if comp and comp[-1] in ".!?\"'" else 0.0
        rewards.append(score * 1.5 - 0.5)
    return rewards


def reward_length(completions: list[str], **kwargs) -> list[float]:
    """1.0 if 10 ≤ len(completion) ≤ 200 chars; linearly degrades outside.

    Penalises degenerate (too short) and rambling (too long) outputs.
    Scaled to [-0.5, 1.0].
    """
    rewards = []
    for comp in completions:
        comp = comp.strip()
        n = len(comp)
        if 10 <= n <= 200:
            score = 1.0
        elif n < 10:
            score = n / 10.0
        else:  # n > 200
            score = max(0.0, 1.0 - (n - 200) / 300.0)
        rewards.append(score * 1.5 - 0.5)
    return rewards


def reward_grounding(completions: list[str], prompts: list[str] | None = None, **kwargs) -> list[float]:
    """Fraction of prompt words appearing in completion (weak grounding check).

    Default 0.5 when no prompt is available.
    Scaled to [-0.5, 1.0].
    """
    rewards = []
    for i, comp in enumerate(completions):
        comp = comp.strip()
        score = 0.5  # default when no prompt given
        if prompts is not None and i < len(prompts):
            prompt_words = set(prompts[i].lower().split())
            comp_words = set(comp.lower().split())
            if prompt_words:
                overlap = len(prompt_words & comp_words) / len(prompt_words)
                score = min(1.0, overlap * 3.0)  # scale up small overlaps
        rewards.append(score * 1.5 - 0.5)
    return rewards


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_prompt_dataset(prefs_dir: Path, split: str = "train") -> Dataset:
    """Load prompts for RL training.

    RLOOTrainer expects a dataset with a "prompt" column (plain strings).
    We extract prompts from our preference pairs — the RL policy will generate
    new completions from these prompts (on-policy generation).
    """
    path = prefs_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Preference file not found: {path}\n"
            "Run `python data/make_preferences.py` first."
        )
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append({"prompt": rec["prompt"]})
    print(f"  {split:5s} prompts: {len(records):>6,}")
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PPO-style RL training using TRL RLOOTrainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_name", default="gpt2",
                        help="Base model or SFT checkpoint to train")
    parser.add_argument("--sft_model_path",
                        default="../Week3/fine_tuning_project/results/sft-gpt2",
                        help="SFT checkpoint path (used as both policy init and KL reference)")
    parser.add_argument("--prefs_dir", default="data/preferences",
                        help="Directory with train.jsonl / eval.jsonl")
    parser.add_argument("--output_dir", default="results/ppo-gpt2")
    parser.add_argument("--max_completion_length", type=int, default=64,
                        help="Max tokens to generate per completion")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_generations", type=int, default=2,
                        help="Completions per prompt for leave-one-out baseline (K≥2)")
    parser.add_argument("--kl_coef", type=float, default=0.05,
                        help="KL penalty coefficient β (higher = stay closer to reference)")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--swanlab", action="store_true",
                        help="Enable SwanLab logging")
    parser.add_argument("--run_name", default=None,
                        help="SwanLab run name (defaults to output_dir basename)")
    args = parser.parse_args()

    prefs_dir = Path(args.prefs_dir)

    # ── model path resolution ─────────────────────────────────────────────────
    model_path = args.sft_model_path
    if not Path(model_path).exists():
        print(f"Warning: SFT checkpoint not found at {model_path}. Using {args.model_name}.")
        model_path = args.model_name

    # ── datasets ──────────────────────────────────────────────────────────────
    print("Loading prompt datasets...")
    train_ds = load_prompt_dataset(prefs_dir, split="train")
    eval_ds  = load_prompt_dataset(prefs_dir, split="eval")

    if args.max_samples:
        train_ds = train_ds.select(range(min(args.max_samples, len(train_ds))))
        eval_ds  = eval_ds.select(range(min(max(1, args.max_samples // 10), len(eval_ds))))

    print(f"Train: {len(train_ds):,}  Eval: {len(eval_ds):,}")

    # ── tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.chat_template = CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-padding is required for batch generation in RLOO
    tokenizer.padding_side = "left"

    # ── truncate prompts to fit within GPT-2's 1024 position limit ────────────
    max_prompt_tokens = 1024 - args.max_completion_length - 4  # 4-token safety margin

    def _truncate(example):
        ids = tokenizer(example["prompt"], add_special_tokens=False,
                        truncation=True, max_length=max_prompt_tokens)["input_ids"]
        example["prompt"] = tokenizer.decode(ids, skip_special_tokens=True)
        return example

    train_ds = train_ds.map(_truncate)
    eval_ds  = eval_ds.map(_truncate)

    # ── RLOO config ───────────────────────────────────────────────────────────
    # Key knobs explained in docs/ppo_knobs.md
    rloo_config = RLOOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        # RLOO-specific (note: TRL calls the KL coef 'beta' not 'kl_coef'):
        num_generations=args.num_generations,     # K completions per prompt
        max_completion_length=args.max_completion_length,
        beta=args.kl_coef,                        # KL penalty coefficient β
        model_init_kwargs={"torch_dtype": torch.float32},
        temperature=0.9,                           # sampling temperature for rollouts
        # Metrics to watch:
        #   rewards/mean   : average proxy reward per step
        #   kl             : KL divergence from reference model
        #   entropy        : policy entropy (drops → mode collapse, rises → random)
        max_grad_norm=0.5,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        bf16=False,
        fp16=False,
    )

    # ── policy model ──────────────────────────────────────────────────────────
    # RLOO needs the policy to generate on-policy completions.
    # The reference model (for KL) is loaded internally from the same checkpoint.
    print(f"Loading policy from: {model_path}")
    policy = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    policy.config.pad_token_id = tokenizer.pad_token_id

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
    # Pass all three component functions so TRL logs each separately:
    #   rewards/reward_format / rewards/reward_length / rewards/reward_grounding
    # RLOOTrainer calls each func(completions, prompts) → list[float]
    trainer = RLOOTrainer(
        model=policy,
        reward_funcs=[reward_format, reward_length, reward_grounding],
        args=rloo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=callbacks or None,
    )

    # ── training ──────────────────────────────────────────────────────────────
    print("\nStarting PPO/RLOO training...")
    print(f"  Model         : {model_path}")
    print(f"  Train prompts : {len(train_ds):,}")
    print(f"  Generations/prompt: {args.num_generations}  (RLOO leave-one-out baseline)")
    print(f"  KL coef β     : {args.kl_coef}")
    print(f"  Max completion: {args.max_completion_length} tokens")
    print()
    print("Logged metrics (SwanLab — watch these during training):")
    print("  rewards/reward_format    : format score  (ends with punctuation?)")
    print("  rewards/reward_length    : length score  (10–200 chars?)")
    print("  rewards/reward_grounding : grounding score (prompt words in response?)")
    print("  kl                       : KL to reference (spikes → reward hacking)")
    print("  entropy                  : policy entropy (should stay above ~1.0)")
    print("  objective/kl_coef        : KL penalty coefficient (adaptive if enabled)")
    print()

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved PPO/RLOO model → {args.output_dir}")


if __name__ == "__main__":
    main()
