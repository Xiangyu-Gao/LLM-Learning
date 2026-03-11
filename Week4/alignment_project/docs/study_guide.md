# Study Guide: Alignment — DPO vs PPO vs GRPO (Week 4)

A practical walkthrough of every day's work, the concepts behind each script,
and the interview Q&A pairs you should be able to answer cold.

---

## Day 8 — Reward Modeling Intuition

### Goal
Understand reward models as "learned preference functions," not magic.

### What to do
```bash
# Generate preference pairs from Week 3 data
python data/make_preferences.py --max_per_source 200
```

### What to understand

**Why reward models exist:**  SFT treats every demonstration equally — it
minimises cross-entropy over all responses without distinguishing good from bad.
A reward model introduces *contrastive* signal: "this response is better than
that one."  The RM is trained with the Bradley-Terry loss:

```
L_RM = -log σ(r(x, y_chosen) − r(x, y_rejected))
```

**Our preference pairs use two heuristics:**
1. **Format correctness** (ifeval): gold response vs. degraded (CAPS/truncated/stripped)
2. **Groundedness** (trivia_qa): correct answer vs. random distractor

These are noisy proxies — production systems use human comparisons or model-assisted labels.

### Read
- `data/make_preferences.py` — the full pipeline with inline explanations
- `docs/reward_modeling_intuition.md` — RM architecture, failure modes, analogies

### Interview pair
**Q: Why not just SFT more?**
A: SFT can only push the model *toward* demonstrations — it has no mechanism to
push *away* from undesired completions.  Every response is weighted equally.
Preference training adds a contrastive gradient: "this direction is better, that
direction is worse."  That lets the model navigate trade-offs (conciseness vs.
accuracy, helpfulness vs. safety) that SFT fundamentally cannot express.

---

## Day 9 — DPO

### Goal
Know DPO deeply enough to derive it verbally in an interview.

### What to do
```bash
# Train DPO on preference pairs
python src/train_dpo.py --max_samples 200 --epochs 1

# With config file:
python src/train_dpo.py --config configs/dpo.yaml
```

### What to understand

**The DPO derivation (verbal version):**
1. Start with the RLHF objective: maximise E[r(x,y)] − β·KL[π || π_ref]
2. The optimal policy has the closed-form: π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)
3. Rearrange to express the reward as: r*(x,y) = β·log[π*(y|x)/π_ref(y|x)] + β·log Z(x)
4. Plug this into Bradley-Terry: P(y_w ≻ y_l) = σ(r_w − r_l)
5. The log Z(x) cancels in the difference → no reward model needed
6. Final loss: L_DPO = −E[log σ(β·(log π(y_w|x)/π_ref(y_w|x) − log π(y_l|x)/π_ref(y_l|x)))]

**Key metrics to watch during training:**
- `rewards/chosen` — log-prob ratio for preferred responses (should increase)
- `rewards/rejected` — log-prob ratio for dispreferred (should decrease)
- `rewards/margins` — chosen − rejected (should be positive and growing)
- `logits/kl` — KL from reference (should stay bounded)

**β (beta) = 0.1 by default:**  Higher β → stay closer to SFT reference.
Lower β → stronger preference optimisation but risk of mode collapse.

### Read
- `src/train_dpo.py` — full implementation with docstring derivation
- `configs/dpo.yaml` — hyperparameter choices explained

### Interview pair
**Q: What is DPO optimising?**
A: A closed-form objective that pushes chosen > rejected relative to a reference
policy, without training an explicit reward model.  It's equivalent to the
KL-constrained RLHF objective but solved analytically: the Bradley-Terry
preference model + the optimal policy form lets you write the loss purely in
terms of policy and reference log-probs.

---

## Day 10 — PPO (Conceptual + Minimal Run)

### Goal
Understand why PPO exists and what it buys you over DPO.

### What to do
```bash
# Run PPO/RLOO training with proxy reward
python src/train_ppo.py --max_samples 100 --epochs 1
```

**Note:** Our implementation uses TRL's RLOOTrainer (REINFORCE Leave-One-Out),
which shares PPO's key ideas (on-policy sampling, KL penalty, policy gradient)
but is simpler (no value network, no clipping).

### What to understand

**PPO's advantage over DPO:**  PPO is *online* — it generates fresh completions
from the current policy, scores them, and updates.  This means:
- The policy can explore beyond the fixed preference dataset
- Reward signals can come from any computable function (code tests, math verifiers)
- The policy continuously adapts its own distribution

**The control knobs (see `docs/ppo_knobs.md`):**
1. **KL coefficient (β):** How much to penalise deviation from SFT reference
2. **Clip ratio (ε):** Max policy change per update step (PPO-specific)
3. **Value loss coefficient:** Weight of the value head loss (PPO-specific; RLOO has none)
4. **GAE λ:** Bias-variance trade-off for advantage estimation
5. **Entropy bonus:** Prevents mode collapse by encouraging diverse outputs
6. **Temperature:** Controls rollout diversity during generation

**Metrics to watch:** reward/mean (should increase), KL (should stay bounded),
entropy (should stay > 1.0 to avoid mode collapse).

### Read
- `src/train_ppo.py` — RLOO implementation with detailed docstring
- `docs/ppo_knobs.md` — every hyperparameter explained with healthy ranges

### Interview pair
**Q: Why is PPO hard in LLMs?**
A: Five compounding challenges: (1) credit assignment — one reward for a
200-token response, (2) value function training instability, (3) on-policy
bottleneck — expensive generation before every gradient step, (4) four+
interacting hyperparameters that don't transfer across model sizes, (5)
reward model staleness as the policy distribution shifts.

---

## Day 11 — GRPO (The On/Off-Policy Confusion Killer)

### Goal
Be able to explain on-policy vs. off-policy precisely and correct misconceptions.

### What to do
```bash
# Run GRPO training
python src/train_grpo.py --max_samples 100 --epochs 1
```

### What to understand

**GRPO's key innovation:** Replace the value function with group-relative
advantage estimation.  For each prompt, generate G completions and normalise
rewards within the group:

```
A_i = (r_i − mean(r_1..r_G)) / std(r_1..r_G)
```

No value head needed → simpler architecture, fewer hyperparameters, but you
need G ≥ 2 completions per prompt (more compute per step).

**The rollout loop (logged explicitly in our implementation):**
1. Generate G completions per prompt using CURRENT policy π_θ
2. Score each completion with reward_func
3. Group-normalise advantages
4. Compute importance ratio: π_θ(y|x) / π_old(y|x)
5. Clipped policy gradient + KL penalty → update π_θ
6. Go to step 1 with UPDATED policy (→ on-policy)

**On-policy vs. off-policy classification:**

| Method | On/Off-Policy | Why |
|--------|--------------|-----|
| SFT | Off-policy (behaviour cloning) | Trains on fixed demonstrations |
| DPO | Off-policy / offline | Trains on fixed preference dataset |
| PPO | On-policy | Generates fresh rollouts from current policy |
| GRPO | On-policy | Same as PPO — fresh rollouts every step |

**Nuance:** If `ppo_epochs > 1` (multiple gradient steps per rollout batch),
later updates are technically "near-on-policy."  The clip ratio keeps this safe.

### Read
- `src/train_grpo.py` — full implementation with RolloutLogger callback
- `docs/on_vs_off_policy.md` — detailed classification with visual diagram

### Interview pair
**Q: Is GRPO on-policy?**
A: Yes.  At the start of every step, GRPO generates G completions from the
CURRENT policy.  The "old policy" in the importance ratio is just a snapshot
from the start of the same step, not a separate deployment.  If rollouts are
reused across multiple gradient steps, it becomes "near-on-policy" — the PPO
clip handles this safely.

---

## Day 12 — Comparison Table

### Goal
Produce the tradeoff explanation that ties everything together.

### Read
- `docs/alignment_tradeoffs.md` — the full 1-page comparison
- `README.md` — condensed tradeoff table

### Summary of when to use each

- **DPO:** You have preference pairs + want simplicity.  No generation loop,
  no value head, no reward server.  2× model memory.
- **PPO:** You have a computable reward function (code tests, math verifiers)
  and need the policy to explore.  Most control knobs but hardest to tune.
- **GRPO:** Same use cases as PPO but you want to skip the value function.
  Group normalisation is simpler and often more stable.

### Interview pair
**Q: When would you choose DPO over PPO?**
A: When you have preference pairs and want simplicity + stability.  DPO has no
generation loop, no value head, no KL controller — it's a single supervised
training run.  PPO when you can define/learn a reward and need explicit
constraint control or exploration beyond the fixed dataset.

---

## Day 13 — Eval Pass + Ablations

### Goal
Validate that repo claims are backed by actual runs.

### What to do
```bash
# Run the full evaluation suite
bash scripts/eval_all.sh --max_samples 200

# Results appear in:
#   results/eval_results.tsv    (machine-readable)
#   results/example_table.md    (Markdown table)
```

### Metrics evaluated
1. **Format pass rate** — does the response end with sentence-final punctuation?
2. **Win rate** — does the model prefer chosen over rejected?
3. **Grounded QA** — does the correct answer appear in the response?
4. **Refusal rate** — does the model refuse harmful prompts?

### Interview pair
**Q: What's the fastest way to improve format adherence?**
A: SFT with strict templates + eval gating.  Format is surface-level behaviour
— the model just needs to see enough examples.  50–100 SFT examples often
suffice.  Alignment helps when the format trade-off is complex, but for simple
rules SFT is faster and more reliable.

---

## Day 14 — Polish

### Goal
Ship a repo you'd be proud to paste into a take-home.

### Checklist
- [x] Clean README with tradeoffs focus (not bragging numbers)
- [x] `requirements.txt` with pinned minimum versions
- [x] Quickstart commands (full run + smoke test)
- [x] "Common Issues" section (OOM, NaN, tokenizer mismatch)
- [x] "Design Choices" section (prompt template, loss masking, eval contract)
- [x] All interview Q&A pairs documented in source files and this guide

---

## All Interview Q&A (Quick Reference)

| Day | Question | Core Answer |
|-----|----------|-------------|
| 8 | Why not just SFT more? | SFT can't express trade-offs; preferences shape a direction in behaviour space |
| 9 | What is DPO optimising? | Closed-form KL-constrained objective; chosen > rejected relative to reference, no explicit RM |
| 10 | Why is PPO hard in LLMs? | Credit assignment, reward hacking, instability, managing KL to avoid drift |
| 11 | Is GRPO on-policy? | Yes — fresh rollouts from current policy each step; near-on-policy if rollouts reused |
| 12 | When choose DPO over PPO? | Preference pairs + simplicity; PPO when you need online reward + exploration |
| 13 | Fastest way to improve format? | SFT with strict templates + eval gating; alignment helps for complex trade-offs |
| 14 | Explain project in 90 seconds | Problem → methods compared → eval contract → key tradeoffs learned |

---

## Key Commands Summary

```bash
# Activate environment
conda activate llm-learning
cd Week4/alignment_project

# Preference data
python data/make_preferences.py --max_per_source 200

# DPO
python src/train_dpo.py --config configs/dpo.yaml
python src/train_dpo.py --max_samples 200 --eval_win_rate

# PPO/RLOO
python src/train_ppo.py --max_samples 100 --kl_coef 0.1

# GRPO
python src/train_grpo.py --max_samples 100 --num_generations 4

# Eval
bash scripts/eval_all.sh --max_samples 200
```

---

## References

- [DPO paper](https://arxiv.org/abs/2305.18290) — Rafailov et al. 2023
- [PPO paper](https://arxiv.org/abs/1707.06347) — Schulman et al. 2017
- [GRPO / DeepSeekMath](https://arxiv.org/abs/2402.03300) — Shao et al. 2024
- [InstructGPT](https://arxiv.org/abs/2203.02155) — Ouyang et al. 2022
- [TRL documentation](https://huggingface.co/docs/trl) — DPOTrainer, RLOOTrainer, GRPOTrainer
- [RLOO paper](https://arxiv.org/abs/2402.14740) — Ahmadian et al. 2024
