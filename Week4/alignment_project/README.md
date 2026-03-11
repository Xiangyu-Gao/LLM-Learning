# Alignment Project — Week 4

DPO vs PPO vs GRPO: a hands-on comparison of LLM alignment methods, all applied
to GPT-2 with the same preference data, proxy reward, and evaluation suite.

---

## Project Structure

```
alignment_project/
├── configs/
│   └── dpo.yaml                   # DPO training configuration
├── data/
│   ├── make_preferences.py        # Day 8: Build (chosen, rejected) pairs
│   └── preferences/               # Generated train.jsonl + eval.jsonl
├── docs/
│   ├── reward_modeling_intuition.md   # Day 8: Why reward models exist
│   ├── ppo_knobs.md                   # Day 10: PPO hyperparameter guide
│   ├── on_vs_off_policy.md            # Day 11: On/off-policy clarity
│   ├── alignment_tradeoffs.md         # Day 12: DPO vs PPO vs GRPO table
│   └── study_guide.md                 # Full walkthrough guide
├── scripts/
│   └── eval_all.sh                # Day 13: Run eval suite across all models
├── src/
│   ├── train_dpo.py               # Day 9: DPO trainer
│   ├── train_ppo.py               # Day 10: PPO/RLOO trainer
│   └── train_grpo.py              # Day 11: GRPO trainer
├── results/                       # Outputs, checkpoints, eval tables
├── requirements.txt
└── README.md
```

---

## Design Choices

### Preference Data
Preference pairs are constructed from the Week 3 SFT datasets using
deterministic heuristics — not human labels.  This is noisy but sufficient
to demonstrate the alignment pipeline:

- **ifeval pairs:** format correctness (gold response vs. degraded version)
- **trivia_qa pairs:** groundedness (correct answer vs. random distractor)

See `data/make_preferences.py` for the full rationale.

### Prompt Template
Same `<|user|>` / `<|assistant|>` chat template as Week 3 to keep tokenisation
consistent across SFT → DPO → PPO → GRPO.

### Loss Masking
DPO applies masking via TRL's DPOTrainer (chosen/rejected pairs).  PPO and
GRPO generate completions on-policy, so no explicit masking is needed — the
reward signal drives learning.

### Proxy Reward (PPO & GRPO)
A simple heuristic combining format score, length score, and prompt-overlap
grounding.  Same function in both trainers for fair comparison.

### Reference Model
All alignment methods use the Week 3 SFT checkpoint as the KL anchor / reference
policy.  If it doesn't exist, the scripts fall back to base GPT-2.

---

## Key Tradeoffs

| Method | Data Needs | Stability | Infra Complexity | When to Use |
|--------|-----------|-----------|------------------|-------------|
| **SFT** | Demonstrations | High | Low | Format, style, starting point |
| **DPO** | Preference pairs | Medium | Low (2× model) | Have pairs, want simplicity |
| **PPO** | Prompts + reward fn | Low–Med | High (gen + reward + value) | Online reward, need exploration |
| **GRPO** | Prompts + reward fn | Medium | Medium (no value head) | Same as PPO, simpler setup |

See `docs/alignment_tradeoffs.md` for the full comparison.

---

## Quickstart

```bash
# 1. Environment
conda activate llm-learning
cd Week4/alignment_project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate preference data (uses Week 3 data)
python data/make_preferences.py --max_per_source 200

# 4. Train DPO (fastest alignment run)
python src/train_dpo.py --max_samples 200 --epochs 1

# 5. Train PPO/RLOO
python src/train_ppo.py --max_samples 100 --epochs 1

# 6. Train GRPO
python src/train_grpo.py --max_samples 100 --epochs 1

# 7. Run full eval suite
bash scripts/eval_all.sh --max_samples 200
```

Smoke test (verify everything runs in ~2 min):
```bash
python data/make_preferences.py --max_per_source 50
python src/train_dpo.py --max_samples 32 --epochs 1
python src/train_ppo.py --max_samples 20 --epochs 1
python src/train_grpo.py --max_samples 20 --epochs 1
bash scripts/eval_all.sh --max_samples 50
```

---

## Common Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| `FileNotFoundError: No parquet files` | Week 3 data not found | Pass `--w3_data_dir` pointing to Week 3's `data/` folder |
| OOM on GPU | PPO/GRPO generate G completions per prompt | Reduce `--batch_size` or `--num_generations`; use CPU with `--device cpu` |
| NaN loss in DPO | Chosen and rejected too similar | Increase preference margin diversity; check `--beta` isn't too low |
| Tokenizer mismatch | Different chat template between SFT and alignment | All scripts inject the same `CHAT_TEMPLATE`; ensure `--sft_model_path` is correct |
| RLOO requires `num_generations ≥ 2` | Leave-one-out baseline needs at least 2 samples | Set `--num_generations 2` (default) |
| Reward hacking (KL explodes) | β too low | Increase `--kl_coef` (try 0.1–0.3) |

---

## Evaluation Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Format pass rate** | Fraction of generated responses ending with sentence-final punctuation |
| **Win rate** | Fraction of eval pairs where model assigns higher log-prob to chosen vs rejected |
| **Grounded QA** | Fraction of TriviaQA answers appearing verbatim in the generated response |
| **Refusal rate** | Fraction of responses containing refusal keywords |

Results are saved to `results/eval_results.tsv` and `results/example_table.md`.

---

## Interview: Explain This Project in 90 Seconds

> **Problem:** SFT can copy style but can't resolve preference trade-offs — it
> treats every demonstration equally.  We need alignment methods that push the
> model toward preferred and away from dispreferred behaviours.
>
> **Methods compared:** DPO (offline, uses preference pairs directly), PPO/RLOO
> (online, generates completions and scores them with a proxy reward), and GRPO
> (online, like PPO but replaces the value function with group-relative baselines).
>
> **Evaluation contract:** Four metrics — format adherence, preference win-rate,
> grounded QA accuracy, and refusal correctness — measured identically across all
> methods for fair comparison.
>
> **Key tradeoffs learned:** DPO is simpler and more stable but limited to offline
> preference data.  PPO gives the most control but is hardest to tune.  GRPO sits
> in between — on-policy training without a value head.  The right choice depends
> on whether you have a computable reward signal or fixed preference pairs.

---

## References

- [DPO paper](https://arxiv.org/abs/2305.18290) — Rafailov et al. 2023
- [PPO paper](https://arxiv.org/abs/1707.06347) — Schulman et al. 2017
- [GRPO / DeepSeekMath](https://arxiv.org/abs/2402.03300) — Shao et al. 2024
- [InstructGPT](https://arxiv.org/abs/2203.02155) — Ouyang et al. 2022
- [TRL documentation](https://huggingface.co/docs/trl) — DPOTrainer, RLOOTrainer, GRPOTrainer
