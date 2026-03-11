# Reward Modeling: Learning a Preference Function

## What is a Reward Model?

A reward model (RM) is a neural network trained to predict **which of two completions a human would prefer**.  Instead of predicting the next token, it outputs a single scalar: the "goodness" of a (prompt, response) pair.

```
RM(prompt, completion) → scalar reward r ∈ ℝ
```

Concretely, an RM is a causal language model with its language-modelling head replaced by a linear layer that produces one number.  The final-token hidden state is used as the pooled representation.

---

## Why Do We Need One?

### The SFT Ceiling

SFT (behaviour cloning) maximises:

```
L_SFT = -E[log π(y | x)]
```

Every demonstration in the dataset is treated as **equally correct**.  This has two problems:

| Problem | Why SFT Can't Fix It |
|---------|----------------------|
| **Trade-offs** | "Be concise AND thorough" are both in the corpus; SFT averages them, producing neither. |
| **Negatives** | SFT can only push the model *toward* demonstrations, never *away* from bad behaviours. |
| **Distribution** | The model learns to mimic the *average* annotator, not a principled notion of quality. |

### The Preference Advantage

A reward model turns comparisons into a score.  Given two responses A and B with A ≻ B (A preferred), training minimises the **Bradley-Terry loss**:

```
L_RM = -log σ(r(x, A) − r(x, B))
```

This is a binary cross-entropy over *relative* scores — the RM only needs to say "A > B", not assign absolute quality.  This is much easier for human annotators (and heuristics) to provide.

---

## Interview Q&A

### Q: Why not just SFT more?

**A:**  More SFT gives you more of the *same* signal — cross-entropy on demonstrations.  There are three fundamental limits:

1. **No contrastive signal.** SFT has no way to say "response A is better than B."  Every token prediction is equally weighted regardless of the quality of the demonstration.

2. **Cannot express trade-offs.** Imagine the training set contains both verbose and concise answers to similar prompts.  SFT will average them, producing a response that is neither.  A preference model can learn that conciseness is preferred *in context X* and thoroughness in *context Y*.

3. **Behaviour space vs. reward space.** SFT moves the model in *behaviour space* (towards demonstrated completions).  Alignment requires moving in *reward space* (towards higher-quality responses, however quality is defined).  Once you have a reward model, RL algorithms (PPO, GRPO) or implicit-reward methods (DPO) can optimise it directly.

**Analogy:** SFT is like learning to cook by copying recipes exactly.  A reward model is like learning to taste — once you can evaluate, you can improve beyond your recipe book.

---

### Q: How do you train a reward model?

**A:**  Collect preference pairs (x, y_chosen, y_rejected), then train with the Bradley-Terry pairwise ranking loss:

```python
r_chosen  = rm(x, y_chosen)   # scalar
r_rejected = rm(x, y_rejected) # scalar
loss = -F.logsigmoid(r_chosen - r_rejected).mean()
```

The model is a pretrained LM backbone with the final projection layer swapped from vocab-size output to a single scalar.  Good initialisations are SFT checkpoints because the backbone already understands language quality.

---

### Q: What are the failure modes of reward models?

**A:**  Several, all important in interviews:

- **Reward hacking / overoptimisation:** The policy learns to game the RM rather than satisfy the underlying human preference.  The RM is an imperfect proxy; optimising it too hard breaks it.  Solution: KL penalty against the SFT policy.
- **Distribution shift:** The RM was trained on completions from an earlier policy.  As RLHF training proceeds, the new policy produces OOD completions the RM can't score reliably.  Solution: iterative RM updates.
- **Annotation bias:** If preference data comes from a single annotator pool, the RM encodes their biases.  Solution: diverse annotators, constitutional AI.
- **Length bias:** RMs often prefer longer responses.  This is an artefact of the annotation interface, not a genuine quality signal.  Solution: length normalisation or penalisation.

---

## Architecture Diagram

```
Prompt + Completion
        │
        ▼
┌─────────────────────┐
│   Transformer       │  (initialised from SFT checkpoint)
│   Backbone          │
└─────────────────────┘
        │
        │ last-token hidden state (d_model,)
        ▼
┌─────────────────────┐
│   Linear(d, 1)      │  (reward head — replaces LM head)
└─────────────────────┘
        │
        ▼
     scalar r
```

---

## This Project

In `data/make_preferences.py` we skip human annotation and use **deterministic heuristics** to generate (chosen, rejected) pairs:

- **ifeval pairs** — format correctness: the gold response is *chosen*; a CAPS / truncated / stripped version is *rejected*.
- **trivia_qa pairs** — groundedness: the correct answer is *chosen*; a random distractor is *rejected*.

These are *noisy* preferences — suitable for understanding the pipeline, not for production.  A production system would use crowd-sourced comparisons or a strong model (e.g., GPT-4) as a preference labeller.
