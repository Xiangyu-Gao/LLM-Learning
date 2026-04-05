# Week 9 — Interview Q&A: RLHF Mechanics

These are high-probability questions at research/engineering roles at Anthropic, OpenAI,
Google DeepMind, Meta FAIR, and top AI startups.

---

## Core Theory

**Q: Derive the policy gradient theorem from first principles.**

Start from the objective J(θ) = E_{τ~π_θ}[R(τ)].

```
∇_θ J(θ) = ∇_θ ∫ p_θ(τ) R(τ) dτ
           = ∫ ∇_θ p_θ(τ) R(τ) dτ
           = ∫ p_θ(τ) ∇_θ log p_θ(τ) R(τ) dτ    [log-derivative trick]
           = E_{τ~π_θ} [ ∇_θ log p_θ(τ) · R(τ) ]
```

Since log p_θ(τ) = Σ_t log π_θ(a_t|s_t) + const (transitions cancel):

```
∇_θ J(θ) = E [ Σ_t ∇_θ log π_θ(a_t|s_t) · R(τ) ]
```

Replace R(τ) with Q^π(s_t, a_t) (removes future reward causality irrelevant to action).

---

**Q: Why does subtracting a baseline not change the expected gradient?**

We want to show E_a[∇ log π(a|s) · b(s)] = 0 for any b(s) not depending on a.

```
E_a[∇ log π(a|s) · b(s)] = b(s) · E_a[∇ log π(a|s)]
                           = b(s) · ∇_θ E_a[1]          [swap ∇ and E]
                           = b(s) · ∇_θ 1
                           = 0
```

The key step is that ∇_θ Σ_a π_θ(a|s) = ∇_θ 1 = 0 (probabilities sum to 1).

This means V(s_t) can always be subtracted for free. The optimal baseline
minimises Var[∇ log π · (G_t - b)] → optimal b* = E[G_t^2 · ||∇ log π||^2] /
E[||∇ log π||^2], which is approximately V^π(s_t).

---

**Q: Why does PPO use clipping instead of a hard KL constraint?**

A hard KL constraint (TRPO) requires:
```
max_θ L(θ)   s.t.  KL(π_old || π_θ) ≤ δ
```
This needs computing the Fisher information matrix (or conjugate gradients),
which is O(n²) in parameters — infeasible for large models.

PPO approximates the trust region constraint with a simple clip:
```
L^CLIP = E[ min( r_t A_t,  clip(r_t, 1-ε, 1+ε) A_t ) ]
```

This is O(1) additional cost over vanilla PG, scales to billions of parameters,
and empirically achieves similar stability to TRPO.

---

**Q: What happens when ε (clip range) is too small or too large?**

- ε → 0: The ratio is always clipped immediately. No meaningful update can occur.
  Learning stops. (Gradient is zeroed out for all samples.)

- ε → ∞: Equivalent to unclipped REINFORCE with importance sampling. Catastrophic
  updates possible when r_t is very large. Training diverges.

- ε ≈ 0.1–0.2: Sweet spot. Allows meaningful updates but prevents overshooting.

---

**Q: What is GAE and what does λ control?**

Generalised Advantage Estimation:

```
A_t^GAE(λ) = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

λ controls the **bias-variance tradeoff**:

- λ = 0 → A_t = δ_t = TD error. One-step estimate. High bias (V may be wrong),
  low variance (uses only one reward sample).

- λ = 1 → A_t = G_t - V(s_t) = Monte-Carlo return minus baseline. No bias
  (uses true return), high variance (sums many random rewards).

- λ = 0.95 → Weighted sum of all TD-errors. Empirically best for most tasks.

---

**Q: Why does high advantage variance destabilise LLM fine-tuning specifically?**

Three reasons compound:

1. **Scale**: gradients are summed over billions of parameters. High-variance
   gradients cause large random fluctuations in each weight.

2. **Auto-regressive structure**: a bad update at position t corrupts all
   downstream positions (token 2 depends on token 1, etc.).

3. **Softmax sensitivity**: near the optimum, the policy's logit differences are
   large. A gradient perturbation can flip a high-confidence token probability,
   causing a cascade of errors.

---

## PPO vs GRPO

**Q: What is the fundamental difference between PPO and GRPO?**

| Aspect | PPO | GRPO |
|--------|-----|------|
| Baseline | Learned value function V_φ(s) | Group mean of rewards |
| Critic | Separate network, same size as policy | None |
| Advantage | A = Q - V_φ (GAE) | A_i = (r_i - μ_g) / σ_g |
| Memory | 2× policy size (policy + critic) | 1× policy size |
| Stability | Requires stable V_φ training | Stable by construction |
| Off-policy | On-policy (limited by clip) | More off-policy (no V_φ) |

---

**Q: Why does GRPO not need a value function?**

The value function in PPO serves as a **variance-reducing baseline** — it estimates
the expected return from state s_t.

GRPO replaces this with the **empirical mean reward within the group**:
```
μ_g = (1/G) Σ_{i=1}^G r(x_i | p)
```

This is an unbiased estimate of V(p) (the expected reward given prompt p).
With G ≥ 4, it provides a low-variance baseline without needing to train a
separate network.

The tradeoff: GRPO requires G forward passes per prompt (vs 1 for PPO).
At G=8, this is 8× more compute per prompt — but avoids the full critic
training cost. At large scale this is usually cheaper overall.

---

**Q: Why is GRPO considered "off-policy"?**

PPO is "on-policy" in the sense that data is collected from the current policy.
But when n_epochs > 1, the policy changes after each gradient step — so by
epoch 2, the data is already off-policy relative to the updated policy.

GRPO makes this more explicit: the group rollouts are all collected from π_θ_old,
and then the policy is updated using importance sampling (the ratio r_t = π_θ/π_θ_old).
The clipping ensures the ratio stays bounded, limiting the off-policy error.

The key distinction from a true off-policy algorithm (like DQN with replay buffer):
GRPO does not reuse data across multiple outer steps. The off-policy-ness is
controlled and short-lived.

---

## RLHF Pipeline

**Q: Walk me through the full RLHF pipeline.**

**Stage 1 — SFT**: Train on human-written demonstrations.
```
L_SFT = -E_{x~D_SFT} [log π_θ(x)]
```
Output: a model that can follow instructions but is not yet aligned.

**Stage 2 — Reward Modelling**: Collect human preference pairs (w preferred over l).
Train r_φ with Bradley-Terry loss:
```
L_RM = -E [(w,l)] [log σ(r_φ(w) - r_φ(l))]
```
Output: a model that assigns scalar scores reflecting human preferences.

**Stage 3 — RL Fine-tuning**: Optimise the policy against r_φ with a KL constraint:
```
r_eff(x) = r_φ(x) - β · KL(π_SFT(x) || π_θ(x))
```
Use PPO or GRPO to maximise r_eff.

---

**Q: Why is the KL penalty to the reference model critical?**

The reward model r_φ was trained on outputs from the SFT distribution.
As the policy moves away from SFT, it generates outputs that r_φ has never seen.
The RM is unreliable in this out-of-distribution regime — it may assign high scores
to degenerate outputs.

The KL penalty keeps the policy within the "reliable region" of the RM:
```
β · KL(π_ref || π_θ)
```

Without it: the policy quickly finds RM exploits → reward hacking.
Too much: the policy never improves beyond SFT → wasted compute.

---

**Q: What is reward hacking? How do you detect it?**

Reward hacking occurs when the policy finds ways to maximise r_RM that do not
correspond to true quality. The optimised reward ≠ the true objective.

**Detection signals:**
1. Automated reward rises sharply while human eval scores fall
2. Policy entropy collapses (policy becomes deterministic/repetitive)
3. Outputs become formulaic: always start/end with specific phrases
4. KL divergence from reference grows without bound
5. Reward model score keeps rising but responses feel worse qualitatively

**Prevention:**
- KL constraint (necessary but not sufficient)
- RM ensemble (harder to hack multiple independent RMs)
- Process reward models (reward steps, not just outcomes)
- Iterative RM retraining on policy outputs
- Frequent human evaluation as a ground-truth signal

---

**Q: Why do reward models overfit?**

The RM is trained on a finite preference dataset.  It learns to assign high scores
to surface features that correlate with human preference in the training data
(e.g., specific formatting, confident tone, certain vocabulary).

These surface features are spurious correlates — they happen to co-occur with
quality in the training data but are not causal.

When the policy optimises hard against the RM, it finds these spurious correlates
and amplifies them.  The result: high RM score, low actual quality.

---

**Q: What is the Bradley-Terry model?**

A pairwise comparison model. Given items i and j with latent strengths s_i, s_j:

```
P(i preferred over j) = σ(s_i - s_j) = exp(s_i) / (exp(s_i) + exp(s_j))
```

In RLHF, s_i = r_φ(x_i).  The loss is negative log-likelihood:

```
L = -E [log σ(r_φ(preferred) - r_φ(rejected))]
```

Assumptions: preference noise is logistic, preferences are transitive.
These assumptions are violated by real human preferences (humans are inconsistent
and non-transitive) — this is why RM overfitting is an inherent problem.

---

## Practical Engineering

**Q: What metrics would you log during RLHF training?**

- `reward/mean`, `reward/std` — learning signal quality
- `kl_from_ref` — drift from reference model (alert if > 0.5)
- `policy_entropy` — exploration health (alert if dropping toward 0)
- `clip_fraction` — PPO health (should decrease over time)
- `value_loss` (PPO only) — critic stability
- `gradient_norm` — training stability (alert if > 10)
- `reward_std_within_group` (GRPO) — diversity of rollouts
- Automated quality proxy (perplexity on held-out set, format score, etc.)

---

**Q: How do you tune the KL coefficient β?**

Start with β = 0.04–0.1.  Then monitor:

- Too low: KL grows unboundedly → increase β
- Too high: reward never improves beyond SFT baseline → decrease β
- Adaptive β: use a PID controller that targets a fixed KL budget δ:
  ```
  β ← β + α · (KL - δ)
  ```
  This is used in practice by several labs (including in the original InstructGPT paper).

---

**Q: Why do larger models require smaller learning rates in RLHF?**

Three reasons:

1. **Steeper loss landscape**: larger models have more parameters creating
   sharper curvature. A large step in a bad direction is more destructive.

2. **Gradient amplification**: with billions of parameters, even a small
   per-parameter gradient can sum to a large weight change.

3. **Emergent brittleness**: larger models have more complex internal structure.
   A bad RLHF update can destroy capabilities learned during pretraining
   (catastrophic forgetting) — harder to recover than in smaller models.

Rule of thumb: use 1/10 to 1/100 of the pretraining learning rate.

---

## DPO vs PPO vs GRPO

**Q: Compare DPO, PPO, and GRPO.**

| | DPO | PPO | GRPO |
|---|---|---|---|
| **Requires RL** | No | Yes | Yes |
| **Value function** | No | Yes | No |
| **Reward model** | Implicit | Explicit | Explicit |
| **KL control** | Built-in (via β) | Explicit penalty | Explicit penalty |
| **On-policy data** | Not needed | Required | Required |
| **Compute** | Low (SFT-like) | High (policy + critic) | Medium (G rollouts) |
| **Stability** | High | Medium | High |
| **Best for** | Offline preference data | Complex rewards | Simple/sparse rewards |

DPO (Direct Preference Optimisation) reformulates RLHF as a supervised problem:
it shows that the optimal PPO policy under the KL-penalised objective has a
closed-form solution, allowing the RM and RL to be merged into a single
supervised loss on preference pairs.

Practical tradeoff: DPO is simpler but less flexible (can't use arbitrary reward
functions). PPO/GRPO are more powerful but require online rollout infrastructure.

---

## Conceptual Depth

**Q: Is RLHF truly optimising human values?**

No — for several fundamental reasons:

1. **Proxy problem**: RLHF optimises r_RM(x), not actual human preference.
   r_RM is a learned approximation — it overfits to training data and
   extrapolates poorly.

2. **Annotation noise**: human preferences are inconsistent, context-dependent,
   and not always representative of well-considered values.

3. **Goodhart's Law**: optimising any proxy metric causes it to diverge from
   the true objective at the optimum.

4. **Distributional shift**: the policy generates outputs that annotators have
   never seen. Preferences may not generalise to novel outputs.

What RLHF actually does: makes the model *appear* more helpful, harmless, and
honest on the distribution of tasks similar to the training prompts.
This is valuable but not the same as optimising "true values."

---

**Q: Why is alignment fundamentally unstable?**

Several reasons compound:

1. **Moving target**: human preferences change; the optimal policy today is
   not optimal tomorrow.

2. **Model-shifting reward**: as the policy changes, RM scores become less
   calibrated (RM was trained on the old policy's distribution).

3. **Gradient instability at scale**: see "high variance" question above.

4. **Non-convex optimisation**: the RLHF objective has many local optima,
   including reward-hacking optima that are hard to escape once reached.

5. **KL-reward tradeoff is task-specific**: the right β changes as the task
   distribution changes — there is no universally correct value.
