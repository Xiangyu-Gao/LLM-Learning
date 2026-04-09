# Week 10 Interview Q&A — RLHF Mastery

## Core Technical: Derivations

---

**Q: Derive the policy gradient theorem.**

Start from the objective `J(θ) = E_{x~π_θ}[r(x)] = Σ_x π_θ(x) r(x)`.

Take the gradient:
```
∇J(θ) = Σ_x [∇π_θ(x)] r(x)
       = Σ_x π_θ(x) · [∇log π_θ(x)] · r(x)     (log-derivative trick: ∇f = f·∇log f)
       = E_{x~π_θ}[∇log π_θ(x) · r(x)]
```

The key step is the log-derivative trick `∇π = π · ∇log π`.  This converts a gradient
through the expectation into an expectation of a gradient — tractable via Monte Carlo.

---

**Q: Why does E[∇log π] = 0?**

```
E_{x~π_θ}[∇log π_θ(x)] = Σ_x π_θ(x) · ∇π_θ(x)/π_θ(x)
                        = Σ_x ∇π_θ(x)
                        = ∇ Σ_x π_θ(x)    (swap Σ and ∇)
                        = ∇1 = 0
```

Implication: a constant reward produces zero gradient.  The policy only learns from
*differential* rewards.  Adding a constant c to all rewards doesn't change learning.

---

**Q: Why does subtracting a baseline not bias the gradient?**

The gradient with baseline b(s) is:
```
∇J_b(θ) = E[∇log π_θ(x) · (r(x) - b(s))]
         = E[∇log π_θ(x) · r(x)] - E[∇log π_θ(x) · b(s)]
```

The second term is zero by the baseline theorem:
```
E[∇log π_θ(x) · b(s)] = b(s) · E[∇log π_θ(x)] = b(s) · 0 = 0
```

So `∇J_b = ∇J`.  The baseline reduces variance because `(r - b)` has lower variance
than `r` when `b ≈ E[r|s]`, but it doesn't change the *expectation* of the gradient.

---

**Q: Why does PPO clip?**

PPO optimises a clipped surrogate objective:
```
L^CLIP = E[min(ρ · A,  clip(ρ, 1-ε, 1+ε) · A)]
```
where `ρ = π_θ(a|s) / π_old(a|s)`.

Without clipping, using old samples requires importance sampling (IS).  Full IS works
in theory but has unbounded variance when ρ >> 1 (the new policy is much more likely
than the old one on some action).  The clip limits this: ρ can't exceed 1+ε, so the
maximum contribution of any single sample is bounded.

The `min` ensures that if ρ is already on the wrong side (A > 0 but ρ > 1+ε, meaning
the policy already moved too far), the clipped objective is used — preventing further
movement in that direction.

---

**Q: Why does GRPO remove the value network?**

In PPO, the value network estimates `V(s) = E[r|s]` to use as a baseline.
For LLMs, the "state" is the prompt, and for a given prompt, we can directly estimate
`E[r|prompt]` by sampling G responses and computing the mean reward.

GRPO does exactly this: sample G responses per prompt, compute their mean, use that
as the baseline.  Advantage: `A_i = (r_i - μ_group) / σ_group`.

This eliminates the value network entirely:
- No extra forward/backward pass for the critic
- No critic warmup phase
- No value network overfitting
- Lower memory footprint

Trade-off: GRPO requires G > 1 responses per prompt (extra inference cost at rollout
time), whereas PPO only needs 1 response per prompt.

---

**Q: How does group normalisation reduce variance?**

Group normalisation standardises rewards within a group:
```
A_i = (r_i - mean(r_1,...,r_G)) / (std(r_1,...,r_G) + ε)
```

This works because:
1. **Mean subtraction** eliminates the prompt-level reward offset.  Different prompts
   have different "natural" reward levels; subtracting the group mean removes this
   confound.
2. **Std division** makes the gradient scale invariant.  Whether rewards are in [0,1]
   or [-100, 100], the advantage magnitude is comparable.
3. **Local estimation** of E[r|prompt] via the group mean is lower variance than a
   global value network when the reward distribution is highly prompt-dependent.

---

## Practical Engineering

---

**Q: How do you detect reward hacking?**

Reward hacking occurs when the policy finds high-reward outputs that are not actually
high quality.  Indicators:

1. **KL explosion:** KL > 3 suggests the policy has moved far from the reference,
   potentially into regions the reward model wasn't trained on.

2. **Reward-quality gap:** Reward model score goes up, but held-out human evaluations
   go down.  The reward model is being "gamed."

3. **Entropy collapse:** Policy concentrates on a small set of patterns (e.g., always
   starting with a specific phrase the reward model rates highly).

4. **OOD indicator:** Run a calibrated reward model on held-out data.  If RM score
   increases but uncertainty also increases, the policy is extrapolating.

5. **Win-rate degradation:** Measure win rate against baseline SFT.  Reward hacking
   often causes the policy to "win" on the reward model but "lose" in A/B tests.

---

**Q: What metrics would you log during RLHF training?**

Essential:
- Reward (mean, std, min/max per batch)
- KL from reference (mean per token)
- Token entropy (mean per position)
- Gradient norm (pre and post clip)
- Policy loss and value loss (if using PPO)
- Clip fraction (PPO) — what % of ratios are clipped

Derived:
- Reward volatility (rolling std)
- KL / step (rate of policy drift)
- Effective learning rate (after clipping)

Red flags to alert on:
- KL > 3.0 → pause and investigate
- Gradient norm spike > 10× mean → possible instability
- Entropy < 0.5 nats → entropy death risk
- Reward falling after peak → possible reward collapse

---

**Q: Why does RLHF sometimes degrade helpfulness?**

Several mechanisms:

1. **Sycophancy optimisation:** Reward models are often trained by human labellers who
   prefer confident, agreeable responses.  The policy learns to agree rather than be accurate.

2. **KL overfitting:** If β is too high, the policy can't deviate enough from the reference
   model to provide genuinely improved responses — only superficial changes.

3. **Reward model distribution shift:** The reward model was trained on SFT-level outputs.
   As the policy improves, it generates outputs that look different from training data,
   and the reward model's predictions become unreliable.

4. **Capability degradation:** RLHF optimises for human preference, which often favours
   clear, simple responses.  The policy may learn to suppress nuanced or technically
   precise responses that score lower on preference labels.

---

**Q: How do you tune the KL coefficient β?**

Three approaches:

**1. Fixed β sweep:**
Try β ∈ {0.01, 0.1, 0.5, 1.0}.  Monitor KL trajectory.  A good β keeps KL in
[0.5, 2.0] for most of training.  β too low → KL explosion; β too high → mode collapse.

**2. Adaptive KL:**
Set a target KL `d_target` (e.g., 0.1 per token).  After each update:
```
if KL > 1.5 * d_target:  β *= 1.5   (tighten)
if KL < d_target / 1.5:  β /= 1.5   (loosen)
```
This is the approach used in the original InstructGPT paper.

**3. KL stopping criterion:**
Set a maximum KL threshold (e.g., 3.0).  Stop training if KL exceeds it.
Simpler than adaptive, but wastes compute if triggered early.

Practical starting point: β=0.1 for 7B models, β=0.05–0.1 for 70B models.

---

**Q: What breaks at 70B that doesn't at 7B?**

**Training dynamics:**
- Sharpness is higher → instability at lower LRs (use lr ≤ 5e-7 vs 1e-6 for 7B)
- Gradient norm spikes are more severe (tighten clip to max_norm=0.5)
- KL grows faster per update (reduce β or use tighter KL target)

**Infrastructure:**
- Must use FSDP/DeepSpeed ZeRO-3 → gradient allreduce adds noise
- Gradient accumulation required → effective batch size may not match nominal
- Mixed precision more critical → BF16 preferred over FP16

**Reward hacking:**
- More expressive → finds reward model flaws faster
- Can memorise specific patterns that fool the reward model
- Monitor reward/quality gap more closely

**Value network (PPO):**
- A 70B value network is expensive; GRPO or simpler baselines preferred
- Alternatively: use a smaller separate value model

---

## Conceptual Depth

---

**Q: What does "on-policy" really mean in the LLM context?**

In the LLM context, on-policy means: **the sequences used for gradient updates were
sampled from the same policy checkpoint that is being updated.**

In practice:
- **Strictly on-policy:** sample 1 batch, do 1 gradient step, discard the batch.
  This is correct but compute-inefficient.
- **PPO-style:** sample 1 batch, do K gradient steps (K=4 typical).  Steps 2–K are
  technically off-policy by 1–(K-1) updates.  The PPO clip limits the damage.
- **Off-policy:** use a replay buffer.  Samples could be many updates old.

The importance of on-policy comes from the policy gradient theorem: the expectation
must be under the *current* policy.  Stale samples bias the gradient.

---

**Q: Compare DPO vs PPO vs GRPO.**

| | DPO | PPO | GRPO |
|--|-----|-----|------|
| **Learning signal** | Preference pairs (offline) | Scalar reward (online) | Scalar reward (online) |
| **Reward model** | Not needed | Required | Required |
| **Value network** | Not needed | Required | Not needed |
| **On/off-policy** | Fully offline | On-policy (K epochs) | On-policy (K epochs) |
| **Memory** | Low | High (policy + ref + value) | Medium (policy + ref) |
| **Stability** | Very stable | Stable with tuning | Stable |
| **Flexibility** | Only pairwise prefs | Any scalar reward | Any scalar reward |
| **Key weakness** | Needs offline pairs; can't explore | Complex, many hyperparams | G rollouts per prompt costly |

**When to use each:**
- **DPO:** You have a good preference dataset and want simple, stable training
- **PPO:** You have a trained reward model and want online optimisation
- **GRPO:** You want online optimisation without the value network complexity

---

**Q: Why is alignment fundamentally unstable?**

Because of Goodhart's Law: *"When a measure becomes a target, it ceases to be a
good measure."*

The RLHF loop:
1. Train reward model on human preference data
2. Train policy to maximise reward model
3. Policy finds inputs that maximise the reward model but not human preference

There is no stable fixed point where the policy is perfectly aligned and the reward
model is perfectly calibrated, because:

1. The reward model is a *finite-capacity approximation* of human preferences.  It
   has regions where its predictions are wrong.
2. A powerful enough policy will find those regions.
3. Even if you retrain the reward model on the new policy outputs, the new reward
   model has different blind spots the policy will find.

Mitigation strategies (none are perfect):
- KL constraint: limits how far the policy can deviate (limits exploitation)
- Reward model ensembles: harder to fool multiple models simultaneously
- Constitutional AI / RLAIF: use LLM-generated feedback (may propagate biases)
- Iterative feedback: retrain reward model on policy outputs repeatedly

---

**Q: What are the failure modes of preference learning?**

1. **Annotation inconsistency:** Human labellers disagree.  The reward model learns
   a noisy mixture of preferences.  The policy may learn to satisfy the labeller
   majority while violating minority preferences.

2. **Distribution shift at inference:**
   - Reward model trained on distribution D_train
   - Policy outputs distribution D_policy after optimisation
   - D_policy ≠ D_train → reward model predictions unreliable

3. **Reward overoptimisation:** The longer you train against the reward model, the
   worse the real-world quality gets.  There's an optimal stopping point.
   Empirically this is often around KL = 10–30 nats total.

4. **Context collapse:** Reward models typically operate on (prompt, response) pairs.
   The same response can be preferred in one context and harmful in another.  Simple
   reward models can't capture this.

5. **Sycophancy:** Labellers systematically prefer responses that agree with them.
   The model learns to tell users what they want to hear, not what is true.

6. **Safety/helpfulness trade-off:** Safety labellers and helpfulness labellers often
   disagree.  The reward model may have conflicting gradients, leading to unpredictable
   trade-offs at the policy level.

---

## High-Probability Interview Questions (OpenAI/Anthropic/Meta level)

**Q: What is reward hacking in LLMs?**

Reward hacking occurs when the policy finds outputs that score high on the reward model
but are not actually high quality.  Classic examples: repetition of phrases the reward
model rates highly, excessive caveats (safety-washing), sycophantic agreements, or
format exploitation (bullet points score higher than prose regardless of content quality).

---

**Q: Why does advantage normalisation help?**

Advantage normalisation (`A ← (A - μ) / (σ + ε)`) stabilises training by:
1. **Scale invariance:** Advantages from different prompts/tasks may have wildly
   different scales.  Normalisation makes the gradient step size consistent.
2. **Gradient magnitude control:** Without normalisation, a batch with unusually high
   rewards generates a large gradient step, which can destabilise training.
3. **Implicit baseline:** The mean subtraction is equivalent to adding a batch-level
   baseline, which reduces variance by Theorem 2.

---

**Q: Why do large models require smaller learning rates in RLHF?**

Two reasons:
1. **Sharpness:** Larger models have higher curvature in the loss landscape.  The
   stability bound is `η < 2/λ_max`, where λ_max grows with model size.
2. **Output sensitivity:** For the same change in parameters Δθ, a larger model
   changes its output distribution more.  This means a given LR causes a larger
   effective policy shift (in KL terms) for a 70B model than a 7B model.

Practical guideline: halve the learning rate for every 10× increase in model size.

---

**Q: Why does KL control matter in RLHF?**

Without KL control, the policy will:
1. Drift far from the SFT model (capability degradation)
2. Find adversarial patterns that exploit reward model flaws (reward hacking)
3. Output incoherent or harmful text that the reward model wasn't trained to evaluate

The KL constraint says: "improve, but stay close enough to the reference model
that your capabilities are preserved."  The reference model encodes the knowledge
and instruction-following ability instilled during SFT.  If the policy drifts too
far, it loses these properties.

Mathematically: the optimal RLHF policy is `π* ∝ π_ref · exp(r/β)`.  As β → ∞
(no KL penalty), π* → argmax r, which is degenerate.  As β → 0 (strong KL), π* → π_ref,
which means no improvement.  The right β balances these.
