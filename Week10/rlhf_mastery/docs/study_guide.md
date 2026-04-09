# Week 10 Study Guide — RLHF Mastery

## Theme: Turn Knowledge into Interview Ammunition

This week synthesises everything from Week 9 into deep, defensible understanding.
The experiments are designed so that you have *numerical evidence* to back up every
claim you make in an interview.

---

## Day 1: Scaling Behavior

### Why Does Scale Change Training Dynamics?

The core claim is: **"Scaling changes optimization curvature."**

To understand this, you need to think about the loss landscape as a surface in
parameter space.  The curvature of that surface is characterised by the Hessian
matrix H = ∇²L.  The maximum eigenvalue λ_max of H is called the *sharpness*.

**Key fact:** For a gradient step of size η, the update overshoots a minimum if:
```
η · λ_max > 2
```
This is the stability condition for gradient descent.  Larger sharpness → smaller
maximum stable learning rate.

**Why does model size increase sharpness?**

1. **More parameters = more directions to overshoot in.**  A larger parameter space
   has more axes of curvature.  The maximum eigenvalue tends to grow with model size.

2. **Expressivity amplifies reward signal.**  A larger model can change its output
   distribution more rapidly per gradient step.  In RLHF terms: a 70B model can
   move its policy distribution more than a 7B model for the same Δθ in parameter
   space.

3. **Empirical observation:** The optimal learning rate in published RLHF work scales
   roughly as 1/√(n_params), though exact scaling varies by architecture.

### What Metrics Signal Instability?

| Metric | Stable | Warning | Unstable |
|--------|--------|---------|----------|
| KL from reference | < 1.0 | 1–3 | > 3 |
| Reward trend | Rising or flat | Plateaued | Falling after peak |
| Gradient norm | Steady | 2× normal | 10× normal spike |
| Reward volatility (rolling std) | Low | Rising | Very high |

### Practical Takeaway

When scaling from 7B to 70B in RLHF:
- Reduce learning rate by at least 2–5×
- Tighten the KL constraint (smaller β, or use adaptive KL)
- Monitor gradient norms — a spike > 5× mean is a warning
- Use gradient clipping (max_norm=1.0 is standard)

---

## Day 2: On-Policy vs Off-Policy

### The Core Distinction

**On-policy:** Every gradient update uses samples from the *current* policy π_θ.

**Off-policy:** Gradient updates use samples from a *different* (older) policy π_old.

The policy gradient theorem requires:
```
∇J(θ) = E_{x ~ π_θ}[∇log π_θ(x) · r(x)]
```
The expectation **must** be under the *current* policy.  If you use samples from
an older policy, you introduce bias.

### Importance Sampling Correction

To use off-policy samples, reweight them with the importance ratio:
```
ρ(x) = π_θ(x) / π_old(x)
```

For sequences: `ρ = exp(Σ_t log π_θ(x_t) - log π_old(x_t))`

The corrected gradient is:
```
∇J ≈ (1/N) Σ_i ρ(x_i) · ∇log π_θ(x_i) · r(x_i)
```

**Problem:** When the policies differ a lot, ρ can be very large (high variance)
or very small (effectively ignoring good samples).  The variance of the IS
estimator is proportional to E[ρ²]/E[ρ]² — the effective sample size shrinks.

### Why PPO Clips Instead of Just IS-Correcting

PPO uses a clipped IS ratio:
```
L^CLIP(θ) = E[min(ρ · A, clip(ρ, 1-ε, 1+ε) · A)]
```

The clip **limits the trust region** rather than correcting the bias perfectly.
This is a deliberate design choice:
- Perfect IS correction (full ratio) → high variance, unstable
- Clipped IS → some bias remains, but variance is bounded
- The clip prevents any single sample from dominating the gradient

In PPO, ε=0.2 is the standard.  This means the policy can only change by ±20%
relative to the old policy per update.

### Why GRPO Is "Partially Off-Policy"

GRPO generates G responses per prompt using the current policy, then runs K
gradient epochs over those G responses.

- **Epoch 1:** Samples are exactly on-policy (sampled from the policy before
  the update).
- **Epoch 2+:** The policy has been updated once, so the samples are now
  off-policy by 1 step.

GRPO handles this with the same PPO clip.  This is why people say "GRPO is
on-policy at generation but off-policy within the optimization loop."

### The Staleness Problem

If you store experience in a replay buffer and sample it K steps later:
- The IS ratio ρ = π_θ / π_old grows as the policy updates
- After many updates, ρ can be >> 1, making gradients explode
- This is why PPO only runs K=4 epochs on fresh data (not an infinite replay buffer)

---

## Day 3: KL vs Entropy Regularization

### The Full RLHF Objective

```
J(θ) = E_{x~π_θ}[r(x)] - β · KL(π_θ || π_ref) + α · H(π_θ)
```

where:
- `r(x)` = reward model score
- `KL(π_θ || π_ref) = E_{x~π_θ}[log π_θ(x) - log π_ref(x)]`  ≥ 0
- `H(π_θ) = -E_{x~π_θ}[log π_θ(x)]`  ≥ 0

### What Each Term Does

**The reward term** `E[r(x)]`:
- Drives the policy toward high-reward sequences
- Necessary but not sufficient — the policy will overfit to the reward model

**The KL penalty** `-β · KL`:
- Penalises divergence from the reference policy
- Prevents reward hacking: if π_θ drifts far from π_ref, KL grows
- Acts as an "anchor" — the reference model encodes alignment properties
- If β is too large: policy can't improve (mode collapse)
- If β is too small: policy escapes to out-of-distribution regions (reward hacking)

**The entropy bonus** `+α · H`:
- Encourages policy diversity
- Prevents entropy death: without it, the policy collapses to one sequence
- The entropy gradient is `∇H = -∇E[log π] = -E[∇log π + (∇log π)·log π]`
  which effectively *penalises* high-confidence predictions
- If α is too large: policy becomes too random (doesn't exploit reward signal)

### The Mathematical Relationship

Notice that:
```
H(π_θ) = -E[log π_θ]
KL(π_θ||π_ref) = E[log π_θ] - E[log π_ref]
```

So:
```
-β·KL + α·H = -β·E[log π_θ] + β·E[log π_ref] - α·E[log π_θ]
            = -(β+α)·E[log π_θ] + β·E[log π_ref]
```

The second term is constant w.r.t. θ.  So combined, they give a *modified KL
coefficient* of (β+α).  Entropy bonus is equivalent to reducing the KL coefficient!

**But they're controlled separately in practice** because:
1. β and α have different default scales
2. They target different failure modes conceptually
3. Adaptive KL often adjusts β dynamically; α is usually fixed

### Failure Modes Without Each Term

| Missing | Failure Mode | Signature |
|---------|-------------|-----------|
| KL penalty | KL explosion / reward hacking | KL grows → reward looks high but on OOD text |
| Entropy bonus | Entropy death | Entropy → 0, model repeats one sequence |
| Both | Both failures (+ no anchor) | Catastrophic divergence |

---

## Day 4: Instability Taxonomy

### The Five Failure Modes

#### 1. Reward Collapse
**What:** Reward rises then sharply falls.

**Why:** Learning rate too high → gradient step overshoots the reward peak →
policy lands in a low-reward region and can't easily recover.

**Diagnostic:** Gradient norm spikes *before* the reward crash.  The spike
is the early warning signal.  By the time reward crashes, it's too late.

**Fix:** Reduce LR, add LR warmup, tighten gradient clipping.

#### 2. Mode Collapse
**What:** Reward plateaus below the theoretical maximum.  Entropy decays.

**Why:** β too large → KL penalty dominates the reward signal → policy
can't move far enough from the reference to reach high-reward regions.

The policy is "stuck near the reference" — it knows the right behavior
exists (the reward gradient is non-zero) but can't afford the KL cost.

**Diagnostic:** KL ≈ 0 throughout training, even though the reward is below
the maximum.  Policy outputs are near-identical to the reference model.

**Fix:** Reduce β, use adaptive KL that decreases β if KL is too low.

#### 3. KL Explosion
**What:** KL grows without bound.

**Why:** No KL penalty, or β too small → policy can drift arbitrarily.

In LLMs: the policy discovers that high rewards can be obtained by
outputting text the reward model was never trained to evaluate.  Reward
appears high, but the outputs are gibberish or adversarial patterns.

This is Goodhart's Law: when the proxy (reward model) is optimized,
it ceases to correlate with the true objective (human preference).

**Diagnostic:** KL continuously increasing, often past 5–10.

**Fix:** Add KL penalty, use KL stopping criterion (e.g., stop if KL > 3).

#### 4. Entropy Death
**What:** Entropy collapses to near 0.  Policy outputs one sequence.

**Why:** Very peaked reward function (most sequences get low reward, one
gets very high reward) combined with no entropy regularization.  The
policy converges to the single best sequence and stays there.

In LLMs: degenerate repetition ("I am a helpful assistant" looping),
or the model outputs a single token repeatedly.

**Diagnostic:** Entropy < 0.1 nats, reward appears high but on a single
sequence that might not generalise.

**Fix:** Entropy bonus α > 0, temperature scaling in sampling,
reward reshaping to be smoother.

#### 5. Gradient Spikes
**What:** Gradient norm has sharp isolated spikes.  Training is noisy.

**Why:** Batch size too small → gradient variance high → occasional
unlucky batch where one sample has very high reward AND high log-prob
gradient → catastrophically large step.

**Diagnostic:** Gradient norm time series shows clear spikes (10–100×
the mean).  Reward and KL are noisy, sometimes with sudden jumps.

**Fix:** Larger batch size, gradient clipping, reduce LR.

### How to Diagnose in Practice

```
if KL > 5:
    → KL explosion (check: is β set? is reward model being hacked?)
elif entropy < 0.2:
    → Entropy death (add entropy bonus)
elif reward rose then fell:
    if gradient spikes visible before fall:
        → Reward collapse (reduce LR)
    else:
        → Reward hacking causing OOD collapse
elif KL ≈ 0 and reward below max:
    → Mode collapse (reduce β)
elif gradient norm has spikes:
    → Gradient spikes (increase batch, add clipping)
```

---

## Day 5: Core Theorems (Numerical Verification)

### Theorem 1: E[∇log π] = 0

**Statement:** For any normalised distribution π_θ:
```
E_{x~π_θ}[∇_θ log π_θ(x)] = 0
```

**Proof:**
```
E_{x~π_θ}[∇log π_θ(x)] = Σ_x π_θ(x) · ∇log π_θ(x)
                        = Σ_x π_θ(x) · ∇π_θ(x) / π_θ(x)
                        = Σ_x ∇π_θ(x)
                        = ∇ Σ_x π_θ(x)    [swap Σ and ∇, valid under regularity]
                        = ∇ 1
                        = 0
```

**Implication:** A constant reward r(x) = c produces zero gradient.
The policy only learns from *differential* rewards.
Adding a constant to all rewards doesn't change the gradient.

### Theorem 2: Baseline Theorem

**Statement:** For any function b(s) that doesn't depend on the action:
```
E_{x~π_θ}[∇log π_θ(x) · b(s)] = 0
```

**Proof:**
```
E_{x~π_θ}[∇log π_θ(x) · b(s)] = b(s) · E_{x~π_θ}[∇log π_θ(x)]    [b(s) is constant w.r.t. x]
                                = b(s) · 0    [Theorem 1]
                                = 0
```

**Implication:** You can subtract any state-dependent baseline from the
reward without biasing the gradient estimate.  This *reduces variance*
because the advantage `A = r - b` has lower variance than r alone
when b(s) ≈ V(s).

### Theorem 3: Policy Gradient Theorem

**Statement:**
```
∇J(θ) = ∇ E_{x~π_θ}[r(x)] = E_{x~π_θ}[∇log π_θ(x) · r(x)]
```

**Derivation:**
```
∇J(θ) = ∇ Σ_x π_θ(x) r(x)
       = Σ_x [∇π_θ(x)] r(x)                         [r(x) doesn't depend on θ]
       = Σ_x π_θ(x) · [∇π_θ(x) / π_θ(x)] · r(x)   [multiply and divide by π_θ]
       = Σ_x π_θ(x) · [∇log π_θ(x)] · r(x)          [∂log f = ∂f/f]
       = E_{x~π_θ}[∇log π_θ(x) · r(x)]
```

**This is the fundamental result** that makes all policy gradient methods work.
It converts a gradient through the expectation (intractable) into an
expectation of a gradient (tractable via Monte Carlo sampling).

### Theorem 4: PPO vs GRPO

| Property | PPO | GRPO |
|----------|-----|------|
| Baseline | Value network V(s) | Group mean reward |
| Advantage | A = r - V(s) | A_i = (r_i - μ_group) / σ_group |
| Value network | Required | Not required |
| KL control | Clip + KL penalty | Clip + KL penalty |
| Stability | High (value net helps) | High (group norm stabilises) |
| Memory | More (value net params) | Less |
| Why it works | Value net estimates E[r|s] → low variance baseline | Group mean is local estimate of E[r|prompt] |

**GRPO advantage normalisation:**
```
A_i = (r_i - mean({r_1,...,r_G})) / (std({r_1,...,r_G}) + ε)
```

This is essentially *reward whitening* within a group.  It removes the
need for a value network because for the same prompt, `E[r|prompt]` is
approximated by the group mean.

### Theorem 5: DPO

**The key insight:** In RLHF, the optimal policy under a KL-constrained
reward maximisation has a closed form:
```
π*(x) = π_ref(x) · exp(r(x)/β) / Z(x)
```

This implies:
```
r(x) = β · log(π*(x)/π_ref(x)) + β · log Z(x)
```

For a preference pair (y_w preferred over y_l | prompt x):
```
P(y_w ≻ y_l) = σ(r(y_w|x) - r(y_l|x))
             = σ(β·log π*(y_w)/π_ref(y_w) - β·log π*(y_l)/π_ref(y_l))
```

DPO directly maximises this without ever training a reward model:
```
L_DPO(θ) = -E_{(x,y_w,y_l)}[log σ(β·(log π_θ(y_w|x)/π_ref(y_w|x)
                                     - log π_θ(y_l|x)/π_ref(y_l|x)))]
```

**Trade-offs vs PPO:**
- DPO: offline, no rollouts, more stable training, but requires offline preference data
- PPO: online, can use synthetic rewards, more general, but requires reward model and RL loop

---

## Key Interview Patterns

### "Derive X" questions

For any "derive X" question:
1. Start from first principles (what are we optimising?)
2. Take the gradient through the expectation
3. Apply the log-derivative trick: ∇π = π · ∇log π
4. Interpret the result as an expectation

### "Why does X work?" questions

Structure:
1. What problem does X solve?
2. What would happen without X?
3. What is the mathematical mechanism?
4. What is the empirical evidence?

### "What breaks at 70B?" questions

Scale magnifies every failure mode:
- **Sharpness** → instability at lower LRs
- **Capacity** → reward hacking is easier (model finds more adversarial patterns)
- **Drift** → same Δθ moves the policy further in output space
- **Memory** → gradient accumulation required → effective batch size changes
- **FSDP/TP** → gradient synchronisation adds noise

---

## Conceptual Depth Questions

### "Is RLHF truly optimising human values?"

No — it's optimising a *proxy* of human values (the reward model).  Three layers
of approximation:
1. Human labellers approximate their own preferences
2. The reward model approximates labeller preferences
3. The policy approximates the reward model's gradient

Each layer introduces noise and potential misalignment.  This is why RLHF can
*degrade* helpfulness: the model learns to produce text that scores well on the
reward model, not text that is actually helpful.

### "Why is alignment fundamentally unstable?"

The policy gradient pushes the policy toward high-reward sequences.  If the
reward model has any flaw (and it always does), the policy will eventually
find and exploit it.  This is Goodhart's Law: the proxy ceases to correlate
with the true objective when maximised.

Stability requires either:
- A perfect reward model (impossible)
- A strong enough KL constraint to prevent exploration (limits improvement)
- Continuous updating of the reward model (DPO online, RLHF-v, etc.)

### "What are the failure modes of preference learning?"

1. **Annotation inconsistency:** Different humans prefer different things.  The
   reward model learns a noisy average that may not match any individual.
2. **Distribution shift:** Reward model is trained on SFT outputs, but the
   policy quickly moves out of that distribution.
3. **Reward overoptimisation:** Extended training on the reward model signal
   degrades quality (too much optimisation against a proxy).
4. **Context collapse:** Simple reward models ignore context; the same response
   can be good in one context and harmful in another.
5. **Sycophancy:** The model learns to agree with the user rather than be correct,
   because agreement is systematically preferred in labelling.
