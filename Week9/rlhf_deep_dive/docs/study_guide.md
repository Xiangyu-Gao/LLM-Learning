# Week 9 Study Guide — RLHF Mechanics Deep Dive

## What You Built

A ground-up implementation of the core algorithms behind modern LLM alignment training:

| Day | Algorithm | Key Insight |
|-----|-----------|-------------|
| 1 | PPO on a 2-arm bandit | Clipping prevents destructive updates |
| 2 | Advantage estimation (raw / centred / GAE) | Baselines reduce variance without bias |
| 3 | Full RLHF pipeline (SFT → RM → PPO) | KL penalty anchors the policy to SFT |
| 4 | GRPO | Group normalisation replaces the value function |
| 5 | KL coefficient study | β controls the bias-variance tradeoff of the reference constraint |
| 6 | Reward hacking simulation | Optimising a proxy metric degrades true quality |

---

## Day 1: PPO from First Principles

### The Policy Gradient Theorem

The central result of RL:

```
∇_θ J(θ) = E_{τ~π_θ} [ Σ_t ∇_θ log π_θ(a_t|s_t) · Q^π(s_t, a_t) ]
```

Intuition: push up the log-probability of actions proportionally to how good they are.

### REINFORCE

The simplest policy gradient algorithm:

```
θ ← θ + α · ∇ log π_θ(a) · G_t
```

where G_t is the Monte-Carlo return from step t.

**Problem**: G_t has high variance because it includes all future randomness.

### Importance Sampling Ratio

When we want to reuse old data collected under π_old:

```
r_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
```

This ratio corrects for the distributional mismatch.  If r_t >> 1, the new policy
assigns much higher probability to this action — a "large update" warning sign.

### PPO: Why Clip?

**The problem**: unconstrained importance sampling can cause catastrophic policy updates.
If the ratio r_t becomes very large, the gradient is huge and the policy overshoots.

**The solution**: clip the ratio:

```
L^CLIP = E[ min( r_t · A_t,  clip(r_t, 1-ε, 1+ε) · A_t ) ]
```

- If A_t > 0 (good action): ratio can increase at most to 1+ε before clipping kicks in.
- If A_t < 0 (bad action): ratio can decrease at most to 1-ε before clipping kicks in.

**What happens if ε → 0?**  No update is allowed (ratio is locked at 1).  The policy
can't learn at all.

**What happens if ε → ∞?**  Equivalent to unconstrained REINFORCE — can diverge.

### Clipping Frequency Over Training

Early training: high clipping frequency (large updates attempted).
Late training: clipping frequency drops to near zero (policy near optimal, small updates).
A sudden spike in clipping frequency late in training is a warning sign of instability.

---

## Day 2: Advantage Estimation

### Why Advantage, Not Return?

The advantage function:

```
A^π(s, a) = Q^π(s, a) - V^π(s)
```

measures how much better action a is compared to the average action from state s.

Using A instead of Q or G:
- Same expected gradient (baseline theorem — proved below)
- Much lower variance

### The Baseline Theorem

**Claim**: subtracting any function b(s) that does not depend on a leaves the
expected gradient unchanged:

```
E_a [ ∇ log π(a|s) · b(s) ] = b(s) · E_a [ ∇ log π(a|s) ]
                              = b(s) · ∇ E_a[1]
                              = b(s) · ∇ 1
                              = 0
```

The key step: E_a[∇ log π(a|s)] = ∇ Σ_a π(a|s) = ∇ 1 = 0.

This means any function of state alone can be subtracted — it reduces variance for free.

### Generalised Advantage Estimation (GAE)

GAE interpolates between two extremes using λ ∈ [0, 1]:

```
A_t^GAE(λ) = Σ_{l=0}^∞ (γλ)^l · δ_{t+l}
δ_t = r_t + γ V(s_{t+1}) - V(s_t)   (TD error)
```

| λ | Effective method | Bias | Variance |
|---|-----------------|------|----------|
| 0 | TD(0) | High | Low |
| 0.95 | GAE (default) | Medium | Medium |
| 1 | Monte-Carlo | None | High |

Why does **high variance** destabilise LLM fine-tuning?

Large gradient variance → large gradient updates → policy overshoot → reward collapse.
This is amplified at scale because the parameter space is enormous (billions of weights).

---

## Day 3: RLHF Pipeline Internals

### Stage 1: Supervised Fine-Tuning (SFT)

Train on human-written demonstrations:

```
L_SFT = -E_{x~D} [ log π_θ(x) ]
```

This creates the starting point for RL.  The SFT model also becomes the **reference model**
(frozen) used to compute the KL penalty.

### Stage 2: Reward Model (Bradley-Terry)

Given preference pairs (preferred sequence w, rejected sequence l):

```
P(w > l) = σ(r_φ(w) - r_φ(l))
L_RM = -E [ log σ(r_φ(w) - r_φ(l)) ]
```

The Bradley-Terry model assumes preferences are transitive and follow a logistic distribution.

**Why does the reward model overfit?**
The training dataset of preferences is finite.  The RM learns to assign high reward to
patterns in the preference data, not to true human values.  When the policy optimises
strongly against the RM, it exploits these spurious correlations.

### Stage 3: PPO + KL Penalty

The effective reward during RL:

```
r_eff(x) = r_RM(x) - β · KL(π_ref(x) || π_θ(x))
```

The PPO objective optimises:

```
L = E[ min(r_t · A_t, clip(r_t, 1-ε, 1+ε) · A_t) ]
```

where A_t is computed from r_eff.

### Why Is the KL Term Critical?

Without KL:
- The policy drifts from the SFT distribution
- It finds "reward hacking" strategies that score high on r_RM but are low quality
- The RM is no longer a reliable signal (it was trained on SFT-like outputs)

With KL:
- The policy stays close to the SFT model
- RM scores remain calibrated
- The tradeoff is controlled by β (larger β = more conservative)

### Reward Hacking in Real LLMs

Real examples:
- Adding filler words ("Great question! Absolutely!") to boost reward
- Generating overly long responses (if length correlates with reward)
- Confidently stating wrong answers (if confident tone is rewarded)
- Sycophancy: agreeing with the user's premise regardless of correctness

---

## Day 4: GRPO Mechanics

### Motivation: Why Remove the Value Function?

PPO requires a separate value network V_φ(s).  For LLMs:
- V_φ must also be a large model (same size as policy for accurate estimates)
- Training V_φ adds memory and compute overhead
- V_φ estimates can be unstable early in training, destabilising the policy

GRPO's insight: **use the reward signal itself to estimate the baseline**, by sampling
G rollouts per prompt and normalising within the group.

### Algorithm

```
For each prompt p:
  1. Sample x_1, ..., x_G ~ π_θ(· | p)
  2. Score: r_i = R(x_i, p)
  3. Normalise:  A_i = (r_i - mean_g(r)) / (std_g(r) + ε)
  4. Update:     L = E[ min(ratio · A_i, clip(ratio, 1-ε, 1+ε) · A_i) ]
               + β · KL(π_ref || π_θ)
```

### Why Group Normalisation Stabilises

The group mean is an **unbiased estimate** of the expected reward under π_θ for prompt p.
Subtracting it follows the baseline theorem: it reduces variance without introducing bias.

With G=8:
- If all 8 outputs get reward 0 → all advantages are 0 → no gradient → no update
- If 4 get reward 1 and 4 get reward 0 → clear signal: push up the rewarded ones

**Why is GRPO more stable for LLMs?**

1. No value function to train and potentially destabilise
2. Group normalisation automatically adapts to the scale of rewards
3. The baseline is computed from fresh samples (no staleness)
4. Simpler hyperparameter space

### Why GRPO is "Off-Policy"

The importance sampling ratio r_t = π_θ / π_θ_old accounts for the fact that samples
were collected under the old policy.  Because we run multiple gradient steps on the same
batch (n_epochs > 1), the policy moves away from π_θ_old during the epochs — making it
technically off-policy by the last epoch.

The clipping mechanism limits how far off-policy we go.

---

## Day 5: KL Control

### The Four Regimes

| β | Behaviour | Risk |
|---|-----------|------|
| 0.00 | No constraint on drift | Reward hacking, RM unreliable |
| 0.01 | Weak constraint | Slow drift, instability at scale |
| 0.10 | Balanced (typical) | Good tradeoff |
| 1.00 | Strong constraint | Stays near SFT, slow improvement |

### Entropy Collapse

When the policy becomes near-deterministic for some positions, the entropy at those
positions approaches zero.  This is called **mode collapse**.

Warning signs:
- Entropy drops rapidly in early training
- Reward plateaus but KL keeps growing
- The policy outputs repetitive or formulaic text

### KL vs Trust Region

PPO's clipping is a **hard** trust region: it prevents any individual update from
moving the policy ratio outside [1-ε, 1+ε].

The KL penalty is a **soft** constraint: it adds a cost for drift but does not
hard-block large updates.  Both are needed because:

- Clipping controls per-step stability
- KL controls long-term drift from the reference model

---

## Day 6: Reward Hacking

### The Core Insight

Goodhart's Law: *When a measure becomes a target, it ceases to be a good measure.*

In RLHF: the policy optimises the reward model r_RM.  But r_RM is an approximation
of human preference.  As the policy moves away from the SFT distribution, r_RM
becomes less calibrated — and the policy finds edge cases that score high on r_RM
but are not actually preferred by humans.

### Three Hacking Strategies Implemented

**1. Repetition reward**

The reward is `count(token_0) / T`.  The optimal policy under this reward:
always output token_0 at every position.  The output is completely useless
but achieves maximum reward.

Observed: entropy collapses immediately, diversity drops to zero.

**2. Keyword reward**

The reward is `1 if target_token in sequence`.  The optimal hack:
put the target token at position 0 (first), fill rest randomly.
Reward approaches 1.0 because the first position always has the target.

Observed: policy specialises position 0 heavily while ignoring other positions.

**3. Format reward**

The reward is `1 if sequence starts with OPEN and ends with CLOSE`.
The hack: always bracket the output.  Middle content is irrelevant.

Observed: strong bimodal advantage distribution (have bracket / don't have bracket).

### Detection Signals

When reward hacking is occurring:
1. Reward increases rapidly while an unmeasured quality metric falls
2. Token entropy collapses at specific positions
3. Repetition score rises (policy becomes near-deterministic)
4. KL grows fast (policy drifts from reference)
5. Human evaluators report quality degradation despite high automated scores

### Mitigation

- KL penalty (β > 0): limits drift, reduces but doesn't eliminate hacking
- Reward model ensembles: harder to simultaneously hack multiple RMs
- Constitutional AI / process reward models: reward process not just outcome
- Frequent human evaluation: detect hacking before it compounds
- Reward model updates: retrain RM on policy outputs periodically

---

## Mental Models Summary

```
REINFORCE       — unbiased, high variance, slow
PPO             — clips ratio → stable, needs value function
GRPO            — group normalises → stable, no value function
KL penalty      — soft anchor to reference model
Reward hacking  — optimising proxy ≠ optimising true objective
GAE(λ)          — interpolates MC and TD, λ≈0.95 is sweet spot
```

## Key Numbers to Remember

- PPO clip ε: 0.1–0.2
- GAE λ: 0.95–0.97
- KL coefficient β: 0.01–0.1 (depends on scale)
- GRPO group size G: 4–16 (typical)
- SFT → RM → PPO: the three-stage RLHF pipeline
