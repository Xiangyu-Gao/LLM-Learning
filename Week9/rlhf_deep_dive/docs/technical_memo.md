# PPO vs GRPO for Large Language Model Alignment: A Stability Perspective

*Week 9 Technical Memo — RLHF Mechanics Deep Dive*

---

## Abstract

Reinforcement Learning from Human Feedback (RLHF) has become the dominant paradigm for
aligning large language models (LLMs) with human intent.  At its core, RLHF is a policy
optimisation problem: given a reward signal derived from human preferences, update the
language model's parameters to increase expected reward while remaining "close" to a
reference policy.  Two algorithms dominate modern RLHF pipelines: Proximal Policy
Optimisation (PPO) and Group Relative Policy Optimisation (GRPO).  This memo examines
their gradient-level mechanics, the role of KL regularisation, and why GRPO has become
attractive at scale — despite being a seemingly crude approximation of PPO.

---

## 1. The Policy Gradient Foundation

Both PPO and GRPO rest on the policy gradient theorem.  For a language model generating
sequences x = (x_1, ..., x_T) from prompt p, the objective is:

```
J(θ) = E_{x ~ π_θ(·|p)} [ R(x, p) ]
```

where R is the reward signal (e.g., a reward model score minus a KL penalty).

The gradient of J with respect to the policy parameters θ is:

```
∇_θ J(θ) = E_{x ~ π_θ} [ ∇_θ log π_θ(x|p) · R(x, p) ]
           = E_{x ~ π_θ} [ Σ_{t=1}^{T} ∇_θ log π_θ(x_t | x_{<t}, p) · A(x, t) ]
```

The second form uses the advantage function A(x, t) in place of R.  A captures how
much better token x_t is at step t, relative to the expected value.  Critically,
subtracting any baseline b(s_t) from R does not change the expected gradient (the
baseline theorem), but dramatically reduces gradient variance.  The optimal baseline
is the state-value function V^π(s_t) — approximated by a critic network in PPO,
and by group statistics in GRPO.

---

## 2. PPO: The Clipped Surrogate

The core challenge in policy optimisation is the **step-size problem**: how far should
we move the policy at each update?  Moving too far can catastrophically destroy
previously learned capabilities.  Moving too conservatively wastes compute.

### 2.1 Importance Sampling Ratio

PPO separates data collection from optimisation.  Data is collected from the old policy
π_θ_old, then the policy is updated for n_epochs steps.  Because the policy changes
between collection and update, we must correct for this distributional mismatch:

```
r_t = π_θ(x_t | x_{<t}, p) / π_θ_old(x_t | x_{<t}, p)
```

The unclipped gradient estimator is:

```
L^IS = E [ r_t · A_t ]
```

This is unbiased but can have enormous variance when r_t is large.

### 2.2 The Clipped Objective

PPO's key contribution is to clip the ratio:

```
L^CLIP = E [ min( r_t · A_t,  clip(r_t, 1-ε, 1+ε) · A_t ) ]
```

The clip operation makes the objective pessimistic:
- If A_t > 0 (action was good): the objective stops benefiting once the policy assigns
  probability > (1+ε) times what the old policy did.
- If A_t < 0 (action was bad): the objective stops penalising once the policy assigns
  < (1-ε) times the old probability.

This implements an implicit trust region: the policy cannot move the token
probabilities outside a multiplicative factor [1-ε, 1+ε] in a single update.

### 2.3 The Value Function Requirement

PPO requires a critic V_φ(s_t) to compute the advantage:

```
A_t^GAE = Σ_{l=0}^{T-t} (γλ)^l · (r_{t+l} + γV_φ(s_{t+l+1}) - V_φ(s_{t+l}))
```

For LLMs, s_t = (p, x_1, ..., x_{t-1}).  Computing V_φ(s_t) requires a model of
comparable capacity to the policy — because the state space is identical.  This means:

- **Memory cost**: 2× the policy parameters (policy + critic, or critic head added to policy)
- **Optimisation complexity**: two interleaved optimisation problems (actor + critic)
- **Instability risk**: if V_φ is inaccurate early in training, the advantage estimates
  are noisy, destabilising the policy update

---

## 3. GRPO: Group Relative Policy Optimisation

GRPO (DeepSeek-R1, 2025) replaces the learned critic with a statistical baseline
computed from G parallel rollouts of the same prompt.

### 3.1 Algorithm

For each prompt p in a batch:

```
1. Sample x_1, ..., x_G ~ π_θ(·|p)
2. Score: r_i = R(x_i, p)  for i = 1, ..., G
3. Normalise within group:
   A_i = (r_i - μ_g) / (σ_g + ε)
   where μ_g = mean({r_i}), σ_g = std({r_i})
4. PPO clipped update using A_i as the advantage for sequence x_i
5. KL penalty: subtract β · KL(π_ref || π_θ) from the reward
```

### 3.2 Why Group Normalisation Works

The group mean μ_g is an **unbiased estimate of V^π(p)** — the expected reward
for prompt p under the current policy.  By the baseline theorem, subtracting μ_g
does not change the expected gradient but reduces its variance.

The group standard deviation σ_g normalises the scale of the advantage signal.
This is analogous to advantage normalisation in PPO, but computed locally per prompt
rather than globally across the batch.

**Variance of the estimator**: with G rollouts, the variance of μ_g is Var[R] / G.
With G=8, this is 8× lower variance than using a single sample as the baseline.
The value function V_φ in PPO achieves lower variance only if it is well-trained —
an assumption that frequently fails early in training.

### 3.3 The Off-Policy Dimension

Because GRPO samples G rollouts and then performs multiple gradient steps on them,
the policy moves away from π_θ_old during the update.  By the final epoch, the
data is off-policy.  The importance sampling ratio r_t = π_θ / π_θ_old corrects
for this, and clipping bounds the correction.

GRPO is therefore a **limited off-policy algorithm**: off-policy within a group of
rollouts, but fully on-policy across outer steps.  This is a deliberate design
choice — it allows efficient use of G rollouts without the instability of large
replay buffers.

---

## 4. KL Regularisation: The Invisible Stabiliser

Both PPO and GRPO include a KL penalty to the reference policy:

```
r_eff(x) = r_RM(x) - β · KL(π_ref(x || π_θ(x))
```

where KL is computed token-by-token over the generated sequence.

### 4.1 Why KL is Necessary

The reward model r_RM was trained on outputs from the SFT distribution.
As the policy moves into new territory (high KL from reference), the RM
operates out-of-distribution.  Its scores become unreliable — potentially
assigning high values to degenerate outputs that happen to pattern-match
features associated with quality in the training data.

The KL penalty creates a "trust region" for the reward model, not just for
the policy.  The RM is trusted to be calibrated within β-KL of the reference.
Beyond that, scores are extrapolations that may not reflect true quality.

### 4.2 The Tradeoff

| β | Effect |
|---|--------|
| β = 0 | Policy drifts freely; RM exploits emerge quickly |
| β too small (0.01) | Slow drift; instability accumulates over many steps |
| β ≈ 0.04–0.1 | Practical sweet spot; controlled improvement |
| β too large (1.0) | Policy barely moves from SFT; reward improvement stalled |

An adaptive β is often used in practice:

```
β_{t+1} = β_t + α · (KL_t - δ_target)
```

This is a PID controller targeting a fixed KL budget δ_target.  It automatically
adjusts β as the policy's tendency to drift changes.

### 4.3 KL vs Entropy Regularisation

A related technique is entropy regularisation:

```
r_eff = r_RM(x) + λ · H(π_θ)
```

**KL** anchors the policy to a specific distribution (the reference model).
**Entropy** only requires diversity, without specifying what the policy should
be diverse about.

KL is stronger: it both prevents collapse *and* prevents drift from reference.
Entropy only prevents collapse.  Both can be used simultaneously.

---

## 5. Why GRPO is Attractive at Scale

### 5.1 Memory Efficiency

PPO requires maintaining a critic network — typically of the same size as the policy
for accurate value estimates.  For a 70B parameter model:

- PPO: 70B (policy) + 70B (critic) = 140B parameters in memory
- GRPO: 70B (policy) + 70B (reference, frozen) = 140B parameters in memory

At first glance this is equivalent.  But:

- The PPO critic requires gradient computation and an optimiser state (Adam: 2× params)
- The GRPO reference is frozen — no gradients, no optimiser state
- Net GRPO advantage: ~2× less optimiser memory for equivalent model size

### 5.2 Optimisation Simplicity

PPO has two interleaved optimisation problems:
1. Policy: maximise L^CLIP
2. Critic: minimise (V_φ(s_t) - G_t)^2

These interact: a bad critic update produces noisy advantages, destabilising the policy.
The policy and critic require separate learning rates, often tuned independently.

GRPO has one optimisation problem:
1. Policy: maximise L^CLIP with group-normalised advantages

No critic to tune. No interleaved instability. The group statistics are computed
analytically and are always consistent with the current batch.

### 5.3 Sparse Reward Compatibility

GAE in PPO requires V_φ(s_t) to be estimated at each intermediate state.
For sparse rewards (where R = 0 for most tokens and a large value at the end),
V_φ must propagate this signal backwards through the value estimates — a
process that is slow and error-prone.

GRPO operates at the sequence level: the entire sequence gets a single reward,
and the group mean normalises it.  This is naturally compatible with sparse rewards
and is why GRPO was effective for reasoning tasks (where correctness is an all-or-nothing reward).

---

## 6. Empirical Signatures

Based on our toy experiments, the following empirical patterns distinguish healthy
from unhealthy RLHF training:

### Healthy training
- Reward increases gradually (not exponentially fast)
- KL from reference grows slowly and plateaus
- Token entropy decreases moderately (not to near zero)
- Clipping fraction decreases over training
- Advantage distribution is approximately zero-mean, unit-variance

### Reward hacking (unhealthy)
- Reward spikes rapidly while unmeasured quality metrics fall
- Entropy collapses at specific positions (policy becomes deterministic)
- KL grows without bound (no effective regularisation)
- Repetition score approaches 1.0
- Advantage distribution becomes bimodal with extreme values

### Over-regularisation (also unhealthy)
- KL stays near zero (policy not learning)
- Reward plateaus at or near the SFT baseline
- Entropy stays near log(V) (policy remains too diffuse)
- Advantage variance is very high (group has similar rewards — no learning signal)

---

## 7. Conclusion

PPO and GRPO share the same theoretical foundation — the policy gradient theorem with
importance sampling — but differ in how they estimate the advantage baseline.  PPO uses
a learned value function (flexible but expensive and fragile); GRPO uses group
statistics (simpler and memory-efficient but requires G rollouts per prompt).

The KL penalty to the reference model is not optional for either algorithm.  Without
it, the reward model becomes unreliable and reward hacking emerges rapidly.  The
coefficient β should be tuned to maintain a KL budget that keeps the policy within
the calibrated region of the reward model.

At scale, GRPO's elimination of the critic and its natural compatibility with sparse
rewards make it an attractive choice.  PPO remains preferred when dense reward
signals are available and accurate value estimation is tractable.  Both algorithms
converge to the same optimal policy under the same reward — the practical difference
is in which gets there more reliably given finite compute and imperfect reward signals.

---

*Implementations of all algorithms discussed here are in `src/` of this project.*
*All experiments use toy environments (vocab=8, seq_len=4) to isolate the algorithmic
properties from the complexity of real LLMs.*
