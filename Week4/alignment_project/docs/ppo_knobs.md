# PPO Hyperparameter Knobs

PPO (and its simpler variant RLOO) has several interacting hyperparameters.
Getting them right is one reason "PPO is hard in LLMs" — each knob pulls in a different direction.

---

## 1. KL Coefficient (β / `kl_coef`)

**What it does:**  Controls how much the policy is penalised for deviating from the reference (SFT) model.

```
effective_reward = reward - β · KL[π(·|x) || π_ref(·|x)]
```

| β value | Effect |
|---------|--------|
| `0.0`   | No KL constraint → reward hacking within a few steps |
| `0.01–0.05` | Aggressive optimisation, mild constraint; risk of degeneration |
| `0.1–0.3` | Balanced; recommended starting range |
| `> 0.5` | Very conservative; policy barely moves from SFT |

**Symptom of β too low:** KL divergence explodes; model generates repetitive or incoherent text.
**Symptom of β too high:** KL stays near 0; reward barely improves; training is effectively doing nothing.

**Adaptive KL (PPO-style):** Some implementations adjust β automatically to keep KL near a target value (e.g. `kl_target=6.0`).  This is the InstructGPT approach.

---

## 2. PPO Clip Ratio (ε / `clip_range`)

**What it does:**  Limits how much the policy can change in a single update step.

```python
ratio = π(y|x) / π_old(y|x)   # current vs. rollout policy
clipped_ratio = clip(ratio, 1-ε, 1+ε)
loss = -min(ratio * A, clipped_ratio * A)
```

| ε value | Effect |
|---------|--------|
| `0.1`   | Very conservative; small updates per step |
| `0.2`   | Default in most implementations |
| `0.3–0.5` | Larger steps; faster early learning but less stable |

**Note:** RLOO (used in this project) does **not** use PPO clipping — it uses direct policy gradient without the ratio constraint.  This simplifies implementation but can be less stable with large reward variance.

---

## 3. Value Loss Coefficient (`vf_coef`)

**What it does:**  In full PPO, weights the value function loss relative to the policy loss.

```
total_loss = -policy_loss + vf_coef * value_loss - entropy_coef * entropy
```

**Note:** RLOO has **no value function** — the baseline is estimated as the mean reward for the same prompt across K completions.  This removes one source of training instability but increases variance when K is small.

---

## 4. Advantage Estimation (GAE λ)

**What it does:**  In full PPO, Generalised Advantage Estimation (GAE) controls the bias-variance trade-off of advantage estimates.

```
A_t = Σ_{l≥0} (γλ)^l δ_{t+l}    where δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

| λ value | Effect |
|---------|--------|
| `0.0`   | One-step TD advantage (low variance, high bias) |
| `0.5`   | Balanced |
| `1.0`   | Full Monte Carlo return (low bias, high variance) |

**For LLMs:**  Each response is typically treated as a single episode (reward given at EOS).  GAE with γ=1.0 (no discounting) reduces to: advantage at token t = R − baseline.

---

## 5. Entropy Bonus (`entropy_coef`)

**What it does:**  Adds entropy to the objective to prevent mode collapse.

```
objective += entropy_coef * H[π(·|x)]
```

**Why it matters:**  Without entropy regularisation, the policy can collapse to always generating the same high-reward response.  The entropy bonus encourages exploration.

**Symptom of entropy collapse:** All generated responses look the same; the entropy metric drops to near 0.

---

## 6. Number of Rollout Epochs (`ppo_epochs`)

**What it does:**  How many gradient updates to take on a single batch of rollouts.

| Value | Trade-off |
|-------|-----------|
| `1`   | Most conservative; closest to on-policy |
| `2–4` | Standard PPO (4 epochs per rollout batch is the original paper setting) |
| `> 4` | Risk of off-policy degradation — old rollouts become stale |

---

## 7. Rollout Generation Temperature

**What it does:**  Controls diversity of on-policy samples.

| Temperature | Effect |
|-------------|--------|
| `0.7`       | More focused, less diverse completions |
| `1.0`       | Standard sampling |
| `> 1.2`     | Very diverse; may produce incoherent text |

**Tip:**  Higher temperature during rollouts increases exploration but makes advantage estimation noisier.

---

## Summary: What to Watch During Training

| Metric | Healthy Range | Warning |
|--------|--------------|---------|
| `rewards/mean` | Slowly increasing | Flat → reward too sparse or β too high |
| `kl` | 0.1 – 5.0 | > 10 → β too low; reward hacking |
| `entropy` | > 1.0 | < 0.5 → mode collapse |
| `value_loss` | Decreasing | Increasing → value head unstable |
| `policy_loss` | Negative (maximising reward) | Positive and large → clipping too aggressive |

---

## Interview Q&A

### Q: Why is PPO hard in LLMs?

**A:**  At least five compounding challenges:

1. **Credit assignment over long sequences:** A 200-token response gets one reward.  Attributing credit to individual token decisions is ambiguous.
2. **Value function training:** The value head must predict expected return at each position.  LM hidden states encode language, not value — training the value head is a separate optimisation problem that can destabilise the whole system.
3. **On-policy bottleneck:** PPO requires fresh rollouts before every update.  For a 7B model, generating 1024 samples is expensive — training throughput is dominated by generation, not gradient computation.
4. **Four interacting hyperparameters:** β (KL), ε (clip), λ (GAE), and entropy coef all influence each other.  A good set for GPT-2 does not transfer to LLaMA.
5. **Reward model staleness:** The RM was trained on completions from an earlier policy.  As the policy improves, it generates OOD completions the RM cannot score reliably.  This creates a feedback loop: the policy learns to exploit RM blindspots.
