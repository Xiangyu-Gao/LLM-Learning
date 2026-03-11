# Alignment Method Comparison

## 1-Page Comparison Table

| Dimension | SFT | DPO | PPO / RLOO | GRPO |
|-----------|-----|-----|------------|------|
| **Data needs** | Demonstrations (prompt, response) | Preference pairs (chosen, rejected) | Prompts only; reward computed online | Prompts only; reward computed online |
| **Reward model required?** | No | No (implicit) | Yes (or proxy function) | Yes (or proxy function) |
| **Online generation?** | No | No | Yes ← main cost | Yes ← main cost |
| **Value function?** | No | No | Yes (PPO) / No (RLOO) | No (group baseline) |
| **Stability** | High (supervised loss) | Medium (no generation noise) | Low–Medium (reward hacking, value instability) | Medium (group normalisation helps) |
| **Infrastructure complexity** | Low | Low | High (generation server, reward model, value head) | Medium (generation + reward only) |
| **Controllability** | Low (copies distribution) | Medium (β controls KL) | High (many knobs; adaptive KL) | Medium (β + clip ratio) |
| **Failure modes** | Averaging, no trade-offs | Offline mismatch, bad reference | Reward hacking, mode collapse, value divergence | Mode collapse at low G, reward hacking |
| **Memory footprint** | 1× model | 2× model (policy + ref) | 2–3× model (policy + ref + value) | 2× model (policy + ref) |
| **Compute per step** | 1 fwd+bwd | 2 fwd+bwd | G × fwd (generation) + 2 fwd+bwd | G × fwd (generation) + 2 fwd+bwd |
| **Typical β / KL target** | N/A | 0.1 | 0.05–0.2 | 0.05–0.1 |
| **Preferred when…** | Abundant demonstrations | Fixed preference dataset available | Reward can be computed online; code/math tasks | Same as PPO but want to avoid value head |

---

## When to Use Which Method

### Use SFT when:
- You have high-quality demonstrations (human-written or from a strong model).
- Format and style conformance is the primary goal (not resolving trade-offs).
- You need a starting point before any preference alignment.

### Use DPO when:
- You have a good preference dataset (e.g., crowd-sourced comparisons, model-generated).
- You want simplicity: DPO is a single training loop with no generation at train time.
- Compute is limited: 2× memory, no generation overhead.
- The base SFT model is already close to the desired distribution.

### Use PPO / RLOO when:
- The reward function is computable programmatically (code execution, math verification, format checking).
- You want the policy to explore beyond the training distribution.
- Human preference data is scarce but reward signals are available.
- You need to iterate quickly: RLOO is simpler than full PPO.

### Use GRPO when:
- Same conditions as PPO/RLOO, but you want to avoid a value function.
- You have enough compute for G ≥ 4 completions per prompt.
- You want a more stable RL baseline (group normalisation reduces variance).
- Inspired by DeepSeek reasoning models where GRPO showed strong results.

---

## Interview Q&A

### Q: When would you choose DPO over PPO?

**A:**  I'd choose DPO over PPO in three scenarios:

1. **You have a high-quality preference dataset and no online reward signal.**
   DPO uses the dataset directly; PPO would need a reward model trained on that
   same data, adding a second training stage and doubling the engineering burden.

2. **Simplicity matters.**  DPO has no generation loop, no value head, no KL
   controller, no reward server.  It's a single supervised training run.  In
   a production setting this means fewer failure modes, faster iteration, and
   easier debugging.

3. **Compute is constrained.**  PPO/GRPO need to generate completions (the policy
   forward pass without teacher forcing) at every training step.  For a 7B model
   that can be 3–10× slower than a supervised pass.  DPO uses teacher-forced
   forward passes only, which is as fast as SFT.

I'd choose PPO/GRPO when:
- The reward signal is computable online (code test passing, math verifier) and
  human preferences are not available or too noisy.
- I want the model to explore beyond the fixed training distribution.
- I need to optimise a reward that cannot be expressed as a static preference pair
  (e.g., safety + helpfulness jointly).

### Q: What are the practical failure modes unique to DPO?

**A:**
- **Offline distribution mismatch:** DPO's implicit reward is valid only near the
  SFT reference.  If the policy drifts far (low β), log-prob ratios become
  unreliable.  Solution: keep β ≥ 0.05; consider iterative DPO (collect new
  preferences from updated model).
- **Similar chosen/rejected pairs:** DPO's gradient is proportional to
  `r_w − r_l`.  If pairs are very similar, the gradient vanishes.  Solution:
  filter for "margin" diversity.
- **Length bias:** If the preference dataset prefers longer responses, DPO will
  produce verbose outputs.  Solution: length-normalise log-probs.
