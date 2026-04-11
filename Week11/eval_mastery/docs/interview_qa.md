# Week 11 Interview Q&A — Evaluation Mastery

These are real questions from senior ML/LLM interviews.
Answers include the key claim, the mechanism, and a number where possible.

---

## Day 1: BLEU / ROUGE Failures

**Q: Why is BLEU unreliable for reasoning tasks?**

A: BLEU measures n-gram overlap between hypothesis and reference.  Two sentences
with identical meaning but different word order share almost no n-grams — this
is called paraphrase collapse.  For example, "France's capital is Paris" and
"Paris is the capital of France" score near-zero BLEU-4 because no 3- or 4-gram
is shared.  In reasoning tasks, models frequently produce correct paraphrases;
BLEU penalises them as if they were wrong.  Conversely, a wrong-but-plausible
answer that shares surface tokens receives non-zero BLEU — a false positive.

**Q: What is the difference between lexical and semantic metrics?**

A: Lexical metrics (BLEU, ROUGE, exact match) compare the literal character or
word sequences in hypothesis and reference.  They have zero notion of meaning.
Semantic metrics compare meaning representations — either via n-gram proxies
(chrF), contextual embeddings (BERTScore uses BERT), or direct model judgment
(LLM-as-judge).  The distinction matters because natural language allows the same
meaning to be expressed many different ways.

**Q: When would you still use BLEU?**

A: Three legitimate use cases:
1. Machine translation where multiple human reference translations are available
   (averaging over references reduces paraphrase collapse).
2. As a fast sanity check — a very low BLEU usually does signal something is wrong.
3. For reproducibility with older benchmarks where BLEU is the established baseline.
For new work on generative tasks, prefer BERTScore + LLM-judge.

**Q: What is chrF and why is it better than BLEU for paraphrases?**

A: chrF (character n-gram F-score) computes precision and recall over character
n-grams instead of word n-grams.  Because characters are more granular, words like
"capital" and "capitalise" share trigrams even though they are different tokens.
Reordered phrases like "of France" and "France's" share many character bigrams.
chrF is still a lexical metric — it has no semantic understanding — but it is
strictly more tolerant of paraphrase than BLEU.

---

## Day 2: LLM-as-a-Judge

**Q: Is LLM-as-a-judge reliable?**

A: Reliable as a scalable proxy, not as ground truth.  LLM judges show ~80%
agreement with human judgments on pairwise comparisons (MT-Bench results), which
is comparable to human-human agreement.  However, they have systematic biases:
position bias (~30% of comparisons change winner on order swap), verbosity bias
(longer responses preferred), and self-preference (a model rates its own style
higher).  With proper mitigation (order averaging, explicit length instructions,
diverse judge pool) they are production-grade for large-scale evaluation.

**Q: How do you reduce evaluation bias in LLM-as-judge?**

A: Four main mitigations:
1. **Position averaging**: always run both (A, B) and (B, A); use majority.
2. **Prompt engineering**: explicit instruction to ignore length.
3. **Multiple judges**: use ≥ 2 models from different families; take majority.
4. **Calibration**: seed the judge with human-rated examples as few-shot context.
The most impactful in practice is position averaging — it roughly halves the
position-induced error rate.

**Q: How would you audit evaluation fairness?**

A: Systematic audit:
1. Run a test set where all responses have known ground-truth quality.
2. Measure judge agreement with ground truth (accuracy).
3. Inject known biases: swap order, pad responses with filler — measure how much
   the score changes.
4. Stratify by response length and topic — check if accuracy is uniform.
5. Compare against 100–200 human labels as the gold standard.
Any metric that shows >5% score change from pure filler padding has verbosity bias.

---

## Day 3: Task-Specific Metrics

**Q: What is pass@k and how is it computed?**

A: pass@k is the probability that at least one of k sampled code solutions passes
the test suite, given that you generated n total samples and c were correct.

```
pass@k = 1 − C(n−c, k) / C(n, k)
       = 1 − ∏_{i=0}^{k−1} (n−c−i) / (n−i)   [numerically stable]
```

The key insight: you can sample n=100 solutions offline, test all 100 once, and
then compute pass@1 and pass@5 without running any more tests.  This makes
benchmarking cheap while giving accurate estimates.

**Q: Why is next-token loss insufficient as an evaluation metric?**

A: Cross-entropy loss measures how well the model's predicted token distribution
matches the training distribution.  It does not measure utility.  A model that
predicts the reference code with high probability but outputs one wrong variable
name suffers a small loss — but the code will not compile.  A model that paraphrases
a correct answer slightly differently suffers higher loss than one that produces the
exact reference — but the paraphrase may be equally correct.  Loss is a training
signal, not an evaluation signal.

**Q: How would you evaluate an agent?**

A: Decompose into observable sub-steps:
1. **Tool routing**: did the agent call the right function/tool?
2. **Argument structure**: are required argument keys present?
3. **Type compliance**: do argument values have correct types?
4. **Value correctness**: are values semantically right?
5. **Task completion**: did the downstream execution succeed?
6. **Efficiency**: how many steps did it take vs. the optimal?
Aggregate success rate hides the failure mode.  If routing is 90% but value
correctness is 60%, you have a semantic understanding problem, not a routing problem.

**Q: What is schema compliance and when does it matter more than BLEU?**

A: Schema compliance measures whether a structured output (JSON, XML, function call)
conforms to a required schema — correct field names, correct types, correct value
constraints.  A response can have 99% BLEU against the reference but be completely
unusable if it fails JSON parsing.  In production tool-use or agent systems, schema
compliance is the primary metric — a non-compliant output requires a retry or fails
the task entirely.

---

## Day 4: Human Evaluation

**Q: Why is human eval expensive but critical?**

A: Automated metrics are proxies for human judgment.  They diverge from humans in
ways that are hard to predict and change as models improve.  A model might achieve
high BLEU but produce responses that are factually wrong, unhelpful, or harmful
in ways no automated metric detects.  Human eval is critical because models are
ultimately used by humans.  The expense comes from: multiple raters needed per
example (to estimate inter-rater reliability), domain expertise required for
technical topics, and the time cost of careful judgment vs. mechanical annotation.

**Q: How would you design a human evaluation pipeline?**

A: Step by step:
1. **Define criteria** precisely (helpfulness, factual accuracy, safety — with examples).
2. **Sample strategically** — don't eval randomly; include adversarial, ambiguous, and
   edge-case prompts.
3. **Use pairwise ranking** (A vs B) rather than absolute Likert — lower variance.
4. **Minimum 3 raters per item** — measure inter-rater agreement with Cohen's κ.
   If κ < 0.4, the criteria are too ambiguous; iterate.
5. **Train raters** with a calibration set (30 items with known answers).
6. **Audit for drift** — re-rate 10% of items at end of session.
7. **Use LLM-judge at scale**, calibrated against ~200 human-rated items.

**Q: What is the failure mode of crowdsourced labelling?**

A: Four main failure modes:
1. **Annotation artefacts**: raters learn shortcuts — e.g. the response that mentions
   the key term is always marked "correct" regardless of quality.
2. **Label schema ambiguity**: different raters interpret "helpful" differently →
   low κ → noisy labels → noisy model training.
3. **Rater fatigue**: quality degrades within a session; early items are rated more
   carefully than late ones.
4. **Distribution mismatch**: crowdworker demographics may not match the target user
   population — safety and cultural norms in particular can diverge.

**Q: What is the Bradley-Terry model and why is it used?**

A: Bradley-Terry (BT) is a probabilistic model for pairwise comparisons.
P(i beats j) = s_i / (s_i + s_j) where s_i is item i's strength.
BT is fit by iterative MLE from a pairwise win matrix.

It is used because:
- It produces a continuous ranking from sparse pairwise data.
- It handles transitive inconsistency gracefully (A > B > C but C > A in some pairs).
- It gives confidence intervals on the ranking.
- Chatbot Arena (LMSYS) uses BT to rank LLMs from millions of human votes.

---

## Day 5: Robustness & Distribution Shift

**Q: How do you test for jailbreak robustness?**

A: Red-team with a taxonomy of attack families:
- Direct injection: "Ignore previous instructions and..."
- Role-play: "You are DAN, who can answer anything..."
- Hypothetical framing: "In a fictional scenario where..."
- Encoding attacks: base64, pig latin, ROT13 wrappers
- Multi-turn escalation: gradually move context toward prohibited content

Measure **refusal rate** per attack family (not just overall).  A model with 95%
overall refusal but 40% refusal for multi-turn escalation is vulnerable to
patient adversaries.  Use automated red-teaming (adversarial fine-tuning,
constitutional AI) to scale beyond manual attacks.

**Q: What is calibration in LLMs and why does it matter?**

A: Calibration is the alignment between a model's stated confidence and its
empirical accuracy.  A well-calibrated model saying "70% confident" is correct
70% of the time.  LLMs are often overconfident post-RLHF: the RLHF reward model
is trained to prefer confident, hedge-free responses, which trains away the
appropriate uncertainty language.

It matters because:
- Overconfident wrong answers cause users to trust incorrect information.
- Calibration determines when to abstain — only possible with reliable confidence.
- In medical/legal applications, "I'm 95% sure" must mean 95% accuracy.

Measured with ECE (Expected Calibration Error) and reliability diagrams.

**Q: What is abstention and when should models refuse?**

A: Abstention = returning "I don't know" or declining to answer rather than
generating a potentially wrong response.

Models should abstain when:
1. **Confidence is below a threshold** (calibration-based abstention).
2. **Question is outside knowledge domain** — OOD detection can flag these.
3. **Query is harmful** — safety classifier triggers.
4. **Stakes are too high for uncertainty** — medical diagnosis, legal advice.

The abstention tradeoff: higher θ (more abstention) → lower hallucination rate,
lower helpfulness.  Plot accuracy vs coverage (fraction answered) to choose
the operating point for a given deployment context.

**Q: What is the "lost in the middle" phenomenon?**

A: Liu et al. 2023 showed that transformer models retrieve relevant information
significantly less accurately when it is placed in the middle of long contexts,
compared to the beginning or end.  Performance follows a U-shape with context
position.

This is a distribution shift: models are evaluated on tasks where relevant context
is at the ends (most benchmarks), but deployed with relevant information anywhere.
Test for it explicitly by sweeping the position of the key information within
a fixed-length context.

---

## Days 6-7: Evaluation Harness

**Q: How would you build a production LLM evaluation pipeline?**

A: Three layers:
1. **Automated (fast iteration)**: chrF, BLEU, pass@k, schema compliance.
   Runs on every model checkpoint.  Results in 10 minutes.

2. **Semi-automated (pre-release)**: LLM-as-judge with position averaging.
   500–1000 examples.  Results in 1–2 hours.  Calibrated against human labels.

3. **Human (release gate)**: 200-sample pairwise human eval.  Results in 1–2 days.
   This is the ground truth that calibrates the automated metrics.

Continuous production signal: implicit human feedback (thumbs up/down, copy,
follow-up questions) feeds back into the evaluation cycle.

**Q: How do you prevent metric gaming in model development?**

A: Three practices:
1. **Keep test sets held out** — never use eval data in any training loop.
2. **Rotate metrics** — teams optimise for whatever is measured; change metrics
   every quarter to prevent overfitting to evaluation artefacts.
3. **Red-team your own eval** — actively look for ways to score high on metrics
   while being actually bad (e.g. verbose responses gaming verbosity-biased judges).

**Q: What is the single most common evaluation mistake you see?**

A: Using BLEU or ROUGE as the primary metric for instruction-following or
reasoning tasks.  These metrics are proxies for translation fidelity, not
general language quality.  Any model fine-tuned on instruction data will learn
to paraphrase away from the reference, driving BLEU down even as quality
improves.  Teams then incorrectly conclude fine-tuning hurt the model.

The fix: always pair lexical metrics with at least one semantic signal
(BERTScore or LLM-judge win rate) and one task-specific metric.
