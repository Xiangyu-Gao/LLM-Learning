# Week 11 Study Guide — Evaluation Mastery

## Theme: Most candidates fail here. You will not.

LLM evaluation is the discipline of measuring whether a model is actually
good — not just whether it fits training data.  It is arguably the most
important skill in applied ML, and also the most commonly misunderstood.

---

## Day 1: Why BLEU / ROUGE Fail

### The Core Problem: Surface Form ≠ Semantic Equivalence

BLEU (Bilingual Evaluation Understudy, Papineni et al. 2002) was designed
for machine translation in the era when translations were relatively literal.
It counts n-gram overlap between a hypothesis and one or more references.

**BLEU formula:**
```
BLEU = BP × exp( (1/N) × Σ_{n=1}^{N} log p_n )

p_n  = modified n-gram precision (clipped to reference count)
BP   = brevity penalty = exp(1 - r/c) if c < r, else 1
```

### Why It Fails for Generative Reasoning

**Paraphrase collapse:** Two sentences with identical semantics but different
word order share almost no n-grams.

```
Reference:  "Paris is the capital of France."
Hypothesis: "France's capital city is Paris."

1-gram overlap: {paris, capital, france} → 3/4 = 0.75
2-gram overlap: {} → 0/3 = 0.0      ← kills BLEU
4-gram BLEU ≈ 0.0
```

The model gave the correct answer.  BLEU says it failed.

**False positives:** Wrong-but-plausible answers sharing tokens with the
reference receive non-zero BLEU.

```
Reference:    "Paris is the capital of France."
Wrong answer: "Paris is the largest city in France."

Shared n-grams: "Paris is the", "is the", "of France" → BLEU > 0
```

The model gave a wrong answer.  BLEU gives partial credit.

### ROUGE-L

ROUGE-L uses Longest Common Subsequence (LCS) instead of contiguous n-grams.
This is slightly more tolerant of reordering but still purely lexical.

```
LCS("france's capital is paris", "paris is the capital of france") = 3
("is", "capital", "paris" appear in both, in order)

ROUGE-L F = 2 × P × R / (P + R)
```

### chrF: A Better Proxy

chrF (Character n-gram F-score) uses character-level n-grams.  Because
characters are more granular than words, they survive reordering:

```
"france" and "of france" share: "fra", "ran", "anc", "nce"
"capital" and "capitals" share: "cap", "api", "pit", "ita", "tal"
```

chrF is not semantic — it does not understand meaning.  But it is strictly
more robust than BLEU for paraphrases.

### BERTScore (the real fix)

BERTScore (Zhang et al. 2020) encodes hypothesis and reference with a
pretrained BERT model, then computes precision/recall/F1 using cosine
similarity between contextual token embeddings.

Because BERT understands context, "Paris is France's capital" and
"The capital of France is Paris" will have similar embeddings.

**When to use what:**
| Metric   | Speed | Paraphrase | Semantic | Use case |
|----------|-------|------------|----------|----------|
| BLEU     | Fast  | Poor       | No       | MT sanity check, old papers |
| ROUGE-L  | Fast  | Poor       | No       | Summarisation (w/ human labels) |
| chrF     | Fast  | OK         | No       | MT without reference embeddings |
| BERTScore| Med   | Good       | Partial  | Generation, dialogue, QA |
| LLM-judge| Slow  | Good       | Yes      | Final production eval |

---

## Day 2: LLM-as-a-Judge

### Why It Works

GPT-4-style evaluation can assess nuanced quality dimensions that no
automated metric captures: helpfulness, safety, coherence, instruction
following.  At scale, LLM judges agree with human judgments ~80% of the
time on pairwise comparisons (MT-Bench, Chatbot Arena).

### Why It Fails

**Position bias:**
In a prompt like "Response A: ... Response B: ..., which is better?", the
model preferentially selects A.  MT-Bench found this changes ~30% of
outcomes when the order is swapped.

**Verbosity bias:**
Longer responses are rated as more helpful, independent of content quality.
A padded response with filler sentences beats a concise, correct answer.

**Self-preference:**
Models rate outputs that match their own style and vocabulary as higher
quality.  GPT-4 as judge shows a measurable preference for GPT-4 outputs
over equally-good Claude outputs.

### Mitigations

1. **Position averaging:** Run both (A, B) and (B, A) orderings.  Take
   the majority verdict.  This halves position bias.

2. **Length normalisation:** Add explicit instructions: "Evaluate quality
   independently of response length.  A concise correct answer beats a
   verbose correct answer."

3. **Multiple judges:** Use ≥ 3 independent judge calls and take majority.

4. **Reference calibration:** Provide the judge with a set of
   human-rated examples as few-shot context.

5. **Diverse judge pool:** Use 2-3 different model families as judges.
   Differences reveal systematic biases.

### Prompt Template (Anti-Bias)

```
You are an impartial evaluator. Your task is to assess the quality of
a response to the question below.

DO NOT prefer responses because they are longer.
DO NOT prefer the first response because it appears first.
Judge solely on: accuracy, completeness, clarity.

Question: {question}

[Response 1]
{response_a}

[Response 2]
{response_b}

Which response is better? Reply with only "Response 1", "Response 2",
or "Tie", then explain your reasoning in one sentence.
```

---

## Day 3: Task-Specific Metrics

### Why Next-Token Loss Is Insufficient

Cross-entropy loss = −(1/T) Σ_t log p(y_t | y_{<t})

This measures how well the model predicts the training tokens.  It does
not measure whether the output is correct, useful, or safe.

Two models can have identical loss but one produces valid JSON and the
other produces malformed output.

### pass@k (HumanEval, Chen et al. 2021)

For code generation: sample n solutions, run tests on all n, let c pass.

```
pass@k = 1 − C(n−c, k) / C(n, k)

Numerically stable:
pass@k = 1 − ∏_{i=0}^{k−1} (n−c−i) / (n−i)
```

**Intuition:** pass@k is the probability that at least one of k randomly
chosen samples from your n solutions passes the test suite.

**Key insight:** With n=100 samples, you can estimate pass@1 accurately
without running tests 100 times per user query at inference time.  You
estimate it once during offline evaluation.

**Example:**
- n=10 samples, c=2 correct, k=1: pass@1 = 1 − (8/10) = 0.20
- n=10 samples, c=2 correct, k=5: pass@5 = 1 − C(8,5)/C(10,5) = 1 − 56/252 ≈ 0.78

### Schema Compliance

For structured output tasks (tool use, data extraction, function calling),
schema compliance is the primary metric.

A response with 99% BLEU score that fails JSON parsing has zero utility.
Schema evaluation is binary: either the output is valid or it is not.

Decomposed evaluation catches *where* failures occur:
```
Parse error rate   → model doesn't know JSON syntax
Missing field rate → model understands JSON but misses requirements
Type error rate    → model produces wrong types for fields
Value error rate   → correct types but wrong constraint (e.g. age < 0)
```

### Tool-Call / Agent Evaluation

Decompose success into observable sub-steps:
1. **Routing**: Did the model call the right function?
2. **Keys**: Are the required argument names present?
3. **Types**: Do argument values have the correct types?
4. **Values**: Are the values semantically correct?
5. **Downstream**: Did the downstream system accept and execute successfully?

Aggregate "success rate" hides which component is failing.
A model with 60% full success might have 90% routing accuracy and 67%
value accuracy — these imply very different fixes.

---

## Day 4: Human Evaluation

### Cohen's Kappa

Cohen's κ corrects observed agreement for chance agreement:

```
κ = (p_o − p_e) / (1 − p_e)

p_o = fraction of items where both raters agree
p_e = expected agreement by chance = Σ_k p_{k,1} × p_{k,2}
```

**Interpretation:**
| κ range | Agreement level |
|---------|----------------|
| < 0.20  | Poor           |
| 0.21–0.40 | Fair         |
| 0.41–0.60 | Moderate     |
| 0.61–0.80 | Substantial  |
| > 0.80  | Almost perfect |

In practice, NLP annotation tasks achieve κ = 0.6–0.8 for clear criteria
and κ < 0.4 for ambiguous tasks (e.g. "is this response harmful?").

**Key lesson:** Low κ usually means the criteria definition is ambiguous,
not that the raters are incompetent.  Fix the rubric first.

### Bradley-Terry Model

Convert pairwise comparisons into a global ranking.

**Setup:** n items, wins[i, j] = times item i beat item j.

**Model:** P(i beats j) = s_i / (s_i + s_j) where s_i is item i's strength.

**Fitting (iterative MLE):**
```
s_i^{new} = W_i / Σ_{j≠i} n_{ij} / (s_i + s_j)

where W_i = total wins for item i
      n_{ij} = total comparisons between i and j
```

Repeat until convergence.  Normalise so Σ s_i = n.

**Why it matters:**
- Chatbot Arena (LMSYS) uses Bradley-Terry to rank LLMs from pairwise human votes.
- BT handles transitive inconsistency: if A > B and B > C but C > A in some
  comparisons, BT still produces a reasonable ranking.
- BT gives confidence intervals; raw win rate does not.

### Pairwise vs. Likert

| Format | Pros | Cons |
|--------|------|------|
| Absolute Likert (1–5) | Easy to collect | High variance; scale ambiguity |
| Pairwise (A vs B) | Lower variance; forces comparison | O(n²) pairs for n items |
| Ranked list | Complete ordering | Cognitively demanding |

**Recommendation:** Use pairwise ranking for the primary human eval signal,
Likert for quick crowd-sourced screening.

---

## Day 5: Distribution Shift & Robustness

### What Is Distribution Shift?

The training distribution ≠ the deployment distribution.  Examples:
- **Temporal:** A model trained on 2022 text queried about 2024 events.
- **Domain:** A model trained on English queried in French.
- **Adversarial:** A user intentionally crafts inputs to confuse the model.
- **Context length:** Model trained on sequences of ≤ 2048 tokens deployed
  with 8192-token inputs.

### Adversarial Robustness

Attack families:
- **Prompt injection:** "Ignore previous instructions and..."
- **Jailbreaks:** Role-playing, hypothetical framing, encoding attacks.
- **Multi-turn:** Gradual context manipulation over several turns.
- **Indirect injection:** Malicious content in retrieved documents (RAG).

**Evaluation approach:**
```
robustness@k = fraction of k attacks where model behaved as expected
```

A model with 90% clean accuracy but 30% robustness to injection is
unsuitable for production.

### Long-Context Degradation ("Lost in the Middle")

Liu et al. 2023 showed that models struggle to retrieve information placed
in the middle of long contexts.  Performance follows a U-shape: best at the
start and end, worst in the middle.

**Test procedure:**
1. Embed the relevant answer in positions: beginning, 25%, 50%, 75%, end.
2. Measure retrieval accuracy at each position.
3. A robust model should be flat; a degrading model U-shapes.

### Calibration

A model is **well-calibrated** if its stated confidence matches its
empirical accuracy across many examples.

```
Perfect calibration: P(correct | confidence = 0.7) = 0.70
Overconfidence:      P(correct | confidence = 0.7) = 0.55
Underconfidence:     P(correct | confidence = 0.7) = 0.85
```

**Expected Calibration Error (ECE):**
```
ECE = Σ_b |B_b| / n × |acc(B_b) − conf(B_b)|
```
where B_b is the set of samples in confidence bin b.

**Post-RLHF models tend to be overconfident** because RLHF reward models
are trained to prefer confident, direct responses — this trains away hedging.

### Abstention

**Abstention** = the model returning "I don't know" instead of guessing.

- **Benefits:** Reduces hallucination; safer in high-stakes domains.
- **Costs:** Reduces helpfulness; users dislike over-refusal.
- **Tuning:** Set a confidence threshold θ.  Abstain if max_prob < θ.

The θ tradeoff curve (accuracy vs coverage) is the abstention evaluation.
Plot accuracy vs fraction of questions answered to choose operating point.

---

## Days 6-7: Building the Harness

### What a Production Eval Harness Needs

```
eval/
  metrics.py      — automated metrics (BLEU, chrF, BERTScore, pass@k, ECE)
  judge.py        — LLM-as-judge wrapper with bias mitigation
  schema_check.py — structured output validation
  adversarial.py  — robustness test suite

harness.py        — orchestration: run all metrics, aggregate, report
```

### Evaluation Cadence

| Stage | Frequency | Method |
|-------|-----------|--------|
| Development | Every commit | Automated (chrF + pass@k) |
| Pre-release | Weekly | Automated + LLM-judge |
| Major release | Per release | LLM-judge + 200-sample human eval |
| Production | Continuous | Implicit (thumbs up/down, engagement) |

### The Golden Rule of Evaluation

**Your evaluation measures what you tell it to measure.**

- If you measure BLEU, you incentivise literal outputs.
- If you measure win rate on a verbosity-biased judge, you incentivise
  padding.
- If you measure task completion, you incentivise task success.

Choose your metrics deliberately.  Your models will optimise for whatever
you measure — including its failure modes.

---

## Mental Model Summary

```
Correct?  →  Semantic similarity / LLM-judge
Useful?   →  Task-specific (pass@k, schema, tool-call)
Reliable? →  Robustness rate × attack types
Honest?   →  Calibration (ECE) + abstention analysis
Fair?     →  Inter-rater agreement (κ) + human eval
```
