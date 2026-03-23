# Week 7 Interview Q&A: Core Agent Mechanics

---

## Day 1 — Agent Loops and ReAct

**Q: What is the difference between ReAct and chain-of-thought prompting?**

A: Chain-of-thought (CoT) generates all reasoning in a single LLM forward pass.
Every intermediate step is an autoregressive token in the same response —
errors at step N propagate to steps N+1 through the end with no correction.

ReAct interleaves reasoning with external action: the model emits one Thought
and one Action, a tool executes and returns an Observation, and then the model
reasons again from that grounded observation. Each tool result can correct
errors from the previous step. CoT is "soft" reasoning; ReAct is "grounded"
reasoning.

**Q: Why does iterative reasoning amplify hallucination?**

A: Each reasoning step conditions on all prior generated tokens. If step K
contains a hallucinated fact, all subsequent steps see it as established truth
in their context. The model does not "flag" prior tokens as uncertain — it
treats them with the same confidence as real information. Over N steps,
small errors compound multiplicatively. This is sometimes called the
"snowball effect": a small hallucination at step 1 can completely derail
the reasoning chain by step 5.

**Q: When should you separate planner and executor?**

A: Separate them when:
- The task has 4+ steps with a predictable structure (e.g., "find A, find B,
  compare them, calculate X").
- Incorrect step ordering has high cost (e.g., you call a paid API with wrong
  parameters).
- You want parallelism: a planner can identify independent sub-tasks that an
  executor can run concurrently.

Keep them unified (pure ReAct) when:
- The task is exploratory (you don't know how many steps are needed).
- Each observation may change the entire strategy.
- The task is short (1-3 steps) and planning overhead outweighs benefit.

**Q: What is the agent loop and what are its terminal conditions?**

A: The loop is: state → LLM → action → tool → observation → state update →
repeat. Terminal conditions are:
1. The model emits a "finish" action (or `stop_reason == "end_turn"` in tool_use).
2. The step counter exceeds max_steps (safety guard).
3. An unrecoverable error (API failure, tool error after retries).

Without explicit terminal conditions, agents can loop indefinitely.

---

## Day 2 — Tool Calling

**Q: Why do structured tool calls outperform regex parsing?**

A: Regex parsing is brittle — it relies on the model outputting a specific
format exactly. Any deviation (extra whitespace, different capitalization,
multi-line action) breaks the parser, causing the agent to fail silently.

Structured tool calls (JSON schemas) let the API validate arguments before
returning a response. The model is trained on schema-following data and is
more reliable in this mode. If a required field is missing, the error is
surfaced explicitly rather than silently dropped. The schema also communicates
intent to the model (via description fields), making it less likely to
hallucinate argument names.

**Q: What happens if the model partially fills a schema?**

A: The Anthropic API returns whatever the model provided in the ToolUseBlock.
If a required field is missing, block.input will be an incomplete dict.
Your code should:
1. Use `.get("field", default)` instead of direct dict access.
2. Validate inputs before calling the tool.
3. Return an error string as the tool_result so the model can retry.
   ("Error: 'query' field is required") — the model usually self-corrects.

**Q: How do you prevent tool injection attacks?**

A: Three layers of defense:
1. **System prompt**: clearly state which tools exist and what they do. This
   anchors the model's tool-use behavior.
2. **Schema validation**: unknown tool names are rejected at the dispatch layer
   with a clear error message. The model cannot "invent" tools.
3. **Input sanitization**: never pass raw user input directly as a tool argument
   without stripping special characters or checking for injection patterns.
   For search queries, truncate to reasonable length. For code/math expressions,
   restrict the character set.

**Q: What is "tool hallucination" and how does it manifest?**

A: Tool hallucination is when the model generates a ToolUseBlock referencing
a tool that does not exist in the schema (e.g., "database_query" when only
"wikipedia_search" and "python_calculator" are defined). In Day 2, our
dispatch_tool() returns an error string for unknown names. The model then
typically falls back to an existing tool on the next step.

With well-designed schemas and clear descriptions, hallucinated tool names
are rare (<5% of calls on haiku-class models). They increase with:
- Ambiguous tool descriptions
- User prompts that mention non-existent tool names
- Very large tool schemas (>10 tools) where the model may interpolate

---

## Day 3 — Planning vs. Reasoning

**Q: Why might explicit planning reduce hallucination?**

A: In a pure ReAct loop, the model must infer the current sub-goal from an
ever-growing context. By step 5, the context contains the original question
plus 4 tool results — the model must attend to all of it to figure out what
to do next. This attention over noisy context increases the chance of the
model drifting off-task.

An explicit plan externalizes the sub-goals. The executor receives only one
atomic task plus prior results, not the full history. This reduces the search
space for the model's next action from "anything consistent with the full
history" to "something that completes this specific task". Reduced search
space = reduced hallucination probability.

**Q: What is search depth vs. reasoning depth?**

A: **Reasoning depth** = the number of logical inference steps performed
within one LLM call (CoT chain length). High reasoning depth means the model
is doing many serial inferences without external grounding, which amplifies
error through the snowball effect.

**Search depth** = the number of external tool calls made before reaching
a final answer. High search depth means the model grounds many inferences
against external reality. Each grounded step resets the error accumulation.

The key insight: you can often convert reasoning depth to search depth by
decomposing a complex inference into multiple simpler tool calls. This trades
API cost for reliability — usually a good trade.

**Q: When is planning overkill?**

A: Planning adds one extra LLM call (the planner). It is overkill when:
1. The task is 1-2 steps (planning overhead > benefit).
2. The task is highly dynamic (each observation completely changes strategy).
3. The correct step sequence cannot be determined from the question alone
   (you need partial results to plan the next step).

For these cases, use adaptive ReAct (greedily pick the next action at each
step) rather than committing to a plan upfront.

---

## Day 4 — Memory

**Q: Why does RAG fail in long-horizon agents?**

A: Standard RAG assumes one retrieval per query, but in a multi-step agent
the retrieval query changes at every step. Three specific failure modes:

1. **Retrieval drift**: the query at step N is about the current sub-task,
   not the original question. Critical memory from earlier steps may score
   low on this query and be missed.

2. **Memory poisoning**: if a tool returns a wrong fact (hallucinated Wikipedia
   content, calculation error), the agent stores it as a memory. Future
   retrievals will surface this wrong fact with the same confidence as correct
   facts — there is no truth score in a vector DB.

3. **Stale cross-task contamination**: in a persistent DB, memories from a
   prior task that happen to be semantically similar to the current query
   will be retrieved, potentially injecting wrong context.

**Q: What is retrieval collapse?**

A: Retrieval collapse occurs when the embedding of the current query is too
generic or "averaged". The query vector sits near the centroid of the
embedding space rather than near any specific memory. The top-k results are
semantically similar to each other but may not be the specific memories
needed for the current step. The retrieved context is plausible but
irrelevant, and the model generates a confident but wrong answer.

Prevention: be specific in retrieval queries. Instead of "what do I know?",
query with "what did we find about [specific topic from current step]?".

**Q: When should memory be symbolic vs. vector?**

A: Use **symbolic** (key-value, SQL, graph) when:
- You need exact, deterministic retrieval (no approximation).
- Data is structured: dates, numbers, entity relationships.
- Correctness is more important than fuzzy matching.

Use **vector** when:
- Queries are in natural language and vocabulary is unpredictable.
- The stored documents are long and the relevant passage is unknown.
- Fuzzy, approximate recall is acceptable.
- You need to find semantically related memories even when exact words differ.

Use a **hybrid**: symbolic index for structured facts, vector index for
unstructured text. The agent queries both and merges results.

---

## Day 5 — Long-Horizon Failure

**Q: Why do agents degrade exponentially with task horizon?**

A: The exponential comes from error independence: if each step has success
probability p, then N independent steps have combined success probability
p^N. In practice steps are NOT independent (errors at step K increase error
risk at step K+1), making the degradation even faster than p^N.

Additionally, attention quality degrades with context length. The model must
attend to O(N) prior tool results to generate step N+1. With long context,
recent relevant information competes with distant but still-visible
irrelevant information for attention weight.

**Q: What are the five major agent failure modes?**

A: (1) **Error accumulation** — wrong intermediate results propagate and
compound. (2) **Overconfident hallucination** — model finishes with wrong
answer expressed confidently. (3) **State drift** — implicit goal representation
diverges from original question under long context. (4) **Tool misuse** —
model confuses tool purposes after many steps in context. (5) **Memory
overload** — context fills with prior tool results, degrading attention quality
and increasing per-step cost.

**Q: What architectural patterns address long-horizon degradation?**

A: Six patterns in order of implementation complexity:

1. **Max-step guardrail**: always set max_steps and handle gracefully.
2. **Tool result validation**: check tool output format before storing it.
3. **Periodic replanning**: every K steps, summarize state and replan.
4. **Step verification**: a critic LLM checks each step's observation
   before it is added to context.
5. **Context compression**: replace raw history with a LLM-generated
   "situation report" to reduce context length while preserving key facts.
6. **DAG task graphs**: replace flat sequences with dependency graphs;
   execute independent nodes in parallel.

**Q: How do you detect that an agent has hallucinated a final answer?**

A: Several heuristics:
- **Confidence calibration**: ask the model to rate its confidence. If it
  rates >90% on a 5-step task, apply additional verification.
- **Source grounding check**: does the final answer cite information that
  appeared in a tool observation? If the answer contains numbers not in
  any observation, flag it.
- **Cross-validation**: run the same question twice with temperature > 0.
  If the answers differ significantly, the agent is in a high-entropy region.
- **Fact verification step**: add a dedicated final step that re-searches
  the key facts in the answer and checks consistency.

---

## Bonus: System Design

**Q: You are building an agent to answer complex research questions requiring
10+ steps. What is your architecture?**

A good answer includes:

1. **Planner**: generates a DAG of atomic sub-tasks from the question.
2. **Executor pool**: runs independent sub-tasks in parallel.
3. **Step verifier**: LLM-based critic that checks each tool result for
   plausibility before it enters the context.
4. **Memory layer**: symbolic store for structured facts (dates, numbers),
   vector store for text summaries.
5. **Context compressor**: after every 5 steps, compress history into a
   summary. Keeps context under the attention-degradation threshold.
6. **Confidence estimator**: at each step, estimate uncertainty. If
   confidence drops below a threshold, trigger replanning.
7. **Human-in-the-loop escalation**: for tasks with >8 steps or confidence
   below X, surface the current state to a human reviewer.

This architecture directly addresses all five failure modes from Day 5.
