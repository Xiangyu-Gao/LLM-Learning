# Week 7 Study Guide: Core Agent Mechanics

## Overview

This week answers: *what is an agent, really?*  Not just "an LLM with tools" —
that description hides all the interesting failure modes. By the end of this
week you should be able to:

1. Explain the agent loop from first principles.
2. Implement it from scratch (Day 1).
3. Understand where it breaks and why (Days 2–5).

---

## Day 1 — The Agent Loop

### Core Abstraction

    state → LLM → action → tool → observation → state update → repeat

Every agent framework (LangChain, AutoGen, CrewAI) is a variation of this
loop with different state representations and tool APIs.

### What "state" means

The agent carries:
- **question**: the original task (never changes)
- **history**: list of (thought, action, observation) triples appended at each step
- **answer**: set only when action == "finish"
- **steps**: loop counter (prevents infinite loops)

**Key insight**: the entire history is re-sent to the LLM on every step.
This means context grows linearly and attention cost grows quadratically.

### ReAct format (text parsing, Day 1 style)

Turn 1 — User asks the question.
Turn 2 — LLM emits: "Thought: I need to look up... Action: search[WW2 end year]"
Turn 3 — We append: "Observation: World War II ended in 1945."
Turn 4 — LLM emits: "Thought: Now calculate sqrt(1945). Action: calculate[sqrt(1945)]"
Turn 5 — We append: "Observation: 44.1020..."
Turn 6 — LLM emits: "Thought: I have the answer. Action: finish[~44.10]"

Each LLM call costs one API round-trip. This is the price of external grounding.

### Why single-step CoT is not the same as multi-step reliability

**Chain-of-thought (CoT)** produces all reasoning in ONE forward pass:

    [question] → LLM → [thought1 thought2 ... thoughtN answer]

Every reasoning token conditions on all prior tokens in the SAME response.
If token 50 is wrong (e.g., a hallucinated number), tokens 51-N are all
conditioned on that error. The model cannot correct itself mid-response.

**Multi-step agent loop** grounds each step against a real tool:

    [question] → LLM → action1 → tool → observation1 → LLM → action2 → ...

A wrong intermediate result gets corrected by the tool's real output, not
propagated. The probability compounding argument:

- Per-step accuracy: 90%
- CoT (5 steps, no grounding): 0.9^5 = 59% — and errors compound
- Agent (5 steps, grounded):   success rate depends only on tool reliability

The agent is more reliable precisely because it breaks the dependency chain.

### Parsing failure: the key fragility of text-based ReAct

The regex parser in day1 expects "Action: verb[arg]". If the model drifts to:
"I will now search for..." (no "Action:" prefix), the parser returns a
PARSE_ERROR. This is the #1 failure mode of text-based ReAct.

Day 2 fixes this with JSON schemas.

---

## Day 2 — Tool Calling Deep Dive

### Why JSON schema tool calling outperforms text parsing

Text parsing approach (Day 1):
- Model must format output as "Action: search[query]"
- One wrong character breaks the parser
- No schema validation — model can hallucinate argument names
- No graceful degradation — a bad format = a skipped step

JSON schema approach (Day 2):
- Model emits a ToolUseBlock with validated JSON fields
- Anthropic API validates the schema BEFORE returning the response
- If the model partially fills a schema, the API handles it
- The application never sees malformed tool calls

### The native tool_use API flow

    client.messages.create(tools=[TOOL_SCHEMAS], ...)

    if response.stop_reason == "tool_use":
        for block in response.content:
            if block.type == "tool_use":
                name   = block.name   # "wikipedia_search"
                inputs = block.input  # {"query": "..."} — validated JSON
                result = dispatch_tool(name, inputs)

        # Append the result as a tool_result message
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": block.id, "content": result}
        ]})

    elif response.stop_reason == "end_turn":
        final_answer = response.content[0].text

### Failure modes under adversarial pressure

**Tool hallucination**: model invents a tool name not in the schema.
- Mitigation: dispatch_tool() returns a clear error string ("Unknown tool X").
  The model receives this as a tool_result and usually self-corrects.

**Argument corruption**: model fills "query" with an expression instead of
a search term, or vice versa. The schema description (in "description" fields)
is your main defense — write clear, concrete examples there.

**Prompt injection**: malicious text in user input tries to override the
system prompt or call non-existent tools. Claude's safety training resists
most naive injections, but they can still affect tool argument content.
Defense: never pass user input directly as a tool argument without sanitization.

### What happens if the model partially fills a schema?

If a required field is missing, the Anthropic API will still return a
ToolUseBlock but with null or empty input. Your dispatch_tool() should
handle missing keys gracefully (use .get() with defaults, not direct indexing).

---

## Day 3 — Planning vs. Reasoning

### Two architectures

**ReactAgent** (inline, no separation):
- One LLM handles both planning and execution.
- Each step: reason about what to do → pick a tool → observe.
- Context: grows with every observation.
- Risk: noisy context can "distract" the model from the original goal.

**PlannerExecutorAgent** (separated):
- Planner LLM: reads the question ONCE with a clean context.
  Outputs a JSON list of atomic sub-tasks.
- Executor: for each sub-task, runs a mini tool-use loop.
  Receives only: (sub-task description) + (prior step results).
- Risk: if the plan is wrong, the executor cannot adapt.

### Why explicit planning reduces hallucination

In ReAct, the model must simultaneously:
1. Maintain an implicit plan (in its attention over all prior context)
2. Choose the next action
3. Keep the original goal in focus despite growing context

This is hard. Each step adds "noise" (tool outputs) that competes with the
original question in the attention mechanism.

In Planner-Executor:
1. The planner solves problem (1) once, with clean context.
2. Each executor call solves problem (2) for one atomic task.
3. Problem (3) is handled structurally — the plan is always visible.

The result: the executor makes fewer "off-topic" tool calls because it has
a single, clear objective rather than inferring it from noisy history.

### When is planning overkill?

Planning adds latency (one extra LLM call) and reduces adaptability (if an
early observation changes the strategy, the planner doesn't know).

Use planning for:
- Tasks with 4+ steps and known structure (e.g., "compare A and B")
- Tasks where sub-goals are parallelizable
- Tasks where incorrect sequencing is catastrophic

Use pure ReAct for:
- Open-ended exploration (unknown number of steps)
- Tasks where observations frequently change strategy
- Short tasks (1-3 steps) where planning overhead exceeds benefit

### What is search depth vs. reasoning depth?

**Reasoning depth**: how many logical inference steps the model must chain
within a single LLM call (CoT depth). Deep CoT = higher hallucination risk.

**Search depth**: how many tool calls the agent makes before reaching an
answer. Each tool call grounds one inference step, limiting error propagation.

Insight: a shallow-reasoning, deep-search agent (many small tool calls) is
often more reliable than a deep-reasoning, shallow-search agent (few tool
calls, lots of internal reasoning).

---

## Day 4 — Memory Types

### The four memory categories

    Short-term scratchpad   In-context message history. Fast, zero cost.
                            Ephemeral — vanishes after the conversation.

    Conversation history    Rolling window of prior turns. Managed truncation
                            (oldest turns dropped first). Risk: truncated
                            context loses critical earlier facts.

    Vector database         Embedded facts in ChromaDB/Pinecone/FAISS.
                            Retrieved by semantic similarity at each step.
                            Persists across conversations. Risk: retrieval
                            drift, memory poisoning, stale facts.

    Stateful symbolic mem   Key-value store, graph DB, SQL. Exact lookup.
                            Best for structured, deterministic data.
                            Risk: schema rigidity, no fuzzy matching.

### How vector memory works (Day 4 implementation)

At each tool call:
1. Store: embed(tool_name + input + output) → vector DB
2. Retrieve: embed(current_context) → top-k similar past observations

The agent receives these as "Relevant memories" in its next prompt.

ChromaDB uses cosine similarity over sentence-transformer embeddings
(all-MiniLM-L6-v2, 384 dimensions). Lower L2 distance = more relevant.

### Why does RAG fail in long-horizon agents?

Three main failure modes:

**Retrieval drift**: at step 5, the query embedding is about the current
sub-task (e.g., "convert km to miles"), not the original question. The memory
from step 1 ("the user wants the answer in minutes") may score low on this
query and be missed.

**Memory poisoning**: if a tool returns a wrong fact and the agent stores it,
that wrong fact will be retrieved with high confidence later. Vector retrieval
has no "trust score" — wrong and right memories are equally retrievable.

**Stale memory**: memories from a previous run of a different task may match
the current query semantically but contain completely wrong facts.

### When to use symbolic vs. vector memory

Use **symbolic** when: you need exact lookup (user ID, database records),
data is structured, correctness is paramount over flexibility.

Use **vector** when: queries are in natural language, vocabulary may not match
exactly, documents are long, fuzzy recall is acceptable.

**Hybrid** (best of both): use vector retrieval for "what do I know about X?"
and symbolic lookup for "what is the exact value of Y?".

---

## Day 5 — Long-Horizon Failure

### The exponential degradation model

If each step has error probability ε, after N steps:
    P(all correct) = (1 - ε)^N

    ε = 0.05: N=5 → 77%, N=10 → 60%
    ε = 0.10: N=5 → 59%, N=10 → 35%
    ε = 0.15: N=5 → 44%, N=10 → 20%

This is exponential decay, not linear. Real agents have ε > 0.10 per step
on ambiguous tasks because:
- Wikipedia disambiguation errors
- Calculator expression format errors
- Model misinterpreting a tool result

### The five failure modes

1. **Error accumulation**: each wrong intermediate result compounds.
   Wrong fact at step 2 → wrong calculation at step 4 → wrong answer.

2. **Overconfident hallucination**: the model finishes with a confident
   but wrong answer. Confidence and correctness are uncorrelated.

3. **State drift**: by step 8, the model's "implicit goal" (inferred from
   attention over all context) has drifted from the original question.
   It answers a slightly different question than the one asked.

4. **Tool misuse**: the model starts using tools incorrectly after many steps —
   sending a calculation expression to the search tool, for example.
   This is a consequence of context pollution (prior tool calls confuse the
   model about what each tool does).

5. **Memory overload**: the context window fills with prior tool results.
   The model must attend to all of them to produce the next action.
   With 12+ tool results in context, attention quality degrades noticeably.

### What the Day 5 plot shows

The scatter of (required_steps, success_rate) should show a downward trend
matching the theoretical (1-ε)^N curve. Deviations above the curve indicate
tasks where structure (predictable sequence) helps the agent. Deviations
below indicate tasks where tool reliability is poor (e.g., ambiguous
Wikipedia queries).

### How to build reliable long-horizon agents

1. **Explicit replanning**: after every K steps, ask the planner to review
   the current state and update the plan.
2. **Step verification**: after each step, ask a critic LLM: "Is this
   observation consistent with our goal?"
3. **Structured task graphs**: instead of a flat sequence, represent
   dependencies as a DAG. Execute nodes in topological order.
4. **Tool result validation**: add schemas/assertions to tool outputs.
   If a Wikipedia summary is under 50 chars, it's probably a disambiguation
   error — retry automatically.
5. **Memory snapshots**: periodically summarize the conversation into
   a compact "situation report" to replace verbose raw history.

---

## Architecture Summary

    Day 1: Text ReAct       state → LLM(text) → parse → tool → observe → repeat
    Day 2: Tool_use         state → LLM(json)  → schema → tool → observe → repeat
    Day 3: Planner-Exec     plan = LLM(question) → for step in plan: LLM(step) → tool
    Day 4: Memory+React     state + VectorDB.retrieve(context) → LLM → tool → store
    Day 5: Failure analysis  measure P(success) vs N and classify failure modes

Each day adds one dimension of complexity. The key insight of the week:

**Agents are reliable only when their reasoning is externally grounded at
every step. Grounding frequency, not model size, is the primary driver of
long-horizon accuracy.**
