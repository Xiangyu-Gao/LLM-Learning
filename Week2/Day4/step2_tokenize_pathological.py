"""
Step 2: Tokenize Pathological Inputs — Why LLMs Can't Count.

Try tokenizing inputs that stress BPE:
  - Large integers, decimals
  - Code snippets
  - Rare names
  - Chemical formulas
  - Spelling tasks

Then write: "Why LLMs can't count reliably."

Core causes:
  1. Numbers are split into arbitrary fragments
  2. No digit-level positional structure
  3. Training objective predicts next token, not arithmetic truth
  4. Attention doesn't enforce algorithmic computation

Math failures are ARCHITECTURAL, not stupidity.
"""

import tiktoken
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"

enc = tiktoken.get_encoding("gpt2")

print(f"""
{'='*65}
{BOLD}TOKENIZE PATHOLOGICAL INPUTS{RESET}
{'='*65}

  BPE was designed for natural language compression.
  What happens when we feed it things that AREN'T natural language?
""")


def show_tokenization(text, label=""):
    """Display tokenization with colored token boundaries."""
    tokens = enc.encode(text)
    decoded = [enc.decode([t]) for t in tokens]
    if label:
        print(f"  {BOLD}{label}{RESET}")
    print(f"    Input:  \"{text}\"")
    # Show with | separators and token IDs
    tok_display = "|".join(decoded)
    print(f"    Tokens: |{tok_display}|")
    print(f"    IDs:    {tokens}")
    print(f"    Count:  {len(tokens)} tokens for {len(text)} characters "
          f"({len(tokens)/len(text):.2f} tok/char)")
    print()
    return tokens


# ===================================================================
# 1. Large integers
# ===================================================================
print(f"{BOLD}1. LARGE INTEGERS{RESET}")
print(f"{'='*65}")
print()

integers = [
    ("42", "Small, common"),
    ("1000", "Round number"),
    ("2026", "Current year"),
    ("12345", "Sequential digits"),
    ("99999", "Repeated digit"),
    ("314159", "Pi digits"),
    ("1000000", "One million"),
    ("123456789", "All digits in order"),
    ("999999999999", "Twelve 9s"),
    ("18446744073709551616", "2^64 (unsigned overflow)"),
]

for num_str, desc in integers:
    tokens = show_tokenization(num_str, f"{num_str} — {desc}")

print(f"  {BOLD}Observation:{RESET}")
print(f"  Numbers get split at ARBITRARY boundaries.")
print(f"  \"12345\" → \"123\" + \"45\", not \"1\" + \"2\" + \"3\" + \"4\" + \"5\"")
print(f"  The model has no concept of place value (ones, tens, hundreds).")
print(f"  Each fragment is an opaque token — \"123\" is NOT \"1\",\"2\",\"3\".")
print()


# ===================================================================
# 2. Decimal numbers
# ===================================================================
print(f"{BOLD}2. DECIMAL NUMBERS{RESET}")
print(f"{'='*65}")
print()

decimals = [
    "3.14",
    "3.14159",
    "3.141592653589793",
    "0.001",
    "1.23e10",
    "$1,234.56",
    "99.99%",
    "-273.15",
    "6.022e23",
]

for d in decimals:
    show_tokenization(d)

print(f"  {BOLD}Observation:{RESET}")
print(f"  The decimal point \".\" sometimes merges with digits, sometimes not.")
print(f"  \"3.14\" might be [\"3\", \".\", \"14\"] — three separate concepts")
print(f"  that the model must learn to compose into a single number.")
print(f"  Scientific notation (\"1.23e10\") adds yet another fragmentation layer.")
print()


# ===================================================================
# 3. Code snippets
# ===================================================================
print(f"{BOLD}3. CODE SNIPPETS{RESET}")
print(f"{'='*65}")
print()

code_snippets = [
    ("x = 42", "Simple assignment"),
    ("for i in range(10):", "Python loop"),
    ("def fibonacci(n):", "Function definition"),
    ("if __name__ == '__main__':", "Main guard"),
    ("import torch.nn.functional as F", "Common import"),
    ("self.attention = nn.MultiheadAttention(embed_dim, num_heads)", "PyTorch layer"),
    ("std::vector<std::string> names;", "C++ template"),
    ("SELECT * FROM users WHERE id = 1;", "SQL query"),
    ("console.log('hello world');", "JavaScript"),
    ("fn main() -> Result<(), Box<dyn Error>> {", "Rust function"),
]

for code, desc in code_snippets:
    show_tokenization(code, desc)

print(f"  {BOLD}Observation:{RESET}")
print(f"  Common Python patterns (\"def\", \"for\", \"import\") are single tokens.")
print(f"  But operators and punctuation fragment heavily:")
print(f"    \"->\" \"==\" \"__\" may or may not merge with adjacent text.")
print(f"  Indentation (spaces/tabs) consumes tokens too —")
print(f"  deeply nested code uses many tokens just for whitespace.")
print()


# ===================================================================
# 4. Rare names
# ===================================================================
print(f"{BOLD}4. RARE NAMES{RESET}")
print(f"{'='*65}")
print()

names = [
    ("John Smith", "Common English name"),
    ("Barack Obama", "Famous person (in training data)"),
    ("Schwarzenegger", "Complex but famous"),
    ("Xiangyu Gao", "Chinese name in Latin script"),
    ("Nguyễn Văn An", "Vietnamese name with diacritics"),
    ("Лев Толстой", "Russian name in Cyrillic"),
    ("村上春樹", "Japanese name (Murakami Haruki)"),
    ("Eyjafjallajökull", "Icelandic volcano"),
    ("Wolframalpha", "Compound word"),
    ("Hugging Face", "AI company"),
]

for name, desc in names:
    show_tokenization(name, desc)

print(f"  {BOLD}Observation:{RESET}")
print(f"  Famous names that appeared frequently in training data →")
print(f"  fewer tokens (\"Obama\" → 1 token). Rare names → many fragments.")
print(f"  Non-Latin names are especially penalized: each character")
print(f"  may become 2-3 tokens, making the model's \"attention budget\"")
print(f"  much smaller for these names.")
print()


# ===================================================================
# 5. Chemical formulas & technical notation
# ===================================================================
print(f"{BOLD}5. CHEMICAL FORMULAS & TECHNICAL NOTATION{RESET}")
print(f"{'='*65}")
print()

formulas = [
    ("H2O", "Water"),
    ("C6H12O6", "Glucose"),
    ("NaHCO3", "Baking soda"),
    ("CH3CH2OH", "Ethanol"),
    ("C8H10N4O2", "Caffeine"),
    ("Fe2O3", "Iron oxide (rust)"),
    ("Ca(OH)2", "Calcium hydroxide"),
    ("E = mc²", "Mass-energy equivalence"),
    ("∫f(x)dx", "Integral notation"),
    ("∇×B = μ₀J + μ₀ε₀∂E/∂t", "Maxwell's equation"),
]

for formula, desc in formulas:
    show_tokenization(formula, desc)

print(f"  {BOLD}Observation:{RESET}")
print(f"  Chemical formulas are fragmented unpredictably.")
print(f"  \"H2O\" might be 1-3 tokens. Subscript numbers merge")
print(f"  with element symbols or not — there's no chemical structure")
print(f"  in the tokenization. The model must learn chemistry from")
print(f"  random byte-pair fragments.")
print()


# ===================================================================
# 6. The spelling problem
# ===================================================================
print(f"{BOLD}6. THE SPELLING PROBLEM{RESET}")
print(f"{'='*65}")
print()
print(f"  \"How many r's in strawberry?\"")
print(f"  This is famously hard for LLMs. Let's see why:")
print()

spelling_tests = [
    "strawberry",
    "mississippi",
    "occurrence",
    "accommodation",
    "onomatopoeia",
    "supercalifragilisticexpialidocious",
]

for word in spelling_tests:
    tokens = enc.encode(word)
    decoded = [enc.decode([t]) for t in tokens]
    n_tokens = len(tokens)

    # Count a specific letter
    # Find most repeated letter
    from collections import Counter
    letter_counts = Counter(word.lower())
    most_common_letter, count = letter_counts.most_common(1)[0]

    print(f"  \"{word}\"")
    print(f"    Tokens: {decoded}")
    print(f"    How many '{most_common_letter}'s? Answer: {count}")

    # Can we see individual letters in the tokens?
    # Check if any token boundary falls between repeated letters
    char_positions = [i for i, c in enumerate(word) if c == most_common_letter]

    # Map character positions to token boundaries
    pos = 0
    token_containing = []
    for t_idx, tok_str in enumerate(decoded):
        for _ in tok_str:
            if pos in char_positions:
                token_containing.append(t_idx)
            pos += 1

    unique_tokens = len(set(token_containing))
    if unique_tokens == 1:
        print(f"    All '{most_common_letter}'s are INSIDE token \"{decoded[token_containing[0]]}\"")
        print(f"    {RED}The model can't see individual letters!{RESET}")
    else:
        print(f"    '{most_common_letter}'s are spread across {unique_tokens} tokens: "
              f"{[decoded[t] for t in sorted(set(token_containing))]}")
        print(f"    {YELLOW}Model must count across token boundaries.{RESET}")
    print()

print(f"  {BOLD}Why is \"how many r's in strawberry\" hard?{RESET}")
print()
tokens = enc.encode("strawberry")
decoded = [enc.decode([t]) for t in tokens]
print(f"    The model sees: {decoded}")
print(f"    NOT: ['s','t','r','a','w','b','e','r','r','y']")
print(f"    The 'r's are buried inside multi-character tokens.")
print(f"    Counting requires CHARACTER-level reasoning,")
print(f"    but the model operates at TOKEN-level.")
print()


# ===================================================================
# 7. Arithmetic fragmentation
# ===================================================================
print(f"{BOLD}7. WHY ARITHMETIC FAILS — A DETAILED EXAMPLE{RESET}")
print(f"{'='*65}")
print()
print(f"  Task: Compute 2847 + 1536 = ?")
print()

# Show how the model sees this
problem = "2847 + 1536 = "
show_tokenization(problem, "The addition problem")

answer = "4383"
show_tokenization(answer, "The expected answer")

full = "2847 + 1536 = 4383"
show_tokenization(full, "Full equation")

print(f"  {BOLD}What the model NEEDS to do:{RESET}")
print(f"    Digit-level addition with carry:")
print(f"      7 + 6 = 13  → write 3, carry 1")
print(f"      4 + 3 + 1 = 8  → write 8")
print(f"      8 + 5 = 13  → write 3, carry 1")
print(f"      2 + 1 + 1 = 4  → write 4")
print(f"    Answer: 4383")
print()
print(f"  {BOLD}What the model ACTUALLY sees:{RESET}")
a_tokens = enc.encode("2847")
b_tokens = enc.encode("1536")
ans_tokens = enc.encode("4383")
a_decoded = [enc.decode([t]) for t in a_tokens]
b_decoded = [enc.decode([t]) for t in b_tokens]
ans_decoded = [enc.decode([t]) for t in ans_tokens]
print(f"    Operand A: {a_decoded}  (NOT individual digits!)")
print(f"    Operand B: {b_decoded}")
print(f"    Answer:    {ans_decoded}")
print()
print(f"    The model must learn a function:")
print(f"      f({a_decoded}, {b_decoded}) → {ans_decoded}")
print(f"    This is a LOOKUP problem, not an algorithm.")
print(f"    There are 10,000 × 10,000 = 100M possible 4-digit additions.")
print(f"    The model can't memorize them all — it must generalize.")
print(f"    But the token boundaries make generalization nearly impossible")
print(f"    because similar numbers have different tokenizations.")
print()


# ===================================================================
# 8. Visualization
# ===================================================================
print(f"{BOLD}8. GENERATING VISUALIZATION...{RESET}")
print(f"{'='*65}")
print()

fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.suptitle("Tokenization Pathology: Why LLMs Fail at Structured Tasks",
             fontsize=15, fontweight='bold')

# --- Panel 1: Token count for different categories ---
ax = axes[0, 0]
categories = {
    'English words': ["the", "hello", "computer", "understanding", "beautiful"],
    'Numbers': ["42", "2026", "12345", "314159", "99999999"],
    'Code': ["def f():", "for i in range(10):", "import torch", "self.attn", "x = 42"],
    'Names (Latin)': ["John", "Obama", "Schwarzenegger", "Einstein", "Eyjafjallajökull"],
    'Names (CJK)': ["村上春樹", "习近平", "김정은", "Лев Толстой", "محمد"],
    'Formulas': ["H2O", "C6H12O6", "NaHCO3", "E=mc²", "∇×B=μ₀J"],
}

cat_names = []
tok_per_char = []
colors_cat = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c']

for (cat, examples), color in zip(categories.items(), colors_cat):
    fertilities = []
    for ex in examples:
        tokens = enc.encode(ex)
        fertilities.append(len(tokens) / len(ex))
    cat_names.append(cat)
    tok_per_char.append(np.mean(fertilities))

bars = ax.barh(cat_names, tok_per_char, color=colors_cat, alpha=0.8)
ax.set_xlabel("Tokens per character (avg)")
ax.set_title("Tokenization Efficiency by Category")
ax.axvline(x=0.3, color='gray', linestyle='--', alpha=0.5, label='English avg')
ax.grid(True, alpha=0.3, axis='x')

# --- Panel 2: Addition problem visualization ---
ax = axes[0, 1]
ax.axis('off')
ax.set_title("Why 2847 + 1536 is Hard", fontsize=11)

# Show what humans see vs what the model sees
y = 0.9
ax.text(0.05, y, "What humans see:", fontsize=11, fontweight='bold', transform=ax.transAxes)
y -= 0.08
ax.text(0.05, y, "  2  8  4  7", fontsize=14, fontfamily='monospace', transform=ax.transAxes,
        color='green')
ax.text(0.55, y, "← individual digits", fontsize=9, transform=ax.transAxes, color='gray')
y -= 0.06
ax.text(0.05, y, "+ 1  5  3  6", fontsize=14, fontfamily='monospace', transform=ax.transAxes,
        color='green')
y -= 0.06
ax.text(0.05, y, "─────────", fontsize=14, fontfamily='monospace', transform=ax.transAxes)
y -= 0.06
ax.text(0.05, y, "  4  3  8  3", fontsize=14, fontfamily='monospace', transform=ax.transAxes,
        color='green')

y -= 0.12
ax.text(0.05, y, "What GPT-2 sees:", fontsize=11, fontweight='bold', transform=ax.transAxes)
y -= 0.08

a_tokens = enc.encode("2847")
b_tokens = enc.encode("1536")
ans_tokens = enc.encode("4383")

cmap = plt.cm.Set2
x_pos = 0.05
for i, tok in enumerate([enc.decode([t]) for t in a_tokens]):
    bbox = dict(boxstyle='round,pad=0.3', facecolor=cmap(i % 8), alpha=0.7)
    ax.text(x_pos, y, f'"{tok}"', fontsize=12, fontfamily='monospace',
            bbox=bbox, transform=ax.transAxes)
    x_pos += len(tok) * 0.06 + 0.08
ax.text(0.55, y, "← opaque chunks", fontsize=9, transform=ax.transAxes, color='gray')

y -= 0.08
x_pos = 0.05
for i, tok in enumerate([enc.decode([t]) for t in b_tokens]):
    bbox = dict(boxstyle='round,pad=0.3', facecolor=cmap((i+3) % 8), alpha=0.7)
    ax.text(x_pos, y, f'"{tok}"', fontsize=12, fontfamily='monospace',
            bbox=bbox, transform=ax.transAxes)
    x_pos += len(tok) * 0.06 + 0.08

y -= 0.10
ax.text(0.05, y, "No digit alignment → can't do column addition!",
        fontsize=10, color='red', fontweight='bold', transform=ax.transAxes)


# --- Panel 3: Spelling problem ---
ax = axes[0, 2]
ax.axis('off')
ax.set_title('Why "How many r\'s in strawberry?" is Hard', fontsize=10)

word = "strawberry"
tokens = enc.encode(word)
decoded = [enc.decode([t]) for t in tokens]

y = 0.9
ax.text(0.05, y, "Character-level view:", fontsize=11, fontweight='bold',
        transform=ax.transAxes)
y -= 0.07
for i, ch in enumerate(word):
    color = '#e74c3c' if ch == 'r' else '#bdc3c7'
    bbox = dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7)
    ax.text(0.05 + i * 0.08, y, ch, fontsize=14, fontfamily='monospace',
            bbox=bbox, transform=ax.transAxes, ha='center')

y -= 0.1
ax.text(0.05, y, f"r count: 3  (visible at character level)",
        fontsize=10, color='green', transform=ax.transAxes)

y -= 0.12
ax.text(0.05, y, "Token-level view (what GPT-2 sees):", fontsize=11,
        fontweight='bold', transform=ax.transAxes)
y -= 0.07
x_pos = 0.05
for i, tok_str in enumerate(decoded):
    bbox = dict(boxstyle='round,pad=0.3', facecolor=cmap(i % 8), alpha=0.7)
    ax.text(x_pos, y, f'"{tok_str}"', fontsize=13, fontfamily='monospace',
            bbox=bbox, transform=ax.transAxes)
    x_pos += len(tok_str) * 0.06 + 0.10

y -= 0.1
ax.text(0.05, y, f"r's hidden inside tokens — model can't \"see\" individual letters!",
        fontsize=10, color='red', transform=ax.transAxes)

y -= 0.12
ax.text(0.05, y, "This is why LLMs need chain-of-thought:", fontsize=10,
        fontweight='bold', transform=ax.transAxes)
y -= 0.07
ax.text(0.05, y, '"s-t-r-a-w-b-e-r-r-y" → spell it out', fontsize=9,
        fontfamily='monospace', transform=ax.transAxes, color='#2980b9')
y -= 0.06
ax.text(0.05, y, "Forces character-level reasoning at token level", fontsize=9,
        transform=ax.transAxes, color='#2980b9')


# --- Panel 4: Tokenization of math expressions ---
ax = axes[1, 0]
expressions = [
    "1+1=2",
    "12+34=46",
    "123+456=579",
    "1234+5678=6912",
    "12345+67890=80235",
]
y_positions = range(len(expressions))
max_tokens = 0
for y, expr in enumerate(expressions):
    tokens = enc.encode(expr)
    decoded = [enc.decode([t]) for t in tokens]
    x = 0
    for i, (tid, tok_str) in enumerate(zip(tokens, decoded)):
        width = max(len(tok_str), 1)
        color = plt.cm.Set3(i % 12)
        ax.barh(y, width, left=x, height=0.6, color=color,
                edgecolor='black', linewidth=0.5)
        display = tok_str if tok_str.strip() else '·'
        ax.text(x + width/2, y, display, ha='center', va='center',
                fontsize=8, fontweight='bold')
        x += width
    max_tokens = max(max_tokens, x)

ax.set_yticks(list(y_positions))
ax.set_yticklabels(expressions)
ax.set_xlabel("Character position")
ax.set_title("Tokenization of Addition Problems")
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# --- Panel 5: Same number, different context ---
ax = axes[1, 1]
number = "1234"
contexts = [
    f"{number}",
    f"x = {number}",
    f"The year {number}",
    f"${number}.00",
    f"{number} tokens",
    f"port {number}",
]
y_positions = range(len(contexts))
for y, ctx in enumerate(contexts):
    tokens = enc.encode(ctx)
    decoded = [enc.decode([t]) for t in tokens]
    x = 0
    for i, tok_str in enumerate(decoded):
        width = max(len(tok_str), 1)
        color = plt.cm.Pastel1(i % 9)
        ax.barh(y, width, left=x, height=0.6, color=color,
                edgecolor='black', linewidth=0.5)
        display = tok_str if tok_str.strip() else '·'
        ax.text(x + width/2, y, display, ha='center', va='center',
                fontsize=7, fontweight='bold')
        x += width

ax.set_yticks(list(y_positions))
ax.set_yticklabels(contexts, fontsize=8)
ax.set_xlabel("Character position")
ax.set_title(f"Same Number \"{number}\" in Different Contexts")
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# --- Panel 6: The 4 architectural causes ---
ax = axes[1, 2]
ax.axis('off')
ax.set_title("Why LLMs Can't Count: 4 Architectural Causes", fontsize=11)

causes = [
    ("1. Numbers are split\n   into arbitrary fragments",
     "\"2847\" → opaque chunks, not digits\nNo place value, no column alignment"),
    ("2. No digit-level\n   positional structure",
     "Position 1 = first TOKEN, not first DIGIT\nTokens have variable character width"),
    ("3. Training objective:\n   predict next token",
     "Not: \"compute arithmetic truth\"\nCorrectness is a side effect, not the goal"),
    ("4. Attention ≠\n   algorithmic computation",
     "Attention is weighted averaging\nAddition requires carry propagation\n(sequential, not parallel)"),
]

y = 0.92
for i, (title, detail) in enumerate(causes):
    color = ['#e74c3c', '#e67e22', '#f1c40f', '#9b59b6'][i]
    bbox = dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.2,
                edgecolor=color, linewidth=2)
    ax.text(0.02, y, title, fontsize=10, fontweight='bold',
            transform=ax.transAxes, va='top', bbox=bbox)
    ax.text(0.45, y, detail, fontsize=8, transform=ax.transAxes,
            va='top', color='#333333', fontfamily='monospace')
    y -= 0.25

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), 'tokenize_pathological.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"  Saved visualization → {save_path}")
print()


# ===================================================================
# WHY LLMs CAN'T COUNT RELIABLY
# ===================================================================
print(f"""
{'='*65}
{BOLD}WHY LLMs CAN'T COUNT RELIABLY{RESET}
{'='*65}

Math failures are {RED}architectural{RESET}, not stupidity.

{BOLD}Cause 1: Numbers are split into arbitrary fragments{RESET}

  BPE tokenization splits numbers at boundaries determined by
  TRAINING DATA FREQUENCY, not mathematical structure.

    "2847" → might be ["28", "47"] or ["284", "7"]
    "1536" → might be ["15", "36"] or ["153", "6"]

  To add these, the model would need to:
    - Reconstruct individual digits from token fragments
    - Align digits by place value (ones, tens, hundreds)
    - Perform carry operations across token boundaries

  This is like asking a human to add "twenty-eight forty-seven"
  to "fifteen thirty-six" — possible, but unnatural and error-prone.

{BOLD}Cause 2: No digit-level positional structure{RESET}

  Position embeddings encode TOKEN positions, not DIGIT positions.
  "123" at position 5 means "the 5th token is '123'" — not
  "digits 1, 2, 3 are at positions 5, 6, 7."

  The model has no built-in concept of:
    - Place value (ones column, tens column, hundreds column)
    - Digit alignment between numbers
    - Right-to-left processing (which addition requires)

{BOLD}Cause 3: Training predicts next token, not arithmetic truth{RESET}

  The loss function is: P(next_token | previous_tokens)

  This rewards the model for predicting what TYPICALLY FOLLOWS
  a sequence, not for computing correct answers.

  If "123 + 456 =" appears rarely in training data, the model
  has little signal to learn the correct answer.
  It may predict "500" (round number bias) or "579" (correct)
  with similar confidence — because both look plausible as
  next-token predictions.

{BOLD}Cause 4: Attention doesn't enforce algorithmic computation{RESET}

  Attention is a soft, parallel, weighted average.
  Addition is a hard, sequential, conditional algorithm:

    Step 1: 7 + 6 = 13 → write 3, carry 1
    Step 2: 4 + 3 + carry = 8 → write 8
    Step 3: 8 + 5 = 13 → write 3, carry 1
    Step 4: 2 + 1 + carry = 4 → write 4

  Each step DEPENDS on the previous carry.
  This is inherently sequential — attention can't parallelize it.
  The model must simulate this algorithm through learned weights,
  layer by layer, using the transformer as a "program."

  Research shows transformers CAN learn addition for numbers
  up to a certain length, but they fail to generalize beyond
  the lengths seen in training. They're memorizing patterns,
  not learning the algorithm.

{'='*65}
{BOLD}THE DEEPER INSIGHT{RESET}
{'='*65}

  {CYAN}Tokenization determines the atoms of thought.{RESET}

  Our character-level model sees: s, t, r, a, w, b, e, r, r, y
  GPT-2 sees:                     str, aw, berry  (or similar)

  A model can only reason about what it can SEE.
  If individual digits are hidden inside tokens, digit-level
  reasoning requires extra work (chain-of-thought, tool use).

  This is why:
    • {GREEN}Chain-of-thought helps{RESET}: "Let me work through this step by step"
      forces the model to externalize intermediate computations
    • {GREEN}Tool use helps{RESET}: a calculator processes digits directly
    • {GREEN}Specialized tokenization helps{RESET}: some models tokenize
      each digit separately for math tasks
    • {GREEN}Scratchpads help{RESET}: writing out intermediate steps gives
      the model "memory" for carry values

  {BOLD}Math failures are not a sign of LLM "stupidity."{RESET}
  They're a predictable consequence of:
    1. Frequency-based tokenization (BPE)
    2. Fixed-depth computation (finite transformer layers)
    3. Soft attention (no hard algorithmic primitives)
    4. Next-token prediction (no arithmetic objective)

  A calculator with 10 tokens (one per digit) and a hardcoded
  addition algorithm will ALWAYS beat a trillion-parameter LLM
  at arithmetic. That's not a flaw — it's a design trade-off.
  LLMs trade arithmetic precision for general intelligence.
""")
