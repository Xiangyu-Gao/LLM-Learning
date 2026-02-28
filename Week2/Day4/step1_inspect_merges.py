"""
Step 1: Inspect BPE Merges — How GPT-2 Sees Text.

Byte-Pair Encoding (BPE) is how real LLMs tokenize text.
Instead of character-level (like our tiny model), BPE learns
to merge frequent byte pairs into larger tokens.

This creates a vocabulary of ~50K tokens where:
  - Common words → single tokens ("the", " and", " is")
  - Rare words → fragmented ("Pneumonoultramicroscopicsilicovolcanoconiosis" → many pieces)
  - Numbers → unpredictable splits ("2026" → "20" + "26", "123" → "12" + "3")

This fragmentation is WHY LLMs struggle with:
  - Counting characters
  - Arithmetic
  - Spelling
  - Rare names
"""

import tiktoken
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"

# ===================================================================
# Load GPT-2 tokenizer
# ===================================================================
print(f"""
{'='*65}
{BOLD}INSPECT BPE MERGES — HOW GPT-2 SEES TEXT{RESET}
{'='*65}
""")

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

print(f"  GPT-2 BPE tokenizer loaded.")
print(f"  Vocabulary size: {vocab_size:,} tokens")
print(f"  (vs our character-level model: 38 tokens)")
print()

# ===================================================================
# 1. What's in the vocabulary?
# ===================================================================
print(f"{BOLD}1. WHAT'S IN THE VOCABULARY?{RESET}")
print(f"{'='*65}")
print()

# Decode every token to see what it looks like
# tiktoken uses byte-level encoding, so some tokens may contain
# raw bytes that aren't valid UTF-8 strings
token_strings = []
token_lengths = []

for i in range(vocab_size):
    try:
        s = enc.decode([i])
        token_strings.append(s)
        token_lengths.append(len(s))
    except Exception:
        token_strings.append(f"<byte:{i}>")
        token_lengths.append(0)

# Length distribution
length_counts = Counter(token_lengths)
print(f"  Token length distribution (in characters):")
print()
for length in sorted(length_counts.keys())[:15]:
    count = length_counts[length]
    bar = '█' * min(count // 200, 50)
    print(f"    {length:>2} chars: {count:>6,} tokens  {bar}")
print()

# Show some example tokens by category
print(f"  {BOLD}Single characters:{RESET}")
singles = [(i, s) for i, s in enumerate(token_strings) if len(s) == 1][:20]
for i, s in singles:
    print(f"    Token {i:>5}: '{s}' (0x{ord(s):02x})")
print(f"    ... ({len([s for s in token_strings if len(s) == 1])} total single-char tokens)")
print()

# Common English words (these got merged early, low token IDs)
print(f"  {BOLD}Common words (early merges → low token IDs):{RESET}")
common_words = [" the", " and", " of", " to", " in", " is", " that", " for",
                " it", " with", " as", " was", " on", " are", " be"]
for word in common_words:
    tokens = enc.encode(word)
    if len(tokens) == 1:
        print(f"    \"{word}\" → token {tokens[0]:>5}  {GREEN}(single token){RESET}")
    else:
        decoded = [enc.decode([t]) for t in tokens]
        print(f"    \"{word}\" → {tokens}  = {decoded}")
print()

# Long tokens (multi-character merges)
print(f"  {BOLD}Longest tokens in vocabulary:{RESET}")
long_tokens = [(i, s) for i, s in enumerate(token_strings)
               if len(s) >= 8 and s.isprintable()]
long_tokens.sort(key=lambda x: len(x[1]), reverse=True)
for i, s in long_tokens[:15]:
    print(f"    Token {i:>5}: \"{s}\" ({len(s)} chars)")
print()


# ===================================================================
# 2. How BPE builds tokens — the merge process
# ===================================================================
print(f"{BOLD}2. HOW BPE BUILDS TOKENS — THE MERGE PROCESS{RESET}")
print(f"{'='*65}")
print()
print(f"  BPE starts with individual bytes (256 base tokens).")
print(f"  Then it iteratively merges the most frequent adjacent pair.")
print()
print(f"  Example: training on \"aaabdaaabac\"")
print(f"    Bytes:  a a a b d a a a b a c")
print(f"    Step 1: Most frequent pair is (a,a) → merge to 'aa'")
print(f"            aa a b d aa a b a c")
print(f"    Step 2: Most frequent pair is (aa,a) → merge to 'aaa'")
print(f"            aaa b d aaa b a c")
print(f"    Step 3: Most frequent pair is (aaa,b) → merge to 'aaab'")
print(f"            aaab d aaab a c")
print(f"    ... and so on until we reach the desired vocabulary size.")
print()

# Show how GPT-2's BPE builds up a word
print(f"  {BOLD}Watch BPE build the word \"understanding\":{RESET}")
print()
word = "understanding"
# Show character-level
print(f"    Characters: {list(word)}")
tokens = enc.encode(word)
decoded = [enc.decode([t]) for t in tokens]
print(f"    GPT-2 BPE:  {decoded}")
print(f"    Token IDs:  {tokens}")
print()

# More examples showing the merge hierarchy
print(f"  {BOLD}Merge examples — how frequency shapes tokenization:{RESET}")
print()
examples = [
    ("the", "Most common English word"),
    ("The", "Capitalized → different token"),
    (" the", "With leading space (most common form)"),
    (" The", "Capitalized with space"),
    ("THE", "ALL CAPS → fragmented"),
    ("there", "Common word"),
    ("therefore", "Less common"),
    ("thermodynamics", "Rare word"),
    ("antidisestablishmentarianism", "Very rare"),
]

for word, desc in examples:
    tokens = enc.encode(word)
    decoded = [enc.decode([t]) for t in tokens]
    n_tok = len(tokens)
    color = GREEN if n_tok <= 2 else (YELLOW if n_tok <= 4 else RED)
    print(f"    \"{word}\"")
    print(f"      → {decoded}  ({color}{n_tok} token{'s' if n_tok > 1 else ''}{RESET})  {DIM}# {desc}{RESET}")
    print()


# ===================================================================
# 3. Numeric tokens — the source of math failures
# ===================================================================
print(f"{BOLD}3. NUMERIC TOKENS — WHY LLMs FAIL AT MATH{RESET}")
print(f"{'='*65}")
print()
print(f"  How does GPT-2 tokenize numbers? Not digit by digit!")
print()

# Show how different numbers get tokenized
print(f"  {BOLD}Single digits:{RESET}")
for d in range(10):
    tokens = enc.encode(str(d))
    print(f"    \"{d}\" → token {tokens[0]:>5}  ({len(tokens)} token)")
print()

print(f"  {BOLD}Two-digit numbers:{RESET}")
for n in [10, 12, 23, 42, 50, 67, 89, 99]:
    tokens = enc.encode(str(n))
    decoded = [enc.decode([t]) for t in tokens]
    color = GREEN if len(tokens) == 1 else RED
    dec_str = str(decoded)
    print(f"    \"{n}\" → {dec_str:>20}  ({color}{len(tokens)} token{'s' if len(tokens) > 1 else ''}{RESET})")
print()

print(f"  {BOLD}Three-digit numbers:{RESET}")
for n in [100, 123, 256, 314, 500, 777, 999]:
    tokens = enc.encode(str(n))
    decoded = [enc.decode([t]) for t in tokens]
    color = GREEN if len(tokens) == 1 else (YELLOW if len(tokens) == 2 else RED)
    dec_str = str(decoded)
    print(f"    \"{n}\" → {dec_str:>25}  ({color}{len(tokens)} token{'s' if len(tokens) > 1 else ''}{RESET})")
print()

print(f"  {BOLD}Larger numbers:{RESET}")
for n in [1000, 2026, 12345, 99999, 1000000, 3141592]:
    s = str(n)
    tokens = enc.encode(s)
    decoded = [enc.decode([t]) for t in tokens]
    print(f"    \"{s}\" → {decoded}")
    print(f"      Token IDs: {tokens}  ({RED}{len(tokens)} fragments{RESET})")
    print()

print(f"  {BOLD}The problem:{RESET}")
print(f"    \"123\" might become [\"12\", \"3\"] or [\"1\", \"23\"]")
print(f"    \"1234\" might become [\"12\", \"34\"] or [\"123\", \"4\"]")
print(f"    There's NO consistent digit-level structure!")
print()
print(f"    To add 123 + 456, the model would need to:")
print(f"      1. Realize \"12\" and \"3\" form the number 123")
print(f"      2. Realize \"45\" and \"6\" form the number 456")
print(f"      3. Perform digit-level addition with carry")
print(f"      4. Produce the right token fragments for 579")
print(f"    All through attention — no calculator, no algorithm.")
print()

# Show the inconsistency: same digits, different tokenization
print(f"  {BOLD}Same digits, different tokenization:{RESET}")
print()
numbers = ["123", "231", "312", "132", "213", "321"]
for n in numbers:
    tokens = enc.encode(n)
    decoded = [enc.decode([t]) for t in tokens]
    print(f"    \"{n}\" → {decoded}")
print()
print(f"  {RED}Same three digits, but BPE splits them differently!{RESET}")
print(f"  The model sees completely different token sequences")
print(f"  for what are structurally similar numbers.")
print()


# ===================================================================
# 4. Frequency drives everything
# ===================================================================
print(f"{BOLD}4. FREQUENCY DRIVES TOKENIZATION{RESET}")
print(f"{'='*65}")
print()
print(f"  BPE merges are learned from training data frequency.")
print(f"  Common substrings get their own tokens. Rare ones don't.")
print()

# Show how tokenization differs for common vs rare words
print(f"  {BOLD}Common words (few tokens) vs rare words (many tokens):{RESET}")
print()
word_pairs = [
    ("computer", "comptroller"),
    ("beautiful", "beauteous"),
    ("information", "informatization"),
    ("python", "pythonic"),
    ("hello", "hullo"),
    ("January", "Januarius"),
]

for common, rare in word_pairs:
    c_tok = enc.encode(common)
    r_tok = enc.encode(rare)
    c_dec = [enc.decode([t]) for t in c_tok]
    r_dec = [enc.decode([t]) for t in r_tok]
    print(f"    \"{common}\" → {c_dec}  ({GREEN}{len(c_tok)} tok{RESET})")
    print(f"    \"{rare}\" → {r_dec}  ({RED}{len(r_tok)} tok{RESET})")
    print()

# Show the "fertility" concept: tokens per character
print(f"  {BOLD}Tokens per character (fertility) across languages:{RESET}")
print()
texts = [
    ("English", "The quick brown fox jumps over the lazy dog"),
    ("Spanish", "El rápido zorro marrón salta sobre el perro perezoso"),
    ("French", "Le rapide renard brun saute par-dessus le chien paresseux"),
    ("German", "Der schnelle braune Fuchs springt über den faulen Hund"),
    ("Chinese", "快速的棕色狐狸跳过了懒惰的狗"),
    ("Japanese", "素早い茶色の狐が怠惰な犬を飛び越える"),
    ("Korean", "빠른 갈색 여우가 게으른 개를 뛰어넘는다"),
    ("Arabic", "الثعلب البني السريع يقفز فوق الكلب الكسول"),
]

for lang, text in texts:
    tokens = enc.encode(text)
    fertility = len(tokens) / len(text)
    bar = '█' * int(fertility * 20)
    color = GREEN if fertility < 0.5 else (YELLOW if fertility < 1.0 else RED)
    print(f"    {lang:<10} {len(text):>3} chars → {len(tokens):>3} tokens  "
          f"(fertility: {color}{fertility:.2f}{RESET})  {bar}")

print()
print(f"  {YELLOW}Non-Latin scripts have much higher fertility (more tokens per char).{RESET}")
print(f"  This means GPT-2 is ~3-4× less efficient for Chinese/Japanese/Korean.")
print(f"  Each token costs the same compute, so these languages get less")
print(f"  \"thinking\" per character of content.")
print()


# ===================================================================
# 5. Visualization
# ===================================================================
print(f"{BOLD}5. GENERATING VISUALIZATION...{RESET}")
print(f"{'='*65}")
print()

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("BPE Tokenization: How GPT-2 Sees Text", fontsize=15, fontweight='bold')

# --- Panel 1: Token length distribution ---
ax = axes[0, 0]
lengths = sorted(length_counts.keys())
counts = [length_counts[l] for l in lengths]
ax.bar(lengths[:20], counts[:20], color='steelblue', alpha=0.8)
ax.set_xlabel("Token length (characters)")
ax.set_ylabel("Number of tokens")
ax.set_title(f"Vocabulary: {vocab_size:,} tokens\nby character length")
ax.grid(True, alpha=0.3, axis='y')

# --- Panel 2: Tokenization of numbers ---
ax = axes[0, 1]
numbers_to_show = list(range(0, 1001, 1))
n_tokens_per_number = []
for n in numbers_to_show:
    tokens = enc.encode(str(n))
    n_tokens_per_number.append(len(tokens))

ax.scatter(numbers_to_show, n_tokens_per_number, s=1, alpha=0.5, c='red')
ax.set_xlabel("Number (0-1000)")
ax.set_ylabel("BPE tokens needed")
ax.set_title("Tokenization of Integers 0-1000")
ax.set_yticks([1, 2, 3, 4])
ax.grid(True, alpha=0.3)

# --- Panel 3: Fertility across languages ---
ax = axes[0, 2]
langs = [l for l, _ in texts]
fertilities = []
for _, text in texts:
    tokens = enc.encode(text)
    fertilities.append(len(tokens) / len(text))

colors = [('#2ecc71' if f < 0.5 else '#f39c12' if f < 1.0 else '#e74c3c') for f in fertilities]
bars = ax.barh(langs, fertilities, color=colors, alpha=0.8)
ax.set_xlabel("Tokens per character (fertility)")
ax.set_title("Fertility Across Languages")
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3, axis='x')

# --- Panel 4: Tokenization visualization for a sentence ---
ax = axes[1, 0]
sentence = "The year 2026 costs $1,234.56"
tokens = enc.encode(sentence)
decoded = [enc.decode([t]) for t in tokens]

# Color each token differently
cmap = plt.cm.Set3
y_pos = 0.5
x_pos = 0.02
for i, tok_str in enumerate(decoded):
    display = tok_str.replace(' ', '·')
    color = cmap(i % 12)
    bbox = dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7)
    ax.text(x_pos, y_pos, display, fontsize=9, fontfamily='monospace',
            bbox=bbox, va='center', transform=ax.transAxes)
    x_pos += (len(display) * 0.045 + 0.03)
    if x_pos > 0.95:
        y_pos -= 0.15
        x_pos = 0.02

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title(f"Tokenization of: \"{sentence}\"\n({len(tokens)} tokens)")
ax.axis('off')

# --- Panel 5: Number fragmentation detail ---
ax = axes[1, 1]
test_nums = ["1", "12", "123", "1234", "12345", "123456", "1234567"]
y_positions = range(len(test_nums))
for y, num_str in enumerate(test_nums):
    tokens = enc.encode(num_str)
    decoded = [enc.decode([t]) for t in tokens]
    x = 0
    for i, (tid, tok_str) in enumerate(zip(tokens, decoded)):
        width = len(tok_str)
        color = plt.cm.Set2(i % 8)
        ax.barh(y, width, left=x, height=0.6, color=color, edgecolor='black', linewidth=0.5)
        ax.text(x + width/2, y, tok_str, ha='center', va='center', fontsize=9, fontweight='bold')
        x += width

ax.set_yticks(list(y_positions))
ax.set_yticklabels(test_nums)
ax.set_xlabel("Character position")
ax.set_title("How BPE Fragments Numbers")
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# --- Panel 6: Tokens needed for n-digit numbers ---
ax = axes[1, 2]
for n_digits in [1, 2, 3, 4, 5, 6]:
    if n_digits <= 4:
        # Check all numbers of this length
        start = 10**(n_digits-1) if n_digits > 1 else 0
        end = 10**n_digits
        tok_counts = [len(enc.encode(str(n))) for n in range(start, min(end, start + 2000))]
        avg_tok = np.mean(tok_counts)
        min_tok = min(tok_counts)
        max_tok = max(tok_counts)
    else:
        # Sample
        nums = np.random.randint(10**(n_digits-1), 10**n_digits, 500)
        tok_counts = [len(enc.encode(str(n))) for n in nums]
        avg_tok = np.mean(tok_counts)
        min_tok = min(tok_counts)
        max_tok = max(tok_counts)

    ax.bar(n_digits, avg_tok, color='steelblue', alpha=0.7, edgecolor='black')
    ax.errorbar(n_digits, avg_tok, yerr=[[avg_tok - min_tok], [max_tok - avg_tok]],
                color='black', capsize=3)

ax.set_xlabel("Number of digits")
ax.set_ylabel("BPE tokens needed")
ax.set_title("Tokens per Number Length")
ax.set_xticks(range(1, 7))
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), 'inspect_merges.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"  Saved visualization → {save_path}")
print()


# ===================================================================
# Summary
# ===================================================================
print(f"""
{'='*65}
{BOLD}STEP 1 SUMMARY: BPE MERGES{RESET}
{'='*65}

{BOLD}How BPE works:{RESET}

  1. Start with 256 byte-level tokens
  2. Count all adjacent pairs in training data
  3. Merge the most frequent pair → new token
  4. Repeat until vocabulary reaches target size (50,257 for GPT-2)

  Result: common substrings → single tokens, rare ones → fragments.

{BOLD}Key observations:{RESET}

  {GREEN}1. Common English words are single tokens{RESET}
     " the", " and", " is" → 1 token each
     Efficient: 1 token ≈ 4-5 characters on average

  {YELLOW}2. Rare words get fragmented{RESET}
     "thermodynamics" → ["therm", "odynamics"] (2 tokens)
     "antidisestablishmentarianism" → many tokens
     More tokens = more compute = less context available

  {RED}3. Numbers are tokenized INCONSISTENTLY{RESET}
     "2026" → ["20", "26"],  "3141592" → ["314", "15", "92"]
     Splits are arbitrary — no digit-level structure!
     No digit alignment → no arithmetic structure

  {RED}4. Non-Latin scripts are penalized{RESET}
     English: ~0.2 tokens/char  (efficient)
     Chinese: ~2.1 tokens/char  (10× less efficient!)
     Same compute budget, less content processed

{BOLD}Why this matters for Step 2:{RESET}

  Tokenization isn't neutral — it shapes what the model CAN learn.
  If numbers are split randomly, arithmetic becomes a pattern-matching
  problem rather than an algorithmic one.
""")
