"""
Step 2: Plot RoPE rotation.

Take a sample embedding vector, apply RoPE for positions 0, 1, 2, ...,
and plot 2D projections of the rotated dimensions.

Key realization: RoPE encodes position as ROTATION, not addition.
  - Each dimension pair (2i, 2i+1) is treated as a 2D plane
  - Position m rotates that plane by angle m * θ_i
  - The SAME vector at different positions traces out a circle
"""

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math
import os

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# RoPE functions (from step1, redefined to avoid import side effects)
# ---------------------------------------------------------------------------
def build_rope_freqs(dim, max_len, base=10000.0):
    """Precompute rotation frequencies: e^(i·m·θ) for each position m and dim pair."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_len).float()
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # (max_len, dim/2)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x, freqs_cis):
    """Apply rotary embedding: treat dim pairs as complex numbers, multiply by rotation."""
    B, C, D = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, C, D // 2, 2))
    x_rotated = x_complex * freqs_cis[:C].unsqueeze(0)
    return torch.view_as_real(x_rotated).reshape(B, C, D).type_as(x)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
EMBED_DIM = 64
MAX_POS = 32

freqs_cis = build_rope_freqs(EMBED_DIM, MAX_POS)

# Create a FIXED embedding vector (same content at every position)
torch.manual_seed(42)
vec = torch.randn(1, 1, EMBED_DIM)  # one vector, (1, 1, E)

# Tile it to all positions: (1, MAX_POS, E)
vec_all_pos = vec.expand(1, MAX_POS, EMBED_DIM).clone()

# Apply RoPE — the SAME vector gets rotated differently at each position
vec_rotated = apply_rope(vec_all_pos, freqs_cis)  # (1, MAX_POS, E)
vec_rotated = vec_rotated[0]  # (MAX_POS, E)
vec_orig = vec[0, 0]  # (E,)

print(f"""
{'='*65}
{BOLD}ROPE ROTATION VISUALIZATION{RESET}
{'='*65}

  Setup: one fixed embedding vector, placed at positions 0..{MAX_POS-1}.
  RoPE rotates each dimension pair (2i, 2i+1) by angle = pos × θ_i.
  The SAME content at different positions → different rotated vectors.
""")

# ---------------------------------------------------------------------------
# Demo 1: Show the rotation in a single dimension pair
# ---------------------------------------------------------------------------
print(f"{BOLD}1. ROTATION IN A SINGLE DIMENSION PAIR{RESET}")
print(f"{'='*65}")
print()

# Pick dim pair 0: dimensions (0, 1)
d0 = vec_rotated[:, 0].numpy()  # x-coordinates at each position
d1 = vec_rotated[:, 1].numpy()  # y-coordinates at each position

orig_x = vec_orig[0].item()
orig_y = vec_orig[1].item()
radius = math.sqrt(orig_x**2 + orig_y**2)

print(f"  Dimension pair (0, 1) — the FASTEST rotating pair:")
print(f"    Original vector:  ({orig_x:.3f}, {orig_y:.3f}),  radius = {radius:.3f}")
print(f"    θ₀ = 1.0 rad/pos (one full rotation every ~6.3 positions)")
print()

print(f"  {'Pos':<6} {'dim 0':>8} {'dim 1':>8} {'angle':>10} {'radius':>8}")
print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
for pos in range(min(8, MAX_POS)):
    x, y = d0[pos], d1[pos]
    angle = math.atan2(y, x)
    r = math.sqrt(x**2 + y**2)
    print(f"  {pos:<6} {x:>8.3f} {y:>8.3f} {angle:>10.3f} rad {r:>7.3f}")

print()
print(f"  {GREEN}Notice:{RESET} radius stays CONSTANT ({radius:.3f}) — only the angle changes.")
print(f"  RoPE is a pure rotation. It doesn't change the vector's magnitude.")
print()

# ---------------------------------------------------------------------------
# Demo 2: Compare fast vs slow dimension pairs
# ---------------------------------------------------------------------------
print(f"{BOLD}2. FAST vs SLOW DIMENSION PAIRS{RESET}")
print(f"{'='*65}")
print()

# θ_i = 1 / 10000^(2i/dim)
freqs = 1.0 / (10000.0 ** (torch.arange(0, EMBED_DIM, 2).float() / EMBED_DIM))

pairs_to_show = [0, 4, 15, 31]  # fast to slow
for pair_idx in pairs_to_show:
    freq = freqs[pair_idx].item()
    period = 2 * math.pi / freq if freq > 0 else float('inf')
    d = pair_idx * 2
    print(f"  Dim pair {pair_idx:>2} (dims {d},{d+1}):  θ = {freq:.6f} rad/pos,  period = {period:.1f} positions")

print()
print(f"  {CYAN}Low-index pairs{RESET} rotate fast → capture FINE position differences.")
print(f"  {CYAN}High-index pairs{RESET} rotate slowly → capture COARSE/long-range position.")
print(f"  This is like Fourier analysis: multiple frequencies = full signal.")
print()

# ---------------------------------------------------------------------------
# Demo 3: Cosine similarity between positions (same content)
# ---------------------------------------------------------------------------
print(f"{BOLD}3. POSITION SIMILARITY (same content, different positions){RESET}")
print(f"{'='*65}")
print()

vec_norm = F.normalize(vec_rotated, dim=-1)
sim_matrix = (vec_norm @ vec_norm.T).numpy()

print(f"  Cosine similarity after RoPE (pos 0 vs others):")
for p in [0, 1, 2, 3, 5, 10, 15, 31]:
    bar_len = int(20 * (sim_matrix[0, p] + 1) / 2)  # scale [-1,1] to [0,20]
    bar = "█" * bar_len + "░" * (20 - bar_len)
    print(f"    pos 0 vs pos {p:>2}: {sim_matrix[0, p]:+.4f}  {bar}")

print()
print(f"  {GREEN}Key:{RESET} Similarity decays with distance — nearby = similar, far = different.")
print(f"  This happens even though the CONTENT is identical at every position.")
print(f"  RoPE makes position visible through rotation, not addition.")
print()

# ---------------------------------------------------------------------------
# Demo 4: Why rotation preserves dot-product relativity
# ---------------------------------------------------------------------------
print(f"{BOLD}4. WHY ROTATION → RELATIVE POSITION{RESET}")
print(f"{'='*65}")
print()

print(f"  The math (for one dimension pair):")
print()
print(f"    q at position m:  q_m = R(m·θ) × q     (rotate by m·θ)")
print(f"    k at position n:  k_n = R(n·θ) × k     (rotate by n·θ)")
print()
print(f"    Dot product:  q_m · k_n = [R(m·θ)×q] · [R(n·θ)×k]")
print(f"                            = q · R((m-n)·θ) × k")
print(f"                              ↑               ↑")
print(f"                           content      relative position")
print()
print(f"  {CYAN}The rotation factors out into content × relative position.{RESET}")
print(f"  This is because R(a)·R(b) = R(a+b) — rotations compose additively.")
print()

# Prove it numerically
torch.manual_seed(7)
q = torch.randn(1, 1, EMBED_DIM)
k = torch.randn(1, 1, EMBED_DIM)

# Put q at position 5, k at position 3 → distance 2
q_at_5 = apply_rope(q.expand(1, 6, -1), freqs_cis)[0, 5]
k_at_3 = apply_rope(k.expand(1, 4, -1), freqs_cis)[0, 3]
dot_5_3 = (q_at_5 * k_at_3).sum().item()

# Put q at position 10, k at position 8 → same distance 2
q_at_10 = apply_rope(q.expand(1, 11, -1), freqs_cis)[0, 10]
k_at_8 = apply_rope(k.expand(1, 9, -1), freqs_cis)[0, 8]
dot_10_8 = (q_at_10 * k_at_8).sum().item()

# Put q at position 20, k at position 18 → same distance 2
q_at_20 = apply_rope(q.expand(1, 21, -1), freqs_cis)[0, 20]
k_at_18 = apply_rope(k.expand(1, 19, -1), freqs_cis)[0, 18]
dot_20_18 = (q_at_20 * k_at_18).sum().item()

print(f"  Proof: same q, same k, same relative distance, DIFFERENT absolute positions:")
print(f"    q@5  · k@3  (dist=2): {dot_5_3:+.6f}")
print(f"    q@10 · k@8  (dist=2): {dot_10_8:+.6f}")
print(f"    q@20 · k@18 (dist=2): {dot_20_18:+.6f}")

if abs(dot_5_3 - dot_10_8) < 1e-4 and abs(dot_10_8 - dot_20_18) < 1e-4:
    print(f"    {GREEN}IDENTICAL — dot product depends only on relative distance!{RESET}")
else:
    print(f"    {RED}Small numerical differences: {abs(dot_5_3 - dot_10_8):.8f}{RESET}")
print()

# Different distance for comparison
q_at_5_k_at_2 = (apply_rope(q.expand(1, 6, -1), freqs_cis)[0, 5] *
                  apply_rope(k.expand(1, 3, -1), freqs_cis)[0, 2]).sum().item()
print(f"    q@5  · k@2  (dist=3): {q_at_5_k_at_2:+.6f}  ← different distance = different dot product")
print()

# ===========================================================================
# Visualization: 6-panel figure
# ===========================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("RoPE: Position as Rotation", fontsize=16, fontweight='bold')

colors = plt.cm.viridis(np.linspace(0, 1, MAX_POS))

# Panel 1: Dim pair 0 (fastest rotation) — 2D trajectory
ax = axes[0, 0]
ax.set_aspect('equal')
# Draw circle
theta_circle = np.linspace(0, 2 * np.pi, 100)
ax.plot(radius * np.cos(theta_circle), radius * np.sin(theta_circle),
        '--', color='gray', alpha=0.3, linewidth=1)
# Plot rotated points
for pos in range(MAX_POS):
    ax.plot(d0[pos], d1[pos], 'o', color=colors[pos], markersize=6, zorder=3)
    if pos < 10 or pos % 5 == 0:
        ax.annotate(str(pos), (d0[pos], d1[pos]), fontsize=6, ha='center', va='bottom',
                    xytext=(0, 4), textcoords='offset points')
# Connect consecutive positions
ax.plot(d0, d1, '-', color='gray', alpha=0.3, linewidth=0.5)
ax.plot(d0[0], d1[0], 'r*', markersize=12, zorder=5, label='pos 0')
ax.set_title(f"Dim pair 0: θ=1.0 rad/pos\n(fastest rotation)")
ax.set_xlabel("Dim 0")
ax.set_ylabel("Dim 1")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# Panel 2: Dim pair 4 (medium rotation)
ax = axes[0, 1]
ax.set_aspect('equal')
d_idx = 4
dx = vec_rotated[:, d_idx*2].numpy()
dy = vec_rotated[:, d_idx*2+1].numpy()
r4 = math.sqrt(dx[0]**2 + dy[0]**2)
ax.plot(r4 * np.cos(theta_circle), r4 * np.sin(theta_circle),
        '--', color='gray', alpha=0.3, linewidth=1)
for pos in range(MAX_POS):
    ax.plot(dx[pos], dy[pos], 'o', color=colors[pos], markersize=6, zorder=3)
    if pos < 10 or pos % 5 == 0:
        ax.annotate(str(pos), (dx[pos], dy[pos]), fontsize=6, ha='center', va='bottom',
                    xytext=(0, 4), textcoords='offset points')
ax.plot(dx, dy, '-', color='gray', alpha=0.3, linewidth=0.5)
ax.plot(dx[0], dy[0], 'r*', markersize=12, zorder=5, label='pos 0')
freq4 = freqs[d_idx].item()
ax.set_title(f"Dim pair {d_idx}: θ={freq4:.3f} rad/pos\n(medium rotation)")
ax.set_xlabel(f"Dim {d_idx*2}")
ax.set_ylabel(f"Dim {d_idx*2+1}")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# Panel 3: Dim pair 15 (slow rotation)
ax = axes[0, 2]
ax.set_aspect('equal')
d_idx = 15
dx = vec_rotated[:, d_idx*2].numpy()
dy = vec_rotated[:, d_idx*2+1].numpy()
r15 = math.sqrt(dx[0]**2 + dy[0]**2)
ax.plot(r15 * np.cos(theta_circle), r15 * np.sin(theta_circle),
        '--', color='gray', alpha=0.3, linewidth=1)
for pos in range(MAX_POS):
    ax.plot(dx[pos], dy[pos], 'o', color=colors[pos], markersize=6, zorder=3)
    if pos < 6 or pos % 10 == 0:
        ax.annotate(str(pos), (dx[pos], dy[pos]), fontsize=6, ha='center', va='bottom',
                    xytext=(0, 4), textcoords='offset points')
ax.plot(dx, dy, '-', color='gray', alpha=0.3, linewidth=0.5)
ax.plot(dx[0], dy[0], 'r*', markersize=12, zorder=5, label='pos 0')
freq15 = freqs[d_idx].item()
ax.set_title(f"Dim pair {d_idx}: θ={freq15:.5f} rad/pos\n(slow rotation)")
ax.set_xlabel(f"Dim {d_idx*2}")
ax.set_ylabel(f"Dim {d_idx*2+1}")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# Panel 4: Rotation angles per position (all dim pairs)
ax = axes[1, 0]
for pair_idx in [0, 2, 4, 8, 15, 31]:
    angles_d = (freqs_cis[:MAX_POS].angle()[:, pair_idx].numpy()) % (2 * np.pi)
    lw = 2.5 if pair_idx in [0, 15, 31] else 1.0
    ax.plot(angles_d, label=f'pair {pair_idx} (θ={freqs[pair_idx]:.4f})', linewidth=lw)
ax.set_xlabel("Position")
ax.set_ylabel("Rotation Angle (rad, mod 2π)")
ax.set_title("Rotation Speed by Dimension Pair")
ax.legend(fontsize=7, loc='upper left')
ax.grid(True, alpha=0.3)

# Panel 5: Cosine similarity heatmap (same content, different positions)
ax = axes[1, 1]
im = ax.imshow(sim_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xlabel("Position")
ax.set_ylabel("Position")
ax.set_title("Cosine Similarity After RoPE\n(identical content at every position)")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Panel 6: Vector norm at each position (should be constant)
ax = axes[1, 2]
norms = vec_rotated.norm(dim=-1).numpy()
ax.plot(norms, 'o-', color='steelblue', markersize=4, linewidth=2)
ax.axhline(y=norms[0], color='red', linestyle='--', alpha=0.5, label=f'Expected: {norms[0]:.3f}')
norm_range = norms.max() - norms.min()
ax.set_ylim(norms[0] - max(0.1, norm_range * 5), norms[0] + max(0.1, norm_range * 5))
ax.set_xlabel("Position")
ax.set_ylabel("Vector Norm")
ax.set_title("Norm Preservation (rotation = isometry)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), 'rope_rotation.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved visualization → {save_path}")
print()

# ===========================================================================
# Summary
# ===========================================================================
print(f"""
{'='*65}
{BOLD}STEP 2 SUMMARY: RoPE ROTATION{RESET}
{'='*65}

{BOLD}What we visualized:{RESET}

  The SAME embedding vector placed at positions 0..{MAX_POS-1}.
  RoPE rotates each dimension pair (2i, 2i+1) independently.

{BOLD}Key observations:{RESET}

  1. {CYAN}Each dim pair traces a circle{RESET}
     The vector tip moves along a circle as position increases.
     The radius is CONSTANT — rotation preserves magnitude.

  2. {CYAN}Different dim pairs rotate at different speeds{RESET}
     Pair 0:  θ = 1.0 rad/pos      (fast — fine position)
     Pair 4:  θ = {freqs[4].item():.4f} rad/pos  (medium)
     Pair 15: θ = {freqs[15].item():.6f} rad/pos  (slow — coarse position)
     Pair 31: θ = {freqs[31].item():.6f} rad/pos  (very slow)

     Like a clock: second hand (fast) + minute hand (slow)
     = unique position for every time.

  3. {CYAN}Similarity decays with distance{RESET}
     Nearby positions → small rotation → high cosine similarity
     Far positions → large rotation → low similarity
     This gives the model a natural sense of "distance".

  4. {CYAN}Norm is perfectly preserved{RESET}
     Rotation is an isometry — it doesn't stretch or shrink.
     Position information is encoded purely in DIRECTION,
     not magnitude. This is crucial for training stability.

{BOLD}The key realization:{RESET}

  {GREEN}Absolute PE:{RESET}  position = something ADDED to the vector
    vec_at_pos = vec + PE[pos]
    This shifts the vector, changing both direction AND magnitude.

  {GREEN}RoPE:{RESET}  position = ROTATION of the vector
    vec_at_pos = Rotate(vec, pos × θ)
    This changes direction but preserves magnitude.
    And crucially: Rotate(q, mθ) · Rotate(k, nθ) depends on (m-n).

  RoPE encodes position in the GEOMETRY of the space itself,
  not as an additive signal that competes with content.
""")
