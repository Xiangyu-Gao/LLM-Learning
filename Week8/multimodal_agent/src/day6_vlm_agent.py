"""
Day 6: VLM Tool Agent
=====================
Extends the Week 7 tool-use agent with vision (image) inputs.

Architecture Recap
------------------
Claude's multimodal API uses **token concatenation**:

    Image → vision encoder → visual tokens (e.g. 1568 tokens for a 1024px img)
    Text  → BPE tokenizer  → text tokens
    [visual tokens] ++ [text tokens] → single transformer decoder → response

Alternative — **latent fusion** (used by LLaVA, InstructBLIP):

    Image → CLIP encoder → image embedding (e.g. 768-dim vector)
    Learned linear projector: R^768 → R^4096 (LLM embedding space)
    Projected embedding PREPENDED to text token embeddings
    → More token-efficient but requires projector training

Why is tool calling harder with VLM?
  1. Partial visual grounding: some of the answer comes from pixels, some from
     tools — the agent must decide where the boundary is.
  2. OCR uncertainty: misread numbers → wrong calculator inputs.
  3. Spatial ambiguity: dimensions labeled non-obviously.
  4. Context splitting: image tokens fill context early, leaving less room for
     multi-turn tool results in long tasks.

Experiments
-----------
  A  Text-only baseline (no image) — establishes tool-use accuracy floor.
  B  Image sufficient — agent should NOT call any tool.
  C  Image + calculator — visual math, answer requires computation.
  D  Image + search — image contains a concept needing factual lookup.
  E  Mixed tool-selection challenge — measures overall VLM tool accuracy.

Test images are generated with matplotlib and saved to results/images/.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from utils import (
    MODEL,
    TOOL_SCHEMAS,
    build_user_message,
    dispatch_tool,
    get_client,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"
IMAGES_DIR = RESULTS_DIR / "images"


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    name: str
    inputs: dict
    result: str


@dataclass
class VLMResult:
    task_id: str
    question: str
    image_path: str | None
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    steps: int = 0
    expected_tool: str | None = None   # None means "no tool should be used"
    correct_tool_choice: bool = False
    image_analysis_snippet: str = ""   # what the model said about the image
    elapsed_sec: float = 0.0


# ─── Test Image Generation ────────────────────────────────────────────────────

def create_test_images(output_dir: Path) -> dict[str, Path]:
    """
    Generate a set of labeled test images using matplotlib.
    Each image is designed to test a different VLM+tool scenario.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images: dict[str, Path] = {}

    # ── 1. Labeled rectangle ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    rect = mpatches.Rectangle(
        (0.15, 0.25), 0.7, 0.45,
        linewidth=3, edgecolor="#1a1a80", facecolor="#c8d8f8",
    )
    ax.add_patch(rect)
    # width arrow + label
    ax.annotate("", xy=(0.85, 0.13), xytext=(0.15, 0.13),
                arrowprops=dict(arrowstyle="<->", color="black", lw=2))
    ax.text(0.5, 0.07, "width = 12 m", ha="center", fontsize=14, fontweight="bold")
    # height arrow + label
    ax.annotate("", xy=(0.92, 0.70), xytext=(0.92, 0.25),
                arrowprops=dict(arrowstyle="<->", color="black", lw=2))
    ax.text(0.97, 0.47, "height\n= 7 m", ha="left", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.47, "Rectangle", ha="center", va="center",
            fontsize=18, color="#1a1a80", fontweight="bold")
    ax.set_xlim(0, 1.15); ax.set_ylim(0, 1.0)
    ax.axis("off")
    ax.set_title("Figure 1 — Labeled Rectangle", fontsize=12, pad=8)
    path = output_dir / "rectangle.png"
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
    images["rectangle"] = path

    # ── 2. Labeled circle ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))
    circle = plt.Circle((0.5, 0.5), 0.35,
                         linewidth=3, edgecolor="#801a1a", facecolor="#f8c8c8")
    ax.add_patch(circle)
    ax.plot([0.5, 0.85], [0.5, 0.5], "k-", linewidth=2)
    ax.text(0.68, 0.55, "r = 5 m", ha="center", fontsize=14, fontweight="bold")
    ax.plot(0.5, 0.5, "ko", markersize=7)
    ax.set_xlim(0, 1.1); ax.set_ylim(0, 1.1)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Figure 2 — Circle with Radius Labeled", fontsize=12, pad=8)
    path = output_dir / "circle.png"
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
    images["circle"] = path

    # ── 3. Bar chart ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Product A", "Product B", "Product C", "Product D"]
    values = [45, 32, 67, 28]
    colors = ["#4472C4", "#ED7D31", "#A9D18E", "#FF6666"]
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_ylabel("Sales (units)", fontsize=12)
    ax.set_title("Monthly Product Sales — Q1 Report", fontsize=13)
    ax.set_ylim(0, 85)
    ax.grid(axis="y", alpha=0.3)
    path = output_dir / "barchart.png"
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
    images["barchart"] = path

    # ── 4. Physics formula reference card ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.set_facecolor("#f0f4f8")
    fig.patch.set_facecolor("#f0f4f8")
    ax.text(0.5, 0.82, "Special Relativity — Reference Card",
            ha="center", fontsize=12, fontweight="bold", color="#333333")
    ax.text(0.5, 0.55, r"$E = mc^2$",
            ha="center", fontsize=32, fontweight="bold", color="#1a1a80")
    ax.text(0.5, 0.30, "where   c = speed of light in vacuum",
            ha="center", fontsize=12, color="#444444")
    ax.text(0.5, 0.12, "First published by Albert Einstein, 1905",
            ha="center", fontsize=10, color="#666666", style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    path = output_dir / "formula_text.png"
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    images["formula_text"] = path

    # ── 5. Grocery price table ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axis("off")
    col_labels = ["Item", "Qty", "Unit Price ($)", "Subtotal ($)"]
    cell_data = [
        ["Apples",  "3", "1.50", "4.50"],
        ["Bananas", "5", "0.75", "3.75"],
        ["Oranges", "2", "2.00", "4.00"],
        ["Grapes",  "1", "3.50", "3.50"],
    ]
    tbl = ax.table(cellText=cell_data, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 1.8)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#4472C4")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Grocery Cart — Calculate Grand Total", fontsize=12, pad=16)
    path = output_dir / "price_table.png"
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
    images["price_table"] = path

    return images


# ─── VLM Tool Agent ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a multimodal assistant with vision and tool-use capabilities.

When an image is provided:
1. FIRST describe what you see (shapes, numbers, labels, text, formulas).
2. Extract all relevant data from the image before deciding on tools.
3. Use the calculator tool when numerical computation is needed.
4. Use the search tool when external factual information is needed.
5. Do NOT use tools when the answer is clearly readable from the image alone.

Always state what you read from the image before calling any tool."""


class VLMToolAgent:
    def __init__(self, client=None):
        self.client = client or get_client()

    def run(
        self,
        task_id: str,
        question: str,
        image_path: "Path | None",
        expected_tool: "str | None",
        max_steps: int = 8,
    ) -> VLMResult:
        start = time.time()
        messages = [build_user_message(question, image_path)]
        tool_calls: list[ToolCall] = []
        answer = "No answer produced."
        image_analysis_snippet = ""

        for step in range(max_steps):
            resp = self.client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )

            text_parts = [
                b.text for b in resp.content if hasattr(b, "text") and b.text
            ]
            text_content = " ".join(text_parts)

            if step == 0 and image_path and text_content:
                image_analysis_snippet = text_content[:200]

            if resp.stop_reason == "end_turn":
                answer = text_content
                break

            if resp.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": resp.content})
                tool_results = []
                for block in resp.content:
                    if block.type == "tool_use":
                        result = dispatch_tool(block.name, block.input)
                        tool_calls.append(
                            ToolCall(name=block.name, inputs=dict(block.input), result=result)
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                messages.append({"role": "user", "content": tool_results})
        else:
            answer = "Max steps reached without final answer."

        # Evaluate whether the agent made the correct tool-use decision
        tools_used = {tc.name for tc in tool_calls}
        if expected_tool is None:
            correct_choice = len(tool_calls) == 0
        else:
            correct_choice = expected_tool in tools_used

        return VLMResult(
            task_id=task_id,
            question=question,
            image_path=str(image_path) if image_path else None,
            answer=answer,
            tool_calls=tool_calls,
            steps=step + 1,
            expected_tool=expected_tool,
            correct_tool_choice=correct_choice,
            image_analysis_snippet=image_analysis_snippet,
            elapsed_sec=time.time() - start,
        )


# ─── Experiments ──────────────────────────────────────────────────────────────

def run_experiment_a(agent: VLMToolAgent) -> list[VLMResult]:
    """A — Text-only baseline. No image. Establishes tool accuracy floor."""
    print("\n── Experiment A: Text-Only Baseline ─────────────────────────")
    tasks = [
        ("A1", "What is 15 × 23?",                       None, "python_calculator"),
        ("A2", "What is the square root of 225?",         None, "python_calculator"),
        ("A3", "Who wrote the play 'Romeo and Juliet'?",  None, "wikipedia_search"),
        ("A4", "What is 2 to the power of 10?",           None, "python_calculator"),
    ]
    return _run_tasks(agent, tasks)


def run_experiment_b(agent: VLMToolAgent, images: dict) -> list[VLMResult]:
    """B — Image is self-sufficient. Agent should NOT call any tool."""
    print("\n── Experiment B: Image Sufficient — No Tool Needed ───────────")
    tasks = [
        ("B1",
         "Look carefully at this bar chart. Which product had the highest sales? "
         "Just read the chart — no tools needed.",
         images["barchart"], None),
        ("B2",
         "What geometric shape is shown? List all dimension labels you can read. "
         "No calculation needed — just describe what you see.",
         images["rectangle"], None),
        ("B3",
         "What mathematical formula is displayed in this image? "
         "Read and transcribe it exactly. No tools needed.",
         images["formula_text"], None),
    ]
    return _run_tasks(agent, tasks)


def run_experiment_c(agent: VLMToolAgent, images: dict) -> list[VLMResult]:
    """C — Visual math: image provides numbers, agent must calculate."""
    print("\n── Experiment C: Image + Calculator ─────────────────────────")
    tasks = [
        ("C1",
         "This image shows a labeled rectangle. "
         "Read the width and height from the image, then use the calculator to find the area (width × height).",
         images["rectangle"], "python_calculator"),
        ("C2",
         "This image shows a circle with a labeled radius r. "
         "Read r from the image, then calculate the area using A = π × r². Use the calculator.",
         images["circle"], "python_calculator"),
        ("C3",
         "This bar chart shows product sales figures. "
         "Read all four values, then use the calculator to find the total units sold.",
         images["barchart"], "python_calculator"),
        ("C4",
         "This image shows a grocery price table with subtotals. "
         "Read each subtotal, then use the calculator to find the grand total.",
         images["price_table"], "python_calculator"),
    ]
    return _run_tasks(agent, tasks)


def run_experiment_d(agent: VLMToolAgent, images: dict) -> list[VLMResult]:
    """D — Image + Search: image shows a concept needing external factual lookup."""
    print("\n── Experiment D: Image + Search ──────────────────────────────")
    tasks = [
        ("D1",
         "The image shows Einstein's famous equation E = mc². "
         "What is the precise value of c (speed of light) in metres per second? "
         "Search Wikipedia for the exact figure.",
         images["formula_text"], "wikipedia_search"),
        ("D2",
         "The image references Albert Einstein and 1905. "
         "What specific theory did Einstein publish in 1905 that introduced E = mc²? "
         "Search for the answer.",
         images["formula_text"], "wikipedia_search"),
    ]
    return _run_tasks(agent, tasks)


def run_experiment_e(agent: VLMToolAgent, images: dict) -> list[VLMResult]:
    """E — Mixed challenge: agent must decide tool need on its own."""
    print("\n── Experiment E: Mixed Tool-Selection Challenge ───────────────")
    tasks = [
        ("E1", "What is 888 ÷ 37? Give the exact decimal result.",
         None, "python_calculator"),
        ("E2",
         "Look at this rectangle. What is its perimeter (2×width + 2×height)?",
         images["rectangle"], "python_calculator"),
        ("E3",
         "What formula is shown in this image?",
         images["formula_text"], None),
        ("E4",
         "Which item in this table costs the most per unit?",
         images["price_table"], None),
        ("E5",
         "Which product had the lowest sales according to this chart?",
         images["barchart"], None),
        ("E6",
         "What is the value of pi to 10 decimal places?",
         None, "wikipedia_search"),
    ]
    return _run_tasks(agent, tasks)


def _run_tasks(agent: VLMToolAgent, tasks: list) -> list[VLMResult]:
    results = []
    for tid, question, image_path, expected in tasks:
        img_label = Path(image_path).name if image_path else "none"
        print(f"  [{tid}] {question[:65]}  [img={img_label}]")
        r = agent.run(tid, question, image_path, expected)
        tools_used = [tc.name for tc in r.tool_calls] or ["(none)"]
        status = "✓ correct" if r.correct_tool_choice else "✗ wrong"
        print(f"        tools={tools_used}  {status}")
        print(f"        answer: {r.answer[:110]}")
        results.append(r)
    return results


# ─── Reporting ────────────────────────────────────────────────────────────────

def compute_summary(results: list[VLMResult]) -> dict:
    n = len(results)
    return {
        "total_tasks": n,
        "tool_selection_accuracy": sum(r.correct_tool_choice for r in results) / n if n else 0,
        "avg_steps": sum(r.steps for r in results) / n if n else 0,
        "avg_tool_calls": sum(len(r.tool_calls) for r in results) / n if n else 0,
    }


def plot_results(all_results: list[VLMResult], out_dir: Path):
    """Bar charts: tool selection accuracy and avg tool calls, by experiment."""
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_ids = sorted({r.task_id[0] for r in all_results})  # A,B,C,D,E
    groups = {e: [r for r in all_results if r.task_id[0] == e] for e in exp_ids}

    accuracies = [sum(r.correct_tool_choice for r in groups[e]) / len(groups[e])
                  for e in exp_ids]
    avg_calls = [sum(len(r.tool_calls) for r in groups[e]) / len(groups[e])
                 for e in exp_ids]

    exp_labels = {
        "A": "A: Text-Only",
        "B": "B: No Tool\nNeeded",
        "C": "C: Calculator\nNeeded",
        "D": "D: Search\nNeeded",
        "E": "E: Mixed\nChallenge",
    }
    x_labels = [exp_labels.get(e, e) for e in exp_ids]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#2ecc71" if a >= 1.0 else ("#f39c12" if a >= 0.5 else "#e74c3c")
              for a in accuracies]
    axes[0].bar(x_labels, accuracies, color=colors, edgecolor="black", linewidth=0.7)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_ylabel("Tool Selection Accuracy")
    axes[0].set_title("Tool Choice Accuracy by Experiment")
    axes[0].axhline(1.0, color="green", linestyle="--", alpha=0.4, label="Perfect")
    for i, a in enumerate(accuracies):
        axes[0].text(i, a + 0.04, f"{a:.0%}", ha="center", fontweight="bold")

    axes[1].bar(x_labels, avg_calls, color="#3498db", edgecolor="black", linewidth=0.7)
    axes[1].set_ylabel("Avg Tool Calls per Task")
    axes[1].set_title("Average Tool Usage by Experiment")
    for i, v in enumerate(avg_calls):
        axes[1].text(i, v + 0.05, f"{v:.1f}", ha="center", fontweight="bold")

    plt.suptitle("Day 6: VLM Tool Agent — Experiment Summary", fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = out_dir / "day6_results.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  [plot] → {save_path}")


def print_failure_analysis(results: list[VLMResult]):
    failures = [r for r in results if not r.correct_tool_choice]
    if not failures:
        print("\n  No tool-selection failures — perfect score!")
        return
    print(f"\n── VLM Tool-Selection Failure Analysis ({len(failures)} failures) ──")
    for r in failures:
        used = [tc.name for tc in r.tool_calls] or ["(none)"]
        print(f"  [{r.task_id}] Expected: {r.expected_tool or 'no-tool'} | Got: {used}")
        print(f"    Q: {r.question[:80]}")
        if r.image_analysis_snippet:
            print(f"    Image read: {r.image_analysis_snippet[:100]}")
        print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(smoke: bool = False):
    print("=" * 60)
    print("Day 6: VLM Tool Agent")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[Setup] Generating test images...")
    images = create_test_images(IMAGES_DIR)
    print(f"  Created {len(images)} images in {IMAGES_DIR}")

    agent = VLMToolAgent()

    if smoke:
        print("\n[SMOKE MODE] Running 2 tasks only...")
        r1 = agent.run("A1", "What is 12 × 7?", None, "python_calculator")
        r2 = agent.run("C1",
                       "This image shows a labeled rectangle. "
                       "Read the width and height, then calculate the area (width × height) "
                       "using the calculator.",
                       images["rectangle"], "python_calculator")
        all_results = [r1, r2]
        for r in all_results:
            print(f"  [{r.task_id}] tools={[tc.name for tc in r.tool_calls]} "
                  f"correct={r.correct_tool_choice}")
            print(f"         ans: {r.answer[:120]}")
    else:
        all_results = (
            run_experiment_a(agent)
            + run_experiment_b(agent, images)
            + run_experiment_c(agent, images)
            + run_experiment_d(agent, images)
            + run_experiment_e(agent, images)
        )

    summary = compute_summary(all_results)
    print("\n── Summary ──────────────────────────────────────────────────")
    print(f"  Tasks:                  {summary['total_tasks']}")
    print(f"  Tool selection accuracy:{summary['tool_selection_accuracy']:.0%}")
    print(f"  Avg steps / task:       {summary['avg_steps']:.1f}")
    print(f"  Avg tool calls / task:  {summary['avg_tool_calls']:.1f}")

    if not smoke:
        print_failure_analysis(all_results)
        plot_results(all_results, RESULTS_DIR)

    out_path = RESULTS_DIR / "day6_results.json"
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, default=str)
    print(f"\n  Results → {out_path}")

    print("\n── Key VLM Takeaways ────────────────────────────────────────")
    print("  1. Token concatenation: image + text processed in one transformer pass.")
    print("  2. Agent must decide: is the answer in the pixels or in tools?")
    print("  3. Common failure: calling calculator with wrong number read from image.")
    print("  4. Best practice: instruct agent to verbalize what it sees FIRST.")
    print("  5. Visual context can both help (gives data) and hurt (distracts).")


if __name__ == "__main__":
    import sys
    main(smoke="--smoke" in sys.argv)
