#!/usr/bin/env python3
"""plot_vs_lmstudio.py — render the MLX-serve vs LM Studio comparison chart
from a CSV produced by tests/bench_vs_lmstudio.sh.

Three panels per chart:
  - "Echo (verbatim recitation)"       → echo prompt — PLD's home turf
  - "Code completion (drafter's turf)" → code prompt — fresh code, no n-gram lookup
  - "Free-form writing (parity)"       → decode (creative essay) prompt

Bar layout depends on --family:
  gemma   → 5 bars: LM Studio MLX | LM Studio GGUF | MLX-serve --no-pld | --pld | --drafter
  qwen36  → 4 bars: LM Studio MLX (baseline) | LM Studio GGUF | MLX-serve --no-pld | --pld

Usage:
  python3 tests/plot_vs_lmstudio.py <csv> <png_out> --family <gemma|qwen36>

Requires matplotlib; install with `pip3 install --user matplotlib`
(or `--break-system-packages` on PEP-668 systems).
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Family-specific layout. Each entry:
#   (variant_filter, label, color)
# variant_filter is matched against the second '/' segment in the CSV `label`
# column (e.g. "lmstudio-baseline", "lmstudio-alt", "mlx-serve") AND the spec
# (third segment: "none", "pld", "drafter").
FAMILIES = {
    "gemma": {
        "title": "MLX-serve vs LM Studio — Gemma 4 (Apple Silicon, decode tok/s)",
        "x_label": lambda key: {
            "gemma4-e2b-4bit":          "E2B (4bit)",
            "gemma4-e4b-4bit":          "E4B (4bit)",
            "gemma4-31b-4bit":          "31B (4bit)",
            "gemma4-26b-a4b-moe-4bit":  "26B-A4B-MoE (4bit)",
        }.get(key, key),
        "model_order": [
            "gemma4-e2b-4bit",
            "gemma4-e4b-4bit",
            "gemma4-31b-4bit",
            "gemma4-26b-a4b-moe-4bit",
        ],
        "variants": [
            ("lmstudio-baseline", "none",    "LM Studio (MLX, baseline)", "#888888", True),
            ("lmstudio-alt",      "none",    "LM Studio (GGUF)",          "#cccccc", False),
            ("mlx-serve",         "none",    "MLX-serve --no-pld",        "#3b82f6", False),
            ("mlx-serve",         "pld",     "MLX-serve --pld",           "#22c55e", False),
            ("mlx-serve",         "drafter", "MLX-serve --drafter",       "#f97316", False),
        ],
    },
    "qwen36": {
        "title": "MLX-serve vs LM Studio — Qwen 3.6 (Apple Silicon, decode tok/s)",
        "x_label": lambda key: {
            "qwen36-27b":      "27B (4bit)",
            "qwen36-35b-a3b":  "35B-A3B (4bit)",
        }.get(key, key),
        "model_order": [
            "qwen36-27b",
            "qwen36-35b-a3b",
        ],
        "variants": [
            ("lmstudio-baseline", "none", "LM Studio (MLX, baseline)", "#888888", True),
            ("lmstudio-alt",      "none", "LM Studio (GGUF)",          "#cccccc", False),
            ("mlx-serve",         "none", "MLX-serve --no-pld",        "#3b82f6", False),
            ("mlx-serve",         "pld",  "MLX-serve --pld",           "#22c55e", False),
        ],
    },
}


def load_csv(path: Path) -> dict:
    """Returns {(model_logical, variant, spec): {prefill,decode,echo}}."""
    data: dict = defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 9 or parts[0] in ("label", ""):
                continue
            label, _engine, _model, _spec, prompt, pf, dc, _pt, _ct, *_ = parts
            bits = label.split("/")
            if len(bits) < 3:
                continue
            logical, variant, spec = bits[0], bits[1], bits[2]
            try:
                pf_v = float(pf or 0)
                dc_v = float(dc or 0)
            except ValueError:
                continue
            key = (logical, variant, spec)
            if prompt == "prefill":
                data[key]["prefill"] = pf_v
            elif prompt == "decode":
                data[key]["decode"] = dc_v
            elif prompt == "echo":
                data[key]["echo"] = dc_v
            elif prompt == "code":
                data[key]["code"] = dc_v
    return data


def render(csv_path: Path, png_out: Path, family: str) -> None:
    if family not in FAMILIES:
        sys.exit(f"Unknown family '{family}'; pick one of: {', '.join(FAMILIES)}")
    cfg = FAMILIES[family]
    data = load_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle(cfg["title"], fontsize=14, fontweight="bold")

    x = np.arange(len(cfg["model_order"]))
    n_variants = len(cfg["variants"])
    width = 0.8 / n_variants

    panels = [
        ("echo",   "Echo (verbatim recitation — PLD's home turf)"),
        ("code",   "Code completion (drafter's intended turf)"),
        ("decode", "Free-form writing (creative essay, parity case)"),
    ]

    for panel_idx, (workload_key, workload_label) in enumerate(panels):
        ax = axes[panel_idx]
        for v_idx, (variant, spec, label, color, is_baseline) in enumerate(cfg["variants"]):
            values, baselines = [], []
            for logical in cfg["model_order"]:
                cell = data.get((logical, variant, spec), {})
                # Find the row marked as baseline within the family, look up
                # its value for this prompt as the comparison point.
                base_variant_spec = next(
                    ((var, sp) for (var, sp, _l, _c, base) in cfg["variants"] if base),
                    None,
                )
                if base_variant_spec:
                    base_cell = data.get((logical,) + base_variant_spec, {})
                else:
                    base_cell = {}
                values.append(cell.get(workload_key, 0))
                baselines.append(base_cell.get(workload_key, 0))
            offset = (v_idx - (n_variants - 1) / 2) * width
            bars = ax.bar(
                x + offset, values, width,
                label=label, color=color, edgecolor="black", linewidth=0.6,
            )
            for bar, val, base in zip(bars, values, baselines):
                if val <= 0:
                    continue
                if is_baseline:
                    txt = f"{val:.1f}"
                    color_txt = "#222"
                else:
                    gain = (val / base - 1) * 100 if base > 0 else 0
                    color_txt = (
                        "#15803d" if gain >= 5 else
                        "#b91c1c" if gain <= -5 else
                        "#525252"
                    )
                    txt = f"{val:.1f}\n{gain:+.0f}%"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    txt,
                    ha="center", va="bottom", fontsize=8, color=color_txt,
                )
        ax.set_xticks(x)
        ax.set_xticklabels([cfg["x_label"](m) for m in cfg["model_order"]], fontsize=9)
        ax.set_ylabel("decode tok/s", fontsize=10)
        ax.set_title(workload_label, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        if panel_idx == 0:
            ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Headroom for the +X% labels above tall bars.
    for ax in axes:
        cur_ylim = ax.get_ylim()
        ax.set_ylim(0, cur_ylim[1] * 1.18)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(png_out, dpi=140, bbox_inches="tight")
    print(f"Wrote {png_out}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render MLX-serve vs LM Studio comparison chart from a "
                    "CSV produced by tests/bench_vs_lmstudio.sh.",
    )
    p.add_argument("csv", type=Path, help="input CSV path")
    p.add_argument("png", type=Path, help="output PNG path")
    p.add_argument("--family", required=True, choices=list(FAMILIES.keys()),
                   help="model family (matches --family of bench_vs_lmstudio.sh)")
    args = p.parse_args()

    if not args.csv.exists():
        sys.exit(f"CSV not found: {args.csv}")
    render(args.csv, args.png, args.family)


if __name__ == "__main__":
    main()
