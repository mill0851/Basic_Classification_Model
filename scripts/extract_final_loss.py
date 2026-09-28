"""Summarize the training loss of every completed run under outputs/.

Each run directory written by scripts/baseline_model_training-eval.py contains a
best_model_state.pt holding both the model weights and the training history dict
built in src/models/ClassificationMLP.py. This script reads only the history,
prints a comparison table, and writes:

    outputs/final_losses.csv           machine-readable summary
    outputs/analysis/final_losses.png  rendered comparison table

Important caveat on what "final" means here. ClassificationMLP.train_classification
deep-copies the history *inside* the "validation improved" branch, after appending
that epoch's losses. So the stored lists are truncated at the best-validation
epoch: any epoch trained after the last improvement is absent from the checkpoint
(it survives only in the rendered training_loss.png). Every number below is
therefore taken at the best-validation epoch, not at the last epoch actually run.

That truncation also makes two values redundant, so neither is reported:
val_loss[-1] always equals best_cls_loss, and len(train_loss) always equals
stop_epoch + 1. Both are still checked, and a violation is warned about, since it
would mean a corrupt checkpoint or a change to the save logic.

Run from anywhere:  python scripts/extract_final_loss.py
"""

import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # render straight to file; never open a window
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
ANALYSIS_DIR = OUTPUTS_DIR / "analysis"
SUMMARY_CSV = OUTPUTS_DIR / "final_losses.csv"
SUMMARY_PNG = ANALYSIS_DIR / "final_losses.png"

COLUMNS = ["dataset", "best_epoch", "best_val_loss", "final_train_loss"]

# Ink and surface tokens. Numbers carry primary ink while headers and the
# footnote recede, so the data stays the loudest thing on the surface.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
RULE_HAIRLINE = "#e1e0d9"
RULE_BASELINE = "#c3c2b7"
BAR_FILL = "#2a78d6"            # sequential blue, step 450
BAR_TRACK = "#eaeae4"


def display_name(run_dir_name):
    """Turn 'gaussian_3d_overlap_results' into '3D Gaussian (Overlap)'.

    Derived rather than looked up so that runs added later are labelled without
    editing this script. Unrecognized names fall back to the bare stem so an
    unexpected directory still shows up in the table instead of raising.
    """
    stem = run_dir_name[:-len("_results")] if run_dir_name.endswith("_results") else run_dir_name
    parts = stem.split("_")
    if len(parts) != 3:
        return stem
    family, dimension, spread = parts
    return f"{dimension.upper()} {family.capitalize()} ({spread.capitalize()})"


def group_key(run_dir_name):
    """'gaussian_3d_overlap_results' -> 'gaussian_3d', used to rule off families."""
    stem = run_dir_name[:-len("_results")] if run_dir_name.endswith("_results") else run_dir_name
    return "_".join(stem.split("_")[:2])


def summarize(checkpoint_path):
    """Read one checkpoint and return its summary row, or None if unusable."""
    # weights_only=False is required, not incidental: the saved dict holds the
    # non-tensor 'history' entry, which the weights-only unpickler rejects.
    # src/models/ClassificationMLP.py:178 loads it the same way.
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    history = checkpoint["history"]

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    stop_epoch = history["stop_epoch"]
    best_val_loss = history["best_cls_loss"]

    run_name = display_name(checkpoint_path.parent.name)

    # The two invariants described in the module docstring. They should be
    # impossible to violate, so say something loudly rather than silently
    # reporting numbers that no longer mean what the header claims.
    if val_loss and val_loss[-1] != best_val_loss:
        print(f"warning: {run_name}: last val_loss {val_loss[-1]!r} != "
              f"best_cls_loss {best_val_loss!r}", file=sys.stderr)
    if len(train_loss) != stop_epoch + 1:
        print(f"warning: {run_name}: recorded {len(train_loss)} epochs but "
              f"stop_epoch is {stop_epoch}", file=sys.stderr)

    return {
        "dataset": run_name,
        # 1-indexed to line up with the training_loss.png x-axis, which the
        # training script plots as range(1, len(losses) + 1).
        "best_epoch": stop_epoch + 1,
        "best_val_loss": best_val_loss,
        "final_train_loss": train_loss[-1] if train_loss else None,
        "_group": group_key(checkpoint_path.parent.name),
    }


def format_loss(value):
    return "n/a" if value is None else f"{value:.6e}"


def render_table_png(rows, destination):
    """Draw the summary as a typeset table and save it to `destination`.

    The losses span five orders of magnitude, so the numerals alone don't convey
    relative scale. A log-scaled bar sits beside the best-val-loss column to make
    that ordering readable at a glance; the exact value stays in the text, so the
    bar is never the only way to read a number.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Lay out in inches so row height stays constant as runs are added.
    fig_width = 11.0
    row_height = 0.36
    top_margin = 1.30                       # title, subtitle, column headers
    bottom_margin = 0.85                    # footnote
    fig_height = top_margin + len(rows) * row_height + bottom_margin

    figure = plt.figure(figsize=(fig_width, fig_height), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    axes = figure.add_axes([0, 0, 1, 1])
    axes.set_axis_off()
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)

    def y_of(inches_from_top):
        return 1.0 - inches_from_top / fig_height

    # Column anchors as a fraction of figure width.
    x_dataset = 0.035
    x_epoch = 0.375                          # right-aligned
    x_best_val = 0.585                       # right-aligned
    bar_x0, bar_x1 = 0.620, 0.775
    x_train = 0.965                          # right-aligned
    rule_x0, rule_x1 = 0.030, 0.970

    axes.text(x_dataset, y_of(0.44), "Training loss by dataset",
              fontsize=17, color=INK_PRIMARY, va="baseline")
    axes.text(x_dataset, y_of(0.72),
              "Baseline MLP (32 hidden units, 2 layers) - values at the "
              "best-validation epoch",
              fontsize=10, color=INK_SECONDARY, va="baseline")

    header_baseline = 1.12
    axes.text(x_dataset, y_of(header_baseline), "Dataset",
              fontsize=9.5, color=INK_SECONDARY, va="baseline")
    axes.text(x_epoch, y_of(header_baseline), "Best epoch",
              fontsize=9.5, color=INK_SECONDARY, va="baseline", ha="right")
    axes.text(x_best_val, y_of(header_baseline), "Best val loss",
              fontsize=9.5, color=INK_SECONDARY, va="baseline", ha="right")
    axes.text(bar_x1, y_of(header_baseline), "log scale",
              fontsize=9.5, color=INK_MUTED, va="baseline", ha="right")
    axes.text(x_train, y_of(header_baseline), "Final train loss",
              fontsize=9.5, color=INK_SECONDARY, va="baseline", ha="right")
    axes.plot([rule_x0, rule_x1], [y_of(1.26)] * 2,
              color=RULE_BASELINE, linewidth=1.0)

    # Map log10(loss) onto bar length. Guard the single-run and all-equal cases,
    # where the span is zero and the ratio would be undefined.
    losses = [row["best_val_loss"] for row in rows
              if row["best_val_loss"] is not None and row["best_val_loss"] > 0]
    log_lo = math.log10(min(losses)) if losses else 0.0
    log_span = (math.log10(max(losses)) - log_lo) if losses else 0.0

    bar_height = 0.10 / fig_height
    for index, row in enumerate(rows):
        baseline = top_margin + index * row_height

        # Hairline between dataset families, not between every row.
        if index > 0 and row["_group"] != rows[index - 1]["_group"]:
            axes.plot([rule_x0, rule_x1], [y_of(baseline - 0.09)] * 2,
                      color=RULE_HAIRLINE, linewidth=0.8)

        text_y = y_of(baseline + 0.19)
        axes.text(x_dataset, text_y, row["dataset"],
                  fontsize=11, color=INK_PRIMARY, va="baseline")
        for x_anchor, value in ((x_epoch, str(row["best_epoch"])),
                                (x_best_val, format_loss(row["best_val_loss"])),
                                (x_train, format_loss(row["final_train_loss"]))):
            # Monospace keeps the digits aligned vertically down each column.
            axes.text(x_anchor, text_y, value, fontsize=11, color=INK_PRIMARY,
                      va="baseline", ha="right", family="monospace")

        value = row["best_val_loss"]
        if value is None or value <= 0:
            continue
        fraction = (math.log10(value) - log_lo) / log_span if log_span > 0 else 1.0
        fraction = max(fraction, 0.015)      # keep the smallest loss visible
        bar_y = y_of(baseline + 0.23)
        axes.add_patch(FancyBboxPatch(
            (bar_x0, bar_y), bar_x1 - bar_x0, bar_height,
            boxstyle="round,pad=0,rounding_size=0.003",
            facecolor=BAR_TRACK, edgecolor="none"))
        axes.add_patch(FancyBboxPatch(
            (bar_x0, bar_y), (bar_x1 - bar_x0) * fraction, bar_height,
            boxstyle="round,pad=0,rounding_size=0.003",
            facecolor=BAR_FILL, edgecolor="none"))

    axes.plot([rule_x0, rule_x1], [y_of(fig_height - 0.62)] * 2,
              color=RULE_HAIRLINE, linewidth=0.8)
    axes.text(x_dataset, y_of(fig_height - 0.44),
              "Bar length is best val loss on a log scale - longer is worse. "
              "Values are taken at the best-validation epoch, not the last epoch run;",
              fontsize=8.5, color=INK_MUTED, va="baseline")
    axes.text(x_dataset, y_of(fig_height - 0.26),
              "epochs after the final improvement are not stored in the checkpoint.",
              fontsize=8.5, color=INK_MUTED, va="baseline")

    figure.savefig(destination, facecolor=SURFACE)
    plt.close(figure)


def main():
    checkpoints = sorted(OUTPUTS_DIR.glob("*_results/best_model_state.pt"))
    if not checkpoints:
        print(f"No checkpoints found matching {OUTPUTS_DIR}/*_results/best_model_state.pt",
              file=sys.stderr)
        return 1

    rows = []
    for checkpoint_path in checkpoints:
        try:
            rows.append(summarize(checkpoint_path))
        except Exception as error:
            # One bad checkpoint should not cost us the other eight.
            print(f"warning: skipping {checkpoint_path}: {error}", file=sys.stderr)

    if not rows:
        print("No checkpoints could be read.", file=sys.stderr)
        return 1

    name_width = max(len(row["dataset"]) for row in rows)
    name_width = max(name_width, len("dataset"))
    header = (f"{'dataset':<{name_width}}  {'best_epoch':>10}  "
              f"{'best_val_loss':>14}  {'final_train_loss':>16}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['dataset']:<{name_width}}  {row['best_epoch']:>10}  "
              f"{format_loss(row['best_val_loss']):>14}  "
              f"{format_loss(row['final_train_loss']):>16}")

    print()
    print("Note: values are taken at the best-validation epoch, not the last epoch run.")
    print("Epochs after the final improvement are not stored in the checkpoint.")

    # newline="" keeps csv from emitting blank rows between records on Windows.
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row[key] is None else row[key]) for key in COLUMNS})

    render_table_png(rows, SUMMARY_PNG)

    print(f"\nWrote {len(rows)} rows to {SUMMARY_CSV}")
    print(f"Wrote table image to {SUMMARY_PNG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
