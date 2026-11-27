"""Render Task05 figures from completed runs and fixed-split evaluation summaries."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter

OUT = Path(__file__).resolve().parent
DATA = OUT.parents[2] / "experiments/general-opd/toolace"
COLORS = ["#6c8091", "#b9662f", "#00796c", "#435491"]
MODELS = ["Base 4B", "SFT 4B", "OPD 4B", "Teacher 27B"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#193249",
        "text.color": "#193249",
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
    }
)


def load(path):
    return json.loads((DATA / path).read_text())


def training():
    runs = [load(f"runs/{name}_summary.json") for name in ("opd", "sft")]
    if any(
        not r["weights_path"] or len(r["history"]) != r["config"]["steps"] for r in runs
    ):
        raise ValueError("training incomplete")
    opd, sft = (r["history"] for r in runs)
    x = [r["step"] for r in opd]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), layout="constrained")
    panels = [
        (opd, "reverse_kl_mean", "Reverse KL per sampled token", "Nats"),
        (opd, "trainer/loss_mean", "OPD trainer loss", "Service mean"),
        (opd, "completion_tokens_mean", "Student response length", "Tokens / response"),
        (opd, "step_seconds", "OPD step duration", "Seconds"),
        (opd, "tokens_per_second", "OPD throughput", "Completion tokens / second"),
        (sft, "loss", "SFT cross entropy", "Nats / supervised token"),
    ]
    for ax, (series, key, title, unit) in zip(axes.flat, panels):
        ax.plot(
            [r["step"] for r in series],
            [r[key] for r in series],
            color=COLORS[1 if series is sft else 2],
            lw=1.8,
            marker=".",
            ms=4,
            label="Mean",
        )
        if key == "reverse_kl_mean":
            ax.plot(
                x,
                [r["reverse_kl_std"] for r in opd],
                color=COLORS[0],
                lw=1.3,
                ls="--",
                label="Token SD",
            )
            ax.legend(frameon=False)
        ax.set(title=title, xlabel="Update", ylabel=unit)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle(
        "ToolACE   |   Training over 25 updates", fontsize=16, fontweight="bold"
    )
    fig.savefig(OUT / "training.png", dpi=165)
    plt.close(fig)


def evaluation():
    models = ["base", "sft", "opd", "teacher"]
    tool = [load(f"eval-results/{m}.json") for m in models]
    gsm = [load(f"eval-results/gsm-{m}.json") for m in models[:3]]
    for series, n in ((tool, 60), (gsm, 30)):
        if any(not r["completed"] or r["n"] != n for r in series):
            raise ValueError("evaluation incomplete")
        if (
            len(
                {
                    (
                        r["config"]["data_hash"],
                        r["config"]["temperature"],
                        r["config"]["max_tokens"],
                    )
                    for r in series
                }
            )
            != 1
        ):
            raise ValueError("inconsistent evaluation settings")
    fig, (ax, err, math) = plt.subplots(
        1,
        3,
        figsize=(13, 4.4),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.35, 1.22, 0.85]},
    )
    keys = ["format_rate", "tool_name_acc", "exact_ast_match"]
    pos = np.arange(len(keys))
    for i, r in enumerate(tool):
        bars = ax.bar(
            pos + (i - 1.5) * 0.20,
            [r[k] for k in keys],
            0.18,
            color=COLORS[i],
            label=MODELS[i],
        )
        ax.bar_label(
            bars, labels=[str(round(r[k] * 60)) for k in keys], fontsize=7.5, padding=1
        )
    ax.set_xticks(pos, ["Format", "Tool name", "Exact AST"])
    ax.set(
        title="ToolACE  ·  60 prompts", ylabel="Share of all prompts", ylim=(0, 1.12)
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        fontsize=8,
    )
    left = np.zeros(4)
    for key, label, color in (
        ("format_error", "Format", "#8ba6bd"),
        ("wrong_tool", "Tool", "#566a9c"),
        ("wrong_args_key", "Argument keys", "#d09053"),
        ("wrong_args_value", "Argument values", "#b85841"),
        ("truncated", "Truncated", "#7e6b85"),
    ):
        values = np.array([r["error_counts"].get(key, 0) for r in tool])
        err.barh(MODELS, values, left=left, color=color, label=label)
        left += values
    err.set(title="ToolACE errors", xlabel="Failed prompts")
    err.invert_yaxis()
    err.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        fontsize=8,
    )
    err.xaxis.set_major_locator(MaxNLocator(integer=True))
    bars = math.bar(range(3), [r["correct"] for r in gsm], color=COLORS[:3], width=0.65)
    math.bar_label(
        bars, labels=[f"{round(r['correct'] * 30)}/30" for r in gsm], padding=2
    )
    math.set_xticks(range(3), ["Base", "SFT", "OPD"])
    math.set(title="GSM8K-mini  ·  30 prompts", ylabel="Accuracy", ylim=(0, 1.12))
    math.yaxis.set_major_formatter(PercentFormatter(1))
    fig.savefig(OUT / "evaluation.png", dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    training()
    evaluation()
