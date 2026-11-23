"""Rebuild the two Task 03 figures with the repository's Python environment."""

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np
from swanlab.proto.swanlab.record.v1.record_pb2 import Record
from swanlab.sdk.internal.core_python.store import DataStoreReader


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUN_ID = "5fgjgxfb"
METRICS = (
    "reward", "format_rate", "degenerate_fraction", "train_datums",
    "rollout/completion_tokens_mean", "trainer/loss_mean",
)
INK, TEAL, OCHRE, GREY = "#193249", "#00796C", "#B76422", "#66758A"


def configure():
    families = {font.name for font in font_manager.fontManager.ttflist}
    for path in ("/mnt/c/Windows/Fonts/Deng.ttf", "/mnt/c/Windows/Fonts/Dengb.ttf",
                 "/mnt/c/Windows/Fonts/msyh.ttc", "/mnt/c/Windows/Fonts/msyhbd.ttc"):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            families.add(font_manager.FontProperties(fname=path).get_name())
    cjk = [f for f in ("Noto Sans CJK SC", "DengXian", "Microsoft YaHei", "PingFang SC") if f in families]
    mpl.rcParams.update({
        "font.family": cjk + ["DejaVu Sans"], "font.size": 11, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": GREY, "ytick.color": GREY,
        "axes.edgecolor": "#BAC4CD", "axes.spines.top": False, "axes.spines.right": False,
        "axes.unicode_minus": False, "figure.facecolor": "white", "savefig.dpi": 165,
    })


def read_training():
    source, = (ROOT / "swanlog").glob(f"run-*-{RUN_ID}/run-{RUN_ID}.swanlab")
    values = {key: {} for key in METRICS}
    reader = DataStoreReader()
    reader.open(source)
    try:
        while (data := reader.scan()) is not None:
            record = Record.FromString(data)
            scalar = record.scalar
            if record.HasField("scalar") and scalar.key in values:
                step = scalar.step + 1
                values[scalar.key][step] = scalar.value.number
    finally:
        reader.close()
    steps = np.arange(1, 101)
    for key in METRICS:
        expected = set(steps) if key != "trainer/loss_mean" else {
            s for s, n in values["train_datums"].items() if n > 0
        }
        if set(values[key]) != expected:
            raise ValueError(f"Unexpected step coverage for {key}")
    return steps, {k: np.array([v.get(s, np.nan) for s in steps]) for k, v in values.items()}


def training():
    steps, series = read_training()
    fig = plt.figure(figsize=(10.4, 7.8), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=[1.15, 1, 1])
    top = fig.add_subplot(grid[0, :])
    top.set_title("平均奖励与格式合规率", loc="left")
    for key, label, color in (("reward", "平均奖励", TEAL), ("format_rate", "格式合规率", OCHRE)):
        values = series[key]
        top.plot(steps, values, color=color, lw=1, alpha=.35)
        moving = np.convolve(values, np.ones(5) / 5, mode="valid")
        top.plot(steps[4:], moving, color=color, lw=2.2, label=label)
    top.set_ylim(0, 1.06)
    top.set_yticks([0, .25, .5, .75, 1])
    top.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    top.legend(loc="lower right", ncols=2, frameon=False, title="粗线为最近 5 步均值")

    panels = (
        ("degenerate_fraction", "组内奖励全相同的题目占比", TEAL),
        ("train_datums", "参与更新的回答数", OCHRE),
        ("rollout/completion_tokens_mean", "平均回答长度（token）", TEAL),
        ("trainer/loss_mean", "loss_mean（有更新的步骤）", GREY),
    )
    axes = [top]
    for i, (key, title, color) in enumerate(panels):
        ax = fig.add_subplot(grid[1 + i // 2, i % 2])
        axes.append(ax)
        values = series[key]
        ax.set_title(title, loc="left")
        if key == "train_datums":
            ax.bar(steps, values, width=.85, color=color)
            ax.yaxis.set_major_locator(MultipleLocator(8))
        else:
            ax.plot(steps, values, color=color, lw=1.15, marker="o", ms=2)
        if key == "degenerate_fraction":
            ax.set_ylim(0, 1.07)
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        if key == "rollout/completion_tokens_mean":
            ax.set_ylim(bottom=0)
        if key == "trainer/loss_mean":
            ax.axhline(0, color="#9BA6B0", lw=.8)
    for ax in axes:
        ax.grid(axis="y", color="#E0E6E8", lw=.75)
        ax.set_axisbelow(True)
        ax.set_xlim(.2, 100.8)
        ax.set_xticks([1, 20, 40, 60, 80, 100])
        ax.tick_params(length=0)
    for ax in axes[-2:]:
        ax.set_xlabel("训练步")
    fig.suptitle("Vision GRPO 训练记录", fontsize=17, fontweight="bold")
    fig.savefig(OUT / "training.png")
    plt.close(fig)


def evaluation():
    records = []
    for model in ("base", "step-100"):
        path = ROOT / "experiments" / "vision-grpo" / f"eval-results-{model}.json"
        records.append(json.loads(path.read_text(encoding="utf-8")))
    base, trained = records
    identities = [{(row["id"], row["gold"]) for row in r["samples"]} for r in records]
    if identities[0] != identities[1] or len(identities[0]) != base["eval_size"] or base["eval_size"] != trained["eval_size"]:
        raise ValueError("Evaluation files do not cover the same questions")

    fig, ax = plt.subplots(figsize=(8.6, 3.1), layout="constrained")
    for i, (key, label) in enumerate((("format_rate", "格式合规率"), ("accuracy", "准确率"))):
        values = [r["metrics"][key] for r in records]
        ax.plot(values, [i, i], color="#ABBAC2", lw=2.5, zorder=2)
        for x, marker, color, name in zip(values, ("o", "D"), (GREY, TEAL), ("基座", "第 100 步"), strict=True):
            ax.scatter(x, i, s=86, color=color, marker=marker, label=name if i == 0 else None, zorder=3)
            ax.annotate(f"{x:.0%}", (x, i), xytext=(0, 11), textcoords="offset points",
                        color=color, ha="center", fontweight="bold")
    ax.set_yticks([0, 1], ["格式合规率", "准确率"])
    ax.set_ylim(-.58, 1.58)
    lo, hi = ax.get_xlim()
    ax.set_xlim(np.floor(lo * 10) / 10, np.ceil(hi * 10) / 10)
    ax.xaxis.set_major_locator(MultipleLocator(.05))
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="x", color="#DFE5E9")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(loc="lower center", ncols=2, frameon=False)
    ax.set_title(f"GeoQA 固定 {base['eval_size']} 题评测", loc="left", fontsize=16, fontweight="bold")
    fig.savefig(OUT / "evaluation.png")
    plt.close(fig)


if __name__ == "__main__":
    configure()
    training()
    evaluation()
