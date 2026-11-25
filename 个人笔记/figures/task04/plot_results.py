"""Rebuild the Task 04 OPD training figure from the local SwanLab run log."""

import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from swanlab.proto.swanlab.record.v1.record_pb2 import Record
from swanlab.sdk.internal.core_python.store import DataStoreReader


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUN_ID = "mgpcfyoz"
METRICS = (
    "data/datums",
    "data/completion_tokens_mean",
    "data/completion_tokens_total",
    "data/completion_tokens_per_second",
    "opd/reverse_kl_mean",
    "opd/reverse_kl_std",
    "time/step_elapsed_time",
    "trainer/loss_mean",
    "trainer/loss_sum",
    "trainer/token_count",
)
INK, TEAL, OCHRE, GREY = "#193249", "#00796C", "#B76422", "#66758A"
TEAL_LIGHT, OCHRE_LIGHT = "#5BA89E", "#D99B66"


def configure():
    families = {font.name for font in font_manager.fontManager.ttflist}
    for path in (
        "/mnt/c/Windows/Fonts/Deng.ttf",
        "/mnt/c/Windows/Fonts/Dengb.ttf",
        "/mnt/c/Windows/Fonts/msyh.ttc",
        "/mnt/c/Windows/Fonts/msyhbd.ttc",
    ):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            families.add(font_manager.FontProperties(fname=path).get_name())
    cjk = [
        f
        for f in ("Noto Sans CJK SC", "DengXian", "Microsoft YaHei", "PingFang SC")
        if f in families
    ]
    mpl.rcParams.update(
        {
            "font.family": cjk + ["DejaVu Sans"],
            "font.size": 10.5,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": GREY,
            "ytick.color": GREY,
            "axes.edgecolor": "#BAC4CD",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "savefig.dpi": 165,
        }
    )


def read_training():
    source, = (ROOT / "experiments" / "general-opd" / "swanlog").glob(
        f"run-*-{RUN_ID}/run-{RUN_ID}.swanlab"
    )
    values = {key: {} for key in METRICS}
    rollout_secs = {}
    reader = DataStoreReader()
    reader.open(source)
    try:
        while (data := reader.scan()) is not None:
            record = Record.FromString(data)
            which = record.WhichOneof("record_type")
            if which == "scalar" and record.scalar.key in values:
                step = record.scalar.step + 1
                values[record.scalar.key][step] = record.scalar.value.number
            elif which == "log":
                m = re.search(r"OPD step (\d+):.*\[(\d+):(\d+)<", record.log.line)
                if m:
                    rollout_secs[int(m.group(1)) + 1] = int(m.group(2)) * 60 + int(m.group(3))
    finally:
        reader.close()

    steps = np.arange(1, 11)
    expected = set(steps.tolist())
    for key in METRICS:
        if set(values[key]) != expected:
            raise ValueError(f"Unexpected step coverage for {key}: {sorted(values[key])}")
    if set(rollout_secs) != expected:
        raise ValueError(f"Unexpected rollout timing coverage: {sorted(rollout_secs)}")

    series = {k: np.array([v[s] for s in steps], dtype=float) for k, v in values.items()}
    series["time/rollout_and_teacher"] = np.array([rollout_secs[s] for s in steps], dtype=float)
    series["time/train_and_sync"] = (
        series["time/step_elapsed_time"] - series["time/rollout_and_teacher"]
    )
    return steps, series


def training():
    steps, s = read_training()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), layout="constrained")
    ax_kl, ax_std, ax_tok, ax_time = axes.flat

    # Panel 1: reverse_kl_mean vs trainer/loss_mean
    rkl = s["opd/reverse_kl_mean"]
    loss = s["trainer/loss_mean"]
    corr = float(np.corrcoef(rkl, loss)[0, 1])
    ax_kl.set_title("Reverse KL 均值与 Trainer Loss 均值", loc="left", fontweight="bold")
    ax_kl.plot(steps, rkl, color=TEAL, lw=2.1, marker="o", ms=5, label="opd/reverse_kl_mean")
    ax_kl.plot(
        steps,
        loss,
        color=OCHRE,
        lw=1.9,
        ls="--",
        marker="s",
        ms=4.5,
        label="trainer/loss_mean",
    )
    for idx, off_rkl, off_loss in (
        (0, (0, 8), (0, -14)),
        (2, (12, 14), (0, -15)),
        (5, (0, 8), (0, -18)),
        (9, (0, 8), (0, -15)),
    ):
        ax_kl.annotate(
            f"{rkl[idx]:.3f}",
            (steps[idx], rkl[idx]),
            xytext=off_rkl,
            textcoords="offset points",
            ha="center",
            fontsize=8.8,
            color=TEAL,
        )
        ax_kl.annotate(
            f"{loss[idx]:.3f}",
            (steps[idx], loss[idx]),
            xytext=off_loss,
            textcoords="offset points",
            ha="center",
            fontsize=8.8,
            color=OCHRE,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85),
        )
    ax_kl.set_ylim(0.06, 0.195)
    ax_kl.legend(loc="upper right", frameon=False, fontsize=9.2)
    ax_kl.text(
        0.02,
        0.06,
        f"Pearson r = {corr:.3f}（loss 含 prompt 零占位，均值略低）",
        transform=ax_kl.transAxes,
        fontsize=8.8,
        color=GREY,
    )

    # Panel 2: reverse_kl_std
    std = s["opd/reverse_kl_std"]
    std_mean = float(np.mean(std))
    ax_std.set_title("Token 级 Reverse KL 标准差", loc="left", fontweight="bold")
    ax_std.plot(steps, std, color=INK, lw=2.0, marker="o", ms=5, label="opd/reverse_kl_std")
    ax_std.axhline(
        std_mean,
        color=TEAL,
        lw=1.2,
        ls=":",
        label=f"10 步均值 = {std_mean:.3f}",
    )
    for idx, dy in ((0, 8), (2, -15), (5, 8), (9, 8)):
        ax_std.annotate(
            f"{std[idx]:.3f}",
            (steps[idx], std[idx]),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            fontsize=8.8,
            color=INK,
        )
    ax_std.set_ylim(0.50, 0.82)
    ax_std.legend(loc="upper right", frameon=False, fontsize=9.2)
    ax_std.text(
        0.02,
        0.06,
        "std 约为均值的 4.9 倍，反映 token 级纠偏高度集中在关键分叉处",
        transform=ax_std.transAxes,
        fontsize=8.8,
        color=GREY,
    )

    # Panel 3: token composition per step
    comp = s["data/completion_tokens_total"]
    prompt_pad = s["trainer/token_count"] - comp
    ax_tok.set_title("单步 Token 构成（16 条轨迹，全部达 max_tokens=512）", loc="left", fontweight="bold")
    ax_tok.bar(steps, comp, width=0.62, color=TEAL_LIGHT, label="Completion（有效训练，恒定 8,192）")
    ax_tok.bar(
        steps,
        prompt_pad,
        bottom=comp,
        width=0.62,
        color=OCHRE_LIGHT,
        label="Prompt 零占位（1,016–2,264）",
    )
    for x, total in zip(steps, s["trainer/token_count"], strict=True):
        ax_tok.annotate(
            f"{int(total):,}",
            (x, total),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8.2,
            color=INK,
        )
    ax_tok.set_ylim(0, 13600)
    ax_tok.set_ylabel("Token 数")
    ax_tok.legend(loc="upper right", frameon=False, fontsize=8.8)

    # Panel 4: step elapsed time breakdown + throughput
    roll = s["time/rollout_and_teacher"]
    other = s["time/train_and_sync"]
    tps = s["data/completion_tokens_per_second"]
    roll_pct = roll.sum() / s["time/step_elapsed_time"].sum() * 100
    ax_time.set_title("单步耗时拆分与整步吞吐", loc="left", fontweight="bold")
    b1 = ax_time.bar(
        steps,
        roll,
        width=0.62,
        color=TEAL_LIGHT,
        label=f"采样 + Teacher 打分（均 {roll.mean():.1f}s，占 {roll_pct:.1f}%）",
    )
    b2 = ax_time.bar(
        steps,
        other,
        bottom=roll,
        width=0.62,
        color=OCHRE_LIGHT,
        label=f"刷新权重 + 反传更新（均 {other.mean():.1f}s）",
    )
    ax_time.set_ylim(0, 210)
    ax_time.set_ylabel("耗时（秒）")

    ax_tps = ax_time.twinx()
    ax_tps.spines["top"].set_visible(False)
    ax_tps.spines["right"].set_visible(True)
    ax_tps.spines["right"].set_color("#BAC4CD")
    l1 = ax_tps.plot(
        steps,
        tps,
        color=INK,
        lw=1.9,
        marker="D",
        ms=4.2,
        label=f"整步吞吐（均 {tps.mean():.1f} tok/s）",
    )
    ax_tps.set_ylim(35, 114)
    ax_tps.set_ylabel("Completion tokens / s", color=INK)
    ax_tps.tick_params(axis="y", colors=INK, length=0)

    ax_time.annotate(
        f"{s['time/step_elapsed_time'][3]:.1f}s",
        (steps[3], s["time/step_elapsed_time"][3]),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        fontsize=8.5,
        color=OCHRE,
        fontweight="bold",
    )
    handles = [b1, b2, l1[0]]
    labels = [h.get_label() for h in handles]
    ax_time.legend(handles, labels, loc="upper right", frameon=False, fontsize=8.3)

    for ax in axes.flat:
        ax.grid(axis="y", color="#E0E6E8", lw=0.75)
        ax.set_axisbelow(True)
        ax.set_xlim(0.35, 10.65)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.tick_params(length=0)
    for ax in (ax_tok, ax_time):
        ax.set_xlabel("训练步")

    fig.suptitle(
        "General OPD 同步版试跑记录（Qwen3.5-4B ← Qwen3.8-27B，DeepMath-103K，10 步）",
        fontsize=14.5,
        fontweight="bold",
    )
    fig.savefig(OUT / "training.png")
    plt.close(fig)


if __name__ == "__main__":
    configure()
    training()
