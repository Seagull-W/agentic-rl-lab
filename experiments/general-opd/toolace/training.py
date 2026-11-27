"""Shared asynchronous update loop and per-stage accounting."""

import argparse
import time
from collections import Counter

import numpy as np
import pyarrow as pa
from pyarrow import json as arrow_json
import pytrio as trio
import swanlab

from toolace import ROOT, STUDENT, TEACHER, read_rows, save_json


def options(method):
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default=STUDENT)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=4e-5)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--swanlab-project", default="agentic-rl-lab-opd")
    p.add_argument(
        "--swanlab-mode", choices=["online", "local", "disabled"], default="online"
    )
    p.add_argument("--name", default=method)
    if method == "opd":
        p.add_argument("--teacher-base-model", default=TEACHER)
        p.add_argument("--group-size", type=int, default=2)
        p.add_argument("--max-tokens", type=int, default=256)
        p.add_argument("--temperature", type=float, default=1.0)
        p.add_argument("--top-p", type=float, default=1.0)
        p.add_argument("--top-k", type=int, default=-1)
        p.add_argument("--kl-penalty-coef", type=float, default=1.0)
        p.add_argument("--concurrency", type=int, default=8)
        p.add_argument("--sampler-refresh-steps", type=int, default=1)
    else:
        p.add_argument("--max-length", type=int, default=1024)
    args = p.parse_args()
    for key in (
        "steps",
        "batch_size",
        "concurrency",
        "sampler_refresh_steps",
        "group_size",
        "max_tokens",
        "max_length",
        "lora_rank",
    ):
        if getattr(args, key, 1) < 1:
            p.error(f"{key} must be positive")
    return args


def datum(inputs, targets, **fields):
    if not inputs or any(len(v) != len(inputs) for v in [targets, *fields.values()]):
        raise ValueError("misaligned target or loss fields")
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(inputs),
        loss_fn_inputs={
            "target_tokens": np.asarray(targets, dtype=np.int64),
            **{k: np.asarray(v, dtype=np.float32) for k, v in fields.items()},
        },
    )


async def train(method, setup):
    a = options(method)
    data_path = ROOT / "datasets/train.jsonl"
    if method == "opd":
        schema = pa.schema([("system", pa.string()), ("user", pa.string())])
        projection = arrow_json.ParseOptions(
            explicit_schema=schema, unexpected_field_behavior="ignore"
        )
        rows = arrow_json.read_json(data_path, parse_options=projection).to_pylist()
    else:
        rows = read_rows(data_path)
    if a.steps * a.batch_size > len(rows):
        raise ValueError("steps × batch exceeds fixed training split")
    path = ROOT / "runs" / f"{a.name}_summary.json"
    if path.exists():
        raise FileExistsError(f"Run exists: {path}. Choose another --name.")
    start = time.monotonic()
    service = trio.ServiceClient()
    trainer = await service.create_lora_training_client_async(
        base_model=a.base_model, rank=a.lora_rank, seed=42
    )
    make_batch = await setup(service, trainer, a)
    run = swanlab.init(
        project=a.swanlab_project,
        name=f"toolace-{a.name}",
        config=vars(a),
        mode=a.swanlab_mode,
        log_dir=str(ROOT / "swanlog"),
    )
    try:
        summary = dict(
            method=method,
            config=vars(a),
            swanlab_url=run.url if a.swanlab_mode == "online" else None,
            history=[],
            weights_path=None,
        )
        totals = Counter()
        adam = trio.AdamParams(learning_rate=a.learning_rate, beta1=0.9, beta2=0.95)
        for step in range(a.steps):
            t = time.monotonic()
            batch = rows[step * a.batch_size : (step + 1) * a.batch_size]
            datums, metrics, usage = await make_batch(step, batch)
            loss_fn = "importance_sampling" if method == "opd" else "cross_entropy"
            fw = await trainer.forward_backward_async(datums, loss_fn=loss_fn)
            out = await fw
            await (await trainer.optim_step_async(adam))
            metrics.update({f"trainer/{k}": float(v) for k, v in out.metrics.items()})
            if method == "sft":
                nll = 0.0
                for d, result in zip(datums, out.loss_fn_outputs, strict=True):
                    lp = np.asarray(result["logprobs"].tolist())
                    mask = np.asarray(d.loss_fn_inputs["weights"].tolist(), dtype=bool)
                    nll -= lp[mask].sum()
                metrics["loss"] = float(nll / metrics["supervised_tokens"])
            elapsed = time.monotonic() - t
            metrics.update(step=step + 1, step_seconds=elapsed, **usage)
            if method == "opd":
                metrics["tokens_per_second"] = (
                    usage["completion_tokens_total"] / elapsed
                )
            totals.update(usage)
            summary["history"].append(metrics)
            summary.update(
                token_totals=dict(totals), elapsed_seconds=time.monotonic() - start
            )
            save_json(path, summary)
            swanlab.log(metrics, step=step + 1)
            print(f"{a.name} {step + 1}/{a.steps} {elapsed:.1f}s", flush=True)
        saved = await (
            await trainer.save_weights_for_sampler_async(
                name=f"toolace-{a.name}-step{a.steps}"
            )
        )
        summary.update(
            weights_path=saved.path, elapsed_seconds=time.monotonic() - start
        )
        save_json(path, summary)
        print(saved.path, flush=True)
    finally:
        swanlab.finish()
