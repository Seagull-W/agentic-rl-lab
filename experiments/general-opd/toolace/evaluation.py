"""Evaluate one model and safely retain completed items across interrupted runs."""

import argparse
import asyncio
import hashlib
import json
import time
from collections import Counter
from statistics import mean

import pytrio as trio

from toolace import ROOT, STUDENT, capped, prompt_ids, read_rows, save_json, save_rows


async def evaluate(task, load, grade):
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default=STUDENT)
    p.add_argument("--model-path", default=None)
    p.add_argument("--name", required=True)
    n = 60 if task == "toolace" else 30
    p.add_argument("--limit", type=int, default=n)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=256 if task == "toolace" else 384)
    p.add_argument("--temperature", type=float, default=0.0, choices=[0.0, 0.01])
    a = p.parse_args()
    if not 1 <= a.limit <= n or min(a.concurrency, a.max_tokens) < 1:
        p.error("invalid sample size, concurrency or token cap")
    rows = load()[: a.limit]
    if len(rows) != a.limit:
        raise ValueError("incomplete evaluation set")
    data_hash = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    config = dict(
        task=task,
        base_model=a.base_model,
        model_path=a.model_path,
        seed=42,
        temperature=a.temperature,
        max_tokens=a.max_tokens,
        data_hash=data_hash,
        enable_thinking=False,
    )
    path = ROOT / "eval-results" / f"{a.name}.jsonl"
    meta = path.with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if meta.exists():
        summary = json.loads(meta.read_text())
        if summary["config"] != config:
            raise ValueError(
                "evaluation name already in use with another configuration"
            )
        if path.exists():
            done = {r["id"]: r for r in read_rows(path)}
        if summary.get("completed") and len(done) == len(rows):
            print(summary)
            return
    elif path.exists():
        raise FileExistsError(path)
    else:
        summary = dict(config=config, completed=False, seconds=0.0)
        save_json(meta, summary)
    start = time.monotonic()
    client = await trio.ServiceClient().create_sampling_client_async(
        base_model=a.base_model, model_path=a.model_path
    )
    tok = client.get_tokenizer()
    sem = asyncio.Semaphore(a.concurrency)
    params = trio.SamplingParams(
        max_tokens=a.max_tokens,
        temperature=a.temperature,
        top_p=1.0,
        top_k=-1,
        seed=42,
        stop=[tok.eos_token],
    )

    async def one(row):
        ids = prompt_ids(tok, row)
        async with sem:
            res = await client.sample_async(
                prompt=trio.ModelInput.from_ints(ids),
                num_samples=1,
                sampling_params=params,
                return_text=False,
            )
        (seq,) = res.sequences
        pred = tok.decode(seq.tokens, skip_special_tokens=True).strip()
        cut = capped(seq, tok, a.max_tokens)
        result = dict(
            id=row["id"],
            user=row["user"],
            gold=row["assistant"],
            prediction=pred,
            prompt_tokens=res.input_tokens or len(ids),
            completion_tokens=len(seq.tokens),
            truncated=cut,
            stop_reason=seq.stop_reason,
            **grade(pred, row["assistant"], cut),
        )
        done[row["id"]] = result
        save_rows(path, [done[r["id"]] for r in rows if r["id"] in done])

    try:
        await asyncio.gather(*(one(r) for r in rows if r["id"] not in done))
        metrics = (
            ("format_rate", "tool_name_acc", "exact_ast_match")
            if task == "toolace"
            else ("correct",)
        )
        summary.update({k: mean(r[k] for r in done.values()) for k in metrics})
        summary.update(
            completed=True,
            n=len(done),
            avg_completion_tokens=mean(r["completion_tokens"] for r in done.values()),
            truncation_rate=mean(r["truncated"] for r in done.values()),
            error_counts=dict(Counter(r["error"] for r in done.values() if r["error"])),
            prompt_tokens_total=sum(r["prompt_tokens"] for r in done.values()),
            completion_tokens_total=sum(r["completion_tokens"] for r in done.values()),
        )
    finally:
        summary["seconds"] += time.monotonic() - start
        save_json(meta, summary)
    print(json.dumps(summary, ensure_ascii=False))
