"""Seed-42 GSM8K held-out slice, using the repository's boxed math-verify scorer."""

import asyncio
import random
import sys

from datasets import load_dataset

from evaluation import evaluate
from toolace import ROOT

sys.path.insert(0, str(ROOT.parents[1] / "gspo"))
from reward import score_answer


def problems():
    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        revision="740312add88f781978c0658806c59bc2815b9866",
        split="test",
    )
    return [
        dict(
            id=f"gsm8k-{i}",
            system=r"Solve briefly. Put the final numeric answer in \boxed{}.",
            user=dataset[i]["question"],
            assistant=dataset[i]["answer"]
            .rsplit("####", 1)[1]
            .strip()
            .replace(",", ""),
        )
        for i in random.Random(42).sample(range(len(dataset)), 30)
    ]


def grade(text, gold, truncated):
    result = score_answer(text, gold, completion_tokens=0)
    return dict(
        correct=result.correct,
        extracted=result.answer,
        error=None if result.correct else ("truncated" if truncated else "incorrect"),
    )


if __name__ == "__main__":
    asyncio.run(evaluate("gsm8k", problems, grade))
