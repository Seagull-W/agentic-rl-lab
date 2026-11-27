"""Score a base model or trio:// sampler on fixed ToolACE calls."""

import asyncio

from evaluation import evaluate
from toolace import ROOT, read_rows, score


if __name__ == "__main__":
    asyncio.run(
        evaluate("toolace", lambda: read_rows(ROOT / "datasets/eval.jsonl"), score)
    )
