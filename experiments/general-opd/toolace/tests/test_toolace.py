"""Run from the repository root after 00-prepare-data.py using unittest discover."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolace import ROOT, read_rows, score
from training import datum


class ToolACETest(unittest.TestCase):
    def test_fixed_gold(self):
        splits = [
            read_rows(ROOT / "datasets" / f"{name}.jsonl") for name in ("train", "eval")
        ]
        self.assertEqual([len(s) for s in splits], [200, 60])
        rows = splits[0] + splits[1]
        self.assertEqual(
            len({" ".join(r["user"].casefold().split()) for r in rows}), 260
        )
        for row in rows:
            self.assertTrue(all(row.values()))
            self.assertTrue(
                score(row["assistant"], row["assistant"], False)["exact_ast_match"]
            )

    def test_error_types(self):
        gold = "[f(x=2, y=True)]"
        cases = [
            ("```python\n[f(y=True, x=2)]\n```", False, None),
            ("[f(x=2, y=1)]", False, "wrong_args_value"),
            ("[f(x=2)]", False, "wrong_args_key"),
            ("[g()]", False, "wrong_tool"),
            ("[f(x=g())]", False, "format_error"),
            ("[f(), g()]", False, "format_error"),
            ("[f(x=", True, "truncated"),
        ]
        for text, capped, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(score(text, gold, capped)["error"], expected)

    def test_datum_alignment(self):
        d = datum(
            [2, 3, 4], [0, 4, 5], logprobs=[0, -0.5, -0.7], advantages=[0, 0.1, 0.2]
        )
        self.assertEqual(d.loss_fn_inputs["target_tokens"].tolist(), [0, 4, 5])
        with self.assertRaises(ValueError):
            datum([2, 3], [3, 4], weights=[1])
