"""Shared ToolACE data, literal-only scoring, and tokenization."""

import ast
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STUDENT = "Qwen/Qwen3.5-4B"
TEACHER = "Qwen/Qwen3.8-27B"


def canonical(value):
    """Compare literal values, treating int and float with equal numeric value as equivalent."""
    if isinstance(value, dict):
        return (
            "dict",
            frozenset((canonical(k), canonical(v)) for k, v in value.items()),
        )
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(canonical(item) for item in value))
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite number")
        return ("number", float(value))
    if value is None or isinstance(value, str):
        return (type(value).__name__, value)
    raise ValueError("unsupported literal")


def _parse_arg_value(raw: str):
    raw = raw.strip()
    if not raw:
        raise ValueError("empty argument value")
    try:
        return ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return raw


def _parse_loose_call(text: str):
    """Fallback parser for unquoted string values in [func(k=v, ...)] or func(k=v, ...)."""
    inner = text[1:-1].strip() if text.startswith("[") and text.endswith("]") else text
    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\(([^()]*)\)", inner, re.S)
    if not match:
        raise ValueError("not a single function call")
    name, arg_str = match[1], match[2].strip()
    if not arg_str:
        return name, {}
    kwargs = {}
    key_pattern = re.compile(r"(?:^|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")
    matches = list(key_pattern.finditer(arg_str))
    if not matches or matches[0].start() != 0:
        raise ValueError("unable to parse keyword arguments")
    for idx, m in enumerate(matches):
        key = m[1]
        if key in kwargs:
            raise ValueError("duplicate keyword")
        val_start = m.end()
        val_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(arg_str)
        val = _parse_arg_value(arg_str[val_start:val_end])
        kwargs[key] = canonical(val)
    return name, kwargs


def _parse_json_call(text: str):
    """Support standard JSON tool-call list: [{"name": ..., "arguments": {...}}]."""
    data = json.loads(text)
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise ValueError("expected single JSON tool call")
    item = data[0]
    name = item.get("name")
    args = item.get("arguments")
    if not isinstance(name, str) or not isinstance(args, dict):
        raise ValueError("invalid JSON tool call schema")
    return name, {k: canonical(v) for k, v in args.items()}


def parse_call(text, *, allow_fence=False):
    text = text.strip()
    if allow_fence:
        fence = re.fullmatch(r"```(?:python|py|json)?\s*\n(.*?)\n?\s*```", text, re.S | re.I)
        if fence:
            text = fence[1].strip()
    if allow_fence and text.startswith("[") and '"name"' in text:
        try:
            return _parse_json_call(text)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    try:
        expr = ast.parse(text, mode="eval").body
    except SyntaxError:
        if allow_fence:
            return _parse_loose_call(text)
        raise
    if isinstance(expr, ast.List):
        if len(expr.elts) != 1:
            raise ValueError("expected a single call in a list")
        call = expr.elts[0]
    elif allow_fence and isinstance(expr, ast.Call):
        call = expr
    else:
        raise ValueError("expected a single call in a list")
    if (
        not isinstance(call, ast.Call)
        or not isinstance(call.func, ast.Name)
        or call.args
    ):
        raise ValueError("expected a tool name and keyword arguments")
    kwargs = {}
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg in kwargs:
            raise ValueError("expanded or duplicate keyword")
        try:
            value = ast.literal_eval(keyword.value)
        except ValueError:
            if allow_fence and isinstance(keyword.value, ast.Name):
                value = keyword.value.id
            elif allow_fence and isinstance(keyword.value, ast.Attribute):
                value = ast.unparse(keyword.value)
            else:
                raise
        kwargs[keyword.arg] = canonical(value)
    return call.func.id, kwargs


PARSE_ERRORS = (SyntaxError, ValueError, TypeError, RecursionError, OverflowError)


def score(prediction, answer, truncated):
    expected, kwargs = parse_call(answer)
    try:
        tool, args = parse_call(prediction, allow_fence=True)
    except PARSE_ERRORS:
        tool, args = None, {}
    name_ok = tool == expected
    exact = name_ok and args == kwargs
    if truncated and not exact:
        error = "truncated"
    elif tool is None:
        error = "format_error"
    elif not name_ok:
        error = "wrong_tool"
    elif args.keys() != kwargs.keys():
        error = "wrong_args_key"
    elif not exact:
        error = "wrong_args_value"
    else:
        error = None
    return dict(
        format_rate=tool is not None,
        tool_name_acc=name_ok,
        exact_ast_match=exact,
        error=error,
    )


def prompt_ids(tokenizer, row):
    messages = [{"role": role, "content": row[role]} for role in ("system", "user")]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tokenizer.encode(rendered, add_special_tokens=False)


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    tmp.replace(path)


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    tmp.replace(path)


def capped(seq, tokenizer, limit):
    if seq.stop_reason.lower() in {"stop", "eos", "stop_sequence", "eos_token"}:
        return False
    return len(seq.tokens) >= limit and seq.tokens[-1] != tokenizer.eos_token_id
