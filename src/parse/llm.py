"""Local-LLM client for citable-sentence classification.

Talks to an OpenAI-compatible chat-completions endpoint (default:
``http://localhost:8080/v1``, e.g. ``llama-server``). Sends a batch of
sentences in a single request and parses back a JSON array of
``{"citable": bool, "reason": str}`` objects, one per sentence.

If the model's response can't be parsed as a length-N JSON array, we fall
back to per-sentence calls for that batch so a single bad response can't
poison the whole pipeline.
"""

from __future__ import annotations

import json
import re
from typing import Any

import requests

from src.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_BATCH_SIZE,
    LLM_MODEL,
    LLM_TIMEOUT,
)

SYSTEM_PROMPT = (
    "You are identifying sentences that need a citation. "
    "A sentence needs a citation if it makes a factual claim, references "
    "prior work, or states an empirical finding that isn't the author's own "
    "contribution in this paper.\n\n"
    "Examples:\n"
    '- "Transformers have become the dominant architecture in NLP." -> citable\n'
    '- "In this paper, we propose a new method." -> not citable\n'
    '- "We trained for 10 epochs on a single GPU." -> not citable'
)

_BATCH_INSTRUCTION = (
    "Return ONLY a JSON array of {n} objects (one per input sentence, in the "
    'same order), each shaped: {{"citable": true|false, "reason": "<short>"}}. '
    "Do not include any commentary or markdown fences."
)

_SINGLE_INSTRUCTION = (
    'Return ONLY a JSON object: {"citable": true|false, "reason": "<short>"}. '
    "Do not include any commentary or markdown fences."
)


def _post_chat(messages: list[dict[str, str]], max_tokens: int) -> str:
    """POST to the chat-completions endpoint and return the message content."""
    url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"
    payload: dict[str, Any] = {
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if LLM_MODEL:
        payload["model"] = LLM_MODEL
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    resp = requests.post(url, json=payload, headers=headers, timeout=LLM_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        if s.endswith("```"):
            s = s[: -3]
    return s.strip()


def _extract_json(text: str, container: str) -> Any:
    """Find the first JSON array (``container='['``) or object (``'{'``) in
    ``text`` and parse it. Raises ``ValueError`` if nothing parses."""
    text = _strip_fences(text)
    open_ch, close_ch = ("[", "]") if container == "[" else ("{", "}")
    start = text.find(open_ch)
    if start < 0:
        raise ValueError(f"no '{open_ch}' in response: {text!r}")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"unterminated JSON in response: {text!r}")


def _normalize_result(obj: Any) -> dict[str, Any]:
    citable = bool(obj.get("citable", False)) if isinstance(obj, dict) else False
    reason = ""
    if isinstance(obj, dict):
        reason = str(obj.get("reason", "") or "")
    return {"citable": citable, "reason": reason}


def classify_one(sentence: str) -> dict[str, Any]:
    """Classify a single sentence. Used as a fallback when batch parsing fails."""
    user = f"{_SINGLE_INSTRUCTION}\n\nSentence: {sentence}"
    try:
        content = _post_chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            max_tokens=120,
        )
        obj = _extract_json(content, container="{")
        return _normalize_result(obj)
    except Exception as e:  # noqa: BLE001 - any failure -> safe default
        return {"citable": False, "reason": f"llm_error: {e}"}


def _classify_one_batch(sentences: list[str]) -> list[dict[str, Any]]:
    """Classify a single batch in one LLM call. Falls back per-sentence on failure."""
    n = len(sentences)
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
    user = f"{_BATCH_INSTRUCTION.format(n=n)}\n\nSentences:\n{numbered}"

    try:
        content = _post_chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            max_tokens=80 * n + 80,
        )
        arr = _extract_json(content, container="[")
        if not isinstance(arr, list) or len(arr) != n:
            raise ValueError(f"expected list of length {n}, got {arr!r}")
        return [_normalize_result(x) for x in arr]
    except Exception:
        return [classify_one(s) for s in sentences]


def classify_batch(
    sentences: list[str], batch_size: int | None = None
) -> list[dict[str, Any]]:
    """Classify ``sentences`` in chunks of ``batch_size``.

    Returns one ``{"citable": bool, "reason": str}`` per input sentence,
    aligned to input order.
    """
    if not sentences:
        return []
    bs = batch_size or LLM_BATCH_SIZE
    out: list[dict[str, Any]] = []
    for i in range(0, len(sentences), bs):
        chunk = sentences[i : i + bs]
        out.extend(_classify_one_batch(chunk))
    return out
