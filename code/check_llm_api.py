#!/usr/bin/env python3
"""Minimal one-shot check that the DashScope API key works.

Reads the key from the first CLI argument, or from DASHSCOPE_API_KEY / LLM_API_KEY
in the environment, then makes a single tiny chat completion call.

Usage:
    python code/check_llm_api.py 'sk-...'
    DASHSCOPE_API_KEY=sk-... python code/check_llm_api.py
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    key = sys.argv[1] if len(sys.argv) > 1 else None
    key = key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("LLM_API_KEY")
    if not key:
        print("FAIL: no API key (pass as argv or set DASHSCOPE_API_KEY)")
        return 1

    base_url = os.environ.get(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = os.environ.get("DELTASG_LLM_MODEL", "qwen3.8-max")

    try:
        from openai import OpenAI
    except ImportError:
        print("FAIL: openai package not installed in this env")
        return 1

    client = OpenAI(api_key=key, base_url=base_url)
    print(f"checking key=...{key[-4:]} model={model} base_url={base_url}")

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "只回复两个字：收到"}],
            max_tokens=8,
        )
    except Exception as exc:  # noqa: BLE001 - surface any API/network error
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1

    print("OK ->", resp.choices[0].message.content.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
