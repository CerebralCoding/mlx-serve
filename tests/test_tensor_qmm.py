#!/usr/bin/env python3
"""Check exact serial/MTP output on a running row-tensor Qwen server.

Start with --mtp --no-pld --no-drafter --prefix-cache-entries 0 and pass
its --log-file here. This tests correctness and engagement, not performance.
"""
import argparse
import json
from pathlib import Path
import urllib.request


def complete(url, prompt, temperature, seed, mtp, budget):
    body = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "seed": seed,
        "max_tokens": budget,
        "enable_mtp": mtp,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
    assert "error" not in result, result
    choice = result["choices"][0]
    count = result["usage"]["completion_tokens"]
    assert 0 < count <= budget, result
    assert choice["message"].get("content"), result
    return choice["message"], choice["finish_reason"], count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:19288")
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    cases = [
        ("Explain why the sky is blue in three sentences.", 0.0, 42, 96),
        ("Explain why the sky is blue in three sentences.", 0.7, 42, 96),
        ("Write a Python function that reverses a string.", 0.7, 7, 96),
        ("Count from one to twenty, separated by commas.", 0.0, 7, 96),
        ("Describe a small garden.", 0.7, 17, 1),
        ("Describe a small garden.", 0.7, 17, 3),
    ]
    for index, (prompt, temperature, seed, budget) in enumerate(cases, 1):
        before = args.log.read_text().count("[spec-stats] mode=mtp")
        serial = complete(args.url, prompt, temperature, seed, False, budget)
        serial_log = args.log.read_text()
        assert serial_log.count("[spec-stats] mode=mtp") == before, "serial opt-out ignored"
        drafted = complete(args.url, prompt, temperature, seed, True, budget)
        after = args.log.read_text()
        assert "[tensor-qmm] tiled row-invariant M5 projections engaged" in after, "tensor path declined"
        if budget > 3:
            assert after.count("[spec-stats] mode=mtp") > before, "MTP silently declined"
        assert serial == drafted, f"case {index}: serial/MTP mismatch\n{serial!r}\n{drafted!r}"
        print(f"PASS {index}: temperature={temperature}, seed={seed}, budget={budget}, tokens={serial[2]}")


if __name__ == "__main__":
    main()
