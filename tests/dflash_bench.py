"""Run TensorFold's benchmark, recording exact requests and complete replies.

The imported harness retains its timing, warmup, seeds and summary calculations.
Full replies and per-request measurements are preserved beside its summary.
Both prompts use chat by default. --native-prompts retains TensorFold's raw
completion code prompt instead. Set --tensorfold-root to a read-only checkout.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
options = argparse.ArgumentParser(add_help=False)
options.add_argument("--native-prompts", action="store_true")
options.add_argument("--tensorfold-root", type=Path, default=Path.home() / "OpenSource/TensorFold")
known, remaining = options.parse_known_args()
if "--help" in remaining or "-h" in remaining:
    print("Adapter options: --native-prompts (keep raw code completion), --tensorfold-root PATH")
sys.argv = [sys.argv[0], *remaining]
source = known.tensorfold_root / "tools/bench_openai.py"
spec = importlib.util.spec_from_file_location("tf_bench", source)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
if not known.native_prompts:
    for prompt in bench.PROMPTS:
        prompt["kind"] = "chat"
        if prompt["name"] == "fibonacci-raw":
            prompt["name"] = "fibonacci-chat-no-think"

original_stream = bench.stream
original_open = bench.urllib.request.urlopen
runs = []
requests = []


def bounded_open(request, timeout=15):
    requests.append({"url": request.full_url, "body": json.loads(request.data)})
    return original_open(request, timeout=min(timeout, 15))


def recorded_stream(base, model, item, tokens, temperature, seed):
    result = original_stream(base, model, item, tokens, temperature, seed)
    runs.append(dict(prompt=item, temperature=temperature, seed=seed, **result))
    if result["ttft_s"] > 15:
        raise RuntimeError(f"First reply exceeded 15 seconds: {result['ttft_s']}")
    if result["tokens"] != tokens:
        raise RuntimeError(f"Expected {tokens} completion tokens, got {result['tokens']}")
    return result


bench.urllib.request.urlopen = bounded_open
bench.stream = recorded_stream
try:
    bench.main()
finally:
    if "--output" in sys.argv:
        output = Path(sys.argv[sys.argv.index("--output") + 1])
        output.with_suffix(".runs.json").write_text(json.dumps(runs, indent=2))
        output.with_suffix(".requests.json").write_text(json.dumps(requests, indent=2))
