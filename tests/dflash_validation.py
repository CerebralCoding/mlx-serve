"""HTTP correctness corpus for DFlash, with recorded replies and one-row comparison.

Run against one server at a time. This checks determinism, exact task answers,
and full-response parity; it is not a speed benchmark or a broad quality score.
"""
import argparse
import json
from pathlib import Path
import urllib.request


def cases():
    out = []
    for i, (a, b) in enumerate([(17, 29), (123, 456), (98, 76), (1001, 999)]):
        out.append({"name": f"arithmetic-{i}", "messages": [{"role": "user", "content":
                    f'Return only a JSON object with integer fields "sum" and "product" for {a} and {b}.'}],
                    "expected": {"sum": a + b, "product": a * b}, "max_tokens": 96})
    out.extend([
        {"name": "sort", "messages": [{"role": "user", "content":
         'Return only a JSON array containing these integers in ascending order, retaining duplicates: 9, -3, 4, 9, 0, -11, 4.'}],
         "expected": [-11, -3, 0, 4, 4, 9, 9], "max_tokens": 96},
        {"name": "extract", "messages": [{"role": "user", "content":
         'Return only JSON with fields "city", "count", "active" from this record: city=Odense; count=17; active=false. Use a string, integer and boolean respectively.'}],
         "expected": {"city": "Odense", "count": 17, "active": False}, "max_tokens": 96},
        {"name": "multi-turn", "messages": [
         {"role": "user", "content": "Remember that project Aster has code 314 and project Birch has code 271."},
         {"role": "assistant", "content": "I have noted both project codes."},
         {"role": "user", "content": 'Birch now has code 828. Return only JSON mapping both project names to their current integer codes.'}],
         "expected": {"Aster": 314, "Birch": 828}, "max_tokens": 96},
        {"name": "code-long", "messages": [{"role": "user", "content":
         "Write a Python function that merges two sorted lists, preserving duplicates, then explain its time and space complexity and give three examples."}], "max_tokens": 384},
        {"name": "chat-long", "messages": [{"role": "user", "content":
         "Explain how matrix multiplication uses a GPU in plain English. Give a worked 2 by 2 numerical example, then explain memory bandwidth and why small matrices can be slower on a GPU."}], "max_tokens": 384},
    ])
    records = [f"Record R{i:03}: code={10000 + i * 37}; category=ordinary." for i in range(96)]
    out.append({"name": "long-context", "messages": [{"role": "user", "content":
                "Read these records:\n" + "\n".join(records) +
                '\nReturn only a JSON object mapping R007, R051 and R089 to their integer codes.'}],
                "expected": {f"R{i:03}": 10000 + i * 37 for i in [7, 51, 89]}, "max_tokens": 96})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base")
    p.add_argument("model")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compare", type=Path)
    p.add_argument("--reps", type=int, default=2)
    args = p.parse_args()
    if args.output.exists():
        p.error(f"refusing to overwrite {args.output}")
    records = []
    try:
        for repeat in range(args.reps):
            for case in cases():
                for temperature in [0, 1]:
                    body = {"model": args.model, "messages": case["messages"],
                            "max_tokens": case["max_tokens"], "temperature": temperature,
                            "top_k": 20, "top_p": 0.95, "seed": 731,
                            "chat_template_kwargs": {"enable_thinking": False}, "stream": False}
                    request = urllib.request.Request(args.base + "/v1/chat/completions",
                              data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(request, timeout=120) as response:
                        result = json.load(response)
                    choice = result["choices"][0]
                    text = choice["message"].get("content") or ""
                    task_pass = None
                    if "expected" in case:
                        try:
                            task_pass = json.loads(text) == case["expected"]
                        except ValueError:
                            task_pass = False
                    row = {"case": case["name"], "temperature": temperature, "repeat": repeat,
                           "text": text, "finish_reason": choice.get("finish_reason"),
                           "usage": result.get("usage"), "task_pass": task_pass}
                    records.append(row)
                    if not text:
                        raise RuntimeError(f"empty response: {row}")
    finally:
        args.output.write_text(json.dumps(records, indent=2) + "\n")
    canonical = {(r["case"], r["temperature"]): r for r in records if r["repeat"] == 0}
    unstable = [r for r in records if r["text"] != canonical[r["case"], r["temperature"]]["text"]]
    mismatches = []
    if args.compare:
        reference = {(r["case"], r["temperature"], r["repeat"]): r
                     for r in json.loads(args.compare.read_text())}
        if len(reference) != len(records):
            raise RuntimeError("reference and candidate have different case counts")
        mismatches = [r for r in records
                      if r["text"] != reference[r["case"], r["temperature"], r["repeat"]]["text"]]
    graded = [r for r in canonical.values() if r["task_pass"] is not None]
    print(json.dumps({"requests": len(records), "repeat_mismatches": len(unstable),
                      "reference_mismatches": len(mismatches) if args.compare else None,
                      "task_pass": sum(r["task_pass"] for r in graded), "task_total": len(graded),
                      "task_failures": [{"case": r["case"], "temperature": r["temperature"]}
                                        for r in graded if not r["task_pass"]]}), flush=True)
    if unstable or mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
