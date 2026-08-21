#!/usr/bin/env python3
"""多机 tp16 推理客户端: 类似单机 launch.py --infer 的文本用例(含 --long-text).

用法: python client.py --host 127.0.0.1 --port 8000 \
        --text-case <text_case.yaml> --long-text-case <long_text_case.yaml> \
        --output-dir <dir> --tag <model_tag>
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import yaml
from openai import OpenAI

CRITICAL_WORDS = [
    "EngineCore encountered an issue",
    "ioctl create queue block timeout",
]

SYSTEM_MESSAGE = {
    "role": "system",
    "content": "You must answer as concisely as possible. Any extra information is unnecessary.",
}


def load_cases(case_file: str) -> list[dict]:
    with open(case_file, "r", encoding="utf-8") as f:
        cases = yaml.safe_load(f)
    return cases or []


def run_text_cases(
    host: str,
    port: int,
    cases: list[dict],
    log_file: str,
    tag: str,
    default_max_tokens: int = 256,
) -> int:
    client = OpenAI(api_key="EMPTY", base_url=f"http://{host}:{port}/v1")
    model = client.models.list().data[0].id

    # Group cases by max_tokens to reduce round-trips (mirrors model_worker.py)
    groups: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for idx, case in enumerate(cases):
        mt = case.get("max_tokens", default_max_tokens)
        groups[mt].append((idx, case))

    responses: dict[int, str] = {}
    for mt, indexed_cases in groups.items():
        for idx, case in indexed_cases:
            messages = [SYSTEM_MESSAGE, {"role": "user", "content": case["question"]}]
            resp = client.chat.completions.create(
                messages=messages,
                model=model,
                max_completion_tokens=mt,
            )
            responses[idx] = resp.choices[0].message.content

    corrected = 0
    with open(log_file, "a", encoding="utf-8") as f:
        for idx, case in enumerate(cases):
            content = responses.get(idx, "") or ""
            for word in CRITICAL_WORDS:
                if word in content:
                    raise RuntimeError(
                        f"client received: {word}, which indicates vllm serve might have crashed. Aborting..."
                    )
            keywords = case.get("keywords", [])
            if any(str(k).lower() in content.lower() for k in keywords):
                corrected += 1

            question_text = case["question"]
            if len(question_text) > 2000:
                question_text = question_text[:2000] + "\n... [truncated]"
            f.write(f"[{tag}] Question: {question_text}\n")
            f.write(f"[{tag}] Response:\n{content}\n")
            f.write("-" * 40 + "\n")
    return corrected


def main() -> int:
    parser = argparse.ArgumentParser(description="Batched text-only inference client (with long-text).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--text-case", required=True)
    parser.add_argument("--long-text-case", default=None)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--tag", default="model")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f"{args.tag}_text_only_inference.log")

    cases = load_cases(args.text_case)
    long_used = "-"
    if args.long_text_case and os.path.exists(args.long_text_case):
        long_cases = load_cases(args.long_text_case)
        if long_cases:
            idx = args.port % len(long_cases)
            picked = dict(long_cases[idx])
            picked.setdefault("max_tokens", 1024)
            cases = list(cases) + [picked]
            long_used = args.long_text_case
            print(
                f"[client] mixed long case #{idx + 1}/{len(long_cases)} "
                f"(max_tokens={picked['max_tokens']}) -> total {len(cases)} cases"
            )

    t0 = time.time()
    corrected = run_text_cases(args.host, args.port, cases, log_file, args.tag)
    duration = time.time() - t0
    total = len(cases)
    print(
        f"[client] {args.tag} text-only inference done: {corrected}/{total} correct, "
        f"{duration:.1f}s -> {log_file}"
    )
    with open(os.path.join(args.output_dir, "inference_summary.txt"), "a", encoding="utf-8") as f:
        f.write(f"{args.tag}\t{corrected}\t{total}\t{duration:.1f}\t{long_used}\n")

    return 0 if corrected == total else 1


if __name__ == "__main__":
    sys.exit(main())
