#!/usr/bin/env python3
"""
Scan a directory of benchmark log files:
  1. Search each file for error patterns (exception types, CUDA errors, OOM, etc.)
  2. If any pattern matches, confirm whether "Traceback" also appears in the file
  3. For each match, capture ±10 lines of context
  4. Extract the vLLM serve command from the first line
  5. Output:
       - Full report with error contexts  (-o report.txt)
       - Clean config lines for bench_vllm.sh (--config configs.txt)

Usage:
    python extract_failed.py ./inference/ -o report.txt --config configs.txt
    python extract_failed.py ./inference/ --config configs.txt   # config-only
"""

import argparse
import glob
import os
import re
import shlex
import sys

# ---------------------------------------------------------------------------
# Error patterns — searched across the ENTIRE file
# Each: (regex, display_name)
# ---------------------------------------------------------------------------
ERROR_PATTERNS = [
    # Python built-in exceptions
    (r"\bAssertionError\b",           "AssertionError"),
    (r"\bValueError\b",               "ValueError"),
    (r"\bRuntimeError\b",             "RuntimeError"),
    (r"\bTypeError\b",                "TypeError"),
    (r"\bKeyError\b",                 "KeyError"),
    (r"\bAttributeError\b",           "AttributeError"),
    (r"\bImportError\b",              "ImportError"),
    (r"\bModuleNotFoundError\b",      "ModuleNotFoundError"),
    (r"\bOSError\b",                  "OSError"),
    (r"\bFileNotFoundError\b",        "FileNotFoundError"),
    (r"\bPermissionError\b",          "PermissionError"),
    (r"\bMemoryError\b",              "MemoryError"),
    (r"\bTimeoutError\b",             "TimeoutError"),
    (r"\bConnectionError\b",          "ConnectionError"),
    (r"\bNotImplementedError\b",      "NotImplementedError"),
    (r"\bIndexError\b",               "IndexError"),
    (r"\bUnboundLocalError\b",        "UnboundLocalError"),
    (r"\bNameError\b",                "NameError"),
    (r"\bSyntaxError\b",              "SyntaxError"),
    (r"\bOverflowError\b",            "OverflowError"),
    (r"\bZeroDivisionError\b",        "ZeroDivisionError"),
    (r"\bSystemError\b",              "SystemError"),
    (r"\bSystemExit\b",               "SystemExit"),
    (r"\bKeyboardInterrupt\b",        "KeyboardInterrupt"),
    (r"\bStopIteration\b",            "StopIteration"),
    (r"\bGeneratorExit\b",            "GeneratorExit"),
    # PyTorch / CUDA
    (r"\btorch\.OutOfMemoryError\b",  "OutOfMemoryError"),
    (r"\bCUDA\s*(out\s+of\s+memory|OOM)\b", "CUDA OOM"),
    (r"\bCUDA\s+error\b",             "CUDA Error"),
    (r"\bcudaError\w*\b",             "CUDA Error"),
    (r"\bCUDNN_STATUS\w*\b",          "cuDNN Error"),
    (r"\bNCCL\s+error\b",             "NCCL Error"),
    (r"\bout\s+of\s+memory\b",        "OOM"),
    # vLLM-specific
    (r"\bValueError.*tensor.parallel\b", "vLLM TP Error"),
    (r"\bcannot\s+be\s+greater\s+than\s+total\b", "vLLM Config Error"),
    (r"\bdoes\s+not\s+support\b",     "Unsupported"),
    (r"\bnot\s+found\b",              "NotFound"),
    (r"\bno\s+such\s+file\b",           "FileNotFound"),
    # HTTP / network
    (r"\bConnection\s+refused\b",     "Connection Refused"),
    (r"\bConnection\s+reset\b",       "Connection Reset"),
    (r"\bHTTP\s+\d{3}\b",             "HTTP Error"),
    # SIG* / crashes
    (r"\bSIGSEGV\b",                  "SIGSEGV"),
    (r"\bSIGKILL\b",                  "SIGKILL"),
    (r"\bSIGTERM\b",                  "SIGTERM"),
    (r"\bSIGABRT\b",                  "SIGABRT"),
    (r"\bSegmentation\s+fault\b",     "Segfault"),
    (r"\bAborted\b",                  "Aborted"),
    (r"\bKilled\b",                   "Killed"),
    (r"\bBus\s+error\b",              "Bus Error"),
    (r"\bhang\b",                     "Hang"),
    (r"\btimeout\b",                  "Timeout"),
    (r"\bdeadlock\b",                 "Deadlock"),
]

CONTEXT_LINES = 10


# ---------------------------------------------------------------------------
# File scanning
# ---------------------------------------------------------------------------

def scan_log_file(path):
    """
    Scan a single log file.
    Returns a dict with:
      - first_line: the first line (expected to contain the vllm command)
      - has_traceback: bool
      - hits: list of (line_number, display_name, matched_text, context_lines)
    Returns None if no error patterns matched.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return None

    if not lines:
        return None

    first_line = lines[0].rstrip("\n").rstrip("\r")
    full_text = "".join(lines)
    has_traceback = "Traceback" in full_text

    # Search for error patterns across all lines
    hits = []
    for lineno, line in enumerate(lines, start=1):
        for pattern, display_name in ERROR_PATTERNS:
            m = re.search(pattern, line, re.IGNORECASE)
            if m:
                # Grab ±CONTEXT_LINES around this hit
                ctx_start = max(0, lineno - 1 - CONTEXT_LINES)
                ctx_end = min(len(lines), lineno + CONTEXT_LINES)
                context = []
                for i in range(ctx_start, ctx_end):
                    prefix = ">" if i == lineno - 1 else " "
                    context.append(
                        "{}{:6d}|{}".format(prefix, i + 1, lines[i].rstrip("\n").rstrip("\r"))
                    )
                hits.append({
                    "line_number": lineno,
                    "error_type": display_name,
                    "matched_text": line.strip(),
                    "context": context,
                })

    if not hits:
        return None

    # Deduplicate hits on the same line with the same error type
    seen = set()
    unique_hits = []
    for h in hits:
        key = (h["line_number"], h["error_type"])
        if key not in seen:
            seen.add(key)
            unique_hits.append(h)

    return {
        "first_line": first_line,
        "has_traceback": has_traceback,
        "hits": unique_hits,
    }


# ---------------------------------------------------------------------------
# Command parsing
# ---------------------------------------------------------------------------

def parse_vllm_command(cmd_line):
    """Parse a vLLM 'serve' command string. Returns dict or None."""
    m = re.match(r"^\[.*?\]\s*command:\s*", cmd_line)
    if m:
        cmd_line = cmd_line[m.end():]

    try:
        tokens = shlex.split(cmd_line)
    except ValueError:
        tokens = cmd_line.split()

    if not tokens or tokens[0] != "vllm":
        return None
    if len(tokens) < 3 or tokens[1] != "serve":
        return None

    model_path = tokens[2]
    idx = 3
    parsed = {"model_path": model_path}

    known_value_flags = {
        "--tp": "tp", "-tp": "tp", "--tensor-parallel-size": "tp",
        "--gpu-memory-utilization": "gpu_mem", "--gpu-memory-utilisation": "gpu_mem",
        "--max-model-len": "max_len",
        "--port": "port",
    }

    ignore_flags = {
        "--host", "-pp", "--pipeline-parallel-size",
        "-dp", "--data-parallel-size", "--distributed-executor-backend",
    }

    boolean_flags = {
        "--trust-remote-code", "--enforce-eager", "--enable-prefix-caching",
        "--enable-chunked-prefill", "--disable-log-requests", "--disable-log-stats",
        "--disable-sliding-window", "--no-enable-prefix-caching",
        "--disable-custom-all-reduce", "--enable-lora", "--fully-sharded-loras",
        "--use-v2-block-manager", "--disable-frontend-multiprocessing",
    }

    extra_parts = []

    while idx < len(tokens):
        tok = tokens[idx]
        if tok in known_value_flags:
            key = known_value_flags[tok]
            idx += 1
            if idx < len(tokens):
                parsed[key] = tokens[idx]
            idx += 1
        elif tok in boolean_flags:
            extra_parts.append(tok)
            idx += 1
        elif tok in ignore_flags:
            idx += 2
        elif tok.startswith("--") or tok.startswith("-"):
            extra_parts.append(tok)
            idx += 1
            if idx < len(tokens) and not tokens[idx].startswith("-"):
                extra_parts.append(tokens[idx])
                idx += 1
        else:
            idx += 1

    for key in ("gpu_mem", "tp", "max_len", "port"):
        val = parsed.get(key, "")
        if val:
            m = re.search(r"[\d.]+", val)
            if m:
                parsed[key] = m.group(0)

    parsed.setdefault("tp", "")
    parsed.setdefault("gpu_mem", "")
    parsed.setdefault("max_len", "")
    parsed.setdefault("port", "")
    parsed["extra_args"] = " ".join(p for p in extra_parts if p)

    return parsed


def format_entry(parsed):
    return "|".join([
        parsed["model_path"], parsed["tp"], parsed["gpu_mem"],
        parsed["max_len"], parsed["port"], parsed["extra_args"],
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Find failed vLLM benchmark logs via error patterns + Traceback."
    )
    parser.add_argument("log_dir", help="Directory containing .log files")
    parser.add_argument("-o", "--output", help="Full report output file")
    parser.add_argument("--config", help="Clean config-only output file")
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        print(f"ERROR: {args.log_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    results = []

    for path in sorted(glob.glob(os.path.join(args.log_dir, "*.log"))):
        info = scan_log_file(path)
        if info is None:
            continue

        parsed = parse_vllm_command(info["first_line"])
        # Extract tp for sorting (default 0 so unknown tp goes first)
        tp_val = 0
        if parsed:
            try:
                tp_val = int(float(parsed.get("tp", 0) or 0))
            except (ValueError, TypeError):
                tp_val = 0

        results.append({
            "file": os.path.basename(path),
            "first_line": info["first_line"],
            "has_traceback": info["has_traceback"],
            "hits": info["hits"],
            "config_line": format_entry(parsed) if parsed else None,
            "model_name": parsed["model_path"] if parsed else "?",
            "tp": tp_val,
        })

    if not results:
        print("No error patterns matched in any .log file.", file=sys.stderr)
        sys.exit(0)

    # Sort by TP ascending
    results.sort(key=lambda r: r["tp"])

    # ------------------------------------------------------------------
    # Build full report
    # ------------------------------------------------------------------
    report = []
    sep = "=" * 72
    sep2 = "-" * 72

    report.append(sep)
    report.append("  Failed benchmark report — {} file(s)  (sorted by TP asc)".format(len(results)))
    report.append("  Generated: {}".format(
        __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    report.append(sep)
    report.append("")

    for r in results:
        tb_status = "YES (confirmed)" if r["has_traceback"] else "NO (error pattern only)"
        report.append(sep2)
        report.append("  FILE       : {}".format(r["file"]))
        report.append("  MODEL      : {}".format(r["model_name"]))
        report.append("  TRACEBACK  : {}".format(tb_status))
        report.append("  ERROR HITS : {}".format(len(r["hits"])))
        report.append("  CONFIG LINE: {}".format(r["config_line"] or "(could not parse)"))
        report.append("  COMMAND    : {}".format(r["first_line"][:120]))
        report.append(sep2)

        for i, hit in enumerate(r["hits"], 1):
            report.append("")
            report.append("  --- Hit #{}/{}  Line {}  [{}] ---".format(
                i, len(r["hits"]), hit["line_number"], hit["error_type"]
            ))
            report.append("  Matched: {}".format(hit["matched_text"][:150]))
            report.append("  Context (±{} lines):".format(CONTEXT_LINES))
            report.append("")
            for ctx_line in hit["context"]:
                report.append("    {}".format(ctx_line))
            report.append("")

    report.append("")
    report.append(sep)
    report.append("  Summary config lines (copy into bench_vllm.sh MODELS array):")
    report.append(sep)
    report.append("")
    for r in results:
        if r["config_line"]:
            report.append(r["config_line"])

    report_text = "\n".join(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report_text + "\n")
        print("Full report written to {}".format(args.output), file=sys.stderr)
    else:
        print(report_text)

    # ------------------------------------------------------------------
    # Clean config file (no comments, no blank lines)
    # ------------------------------------------------------------------
    if args.config:
        config_lines = [r["config_line"] for r in results if r["config_line"]]
        with open(args.config, "w", encoding="utf-8") as f:
            for line in config_lines:
                f.write(line + "\n")
        print("Config lines ({} entries) written to {}".format(
            len(config_lines), args.config), file=sys.stderr)


if __name__ == "__main__":
    main()
