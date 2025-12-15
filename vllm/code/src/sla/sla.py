#!/usr/bin/env python3
import json
import os
import shlex
import sys
import subprocess
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set

ResultKey = Tuple[int, int, int]  # (isl, osl, max_concurrency)

def parse_arguments():
    parser = argparse.ArgumentParser(description='SLA perf tools')
    
    parser.add_argument('--backend', default='vllm', help='backend engine')
    parser.add_argument('--host', default='127.0.0.1', help='server ip')
    parser.add_argument('--port', default='9527', help='server port')
    parser.add_argument('--model', required=True, help='path to model')
    # dataset&scripts params
    parser.add_argument('--dataset-name', choices=['burstgpt', 'random'], default='burstgpt', help='dataset name')
    parser.add_argument('--dataset-path', default=os.path.abspath(os.path.join(os.path.dirname(__file__), "250910_BurstGPT.csv")),
                       help='dataset path')
    parser.add_argument('--combinations', nargs='+', default=["128/128", "2048/128", "1024/1024", "3500/1000"],
                       help='random mode used lists')
    parser.add_argument('--sla-path', default=os.path.abspath(os.path.join(os.path.dirname(__file__), "sla.sh")),
                       help='sla.sh path')
    # test params
    parser.add_argument('--mandatory', type=int, nargs='+', default=[16, 8],
                       help='test boundary')
    parser.add_argument('--max-concurrency-limit', type=int, default=20,
                       help='max concurrency limit')
    parser.add_argument('--find-precise-bs', type=bool, default=False,
                       help='control the threshold to find the exact batch size')
    parser.add_argument('--con-times', type=int, default=10,
                       help='num-prompts is --con-times that of max concurrency')
    # file params
    parser.add_argument('--model-name', type=str, default="", help='if empty, derive from TOKENIZER basename')
    parser.add_argument('--result-jsonl', type=str, default="", help='default: same as bench JSONL')
    parser.add_argument('--log-file', type=str, default="", help='default: model_YYYYMMDD.log in CWD')
    parser.add_argument('--bench-output-jsonl', type=str, default="", help='default: model_bench.jsonl in CWD')
    parser.add_argument('--dataset-jsonl', type=str, default="", help='default computed to model_burst_YYMMDD.jsonl')
    # SLA params
    parser.add_argument('--ttft-ms-max', type=float, default=2005.0,
                       help='mean_ttft_ms < 2005')
    parser.add_argument('--tpot-ms-max', type=float, default=51.0,
                       help='mean_tpot_ms < 51')
    parser.add_argument('--qps-field', default='request_throughput',
                       help='QPS field name')
    # control params
    parser.add_argument('--print-cmd', action='store_true',
                       help='print cmds')
    parser.add_argument('--dry-run', action='store_true',
                       help='dont test anything')
    return parser.parse_args()

def pick_best_sla_qps(
    cache: Dict[ResultKey, Dict],
    isl: Optional[int] = None,
    osl: Optional[int] = None,
    ttft_ms_max: Optional[float] = None,
    tpot_ms_max: Optional[float] = None,
    qps_field: str = "request_throughput"
) -> Tuple[int, Optional[float]]:
    """Pick best QPS from cache under optional latency constraints.

    - If isl/osl are provided, only consider matching pairs; otherwise consider all.
    - If latency thresholds are provided, strictly enforce them: records MUST
      contain the metric and be below the threshold to be considered valid.
    """
    best_con = -1
    best_qps: Optional[float] = None
    for (i, o, c), obj in cache.items():
        if isl is not None and i != isl:
            continue
        if osl is not None and o != osl:
            continue
        mean_ttft = obj.get("mean_ttft_ms")
        mean_tpot = obj.get("mean_tpot_ms")
        if ttft_ms_max is not None:
            if not isinstance(mean_ttft, (int, float)) or float(mean_ttft) >= ttft_ms_max:
                continue
        if tpot_ms_max is not None:
            if not isinstance(mean_tpot, (int, float)) or float(mean_tpot) >= tpot_ms_max:
                continue
        qps_val = obj.get(qps_field)
        if not isinstance(qps_val, (int, float)):
            continue
        if best_qps is None or float(qps_val) > best_qps:
            best_qps = float(qps_val)
            best_con = int(c)
    return best_con, best_qps



def log_both(log_file: str, message: str) -> None:
    print(message)
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception:
        pass


def ensure_pandas_installed(log_file: str) -> None:
    try:
        import pandas  # noqa: F401
        return
    except Exception:
        pass
    log_both(log_file, "[SETUP] Installing pandas...")
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "pandas"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        log_both(log_file, f"[SETUP][ERROR] pip install pandas failed: {res.stderr.strip()}")
        raise SystemExit(1)
    try:
        import pandas  # noqa: F401
    except Exception as e:
        log_both(log_file, f"[SETUP][ERROR] pandas import failed after install: {e}")
        raise SystemExit(1)
    log_both(log_file, "[SETUP] pandas installed.")


def load_cache(jsonl_path: str) -> Dict[ResultKey, Dict]:
    cache: Dict[ResultKey, Dict] = {}
    if not os.path.exists(jsonl_path):
        return cache
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # Support both sglang-bench-style and burst-json lines.
            isl = obj.get("random_input_len")
            osl = obj.get("random_output_len")
            con = (
                obj.get("max_concurrency")
                if obj.get("max_concurrency") is not None
                else obj.get("MAX_CONCURRENCY")
            )
            # Fallback: some burst records may miss isl/osl; treat as (0,0) for keying
            if not isinstance(isl, int):
                isl = 0
            if not isinstance(osl, int):
                osl = 0
            if isinstance(con, (int, float)):
                cache[(int(isl), int(osl), int(con))] = obj
    return cache


def best_from_cache_for_combo(cache: Dict[ResultKey, Dict], isl: int, osl: int) -> Tuple[int, Optional[float]]:
    best_con = -1
    best_qps: Optional[float] = None
    rows: List[Tuple[int, float]] = []
    for (i, o, c), obj in cache.items():
        if i == isl and o == osl:
            qps = obj.get("request_throughput")
            if isinstance(qps, (int, float)):
                rows.append((c, float(qps)))
                if best_qps is None or float(qps) > best_qps:
                    best_qps, best_con = float(qps), c
    return best_con, best_qps


def maybe_log_top3(
    cache: Dict[ResultKey, Dict],
    isl: int,
    osl: int,
    log_file: str,
    ttft_ms_max: Optional[float] = None,
    tpot_ms_max: Optional[float] = None,
) -> None:
    import pandas as pd  # type: ignore
    # Unconstrained top3 (for reference)
    data_all = []
    # SLA-constrained top3
    data_sla = []
    for (i, o, c), obj in cache.items():
        if i != isl or o != osl:
            continue
        qps = obj.get("request_throughput")
        if isinstance(qps, (int, float)):
            data_all.append({"con": c, "qps": float(qps)})
            if is_sla_compliant(obj, ttft_ms_max, tpot_ms_max)[0]:
                data_sla.append({"con": c, "qps": float(qps)})
    if data_all:
        df_all = pd.DataFrame(data_all).sort_values("qps", ascending=False)
        top_all = df_all.head(3).to_dict(orient="records")
        log_both(log_file, f"[TOP3_ALL] {top_all}")
    if data_sla:
        df_sla = pd.DataFrame(data_sla).sort_values("qps", ascending=False)
        top_sla = df_sla.head(3).to_dict(orient="records")
        log_both(log_file, f"[TOP3_SLA] {top_sla}")


def ensure_dirs(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def build_cmd_str(
    backend: str,
    host: str,
    port: int,
    model: str,
    dataset_name: str,
    isl: int,
    osl: int,
    con: int,
    con_times: int,
    tee_log: str,
    bench_output_jsonl: str,
    sla_path: str,
    dataset_path: str,
) -> str:
    # Build env exports for sla.sh. We suppress child output per sla.md
    # by redirecting stdout/stderr to /dev/null at the outer level.
    log_base = tee_log[:-4] if tee_log.endswith(".log") else tee_log
    env_parts: List[str] = []
    env_parts.append(f"LOG={shlex.quote(log_base)}")
    env_parts.append(f"MODEL_PATH={shlex.quote(model)}")
    env_parts.append(f"HOST={shlex.quote(host)}")
    env_parts.append(f"PORT={shlex.quote(str(port))}")
    env_parts.append(f"DATASET_NAME={shlex.quote(dataset_name)}")
    if dataset_name == "random":
        env_parts.append(f"RANDOM_INPUT_LEN={isl}")
        env_parts.append(f"RANDOM_OUTPUT_LEN={osl}")
    env_parts.append(f"RESULT_JSONL={shlex.quote(bench_output_jsonl)}")
    env_parts.append(f"MAX_CONCURRENCY={con}")
    # default  10 times concurrency
    env_parts.append(f"NUM_PROMPT={con * con_times}")
    # Optional dataset CSV if present
    if os.path.exists(dataset_path):
        env_parts.append(f"DATASET_PATH={shlex.quote(dataset_path)}")
    # NUM_PROMPT can be kept small; default is 16 in sla.sh
    env_prefix = " ".join(env_parts)
    cmd = (
        f"bash -lc \"set -o pipefail; {env_prefix} /bin/bash {shlex.quote(sla_path)} >/dev/null 2>&1\""
    )
    return cmd


def run_once(
    isl: int,
    osl: int,
    con: int,
    con_times: int,
    cache: Dict[ResultKey, Dict],
    result_jsonl: str,
    log_file: str,
    bench_output_jsonl: str,
    dry_run: bool = False,
    print_cmd: bool = False,
    backend: str = "vllm",
    host: str = "127.0.0.1",
    port: int = 9527,
    model: str = "",
    dataset_name: str = "burstgpt",
    sla_path: str = "",
    dataset_path: str = "",
) -> Tuple[Optional[Dict], Optional[str]]:
    key = (isl, osl, con)
    if key in cache:
        return cache[key], None
    if dry_run:
        return None, "DRY_RUN mode enabled"

    # If this concurrency already exists in the output jsonl, reuse it to avoid duplicates
    existing = None
    try:
        existing = None
        if os.path.exists(result_jsonl):
            with open(result_jsonl, "r", encoding="utf-8") as f:
                for line in reversed(f.readlines()):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    con_val = obj.get("max_concurrency")
                    if con_val is None:
                        con_val = obj.get("MAX_CONCURRENCY")
                    if isinstance(con_val, (int, float)) and int(con_val) == int(con):
                        existing = obj
                        break
    except Exception as e:
        existing = None
    if existing is not None:
        cache[key] = existing
        return existing, None

    cmd_str = build_cmd_str(
        backend,
        host,
        port,
        model,
        dataset_name,
        isl,
        osl,
        con,
        con_times,
        log_file,
        bench_output_jsonl,
        sla_path,
        dataset_path,
    )

    ensure_dirs(log_file)
    ensure_dirs(bench_output_jsonl)

    if dataset_name == "burstgpt" and isl == 0 and osl == 0:
        dataset_path = dataset_path if os.path.exists(dataset_path) else "BurstGPT dataset"
        log_both(log_file, f"[RUNNING] {dataset_path} con {con}")
    else:
        log_both(log_file, f"[RUNNING] isl {isl} osl {osl} con {con}")
    if print_cmd:
        log_both(log_file, cmd_str)
    rc = os.system(cmd_str)
    if rc != 0:
        return None, f"Benchmark command failed with exit code {rc}"

    def scan_file(path: str) -> Tuple[Optional[Dict], Optional[str]]:
        candidate: Optional[Dict] = None
        try:
            with open(path, "r", encoding="utf-8") as f:
                # Read from the end for efficiency since we want the latest result
                lines = f.readlines()
                for line in reversed(lines):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        obj = json.loads(line)
                        # Prefer matching the concurrency, regardless of isl/osl fields
                        con_val = obj.get("max_concurrency")
                        if con_val is None:
                            con_val = obj.get("MAX_CONCURRENCY")
                        try:
                            if int(con_val) == int(con):
                                return obj, None
                        except Exception:
                            pass
                        # Track last valid JSON as fallback
                        candidate = candidate or obj
                    except (json.JSONDecodeError, ValueError, TypeError) as e:
                        continue
        except (FileNotFoundError, PermissionError, OSError) as e:
            return None, f"Failed to read output file {path}: {e}"
        if candidate is None:
            return None, f"No valid JSON lines found in {path}"
        return candidate, None

    # Source of truth is the bench JSONL
    last_json, scan_error = scan_file(bench_output_jsonl)
    if last_json is None:
        # fallback to log only if bench file not updated
        last_json, scan_error = scan_file(log_file)
    if last_json is None:
        return None, f"No JSON result captured: {scan_error}"

    # Update in-memory cache only; do not append (bench already wrote it)
    cache[key] = last_json
    return last_json, None


def get_qps(result: Optional[Dict]) -> Tuple[Optional[float], Optional[str]]:
    if not result:
        return None, "Result dictionary is None"
    qps = result.get("request_throughput")
    if qps is None:
        return None, "request_throughput field missing from result"
    if not isinstance(qps, (int, float)):
        return None, f"request_throughput has invalid type: {type(qps)} (value: {qps})"
    return float(qps), None


def is_sla_compliant(
    obj: Optional[Dict],
    ttft_ms_max: Optional[float],
    tpot_ms_max: Optional[float],
) -> Tuple[bool, Optional[str]]:
    """Return True only if the record satisfies all provided SLA constraints.

    Strict mode: when a constraint is provided, the corresponding metric must
    exist and be below the threshold.

    Returns:
        (is_compliant, failure_reason)
    """
    if obj is None:
        return False, "Result is None"

    if ttft_ms_max is not None:
        mean_ttft = obj.get("mean_ttft_ms")
        if not isinstance(mean_ttft, (int, float)):
            return False, f"TTFT metric missing or invalid: {mean_ttft}"
        if float(mean_ttft) >= ttft_ms_max:
            return False, f"TTFT violation: {mean_ttft:.1f}ms >= {ttft_ms_max}ms"

    if tpot_ms_max is not None:
        mean_tpot = obj.get("mean_tpot_ms")
        if not isinstance(mean_tpot, (int, float)):
            return False, f"TPOT metric missing or invalid: {mean_tpot}"
        if float(mean_tpot) >= tpot_ms_max:
            return False, f"TPOT violation: {mean_tpot:.1f}ms >= {tpot_ms_max}ms"

    return True, None


def measure_or_get_qps(
    isl: int,
    osl: int,
    con: int,
    con_times: int,
    cache: Dict[ResultKey, Dict],
    result_jsonl: str,
    log_file: str,
    bench_output_jsonl: str,
    ttft_ms_max: Optional[float] = None,
    tpot_ms_max: Optional[float] = None,
    dry_run: bool = False,
    print_cmd: bool = False,
    backend: str = "vllm",
    host: str = "127.0.0.1",
    port: int = 9527,
    model: str = "",
    dataset_name: str = "burstgpt",
    sla_path: str = "",
    dataset_path: str = "",
) -> Tuple[Optional[float], Optional[str]]:
    res, run_error = run_once(isl, osl, con, con_times, cache, result_jsonl, log_file, bench_output_jsonl,
                              dry_run, print_cmd, backend, host, port, model, dataset_name,
                              sla_path, dataset_path)
    if res is None:
        return None, f"run_once failed: {run_error}"

    compliant, sla_error = is_sla_compliant(res, ttft_ms_max, tpot_ms_max)
    if not compliant:
        return None, f"SLA constraint violation: {sla_error}"

    qps, qps_error = get_qps(res)
    if qps is None:
        return None, f"QPS extraction failed: {qps_error}"

    return qps, None


def mandatory_phase(
    isl: int,
    osl: int,
    con_times: int,
    mandatory: List[int],
    cache: Dict[ResultKey, Dict],
    result_jsonl: str,
    log_file: str,
    bench_output_jsonl: str,
    ttft_ms_max: Optional[float] = None,
    tpot_ms_max: Optional[float] = None,
    dry_run: bool = False,
    print_cmd: bool = False,
    backend: str = "vllm",
    host: str = "127.0.0.1",
    port: int = 9527,
    model: str = "",
    dataset_name: str = "burstgpt",
    sla_path: str = "",
    dataset_path: str = "",
) -> Tuple[int, Optional[float]]:
    best_con = -1
    best_qps: Optional[float] = None
    for con in mandatory:
        qps, error = measure_or_get_qps(
            isl,
            osl,
            con,
            con_times,
            cache,
            result_jsonl,
            log_file,
            bench_output_jsonl,
            ttft_ms_max=ttft_ms_max,
            tpot_ms_max=tpot_ms_max,
            dry_run=dry_run,
            print_cmd=print_cmd,
            backend=backend,
            host=host,
            port=port,
            model=model,
            dataset_name=dataset_name,
            sla_path=sla_path,
            dataset_path=dataset_path,
        )
        if qps is None:
            log_both(log_file, f"[MANDATORY] Failed run at con={con}: {error}")
            continue
        if best_qps is None or qps > best_qps:
            best_qps = qps
            best_con = con
    return best_con, best_qps


def peak_search(
    isl: int,
    osl: int,
    con_times: int,
    max_concurrency_limit: int,
    cache: Dict[ResultKey, Dict],
    result_jsonl: str,
    log_file: str,
    bench_output_jsonl: str,
    ttft_ms_max: Optional[float] = None,
    tpot_ms_max: Optional[float] = None,
    dry_run: bool = False,
    print_cmd: bool = False,
    backend: str = "vllm",
    host: str = "127.0.0.1",
    port: int = 9527,
    model: str = "",
    dataset_name: str = "burstgpt",
    find_precise_bs: bool = False,
    sla_path: str = "",  # 新增参数
    dataset_path: str = "",  # 新增参数
) -> Tuple[int, Optional[float]]:
    """
    Fine-grained search based on clinet.md requirements:
    1. Find current best concurrency from existing data
    2. Find con_l = max concurrency < best_con from existing data
    3. Find con_r = min concurrency > best_con from existing data (or max_concurrency_limit)
    4. Binary search between con_l and con_r
    5. Repeat until (con_r - con_l) <= max(con_l/8, 2)
    """
    # Get all measured concurrency values for this combo (for search space)
    measured = sorted([c for (i, o, c) in cache.keys() if i == isl and o == osl])
    
    # Find current best concurrency from existing SLA-compliant data
    best_con = -1
    best_qps: Optional[float] = None
    for c in measured:
        obj = cache.get((isl, osl, c))
        if not obj:
            continue
        compliant, _ = is_sla_compliant(obj, ttft_ms_max, tpot_ms_max)
        if not compliant:
            continue
        q = obj.get("request_throughput")
        if isinstance(q, (int, float)):
            if best_qps is None or float(q) > best_qps:
                best_qps, best_con = float(q), c

    if best_con == -1 or best_qps is None:
        log_both(log_file, "[SEARCH] No SLA-compliant data yet; starting with l=1 and SLA-aware r")
    else:
        log_both(log_file, f"[SEARCH] Initial best: con={best_con} qps={best_qps}")

    while True:
        # Recompute measured sets from cache on each iteration
        measured = sorted([c for (i, o, c) in cache.keys() if i == isl and o == osl])
        measured_sla = sorted([
            c for c in measured
            if is_sla_compliant(cache.get((isl, osl, c)), ttft_ms_max, tpot_ms_max)[0]
        ])

        # Reference point for boundaries: current best if exists, else 1
        best_con_ref = best_con if best_con != -1 else 1

        # SLA-based left boundary: max SLA-compliant concurrency < best_con_ref; else 1
        con_l_candidates = [c for c in measured_sla if c < best_con_ref]
        con_l = max(con_l_candidates) if con_l_candidates else 1

        # SLA-based right boundary: min SLA-compliant concurrency > best_con_ref
        con_r_candidates_sla = [c for c in measured_sla if c > best_con_ref]
        if con_r_candidates_sla:
            con_r = min(con_r_candidates_sla)
        else:
            # Fallback: minimal non-SLA measured concurrency > best_con_ref; else max limit
            con_r_candidates_all = [c for c in measured if c > best_con_ref]
            con_r = min(con_r_candidates_all) if con_r_candidates_all else max_concurrency_limit
        
        log_both(log_file, f"[SEARCH] con_l={con_l} best_con={best_con} con_r={con_r}")
        
        threshold = max(con_l // 8, 1)
        if find_precise_bs:
            threshold = 1
        if (con_r - con_l) <= threshold:
            log_both(log_file, f"[SEARCH] Termination condition met: ({con_r} - {con_l}) <= max({con_l}//8, 1) = {threshold}")
            break
            
        # Find midpoint for next test
        mid = (con_l + con_r) // 2
        
        # If mid equals existing values, adjust to avoid duplication
        if mid == con_l or mid == con_r or mid == best_con:
            # Try to find a gap
            potential_mids = []
            for test_mid in range(con_l + 1, con_r):
                if test_mid not in measured and test_mid != best_con:
                    potential_mids.append(test_mid)
            
            if not potential_mids:
                log_both(log_file, f"[SEARCH] No untested values between {con_l} and {con_r}")
                break
                
            mid = min(potential_mids, key=lambda x: abs(x - (con_l + con_r) // 2))
        
        if mid in measured:
            # Already tested, skip
            measured_set = set(measured)
            for test_mid in range(con_l + 1, con_r):
                if test_mid not in measured_set:
                    mid = test_mid
                    break
            else:
                log_both(log_file, f"[SEARCH] All values between {con_l} and {con_r} already tested")
                break
                
        log_both(log_file, f"[SEARCH] Testing mid={mid}")

        # Measure QPS at midpoint
        q_mid, error = measure_or_get_qps(
            isl,
            osl,
            mid,
            con_times,
            cache,
            result_jsonl,
            log_file,
            bench_output_jsonl,
            ttft_ms_max=ttft_ms_max,
            tpot_ms_max=tpot_ms_max,
            dry_run=dry_run,
            print_cmd=print_cmd,
            backend=backend,
            host=host,
            port=port,
            model=model,
            dataset_name=dataset_name,
            sla_path=sla_path,
            dataset_path=dataset_path,
        )
        measured = sorted(list(set(measured + [mid])))

        if q_mid is None:
            log_both(log_file, f"[SEARCH] Failed con={mid}: {error}")
            continue
            
        log_both(log_file, f"[SEARCH] mid={mid} qps={q_mid}")
        
        # Update best if improved (guard when best_qps is None)
        if best_qps is None or q_mid > best_qps:
            best_qps = q_mid
            best_con = mid
            log_both(log_file, f"[SEARCH] New best: con={best_con} qps={best_qps}")

    log_both(log_file, f"[SEARCH] Final result: con={best_con} qps={best_qps}")
    return best_con, best_qps

def parse_combo(s: str):
    return tuple(map(int, s.split("/")))


def main() -> None:
    args = parse_arguments()

    model_name = args.model_name or os.path.basename(os.path.normpath(args.model)).replace(" ", "_")
    # Unify all read/write to dataset_jsonl
    dataset_jsonl = args.dataset_jsonl or f"{model_name}_{args.dataset_name}_{datetime.now().strftime('%y%m%d')}.jsonl"
    bench_output_jsonl = args.bench_output_jsonl or dataset_jsonl
    result_jsonl = args.result_jsonl or dataset_jsonl
    log_file = args.log_file or f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.log"

    # Ensure pandas is available (used for optional top3 logging)
    ensure_pandas_installed(log_file)

    # Warm cache from the unified dataset jsonl
    cache: Dict[ResultKey, Dict] = load_cache(result_jsonl)

    log_both(log_file, f"Using jsonl: {result_jsonl}")
    log_both(log_file, f"Using log file: {log_file}")

    overall_best: List[Tuple[Tuple[int, int], int, Optional[float]]] = []

    if args.dataset_name == "burstgpt":
        # burstGPT mode: no isl/osl combinations, just test concurrency directly
        isl, osl = 0, 0  # Use (0,0) as placeholder for burstGPT mode
        log_both(log_file, f"\n=== BurstGPT Mode ===")

        mand_best_con, mand_best_qps = mandatory_phase(
            isl,
            osl,
            args.con_times,
            args.mandatory,
            cache,
            result_jsonl,
            log_file,
            bench_output_jsonl,
            ttft_ms_max=args.ttft_ms_max,
            tpot_ms_max=args.ttft_ms_max,
            dry_run=args.dry_run,
            print_cmd=args.print_cmd,
            backend=args.backend,
            host=args.host,
            port=args.port,
            model=args.model,
            dataset_name=args.dataset_name,
            sla_path=args.sla_path,
            dataset_path=args.dataset_path,
        )
        log_both(log_file, f"Mandatory best: con={mand_best_con} qps={mand_best_qps}")

        # Perform fine-grained search based on mandatory phase results
        peak_con, peak_qps = peak_search(
            isl,
            osl,
            args.con_times,
            args.max_concurrency_limit,
            cache,
            result_jsonl,
            log_file,
            bench_output_jsonl,
            ttft_ms_max=args.ttft_ms_max,
            tpot_ms_max=args.ttft_ms_max,
            dry_run=args.dry_run,
            print_cmd=args.print_cmd,
            backend=args.backend,
            host=args.host,
            port=args.port,
            model=args.model,
            dataset_name=args.dataset_name, 
            find_precise_bs=args.find_precise_bs,
            sla_path=args.sla_path,
            dataset_path=args.dataset_path,
        )
        if peak_con != -1:
            log_both(log_file, f"Peak search best: con={peak_con} qps={peak_qps}")
        else:
            log_both(log_file, "Peak search completed without finding better results")

        # Pick final best under SLA constraints from ALL cached data points
        cache_best_con, cache_best_qps = pick_best_sla_qps(
            cache,
            isl=isl,
            osl=osl,
            ttft_ms_max=args.ttft_ms_max,
            tpot_ms_max=args.ttft_ms_max,
            qps_field=args.qps_field
        )
        maybe_log_top3(cache, isl, osl, log_file, ttft_ms_max=args.ttft_ms_max, tpot_ms_max=args.ttft_ms_max)

        final_candidates: List[Tuple[int, Optional[float]]] = []
        if mand_best_con != -1:
            final_candidates.append((mand_best_con, mand_best_qps))
        if peak_con != -1:
            final_candidates.append((peak_con, peak_qps))
        if cache_best_con != -1:
            final_candidates.append((cache_best_con, cache_best_qps))

        best_con = -1
        best_qps: Optional[float] = None
        for con, qps in final_candidates:
            if qps is None:
                continue
            # Ensure the selected point is SLA-compliant
            obj = cache.get((isl, osl, con))
            if not is_sla_compliant(obj, args.ttft_ms_max, args.ttft_ms_max)[0]:
                continue
            if best_qps is None or qps > best_qps:
                best_qps = qps
                best_con = con
        if best_con == -1 or best_qps is None:
            log_both(log_file, "Final best for BurstGPT mode: no SLA-compliant result found")
        else:
            log_both(log_file, f"Final best for BurstGPT mode: con={best_con} qps={best_qps}")
        overall_best.append(((isl, osl), best_con, best_qps))

    else:
        # random mode: use isl/osl combinations
        for combo in args.combinations:
            isl, osl = parse_combo(combo)
            log_both(log_file, f"\n=== Combo isl={isl} osl={osl} ===")

            mand_best_con, mand_best_qps = mandatory_phase(
                isl,
                osl,
                args.con_times,
                args.mandatory,
                cache,
                result_jsonl,
                log_file,
                bench_output_jsonl,
                ttft_ms_max=args.ttft_ms_max,
                tpot_ms_max=args.ttft_ms_max,
                dry_run=args.dry_run,
                print_cmd=args.print_cmd,
                backend=args.backend,
                host=args.host,
                port=args.port,
                model=args.model,
                dataset_name=args.dataset_name,
                sla_path=args.sla_path,
                dataset_path=args.dataset_path,
            )
            log_both(log_file, f"Mandatory best: con={mand_best_con} qps={mand_best_qps}")

            # Perform fine-grained search based on mandatory phase results
            peak_con, peak_qps = peak_search(
                isl,
                osl,
                args.con_times,
                args.max_concurrency_limit,
                cache,
                result_jsonl,
                log_file,
                bench_output_jsonl,
                ttft_ms_max=args.ttft_ms_max,
                tpot_ms_max=args.ttft_ms_max,
                dry_run=args.dry_run,
                print_cmd=args.print_cmd,
                backend=args.backend,
                host=args.host,
                port=args.port,
                model=args.model,
                dataset_name=args.dataset_name, 
                find_precise_bs=args.find_precise_bs,
                sla_path=args.sla_path,
                dataset_path=args.dataset_path,
            )
            if peak_con != -1:
                log_both(log_file, f"Peak search best: con={peak_con} qps={peak_qps}")
            else:
                log_both(log_file, "Peak search completed without finding better results")

            # Pick final best under SLA constraints from ALL cached data points for this combo
            cache_best_con, cache_best_qps = pick_best_sla_qps(
                cache,
                isl=isl,
                osl=osl,
                ttft_ms_max=args.ttft_ms_max,
                tpot_ms_max=args.ttft_ms_max,
                qps_field=args.qps_field
            )
            maybe_log_top3(cache, isl, osl, log_file, ttft_ms_max=args.ttft_ms_max, tpot_ms_max=args.ttft_ms_max)

            final_candidates: List[Tuple[int, Optional[float]]] = []
            if mand_best_con != -1:
                final_candidates.append((mand_best_con, mand_best_qps))
            if peak_con != -1:
                final_candidates.append((peak_con, peak_qps))
            if cache_best_con != -1:
                final_candidates.append((cache_best_con, cache_best_qps))

            best_con = -1
            best_qps: Optional[float] = None
            for con, qps in final_candidates:
                if qps is None:
                    continue
                # Ensure the selected point is SLA-compliant
                obj = cache.get((isl, osl, con))
                if not is_sla_compliant(obj, args.ttft_ms_max, args.ttft_ms_max)[0]:
                    continue
                if best_qps is None or qps > best_qps:
                    best_qps = qps
                    best_con = con
            if best_con == -1 or best_qps is None:
                log_both(log_file, f"Final best for isl={isl} osl={osl}: no SLA-compliant result found")
            else:
                log_both(log_file, f"Final best for isl={isl} osl={osl}: con={best_con} qps={best_qps}")
            overall_best.append(((isl, osl), best_con, best_qps))

    log_both(log_file, "\n=== Summary ===")
    if args.dataset_name == "burstGPT":
        for (isl, osl), con, qps in overall_best:
            if con == -1 or qps is None:
                log_both(log_file, "BurstGPT mode -> no SLA-compliant result found")
            else:
                log_both(log_file, f"BurstGPT mode -> best_con={con} best_qps={qps}")
    else:
        for (isl, osl), con, qps in overall_best:
            log_both(log_file, f"isl={isl} osl={osl} -> best_con={con} best_qps={qps}")


if __name__ == "__main__":
    main()


