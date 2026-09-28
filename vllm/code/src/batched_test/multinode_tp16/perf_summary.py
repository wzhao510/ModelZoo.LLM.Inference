#!/usr/bin/env python3
"""性能压测结果汇总(每次跑完性能总结一次, 供版本升级时做性能看护)。

背景: 多机(tp16/tp32)和单机的性能压测结果散落在各次 run 目录里的 vllm bench JSON
里, 版本升级时没有一张能直接对比的表。本脚本把这些 JSON 解析成固定表头的一行,
追加到同一个 CSV(默认取环境变量 PERF_CSV, 否则用当前目录下的 perf_summary.csv),
于是每跑一次性能就多一行记录, 不同版本/不同日期之间可以直接按
"模型 + 压测参数"对比吞吐和时延。

一行 = 一次 vllm bench 的结果(一个 JSON 文件); 去重主键是
(run_id, model, bench 时间戳, 并发, 条数), 所以同一个 run 目录重复扫描(逐模型汇总一次、
整轮再扫一次)不会写重复行。

用法(一般由 luwu_master.sh / luwu_single.sh / perf_summary.sh 调用):
    python perf_summary.py --scan-dir /sw_home/lli/model_test/tp16_luwu/run_20260921_101010 \
        --csv /sw_home/lli/model_test/tp16_luwu/perf_summary.csv \
        --run-id run_20260921_101010 \
        --model DeepSeek-R1-0528-W8A8 --tp 16 --dp 1 --pp 1 --nodes 2 --gpus-per-node 8 \
        --report

    # 只回看历史(不扫描新目录)
    python perf_summary.py --csv .../perf_summary.csv --show

--report 会打印本次 run 的汇总表, 并在"同一模型 + 同一压测参数 + 同一并发/条数"时
自动带上和上一次 run 的对比(吞吐涨跌百分比), 这就是版本升级时的性能看护视图。
单机(sweep/合并模式)的目录名自带 tp/pp/dp 时, 会自动补 nodes=1 与 gpus_per_node = tp*pp*dp,
所以单机行和多机行在同一张表里能直接对比。
如果扫描目录里有"起了服务/进了压测、但没写进 CSV"的模型(起服务失败/超时/压测失败),
会打印一行 WARN 把模型名点出来, 避免"perf 只有部分模型"这种事后才发现。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
import sys
from datetime import datetime
from importlib.metadata import version as _pkg_version

# CSV 表头(固定顺序): 前面是"这次跑的是什么", 中间是压测结果, 后面是环境(版本)信息
FIELDS = [
    "run_id",              # 本次申请/运行的 run 目录名(run_<时间戳>)
    "scan_time",           # 本行写入 CSV 的时间
    "bench_date",          # vllm bench 自己记录的时间(JSON 里的 date)
    "model",               # 模型清单里的模型名
    "model_id",            # vllm bench 记录的 model_id(通常是模型路径)
    "tp",
    "dp",
    "pp",
    "nodes",
    "gpus_per_node",
    "num_prompts",
    "max_concurrency",
    "in_len_avg",          # 平均输入长度(total_input_tokens/completed)
    "out_len_avg",         # 平均输出长度(total_output_tokens/completed)
    "completed",
    "failed",
    "duration_s",
    "request_throughput",  # req/s
    "output_throughput",   # 输出 token/s(最常用的性能看护指标)
    "total_token_throughput",
    "max_output_tokens_per_s",
    "max_concurrent_requests",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "p99_e2el_ms",
    "vllm_version",
    "vllm_metax_version",
    "torch_version",
    "maca_version",
    "image",
    "host",
    "result_json",         # 该行来源的 JSON(相对扫描目录, 便于回查原始结果)
]

# <model>_tp16_pp1_dp1 这类目录名: 单机 launch.py --perf 的 sweep 产物按并行度分目录,
# 多机 bench.sh 的产物在 <run 目录>/<模型名> 下, 所以这里只是"能推出就用"的兜底
_PARALLEL_DIR_RE = re.compile(r"^(?P<model>.+)_tp(?P<tp>\d+)_pp(?P<pp>\d+)_dp(?P<dp>\d+)$")
# 单机精度侧日志名: <model>[tpXppYdpZ]_serve.log(用来发现"该跑性能但没跑成的模型")
_MODEL_TAG_LOG_RE = re.compile(r"^(?P<model>.+)\[tp\d+pp\d+dp\d+\]_serve\.log$")


def _repo_version(*names: str) -> str:
    for name in names:
        try:
            return str(_pkg_version(name))
        except Exception:  # PackageNotFoundError / 元数据读取失败都当没有
            continue
    return ""


def _maca_version() -> str:
    for path in ("/opt/maca/Version.txt", "/opt/maca/version.txt"):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    return line.split(":", 1)[-1].strip() if ":" in line else line
        except OSError:
            continue
    return os.environ.get("MACA_VERSION", "")


def _image_name() -> str:
    for key in ("LUWU_IMAGE", "IMAGE_NAME", "IMAGE", "CONTAINER_IMAGE", "VLLM_IMAGE"):
        val = os.environ.get(key)
        if val:
            return val
    return ""


def _env_info() -> dict:
    return {
        "vllm_version": _repo_version("vllm"),
        "vllm_metax_version": _repo_version("vllm-metax", "vllm_metax"),
        "torch_version": _repo_version("torch"),
        "maca_version": _maca_version(),
        "image": _image_name(),
        "host": socket.gethostname(),
    }


def _is_bench_result(data: object) -> bool:
    """是否是 vllm bench serve 的结果 JSON(排除 sweep 的 summary.json/列表与 pytorch 格式)。"""
    if not isinstance(data, dict):
        return False
    if "completed" not in data:
        return False
    if not any(k in data for k in ("output_throughput", "total_token_throughput")):
        return False
    return any(k in data for k in ("model_id", "model", "num_prompts"))


def _num(value, digits: int = 2):
    if value is None or value == "":
        return ""
    try:
        number = round(float(value), digits)
    except (TypeError, ValueError):
        return value
    # 并发/条数这类整数列写成 "32" 而不是 "32.0", 方便人看也方便下游按整数处理
    if digits == 0 and float(number).is_integer():
        return int(number)
    return number


def _ratio(total, completed, digits: int = 1):
    try:
        completed = float(completed)
        if completed <= 0:
            return ""
        return round(float(total) / completed, digits)
    except (TypeError, ValueError):
        return ""


def _path_meta(path: str, scan_root: str) -> dict:
    meta = {"model": "", "tp": "", "dp": "", "pp": ""}
    for part in reversed(os.path.dirname(path).split(os.sep)):
        match = _PARALLEL_DIR_RE.match(part)
        if match:
            meta.update(
                model=match.group("model"),
                tp=match.group("tp"),
                pp=match.group("pp"),
                dp=match.group("dp"),
            )
            break
    try:
        meta["rel_json"] = os.path.relpath(path, scan_root)
    except ValueError:
        meta["rel_json"] = path
    return meta


def _build_row(path: str, scan_root: str, data: dict, args, env_info: dict) -> dict:
    path_meta = _path_meta(path, scan_root)
    model = args.model or path_meta["model"]
    model_id = data.get("model_id") or data.get("model") or ""
    if not model and model_id:
        model = os.path.basename(str(model_id).rstrip("/"))
    completed = data.get("completed")
    # 单机(sweep/合并模式)的目录名自带 tp/pp/dp, 而且模型全部在一台机器上:
    # 补齐 nodes/gpus_per_node, 让单机行和多机行在表里长得一样
    nodes = args.nodes or ""
    gpus_per_node = args.gpus_per_node or ""
    if not nodes and path_meta["tp"]:
        nodes = "1"
        if not gpus_per_node:
            try:
                gpus_per_node = str(
                    int(path_meta["tp"]) * int(path_meta["pp"]) * int(path_meta["dp"])
                )
            except ValueError:
                gpus_per_node = ""
    row = {
        "run_id": args.run_id or "",
        "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "bench_date": data.get("date", ""),
        "model": model,
        "model_id": model_id,
        "tp": args.tp or path_meta["tp"],
        "dp": args.dp or path_meta["dp"],
        "pp": args.pp or path_meta["pp"],
        "nodes": nodes,
        "gpus_per_node": gpus_per_node,
        "num_prompts": _num(data.get("num_prompts"), 0),
        "max_concurrency": _num(data.get("max_concurrency"), 0),
        "in_len_avg": _ratio(data.get("total_input_tokens"), completed),
        "out_len_avg": _ratio(data.get("total_output_tokens"), completed),
        "completed": _num(completed, 0),
        "failed": _num(data.get("failed"), 0),
        "duration_s": _num(data.get("duration")),
        "request_throughput": _num(data.get("request_throughput")),
        "output_throughput": _num(data.get("output_throughput")),
        "total_token_throughput": _num(data.get("total_token_throughput")),
        "max_output_tokens_per_s": _num(data.get("max_output_tokens_per_s")),
        "max_concurrent_requests": _num(data.get("max_concurrent_requests"), 0),
        "mean_ttft_ms": _num(data.get("mean_ttft_ms")),
        "p99_ttft_ms": _num(data.get("p99_ttft_ms")),
        "mean_tpot_ms": _num(data.get("mean_tpot_ms")),
        "p99_tpot_ms": _num(data.get("p99_tpot_ms")),
        "mean_itl_ms": _num(data.get("mean_itl_ms")),
        "p99_itl_ms": _num(data.get("p99_itl_ms")),
        "mean_e2el_ms": _num(data.get("mean_e2el_ms")),
        "p99_e2el_ms": _num(data.get("p99_e2el_ms")),
        "result_json": path_meta["rel_json"],
    }
    row.update(env_info)
    return row


def _row_key(row: dict) -> tuple:
    """去重主键: 同一个 run 里同一次 bench 只记一行(按 bench 时间戳+并发+条数)。

    刻意不用文件路径: 逐模型汇总时扫的是 <run>/<模型> 目录, 整轮汇总时扫的是 <run> 目录,
    同一个 JSON 的相对路径不一样; 也不带 model: 逐模型汇总时模型名由 --model 给,
    整轮汇总时只能从路径/JSON 推断, 两者可能不同。用 "时间戳+并发+条数" 做主键,
    重复扫描(含换扫描根)都不会写成两行。
    """
    stamp = str(row.get("bench_date", "") or "")
    if not stamp:  # 极少数 JSON 没有 date 时, 用路径尾两段兜底
        parts = [p for p in str(row.get("result_json", "")).replace("\\", "/").split("/") if p]
        stamp = "/".join(parts[-2:])
    return (
        str(row.get("run_id", "")),
        stamp,
        str(row.get("max_concurrency", "")),
        str(row.get("num_prompts", "")),
    )


def _read_rows(csv_path: str) -> list:
    if not os.path.exists(csv_path):
        return []
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]
    except OSError as exc:
        print(f"[perf-summary] WARN 读取已有 CSV 失败({exc}), 本次按新表处理", file=sys.stderr)
        return []


def _append_rows(csv_path: str, rows: list, has_header: bool) -> None:
    if not rows:
        return
    parent = os.path.dirname(os.path.abspath(csv_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if not has_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDS})


def _write_run_csv(csv_path: str, rows: list, run_id: str) -> str:
    """把本次 run 的行单独留一份在 run 目录里(CSV 主文件是跨 run 累积的)。"""
    selected = [row for row in rows if not run_id or str(row.get("run_id", "")) == run_id]
    if not selected:
        return ""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in selected:
            writer.writerow({key: row.get(key, "") for key in FIELDS})
    return csv_path


def _expected_models(scan_root: str) -> set:
    """这次 run 里"本该有性能数据"的模型(单机/合并模式的目录结构)。

    单机一件事就是: 起了服务(<inference>/<model>[tpXppYdpZ]_serve.log) 或者进了压测
    (<performance>/<model>_tpX_ppY_dpZ/)。把它们和真正写进 CSV 的模型对一下, 就能直接
    在日志里指出"哪些模型这次没有性能数据", 不用人工翻目录。
    """
    found: set = set()
    skipped: set = set()
    for dirpath, dirnames, filenames in os.walk(scan_root):
        base = os.path.basename(dirpath)
        if base == "performance":
            for name in dirnames:
                match = _PARALLEL_DIR_RE.match(name)
                if match:
                    found.add(match.group("model"))
            # harness 的 bench_tasks_result.csv 里 status=skip 的模型(清单里没配
            # benchmark、只校验精度的模型)不算"缺性能数据"。
            status_csv = os.path.join(dirpath, "bench_tasks_result.csv")
            if os.path.exists(status_csv):
                try:
                    with open(status_csv, newline="", encoding="utf-8") as fh:
                        for row in csv.DictReader(fh):
                            if str(row.get("status", "")).strip().lower() != "skip":
                                continue
                            match = _PARALLEL_DIR_RE.match(str(row.get("task_name", "")))
                            if match:
                                skipped.add(match.group("model"))
                except OSError:
                    pass
        elif base == "inference":
            for name in filenames:
                match = _MODEL_TAG_LOG_RE.match(name)
                if match:
                    found.add(match.group("model"))
    return found - skipped


def _cell(value) -> str:
    return "" if value is None else str(value)


def _delta(cur, prev) -> str:
    if cur == "" or prev in ("", None):
        return ""
    try:
        cur_f, prev_f = float(cur), float(prev)
    except (TypeError, ValueError):
        return ""
    if prev_f == 0:
        return ""
    return f"{(cur_f - prev_f) / prev_f * 100:+.1f}%"


def _print_report(rows: list, run_id: str, csv_path: str) -> None:
    """打印汇总表: 吞吐/时延一目了然, 有上一次 run 的数据时带上涨跌。"""
    if not rows:
        print("[perf-summary] 没有可打印的性能记录(先跑一次性能, 或检查 --scan-dir)")
        return

    ordered = sorted(rows, key=lambda r: (str(r.get("bench_date", "")), str(r.get("run_id", ""))))
    previous: dict = {}
    for row in ordered:
        key = (
            str(row.get("model", "")),
            str(row.get("max_concurrency", "")),
            str(row.get("num_prompts", "")),
        )
        row["_prev"] = previous.get(key)
        previous[key] = row

    selected = ordered
    if run_id:
        filtered = [r for r in ordered if str(r.get("run_id", "")) == run_id]
        if filtered:
            selected = filtered
        else:
            print(f"[perf-summary] 本次 run({run_id})在 CSV 里没有记录, 改为打印全部历史")

    header = [
        "run_id", "model", "nodes", "tp/dp/pp", "conc", "prompts", "in/out",
        "output tok/s", "总 tok/s", "mean TTFT", "p99 TTFT", "mean TPOT", "p99 E2EL", "date",
    ]
    table = [
        f"| {' | '.join(header)} |",
        f"|{'---|' * len(header)}",
    ]
    for row in selected:
        prev = row.get("_prev") or {}
        tp, dp, pp = row.get("tp", ""), row.get("dp", ""), row.get("pp", "")
        parallel = f"{tp}/{dp}/{pp}".strip("/") if (tp or dp or pp) else "-"
        out_tp = row.get("output_throughput", "")
        total_tp = row.get("total_token_throughput", "")
        out_delta = _delta(out_tp, prev.get("output_throughput"))
        total_delta = _delta(total_tp, prev.get("total_token_throughput"))
        table.append("| " + " | ".join([
            _cell(row.get("run_id", "")),
            _cell(row.get("model", "")),
            _cell(row.get("nodes", "")),
            parallel,
            _cell(row.get("max_concurrency", "")),
            _cell(row.get("num_prompts", "")),
            f"{row.get('in_len_avg', '')}/{row.get('out_len_avg', '')}",
            f"{out_tp}{f' ({out_delta})' if out_delta else ''}",
            f"{total_tp}{f' ({total_delta})' if total_delta else ''}",
            _cell(row.get("mean_ttft_ms", "")),
            _cell(row.get("p99_ttft_ms", "")),
            _cell(row.get("mean_tpot_ms", "")),
            _cell(row.get("p99_e2el_ms", "")),
            _cell(row.get("bench_date", "")),
        ]) + " |")

    print("[perf-summary] 性能汇总(括号内为同一模型+同一压测参数相对上一次 run 的变化):")
    for line in table:
        print(line)
    print(f"[perf-summary] CSV: {csv_path}(共 {len(rows)} 行, 本次打印 {len(selected)} 行)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="汇总 vllm bench 结果 JSON 到 CSV(版本升级时的性能看护表)"
    )
    parser.add_argument("--scan-dir", action="append", default=[],
                        help="要扫描的目录(可重复), 递归查找 vllm bench 结果 JSON")
    parser.add_argument("--csv", default=os.environ.get("PERF_CSV") or os.path.join(os.getcwd(), "perf_summary.csv"),
                        help="汇总 CSV(默认取环境变量 PERF_CSV, 否则 ./perf_summary.csv)")
    parser.add_argument("--run-id", default="", help="本次 run 目录名(run_<时间戳>), 写进 CSV 便于按次对比")
    parser.add_argument("--model", default="", help="模型名(清单里的名字), 覆盖 JSON/目录名推断")
    parser.add_argument("--tp", default="", help="张量并行度")
    parser.add_argument("--dp", default="", help="数据并行度")
    parser.add_argument("--pp", default="", help="流水并行度")
    parser.add_argument("--nodes", default="", help="节点数")
    parser.add_argument("--gpus-per-node", default="", help="每节点参与卡数")
    parser.add_argument("--run-csv", default="", help="额外把本次 run 的行写一份到这个 CSV(如 run 目录下)")
    parser.add_argument("--report", action="store_true", help="打印本次 run 的汇总表(含与上一次 run 的对比)")
    parser.add_argument("--show", action="store_true", help="打印 CSV 里的全部历史记录")
    parser.add_argument("--quiet", action="store_true", help="只写 CSV, 不打印新增行的明细")
    args = parser.parse_args()

    csv_path = os.path.abspath(os.path.expanduser(args.csv))
    existing = _read_rows(csv_path)
    keys = {_row_key(row) for row in existing}
    env_info = _env_info()

    new_rows: list = []
    for scan_root in args.scan_dir:
        scan_root = os.path.abspath(os.path.expanduser(scan_root))
        if not os.path.isdir(scan_root):
            print(f"[perf-summary] WARN 扫描目录不存在, 跳过: {scan_root}", file=sys.stderr)
            continue
        found = 0
        for dirpath, _dirnames, filenames in os.walk(scan_root):
            for name in sorted(filenames):
                if not name.endswith(".json"):
                    continue
                # bench serve 会额外写 <name>.pytorch.json, sweep 的 summary.json 是列表汇总
                if name.endswith(".pytorch.json") or name == "summary.json":
                    continue
                path = os.path.join(dirpath, name)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except (OSError, ValueError):
                    continue
                if not _is_bench_result(data):
                    continue
                row = _build_row(path, scan_root, data, args, env_info)
                key = _row_key(row)
                if key in keys:
                    continue
                keys.add(key)
                new_rows.append(row)
                found += 1
                if not args.quiet:
                    print(
                        f"[perf-summary] + {row['run_id'] or '-'} {row['model'] or '-'} "
                        f"conc={row['max_concurrency']} prompts={row['num_prompts']} "
                        f"in/out≈{row['in_len_avg']}/{row['out_len_avg']} "
                        f"output_tok/s={row['output_throughput']} "
                        f"total_tok/s={row['total_token_throughput']} "
                        f"({row['result_json']})"
                    )
        if not args.quiet:
            print(f"[perf-summary] 扫描 {scan_root}: 新增 {found} 行")

    try:
        _append_rows(csv_path, new_rows, has_header=bool(existing))
    except OSError as exc:
        print(f"[perf-summary] ERROR 写不了 CSV: {csv_path} ({exc})", file=sys.stderr)
        print("[perf-summary] 提示: 用 --csv 或环境变量 PERF_CSV 指到可写路径"
              "(容器内写 /sw_home/lli/model_test/... 一般没问题)", file=sys.stderr)
        return 1
    all_rows = existing + new_rows
    if new_rows:
        print(f"[perf-summary] 写入 {len(new_rows)} 行 -> {csv_path}(累计 {len(all_rows)} 行)")
    elif not args.quiet:
        print(f"[perf-summary] 没有新增行(重复扫描时属正常), CSV: {csv_path}(累计 {len(all_rows)} 行)")

    if args.run_csv:
        try:
            written = _write_run_csv(
                os.path.abspath(os.path.expanduser(args.run_csv)), all_rows, args.run_id
            )
        except OSError as exc:
            written = ""
            print(f"[perf-summary] WARN 本次 run 的结果留档失败: {exc}", file=sys.stderr)
        if written:
            print(f"[perf-summary] 本次 run 的结果已单独留档: {written}")

    # 缺行提示: 这次 run 里起了服务/进了压测、但最终没写进 CSV 的模型(起服务失败/被超时
    # 打断/压测失败), 直接在日志里点名, 免得事后才发现"perf 只有部分模型"
    if not args.quiet:
        considered = {
            str(row.get("model", ""))
            for row in all_rows
            if not args.run_id or str(row.get("run_id", "")) == args.run_id
        }
        for scan_root in args.scan_dir:
            missing = sorted(_expected_models(scan_root) - considered - {""})
            if missing:
                print(
                    f"[perf-summary] WARN 下面 {len(missing)} 个模型本次没有性能数据"
                    f"(起服务失败/被超时打断/压测失败, 去 run 目录看它们的 *_serve.log):"
                )
                for name in missing:
                    print(f"[perf-summary]   - {name}")

    if args.report or args.show:
        _print_report(all_rows, "" if args.show else args.run_id, csv_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
