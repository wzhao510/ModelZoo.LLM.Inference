#!/bin/bash
# 性能汇总入口: 把 vllm bench 结果 JSON 汇总成 CSV(每次跑完性能总结一次)
# 详细说明(表头含义/去重规则/对比方式)见 perf_summary.py 头部注释
#
# 用法:
#   bash perf_summary.sh --scan-dir <run 目录> [--csv <csv>] [--run-id run_xxx] \
#                        [--model NAME --tp 16 --dp 1 --pp 1 --nodes 2 --gpus-per-node 8] \
#                        [--report] [--run-csv <run 目录/perf_summary.csv>]
#   bash perf_summary.sh --csv <csv> --show          # 回看历史(版本升级前后对比)
#
# CSV 默认取环境变量 PERF_CSV(luwu_master.sh / luwu_single.sh 会传进来), 否则当前目录。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 只需要标准库, 但仍按 config.sh 的思路挑一个可用 python(镜像里 /usr/bin/python3 缺包时用 conda)
_pick_python() {
  local candidate
  for candidate in "${PERF_SUMMARY_PYTHON:-}" \
                   "$(command -v python3 2>/dev/null || true)" \
                   "${CONDA_PREFIX:-/opt/conda}/bin/python3" \
                   "/opt/conda/bin/python3" \
                   "$(command -v python 2>/dev/null || true)"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

PERF_PY="$(_pick_python)" || {
  echo "[perf-summary] ERROR 找不到可用的 python3(可用 PERF_SUMMARY_PYTHON=... 指定)" >&2
  exit 1
}

exec "$PERF_PY" "$SCRIPT_DIR/perf_summary.py" "$@"
