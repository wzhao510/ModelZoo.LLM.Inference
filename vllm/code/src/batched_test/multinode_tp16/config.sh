#!/bin/bash
# =============================================================
# 多机(双机) TP16/TP32 测试通用配置
# 适用于宁夏 10.13.81.57(rank0)+10.13.81.58(rank1), 每节点 8x MXC500X (TP16);
# 上海 C600U 分布式: 每节点 16x MXC600U (spec 118), 两机 TP32 (TP=32 NNODES=2)
# 模型清单(名称/路径/tp/dp/pp/dtype/额外参数)在 configs/models_distributed_2nodes.yaml,
# 本文件只放全局默认值; 所有项均可通过环境变量覆盖
# =============================================================

# ---------- 网络 / 端口 (被占用时覆盖) ----------
MASTER_ADDR="${MASTER_ADDR:-10.13.81.57}"   # rank0 节点 IP
MASTER_PORT="${MASTER_PORT:-8801}"          # 多机通信端口(8800 常被占用)
SERVE_HOST="${SERVE_HOST:-0.0.0.0}"
SERVE_PORT="${SERVE_PORT:-8000}"            # OpenAI 服务端口

# ---------- 分布式 / 性能参数 (模型的全局默认值) ----------
# 每个模型的 tp/dp/pp 以模型清单 YAML 为准, 这里只是缺省回退值
TP="${TP:-16}"
DP="${DP:-1}"
PP="${PP:-1}"
# NNODES 留空时由 start.sh 按 卡数(tp*dp*pp)/GPUS_PER_NODE 自动推导
NNODES="${NNODES:-}"
# 每节点参与该模型的卡数: C500x8 双机 TP16 用默认 8;
# C600U(16 卡/节点)双机 TP32 时设 GPUS_PER_NODE=16
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

# ---------- 模型选择 ----------
MODEL_NAME="${MODEL_NAME:-DeepSeek-R1-0528-W8A8}"

# ---------- 运行环境 ----------
# 调大 fd 上限; 受限容器里可能不允许, 失败不阻断(与 vllm 启动无强绑定)
ulimit -n 65536 2>/dev/null || true
export MACA_SMALL_PAGESIZE_ENABLE=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export MACA_DIRECT_DISPATCH=1
export MACA_MEMCPY_MODE=1
# ---------- FP8 / MCTLASS(C600U Kimi-K2.5-FP8 等) ----------
export VLLM_METAX_SUPPORTS_FP8=1
export MACA_VLLM_ENABLE_MCTLASS_PYTHON_API=1
export MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE=1
export MCCL_USE_MXSML=0
export VLLM_METAX_USE_FP8_WO_A=0
export VLLM_METAX_ENABLE_FP8_WEIGHT=1
export VLLM_METAX_USE_FP8_SPARSE_ATTN_INDEXER=1
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ens5f0np0}"
export MCCL_SOCKET_IFNAME="${MCCL_SOCKET_IFNAME:-$GLOO_SOCKET_IFNAME}"
# 陆吾平台会注入 NETWORK_CONFIG(如 mlx5_0,mlx5_1), 优先使用; 否则用默认 4 卡
export MCCL_IB_HCA="${MCCL_IB_HCA:-${NETWORK_CONFIG:-mlx5_0,mlx5_1,mlx5_2,mlx5_3}}"

# ---------- 模型清单 (YAML 驱动) ----------
# 模型列表(名称/路径/tp/dp/pp/dtype/额外参数/环境变量)统一维护在 YAML 里,
# 格式与 configs/models_single_QA_required.yaml 一致:
#   configs/models_distributed_2nodes.yaml
# 本文件不再硬编码模型表; 用 MODEL_CONFIG 可指定其它清单(绝对路径或相对路径均可)。
# 注意: 解析清单需要能 import yaml 的 python, 见下面 _pick_models_python。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCHED_TEST_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_CONFIG_REQUESTED="${MODEL_CONFIG:-configs/models_distributed_2nodes.yaml}"

# 相对路径按 batched_test 目录解析: MODEL_CONFIG=configs/model.yaml 与
# MODEL_CONFIG=$BATCHED_TEST_DIR/configs/model.yaml 等价; 也兼容脚本目录下的路径
_resolve_model_config() {
  local wanted="$1" candidate
  for candidate in \
    "$wanted" \
    "$BATCHED_TEST_DIR/$wanted" \
    "$BATCHED_TEST_DIR/configs/$(basename "$wanted")" \
    "$SCRIPT_DIR/$wanted"; do
    if [ -f "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

# 选一个能 import yaml 的 python: 镜像里 /usr/bin/python3 常没有 PyYAML,
# PATH 里缺 conda 时会解析失败(旧版直接静默退出, 表现为"没加载模型就结束")
_pick_models_python() {
  local candidate
  for candidate in "${MODELS_PYTHON:-}" \
                   "$(command -v python3 2>/dev/null || true)" \
                   "${CONDA_PREFIX:-/opt/conda}/bin/python3" \
                   "/opt/conda/bin/python3" \
                   "$(command -v python 2>/dev/null || true)"; do
    [ -n "$candidate" ] && [ -x "$candidate" ] || continue
    "$candidate" -c 'import yaml' >/dev/null 2>&1 && { printf '%s\n' "$candidate"; return 0; }
  done
  return 1
}

if ! MODEL_CONFIG="$(_resolve_model_config "$MODEL_CONFIG_REQUESTED")"; then
  echo "[config] ERROR 找不到模型清单: MODEL_CONFIG=$MODEL_CONFIG_REQUESTED" >&2
  echo "[config] 已尝试: $MODEL_CONFIG_REQUESTED, $BATCHED_TEST_DIR/$MODEL_CONFIG_REQUESTED," \
       "$BATCHED_TEST_DIR/configs/$(basename "$MODEL_CONFIG_REQUESTED"), $SCRIPT_DIR/$MODEL_CONFIG_REQUESTED" >&2
  echo "[config] $BATCHED_TEST_DIR/configs/ 下的可用清单:" >&2
  ls -1 "$BATCHED_TEST_DIR"/configs/*.yaml >&2 2>/dev/null || true
  return 1 2>/dev/null || exit 1
fi

if ! MODELS_PY="$(_pick_models_python)"; then
  echo "[config] ERROR 找不到能 import yaml 的 python, 无法解析模型清单: $MODEL_CONFIG" >&2
  echo "[config] PATH=$PATH" >&2
  echo "[config] 解决: 用 MODELS_PYTHON=/opt/conda/bin/python3 指定解释器(或先 pip install pyyaml)" >&2
  return 1 2>/dev/null || exit 1
fi

LOAD_CMD=("$MODELS_PY" "$SCRIPT_DIR/load_models.py" "$MODEL_CONFIG")
# 加超时: 清单在 NFS 上读不动时不要静默卡住(默认 60s, MODELS_LOAD_TIMEOUT 可调)
command -v timeout >/dev/null 2>&1 && LOAD_CMD=(timeout "${MODELS_LOAD_TIMEOUT:-60}" "$MODELS_PY" "$SCRIPT_DIR/load_models.py" "$MODEL_CONFIG")
if ! MODELS_SNIPPET="$("${LOAD_CMD[@]}")"; then
  echo "[config] ERROR 解析模型清单失败或超时: $MODEL_CONFIG (python=$MODELS_PY)" >&2
  return 1 2>/dev/null || exit 1
fi
# 生成 DIST_MODELS/MODEL_PATHS/MODEL_DTYPES/MODEL_TP/... 等数组
eval "$MODELS_SNIPPET"
# 只在最外层脚本首次加载时打印, 避免 start.sh/check.sh/stop.sh 等子脚本重复刷屏
if [ -z "${MODELS_CONFIG_LOADED:-}" ]; then
  echo "[config] 模型清单: $MODEL_CONFIG (共 ${#DIST_MODELS[@]} 个, 默认执行 ${#DIST_DEFAULT_MODELS[@]} 个)"
  export MODELS_CONFIG_LOADED=1
fi

# ---------- 日志目录 (共享挂载, 双机可见) ----------
BASE_LOG_DIR="${BASE_LOG_DIR:-/sw_home/lli/model_test/tp16}"
