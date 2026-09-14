#!/bin/bash
# =============================================================
# 多机(双机) TP16/TP32 测试通用配置
# 适用于宁夏 10.13.81.57(rank0)+10.13.81.58(rank1), 每节点 8x MXC500X (TP16);
# 上海 C600U 分布式: 每节点 16x MXC600U (spec 118), 两机 TP32 (TP=32 NNODES=2)
# 模型清单(名称/路径/tp/dp/pp/dtype/额外参数)在 configs/models_distributed_tp16.yaml,
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
#   configs/models_distributed_tp16.yaml
# 本文件不再硬编码模型表; 用 MODEL_CONFIG 可指定其它清单。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_CONFIG="${MODEL_CONFIG:-$SCRIPT_DIR/../configs/models_distributed_tp16.yaml}"
if [ ! -f "$MODEL_CONFIG" ]; then
  echo "[config] 模型清单不存在: $MODEL_CONFIG" >&2
  return 1 2>/dev/null || exit 1
fi
if ! MODELS_SNIPPET="$(python3 "$SCRIPT_DIR/load_models.py" "$MODEL_CONFIG")"; then
  echo "[config] 解析模型清单失败: $MODEL_CONFIG" >&2
  return 1 2>/dev/null || exit 1
fi
# 生成 DIST_MODELS/MODEL_PATHS/MODEL_DTYPES/MODEL_TP/... 等数组
eval "$MODELS_SNIPPET"

# ---------- 日志目录 (共享挂载, 双机可见) ----------
BASE_LOG_DIR="${BASE_LOG_DIR:-/sw_home/lli/model_test/tp16}"
