#!/bin/bash
# =============================================================
# 多机(双机) TP16/TP32 测试通用配置
# 适用于宁夏 10.13.81.57(rank0)+10.13.81.58(rank1), 每节点 8x MXC500X (TP16);
# 上海 C600U 分布式: 每节点 16x MXC600U (spec 118), 两机 TP32 (TP=32 NNODES=2)
# 所有项均可通过环境变量覆盖
# =============================================================

# ---------- 网络 / 端口 (被占用时覆盖) ----------
MASTER_ADDR="${MASTER_ADDR:-10.13.81.57}"   # rank0 节点 IP
MASTER_PORT="${MASTER_PORT:-8801}"          # 多机通信端口(8800 常被占用)
SERVE_HOST="${SERVE_HOST:-0.0.0.0}"
SERVE_PORT="${SERVE_PORT:-8000}"            # OpenAI 服务端口

# ---------- 分布式 / 性能参数 ----------
TP="${TP:-16}"
DP="${DP:-1}"
PP="${PP:-1}"
NNODES="${NNODES:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

# ---------- 模型选择 ----------
MODEL_NAME="${MODEL_NAME:-DeepSeek-R1-0528-W8A8}"

# ---------- 运行环境 ----------
ulimit -n 65536
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

# ---------- 模型定义 (名称 -> 路径 / dtype / 附加 serve 参数) ----------
declare -A MODEL_PATHS=(
  ["DeepSeek-R1-0528-W8A8"]="/mxstorage/pde_ai/models/llm/DeepSeek/DeepSeek-R1-0528-BF16-W8A8/vllm_quant_model/"
  ["GLM-5.2-W8A8"]="/mxstorage/pde_ai/models/llm/ChatGLM/GLM-5_2-W8A8/"
  # Kimi-K2.6-W8A8(Int4) 在本 vllm-metax 0.25 构建下 auto-detect compressed-tensors 会崩,
  # 暂时用 bf16 版本; 待 W8A8 加载问题修复后可改回:
  #   "/mxstorage/pde_ai/models/llm/Kimi/Kimi-K2.6-W8A8/"
  ["Kimi-K2.6-Int4"]="/mxstorage/pde_ai/models/llm/Kimi/Kimi-K2.6/"
  # 上海 C600U: 600U 节点本地盘 /mnt/hdd1 下的 FP8 权重(陆吾 C600U pod 需能访问该目录)
  ["Kimi-K2.5-FP8"]="/mnt/hdd1/vllm/Kimi-K2.5-FP8/"
)

declare -A MODEL_DTYPES=(
  ["DeepSeek-R1-0528-W8A8"]="bfloat16"
  ["GLM-5.2-W8A8"]="bfloat16"
  ["Kimi-K2.6-Int4"]="bfloat16"
  ["Kimi-K2.5-FP8"]="bfloat16"
)

# 需要时在此追加, 例如:
#   ["DeepSeek-R1-0528-W8A8"]="--speculative-config '{\"method\": \"deepseek_mtp\", \"num_speculative_tokens\": 2}'"
#   ["GLM-5.2-W8A8"]="--speculative-config '{\"num_speculative_tokens\": 1, \"model\": \"/mxstorage/pde_ai/models/llm/ChatGLM/GLM-5_2-W8A8\", \"method\": \"mtp\"}'"
declare -A MODEL_EXTRA_ARGS=(
  ["DeepSeek-R1-0528-W8A8"]=""
  ["GLM-5.2-W8A8"]=""
  ["Kimi-K2.6-Int4"]=""
  ["Kimi-K2.5-FP8"]="--max-num-batched-tokens 8192"
)

# ---------- 日志目录 (共享挂载, 双机可见) ----------
BASE_LOG_DIR="${BASE_LOG_DIR:-/sw_home/lli/model_test/tp16}"
