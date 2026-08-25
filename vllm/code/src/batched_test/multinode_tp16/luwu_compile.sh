#!/bin/bash
# 可选编译步骤: 编译安装 /sw_home/lli/luwu_apply 下的 mcoplib + vllm_metax(覆盖镜像内安装)
# 由 luwu_master.sh / luwu_worker.sh / luwu_single.sh 在 LUWU_COMPILE=1 时调用
# 用法:
#   bash luwu_compile.sh             # 完整编译并安装(master / 单机)
#   bash luwu_compile.sh --install   # 只安装共享 dist 的 wheel(worker, 等 master 编译产物)
set -e

CODE_ROOT="${CODE_ROOT:-/sw_home/lli/luwu_apply}"
MODE="${1:-build}"
# 归一化参数: luwu_worker.sh 以 --install 调用, 脚本内统一按 install 判断
MODE="${MODE#--}"
# 清掉位置参数: 后续 source mcoplib/env.sh 等脚本会用 ${1:-...} 取 MACA_PATH,
# 不清理会把 "--install" 之类的参数当成 MACA_PATH
set --

echo "[luwu-compile] $(date '+%F %T') mode=$MODE CODE_ROOT=$CODE_ROOT"

# 编译锁: 单机/分布式 master 共用同一共享 build/dist 目录, 并发编译会互相
# rm -rf 破坏对方构建(Stale file handle / segfault), 用 mkdir 锁串行化;
# worker 的 --install 模式不编译, 无需加锁
LOCK_DIR="$CODE_ROOT/.luwu_compile.lock"
if [ "$MODE" = "build" ]; then
  # 清理超过 2 小时未释放的陈旧锁(编译任务一般 <2h)
  if [ -d "$LOCK_DIR" ] && find "$LOCK_DIR" -mmin +120 | grep -q .; then
    echo "[luwu-compile] 清理陈旧编译锁"
    rmdir "$LOCK_DIR" 2>/dev/null || true
  fi
  echo "[luwu-compile] 等待编译锁(并发任务串行化)..."
  while ! mkdir "$LOCK_DIR" 2>/dev/null; do
    sleep 30
  done
  trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
  echo "[luwu-compile] 已获取编译锁"
  # 清掉共享 dist 里的旧 wheel, 避免 worker 等到上一次编译的过期产物
  mkdir -p "$CODE_ROOT/mcoplib/dist" "$CODE_ROOT/vLLM-metax/dist"
  rm -f "$CODE_ROOT/mcoplib/dist"/*.whl "$CODE_ROOT/vLLM-metax/dist"/*.whl 2>/dev/null || true
fi

# 共享 NFS 上并行 mxcc 编译不稳定(HeaderSearch 段错误 / Stale file handle / 写 .o.d 失败),
# build 模式把源码拷到容器本地磁盘构建, wheel 产物拷回共享 dist 供 worker 安装
LOCAL_BUILD_ROOT="${LOCAL_BUILD_ROOT:-/tmp/luwu_build}"
SRC_MCOPLIB="$CODE_ROOT/mcoplib"
SRC_VLLM="$CODE_ROOT/vLLM-metax"
if [ "$MODE" = "build" ]; then
  rm -rf "$LOCAL_BUILD_ROOT"
  mkdir -p "$LOCAL_BUILD_ROOT"
  echo "[luwu-compile] 拷贝源码到本地构建目录 $LOCAL_BUILD_ROOT ..."
  cp -a "$SRC_MCOPLIB" "$LOCAL_BUILD_ROOT/mcoplib"
  cp -a "$SRC_VLLM" "$LOCAL_BUILD_ROOT/vLLM-metax"
  SRC_MCOPLIB="$LOCAL_BUILD_ROOT/mcoplib"
  SRC_VLLM="$LOCAL_BUILD_ROOT/vLLM-metax"
fi

# HEAD 中存在但工作区缺失的已跟踪文件(如 op/qk_rms_norm.cu)会导致 CMake
# "Cannot find source file" 构建失败; 构建前恢复这类文件(不影响已修改文件)
restore_deleted_files() {
  git diff --name-only --diff-filter=D -z | xargs -0 -r git checkout HEAD -- 2>/dev/null || true
}

if [ "$MODE" = "install" ]; then
  # 等 master 的编译产物(共享 NFS), 最多等 90 分钟
  MCOPLIB_WHEEL=""
  VLLM_WHEEL=""
  for _ in $(seq 1 180); do
    MCOPLIB_WHEEL=$(ls "$CODE_ROOT/mcoplib/dist"/mcoplib-*.whl 2>/dev/null | head -1)
    VLLM_WHEEL=$(ls "$CODE_ROOT/vLLM-metax/dist"/vllm_metax-*.whl 2>/dev/null | head -1)
    if [ -n "$MCOPLIB_WHEEL" ] && [ -n "$VLLM_WHEEL" ]; then
      break
    fi
    sleep 30
  done
  if [ -z "$MCOPLIB_WHEEL" ] || [ -z "$VLLM_WHEEL" ]; then
    echo "[luwu-compile] ERROR 等待 master 编译产物超时(90min), worker 不自行编译, 请确认 master 编译是否成功"
    exit 1
  fi
fi

if [ "$MODE" = "install" ]; then
  echo "[luwu-compile] 安装共享编译产物:"
  echo "[luwu-compile]   mcoplib: $MCOPLIB_WHEEL"
  echo "[luwu-compile]   vllm:    $VLLM_WHEEL"
  pip uninstall -y vllm vllm_metax mcoplib
  pip install "$MCOPLIB_WHEEL"
  pip install "$VLLM_WHEEL" --no-deps
  VLLM_VER=$(ls "$CODE_ROOT/vLLM-metax/dist"/vllm_metax-*.whl | head -1 | sed -E 's/.*-([0-9]+\.[0-9]+\.[0-9]+)\+.*/\1/')
  pip install "vllm==$VLLM_VER" --no-deps
  echo "[luwu-compile] $(date '+%F %T') 安装完成"
  exit 0
fi

# ---------- build 模式: 完整编译安装 ----------
cd "$CODE_ROOT" || {
  echo "[luwu-compile] ERROR: $CODE_ROOT 不存在; /sw_home/lli 挂载内容可能与预期不符"
  ls -la /sw_home/lli 2>&1 | head -20
  exit 1
}
bash compile_env.sh

# mcoplib/lmdeploy 源码使用 pybind11 2.x 风格注解, pybind11 3.x 会报
# "The number of argument annotations does not match...", 钉到 <3
pip install "pybind11>=2.13,<3" >/dev/null 2>&1 || true

# 0. git safe.directory: NFS 挂载的 /sw_home/lli 属主为 lli, 容器内以 root 运行时
#    git 会报 "dubious ownership", 导致 setuptools-scm 取不到版本、wheel 构建失败。
for d in "$CODE_ROOT"/*/ "$LOCAL_BUILD_ROOT"/*/; do
  d="${d%/}"
  if [ -d "$d/.git" ]; then
    git config --global --add safe.directory "$d"
  fi
done

# 0.5 准备 /root/cu-bridge: 镜像默认没有, env.sh 依赖 $HOME/cu-bridge/CUDA_DIR
if [ ! -e /root/cu-bridge/CUDA_DIR ]; then
  mkdir -p /root/cu-bridge
  ln -sfn /opt/maca/tools/cu-bridge /root/cu-bridge/CUDA_DIR
  ln -sfn /opt/maca/tools/cu-bridge/bin /root/cu-bridge/bin
fi

# 编译安装 mcoplib
if [ ! -d "$SRC_MCOPLIB" ]; then
  echo "[luwu-compile] ERROR: $SRC_MCOPLIB 不存在"
  exit 1
fi
cd "$SRC_MCOPLIB"
restore_deleted_files
pip uninstall -y vllm vllm_metax mcoplib
source env.sh
pip install -r requirements/build.txt
rm -rf build dist .deps   # 清掉旧容器的 FetchContent 缓存, 避免 CMakeCache 路径不匹配
python setup.py bdist_wheel
pip install dist/mcoplib*.whl
mkdir -p "$CODE_ROOT/mcoplib/dist"
cp -a dist/mcoplib-*.whl "$CODE_ROOT/mcoplib/dist/"

# 编译安装 vllm_metax
if [ ! -d "$SRC_VLLM" ]; then
  echo "[luwu-compile] ERROR: $SRC_VLLM 不存在"
  exit 1
fi
cd "$SRC_VLLM"
restore_deleted_files
rm -rf build dist .deps
source env.sh
pip install -r requirements/build.txt
python setup.py bdist_wheel
pip install dist/vllm_metax*.whl --no-deps
pip install "vllm==$(ls dist/vllm_metax-*.whl | sed -E 's/.*-([0-9]+\.[0-9]+\.[0-9]+)\+.*/\1/')" --no-deps
mkdir -p "$CODE_ROOT/vLLM-metax/dist"
cp -a dist/vllm_metax-*.whl "$CODE_ROOT/vLLM-metax/dist/"

echo "[luwu-compile] $(date '+%F %T') 编译安装完成"
