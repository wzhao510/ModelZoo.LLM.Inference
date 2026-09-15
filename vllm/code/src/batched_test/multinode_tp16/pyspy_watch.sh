#!/bin/bash
# =============================================================
# 启动期抓栈看门狗: py-spy dump + /proc 采样
#
# 用途: vllm serve 启动到就绪这段时间如果"几分钟不打印日志", 用它把每个
#       引擎/worker 进程的 Python 栈周期性抓下来, 定位卡在哪一行。
#
# 每轮采样写一个目录(轮次按时间递增):
#   ps.txt              进程 CPU/内存/状态/父子关系快照
#   wchan_<pid>.txt     /proc/<pid>/wchan + 上下文切换次数, 区分"在算"还是"在等"
#   pyspy_<pid>_<名>.txt  py-spy dump 的 Python 栈(装不上/没权限时只有前两项)
#   ../index.txt        每轮的耗时/进程数/py-spy 状态/serve 日志末尾一行
#
# 用法:
#   # start.sh 会自动后台拉起(默认开启)
#   bash pyspy_watch.sh --serve-pid 3510 --serve-log run/xxx/rank0_serve.log \
#        --out run/xxx/pyspy_rank0
#   bash pyspy_watch.sh --once --out /tmp/dump            # 立刻抓一次, 不循环
#
# 可调环境变量:
#   PYSPY_DUMP=0          在 start.sh 里关掉看门狗
#   PYSPY_INTERVAL=30     采样间隔(秒)
#   PYSPY_MAX_MIN=45      最长采样时长(分钟), 到点自动退出
#   PYSPY_START_DELAY=10  首次采样前等待(秒)
#   PYSPY_NATIVE=1        额外抓 native 栈(--native), 更慢但能看到 C 层
#   PYSPY_BIN=py-spy      指定 py-spy 可执行文件
#   PYSPY_AUTO_INSTALL=1  没装 py-spy 时尝试 pip install(离线环境设 0)
#   PYSPY_SILENT_TRIGGER=90  日志静默超过该秒数时加密采样(默认 90)
#   PYSPY_FAST_INTERVAL=10   静默期采样间隔(秒)
# =============================================================
set -uo pipefail

SERVE_PID=""
SERVE_LOG=""
OUT_DIR=""
INTERVAL="${PYSPY_INTERVAL:-30}"
MAX_MIN="${PYSPY_MAX_MIN:-45}"
START_DELAY="${PYSPY_START_DELAY:-10}"
SILENT_TRIGGER="${PYSPY_SILENT_TRIGGER:-90}"
FAST_INTERVAL="${PYSPY_FAST_INTERVAL:-10}"
LABEL=""
ONCE=0
NATIVE="${PYSPY_NATIVE:-0}"
PYSPY_BIN="${PYSPY_BIN:-}"
PYSPY_AUTO_INSTALL="${PYSPY_AUTO_INSTALL:-1}"

usage() {
  sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --serve-pid) SERVE_PID="${2:-}"; shift 2 ;;
    --serve-log) SERVE_LOG="${2:-}"; shift 2 ;;
    --out) OUT_DIR="${2:-}"; shift 2 ;;
    --interval) INTERVAL="${2:-30}"; shift 2 ;;
    --max-min) MAX_MIN="${2:-20}"; shift 2 ;;
    --start-delay) START_DELAY="${2:-10}"; shift 2 ;;
    --silent-trigger) SILENT_TRIGGER="${2:-90}"; shift 2 ;;
    --fast-interval) FAST_INTERVAL="${2:-10}"; shift 2 ;;
    --label) LABEL="${2:-}"; shift 2 ;;
    --native) NATIVE=1; shift ;;
    --once) ONCE=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "[pyspy] 未知参数: $1" >&2; usage 1 ;;
  esac
done

[ -n "$OUT_DIR" ] || OUT_DIR="/tmp/pyspy_dump_$(date +%m%d_%H%M%S)"
mkdir -p "$OUT_DIR" || { echo "[pyspy] 无法创建目录: $OUT_DIR" >&2; exit 1; }
INDEX="$OUT_DIR/index.txt"

log() { echo "[pyspy $(date '+%F %T')] $*" | tee -a "$INDEX"; }

# ---------- py-spy ----------
NONBLOCKING=""
resolve_pyspy() {
  local candidate
  for candidate in "$PYSPY_BIN" \
                   "$(command -v py-spy 2>/dev/null || true)" \
                   "${CONDA_PREFIX:-/opt/conda}/bin/py-spy" \
                   "/opt/conda/bin/py-spy" \
                   "/usr/local/bin/py-spy"; do
    [ -n "$candidate" ] && [ -x "$candidate" ] && { printf '%s\n' "$candidate"; return 0; }
  done
  return 1
}
PYSPY="$(resolve_pyspy || true)"
if [ -z "$PYSPY" ] && [ "$PYSPY_AUTO_INSTALL" != "0" ]; then
  log "未找到 py-spy, 尝试 pip install py-spy (PYSPY_AUTO_INSTALL=0 可跳过)"
  command -v timeout >/dev/null 2>&1 \
    && timeout 180 python3 -m pip install -q py-spy >>"$INDEX" 2>&1 \
    || python3 -m pip install -q py-spy >>"$INDEX" 2>&1
  PYSPY="$(resolve_pyspy || true)"
fi
if [ -n "$PYSPY" ]; then
  log "py-spy: $PYSPY ($("$PYSPY" --version 2>&1 | head -n1))"
  "$PYSPY" dump --help 2>&1 | grep -q -- '--nonblocking' && NONBLOCKING="--nonblocking"
  if [ "$NATIVE" = "1" ]; then
    # py-spy 不允许 native 与 nonblocking 同时用; native 会短暂暂停进程, 只在需要 C 栈时开
    NONBLOCKING=""
    log "PYSPY_NATIVE=1: 抓 C 栈, 采样时会短暂暂停目标进程"
  elif [ -n "$NONBLOCKING" ]; then
    log "使用 --nonblocking(采样时不暂停目标进程)"
  fi
else
  log "WARN 没有可用的 py-spy: 只记录 ps/wchan. 手动安装: pip install py-spy"
  log "WARN 若安装后仍 attach 失败, 容器需要 --privileged 或 --cap-add=SYS_PTRACE"
fi

# ---------- 进程收集 ----------
# serve 进程本身 + 它的所有子孙(mp 起的 EngineCore/Worker_TP*) + 名字被改写
# 成 VLLM::* 的进程; 去重后按 pid 排序
collect_pids() {
  local -a queue=() out=()
  local p c i
  [ -n "$SERVE_PID" ] && [ -d "/proc/$SERVE_PID" ] && queue+=("$SERVE_PID")
  while read -r p; do
    [ -n "$p" ] && [ -d "/proc/$p" ] && queue+=("$p")
  done < <(pgrep -f 'VLLM::' 2>/dev/null || true)
  i=0
  while [ "$i" -lt "${#queue[@]}" ] && [ "$i" -lt 300 ]; do
    p="${queue[$i]}"; i=$((i + 1)); out+=("$p")
    for c in $(cat /proc/"$p"/task/*/children 2>/dev/null); do queue+=("$c"); done
  done
  [ "${#out[@]}" -eq 0 ] && return 0
  printf '%s\n' "${out[@]}" | awk 'NF' | sort -un
}

proc_name() {
  local p="$1" name=""
  name="$(tr '\0' ' ' <"/proc/$p/cmdline" 2>/dev/null \
          | grep -o 'VLLM::[A-Za-z0-9_.:-]*' | head -n1 || true)"
  if [ -z "$name" ]; then
    name="$(basename "$(readlink -f "/proc/$p/exe" 2>/dev/null || echo proc)")"
  fi
  printf '%s\n' "$name" | tr -c 'A-Za-z0-9_.:-' '_' | cut -c1-32
}

is_python_proc() {
  case "$(readlink -f "/proc/$1/exe" 2>/dev/null || true)" in
    *python*) return 0 ;;
    *) return 1 ;;
  esac
}

# serve 日志末尾一行: 用来把"栈快照"和"日志停在哪"对上时间
log_tail_line() {
  [ -n "$SERVE_LOG" ] && [ -f "$SERVE_LOG" ] || { printf 'n/a\n'; return 0; }
  tr '\r' '\n' <"$SERVE_LOG" 2>/dev/null | grep -v '^[[:space:]]*$' | tail -n1 | cut -c1-140
}

# 各 rank 的就绪标记: EngineCore 只在 rank0 打 "init engine ...", headless rank1 靠
# worker 侧这几条(compile_or_warm_up_model 结束时每条 worker 都会打)
engine_ready() {
  [ -n "$SERVE_LOG" ] && [ -f "$SERVE_LOG" ] || return 1
  grep -qaE 'Application startup complete|init engine \(profile, create kv cache, warmup model\)|Graph capturing finished|Kernel JIT monitor activated|Reducing Torch threads from' \
    "$SERVE_LOG" 2>/dev/null
}

sample_once() {
  local round="$1" elapsed="$2" silent="${3:-0}"
  local -a pids=()
  local p name rc
  mapfile -t pids < <(collect_pids)
  [ "${#pids[@]}" -eq 0 ] && return 2

  local dir
  dir="$(printf '%s/round_%02d_t%06ds' "$OUT_DIR" "$round" "$elapsed")"
  mkdir -p "$dir"

  ps -ww -o pid,ppid,stat,etime,time,pcpu,pmem,rss,nlwp,comm,args \
     -p "$(IFS=,; echo "${pids[*]}")" >"$dir/ps.txt" 2>&1 || true

  for p in "${pids[@]}"; do
    {
      echo "# pid=$p name=$(proc_name "$p") $(tr '\0' ' ' <"/proc/$p/cmdline" 2>/dev/null | cut -c1-160)"
      echo "# wchan=$(cat "/proc/$p/wchan" 2>/dev/null)"
      grep -aE '^(State|Threads|VmRSS|voluntary_ctxt_switches|nonvoluntary_ctxt_switches)' \
        "/proc/$p/status" 2>/dev/null
    } >"$dir/wchan_$p.txt" 2>&1 || true
  done

  local pyspy_state="skipped" pyspy_fail_msg="" pyspy_ok=0 pyspy_bad=0
  if [ -n "$PYSPY" ]; then
    for p in "${pids[@]}"; do
      is_python_proc "$p" || continue
      name="$(proc_name "$p")"
      rc=0
      # shellcheck disable=SC2086
      "$PYSPY" dump --full-filenames --pid "$p" $NONBLOCKING $([ "$NATIVE" = "1" ] && echo --native) \
        >"$dir/pyspy_${p}_${name}.txt" 2>&1 || rc=$?
      if [ "$rc" -ne 0 ]; then
        [ -n "$pyspy_fail_msg" ] || pyspy_fail_msg="$(head -n1 "$dir/pyspy_${p}_${name}.txt" | cut -c1-90)"
        pyspy_bad=$((pyspy_bad + 1))
      else
        pyspy_ok=$((pyspy_ok + 1))
      fi
    done
    if [ "$pyspy_ok" -eq 0 ] && [ "$pyspy_bad" -gt 0 ]; then
      # 全失败才报警(个别 launcher/子进程抓不到是正常的)
      pyspy_state="fail"
      log "round=$round t=${elapsed}s py-spy 全部失败: ${pyspy_fail_msg:-见 $dir/pyspy_*.txt}"
    elif [ "$pyspy_bad" -gt 0 ]; then
      pyspy_state="ok(${pyspy_ok}/${pyspy_ok}+${pyspy_bad})"
    fi
  fi

  log "round=$round t=${elapsed}s silent=${silent}s procs=${#pids[@]} pyspy=$pyspy_state dir=$(basename "$dir") log_last=$(log_tail_line)"
  return 0
}

if [ -z "$SERVE_PID" ] && [ "$ONCE" = "0" ]; then
  log "WARN 未指定 --serve-pid: 只按 VLLM::* 匹配进程"
fi
log "开始采样: out=$OUT_DIR interval=${INTERVAL}s max=${MAX_MIN}min label=${LABEL:-<无>} serve_pid=${SERVE_PID:-<未指定>}"

# ---------- 主循环 ----------
start_ts=$(date +%s)
round=0
last_tail="__init__"
last_change_ts=$start_ts
silent_warned=0
sleep "$START_DELAY"
while true; do
  round=$((round + 1))
  now=$(date +%s)
  elapsed=$(( now - start_ts ))

  # 日志静默检测: 启动期真正难查的段是"日志几分钟不动", 这时把采样间隔调密
  cur_tail="$(log_tail_line)"
  if [ "$cur_tail" != "$last_tail" ]; then
    last_tail="$cur_tail"
    last_change_ts=$now
    silent_warned=0
  fi
  silent=$(( now - last_change_ts ))
  if [ "$silent" -ge "$SILENT_TRIGGER" ] && [ "$silent_warned" = "0" ]; then
    silent_warned=1
    log "WARN serve 日志已静默 ${silent}s, 采样间隔切到 ${FAST_INTERVAL}s (最后一行: $cur_tail)"
  fi

  if sample_once "$round" "$elapsed" "$silent"; then
    :
  else
    # 进程还没起来或已经全退了
    if [ -n "$SERVE_PID" ] && ! kill -0 "$SERVE_PID" 2>/dev/null; then
      log "serve 进程 $SERVE_PID 已退出, 停止采样"
      break
    fi
    log "round=$round t=${elapsed}s 未找到 vllm 进程, 等待中"
  fi

  if [ "$ONCE" = "1" ]; then
    log "--once 完成"
    break
  fi
  if engine_ready; then
    log "检测到引擎已就绪(serve 日志), 再抓一轮后停止"
    round=$((round + 1))
    sleep "$FAST_INTERVAL"
    sample_once "$round" "$(( $(date +%s) - start_ts ))" "$silent" || true
    break
  fi
  if [ "$elapsed" -ge $((MAX_MIN * 60)) ]; then
    log "达到最长采样时长 ${MAX_MIN} 分钟, 停止"
    break
  fi
  if [ "$silent" -ge "$SILENT_TRIGGER" ]; then
    sleep "$FAST_INTERVAL"
  else
    sleep "$INTERVAL"
  fi
done
log "采样结束: 共 $round 轮, 结果在 $OUT_DIR"
exit 0
