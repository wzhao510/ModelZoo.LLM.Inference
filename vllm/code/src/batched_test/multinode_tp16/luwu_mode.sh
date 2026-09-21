#!/bin/bash
# 测试模式解析: 本次跑精度(infer) / 性能(perf) / 两者(all)
#
# 优先级: 命令行参数(--infer / --perf / --all) > 环境变量 LUWU_TEST_MODE > 调用方默认值
# 环境变量取值(大小写不敏感, 逗号/空格分隔, 可组合):
#   infer / acc / accuracy / precision / text  -> 只跑精度(client.sh / launch.py --infer)
#   perf  / bench / performance / speed        -> 只跑性能(bench.sh / launch.py --perf)
#   all   / both / full                        -> 精度 + 性能
#
# 结果: 导出 LUWU_TEST_MODE(规范化后的名字)、LUWU_RUN_INFER(0/1)、LUWU_RUN_PERF(0/1)
# 说明: 用户显式给的值(环境变量/命令行)会记在 LUWU_TEST_MODE_INPUT 里, 所以像 run_all.sh
#       这样先 source config.sh(默认 all)再按自己的默认值解析一次时, 不会被默认值顶掉。
# 说明: 取值非法时只告警并回落到默认值, 不中断(stop.sh/status.sh 等也会 source 本文件,
#       不该因为一个拼错的模式名让整个脚本起不来); 多机/单机都用同一套取值, 便于记忆。

luwu_resolve_mode() {
  local default="${1:-all}"
  shift 2>/dev/null || true

  # 只在这份 shell 的第一次解析时把 LUWU_TEST_MODE 当作"用户输入"记下来(之后再解析时
  # LUWU_TEST_MODE 已经是本函数写进去的规范化结果, 再当输入会把默认值顶掉)
  if [ -z "${LUWU_MODE_RESOLVED:-}" ]; then
    if [ -n "${LUWU_TEST_MODE:-}" ]; then
      LUWU_TEST_MODE_INPUT="$LUWU_TEST_MODE"
    fi
    LUWU_MODE_RESOLVED=1
  fi

  local spec="${LUWU_TEST_MODE_INPUT:-$default}"
  local arg
  for arg in "$@"; do
    case "$arg" in
      --infer|--perf|--all|--both)
        spec="${arg#--}"
        [ "$spec" = "both" ] && spec="all"
        LUWU_TEST_MODE_INPUT="$spec"
        ;;
      "" ) ;;
      *)
        echo "[mode] WARN 未知参数 '$arg'(支持 --infer / --perf / --all), 已忽略" >&2
        ;;
    esac
  done

  local run_infer=0 run_perf=0 tok
  for tok in ${spec//,/ }; do
    case "${tok,,}" in
      infer|acc|accuracy|precision|text|infer_only) run_infer=1 ;;
      perf|bench|performance|speed) run_perf=1 ;;
      all|both|full) run_infer=1; run_perf=1 ;;
      *)
        echo "[mode] WARN 无法识别的 LUWU_TEST_MODE 取值 '$tok'" \
             "(支持 infer / perf / all), 已按默认值 '${default}' 处理" >&2
        if [ "$default" = "infer" ]; then run_infer=1; run_perf=0
        elif [ "$default" = "perf" ]; then run_infer=0; run_perf=1
        else run_infer=1; run_perf=1; fi
        spec="$default"
        LUWU_TEST_MODE_INPUT="$default"
        break
        ;;
    esac
  done

  if [ "$run_infer" = "0" ] && [ "$run_perf" = "0" ]; then
    run_infer=1; run_perf=1; spec="all"
  fi

  if [ "$run_infer" = "1" ] && [ "$run_perf" = "1" ]; then
    LUWU_TEST_MODE="all"
  elif [ "$run_perf" = "1" ]; then
    LUWU_TEST_MODE="perf"
  else
    LUWU_TEST_MODE="infer"
  fi
  LUWU_RUN_INFER="$run_infer"
  LUWU_RUN_PERF="$run_perf"
  export LUWU_TEST_MODE LUWU_RUN_INFER LUWU_RUN_PERF LUWU_TEST_MODE_INPUT LUWU_MODE_RESOLVED
}
