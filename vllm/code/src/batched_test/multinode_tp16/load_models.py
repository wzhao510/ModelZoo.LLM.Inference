#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""把分布式模型清单 YAML 转成 bash 数组, 供 config.sh `eval` 加载.

清单格式与 configs/models_single_QA_required.yaml 一致(顶层是模型列表):

- name: <模型名>                # 必填, 需唯一
  model_path: <权重路径>         # 必填
  default: true                 # 可选, 是否进入默认执行列表
  timeout: 3600                 # 可选, 仅记录
  serve_config:
    tp: 16 / dp: 1 / pp: 1      # 并行度, 缺省 1
    gpu_memory_utilization: 0.9 # 可选, 也可写在 extra_args 里
    max_model_len: 4096         # 可选, 也可写在 extra_args 里
    max_num_seqs: 64            # 可选, 也可写在 extra_args 里
    extra_args:                 # flag -> value; value 为 null 表示只有 flag
      --dtype: bfloat16
      --speculative-config: '{"method": "mtp"}'
  extra_env:                    # 可选, flag 形式的环境变量, 仅本模型生效
    VLLM_DISABLE_SHARED_EXPERTS_STREAM: 1

输出 bash 片段:
  DIST_MODELS             清单顺序的全部模型名
  DIST_DEFAULT_MODELS     标注 default: true 的模型名(若都没标注则等于全部)
  MODEL_GPUS              与 DIST_MODELS 同序的卡数(tp*dp*pp, 仅供查看)
  MODEL_PATHS             name -> model_path
  MODEL_DTYPES            name -> dtype(取 --dtype, 缺省 bfloat16)
  MODEL_TP/MODEL_DP/MODEL_PP
  MODEL_GPU_MEM/MODEL_MAX_MODEL_LEN/MODEL_MAX_NUM_SEQS(空串表示用全局默认)
  MODEL_EXTRA_ARGS        name -> 除上述标量外的其余 serve 参数(已按 shell 转义)
  MODEL_ENVS              name -> 模型专属环境变量(已按 shell 转义)

用法:
    eval "$(python3 load_models.py /path/to/models_distributed_tp16.yaml)"
"""
from __future__ import annotations

import argparse
import shlex
import sys

import yaml

# 这些标量参数由 start.sh 用独立变量传给 vllm serve, 从 extra_args 里剥离避免重复传参
SCALAR_FLAGS = {
    "--dtype": "dtype",
    "--gpu-memory-utilization": "gpu_mem",
    "--max-model-len": "max_model_len",
    "--max-num-seqs": "max_num_seqs",
}


def _iter_extra_args(extra) -> list[tuple[str, object]]:
    """把 extra_args 归一化成 (flag, value) 列表; value 为 None 表示只有 flag."""
    if not extra:
        return []
    if isinstance(extra, dict):
        return [(str(flag), value) for flag, value in extra.items()]
    if isinstance(extra, list):
        return [(str(item), None) for item in extra]
    raise TypeError(f"extra_args 需要是 mapping 或 list, 实际是 {type(extra).__name__}")


def _split_extra_args(extra) -> tuple[dict[str, object], list[str]]:
    """拆出 (标量参数, 其余参数 token 列表)."""
    scalars: dict[str, object] = {}
    rest: list[str] = []
    for flag, value in _iter_extra_args(extra):
        if flag in SCALAR_FLAGS:
            scalars[SCALAR_FLAGS[flag]] = value
            continue
        rest.append(flag)
        if value is not None:
            rest.append(str(value))
    return scalars, rest


def _env_tokens(env) -> list[str]:
    """extra_env -> ["KEY=VALUE", ...]."""
    if not env:
        return []
    if isinstance(env, dict):
        return [f"{key}={value}" for key, value in env.items() if value is not None]
    if isinstance(env, list):
        return [str(item) for item in env]
    raise TypeError(f"extra_env 需要是 mapping 或 list, 实际是 {type(env).__name__}")


def _quote_join(tokens: list[str]) -> str:
    return " ".join(shlex.quote(token) for token in tokens)


def _emit_scalar_array(var: str, names: list[str], values: dict[str, str]) -> str:
    body = " ".join(shlex.quote(values[name]) for name in names)
    return f"declare -a {var}=({body})"


def _emit_assoc(var: str, names: list[str], values: dict[str, str]) -> str:
    lines = [f"declare -A {var}=("]
    for name in names:
        lines.append(f"  [{shlex.quote(name)}]={shlex.quote(values[name])}")
    lines.append(")")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="分布式模型清单 YAML -> bash 数组")
    parser.add_argument("model_config", help="模型清单 YAML 路径")
    args = parser.parse_args()

    with open(args.model_config, "r", encoding="utf-8") as fh:
        models = yaml.safe_load(fh)
    if models is None:
        models = []
    if not isinstance(models, list):
        print(f"[load_models] {args.model_config} 顶层需要是模型列表", file=sys.stderr)
        return 1

    paths: dict[str, str] = {}
    dtypes: dict[str, str] = {}
    tps: dict[str, str] = {}
    dps: dict[str, str] = {}
    pps: dict[str, str] = {}
    gpu_mems: dict[str, str] = {}
    max_lens: dict[str, str] = {}
    max_seqs: dict[str, str] = {}
    extras: dict[str, str] = {}
    envs: dict[str, str] = {}

    names: list[str] = []
    defaults: list[str] = []

    for index, model in enumerate(models):
        if not isinstance(model, dict):
            print(f"[load_models] 第 {index + 1} 个条目不是 mapping", file=sys.stderr)
            return 1
        name = str(model.get("name") or "").strip()
        if not name:
            print(f"[load_models] 第 {index + 1} 个条目缺少 name", file=sys.stderr)
            return 1
        if name in paths:
            print(f"[load_models] 模型名重复: {name}", file=sys.stderr)
            return 1

        serve = model.get("serve_config") or {}
        scalars, rest = _split_extra_args(serve.get("extra_args"))

        def pick(key: str, default: str = "") -> str:
            value = serve.get(key)
            if value is None:
                value = scalars.get(key)
            return default if value is None else str(value)

        path = str(model.get("model_path") or "").strip()
        if not path:
            print(f"[load_models] 模型 {name} 缺少 model_path", file=sys.stderr)
            return 1

        names.append(name)
        paths[name] = path
        dtypes[name] = pick("dtype", "bfloat16")
        tps[name] = pick("tp", "1")
        dps[name] = pick("dp", "1")
        pps[name] = pick("pp", "1")
        gpu_mems[name] = pick("gpu_mem")
        max_lens[name] = pick("max_model_len")
        max_seqs[name] = pick("max_num_seqs")
        extras[name] = _quote_join(rest)
        envs[name] = _quote_join(_env_tokens(model.get("extra_env")))
        if model.get("default"):
            defaults.append(name)

    if not defaults:
        defaults = list(names)

    out = [
        _emit_scalar_array("DIST_MODELS", names, {name: name for name in names}),
        _emit_scalar_array(
            "DIST_DEFAULT_MODELS", defaults, {name: name for name in defaults}
        ),
        _emit_scalar_array(
            "MODEL_GPUS",
            names,
            {n: str(int(tps[n]) * int(dps[n]) * int(pps[n])) for n in names},
        ),
        _emit_assoc("MODEL_PATHS", names, paths),
        _emit_assoc("MODEL_DTYPES", names, dtypes),
        _emit_assoc("MODEL_TP", names, tps),
        _emit_assoc("MODEL_DP", names, dps),
        _emit_assoc("MODEL_PP", names, pps),
        _emit_assoc("MODEL_GPU_MEM", names, gpu_mems),
        _emit_assoc("MODEL_MAX_MODEL_LEN", names, max_lens),
        _emit_assoc("MODEL_MAX_NUM_SEQS", names, max_seqs),
        _emit_assoc("MODEL_EXTRA_ARGS", names, extras),
        _emit_assoc("MODEL_ENVS", names, envs),
    ]
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
