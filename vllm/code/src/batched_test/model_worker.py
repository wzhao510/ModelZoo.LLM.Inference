# SPDX-License-Identifier: Apache-2.0
import shlex
import time
import psutil
import os
import abc
import json
import urllib.request
from enum import Enum, auto

import threading
import traceback

import net_utils
from api_client import ChatCompletionClient, EmbeddingClient, cosine_similarity
import pandas as pd


from gpu_manager import GPUManager
from mp_manager import MPClusterManager
from utils import cal_gpu_count

CRITICAL_WORDS = [
    "EngineCore encountered an issue",
    "ioctl create queue block timeout",
]

# 压测子进程(纯 HTTP client)需要继承的环境变量: 与 vllm bench sweep 保持一致,
# 只挑和 MACA/可见卡/VLLM 行为相关的, 避免把无关变量带进去。
BENCH_ENV_KEYS = (
    "MACA_SMALL_PAGESIZE_ENABLE", "MACA_DIRECT_DISPATCH",
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "MACA_VLLM_ENABLE_MCTLASS_PYTHON_API",
    "MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE",
    "CUDA_VISIBLE_DEVICES", "MACA_VISIBLE_DEVICES",
    "VLLM_DISABLE_SHARED_EXPERTS_STREAM",
    "PYTORCH_CUDA_ALLOC_CONF", "DISABLE_MAP2XPU",
)


def select_bench_env(env: dict) -> dict:
    """从环境里挑出压测子进程需要的变量(逻辑同 BenchSweepWorker.select_envs)。"""
    return {key: value for key, value in env.items() if key in BENCH_ENV_KEYS}


class Worker(abc.ABC):
    def __init__(
        self,
        work_dir: str,
        model_cfg: dict,
        gpu_manager: MPClusterManager | GPUManager,
    ):
        self.work_dir = work_dir
        self.model_cfg = model_cfg
        self.gpu_manager = gpu_manager

        self.config_manager = ModelConfigManager(model_cfg)
        self.port_manager = net_utils.PortManager()

        self.port = self.port_manager.get_next_available_port()
        # For mp multi-node distributed init (e.g. master-port). Allocated lazily.
        self.dist_port: int | None = None
        # Track remote headless rank pids (keyed by node index).
        self.remote_rank_pids: dict[int, int] = {}
        self.related_gpu_ids = []
        # 真正开始跑(拿到 GPU)的时间; launch.py 的 per-model 超时从这时算起,
        # 排队/等 GPU 的时间不计入(否则等着自己的卡还没轮到的模型会被判 TIMEOUT)
        self.started_at: float | None = None

    @abc.abstractmethod
    def run(self, stop_event: threading.Event):
        raise NotImplementedError("Worker must implement run method.")

    def _wait_and_allocate_gpus(self, timeout: int = 28800) -> list[int]:
        # Block until required GPUs are allocated
        assert self.related_gpu_ids == [], "GPUs have already been allocated."

        required_gpus = self.config_manager.calc_required_gpus()

        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.stop_event.is_set():
                raise KeyboardInterrupt("Stop event set, terminating service check.")

            print(
                f"[{self.model_cfg['name']}] Trying to allocate {required_gpus} GPUs..."
            )
            occupied_gpus = self.gpu_manager.allocate(required_gpus)

            if len(occupied_gpus) > 0:
                print(
                    f"[{self.model_cfg['name']}] Allocated resources: {occupied_gpus}"
                )
                self.related_gpu_ids = occupied_gpus
                self.started_at = time.time()
                return
            time.sleep(10)

        raise TimeoutError(
            f"[{self.model_cfg['name']}] Failed to allocate {required_gpus} GPUs within {timeout} seconds."
        )

    def _cleanup(self):
        self.port_manager.release_port(self.port)
        if self.dist_port is not None:
            self.port_manager.release_port(self.dist_port)
        self.gpu_manager.release(self.related_gpu_ids)


class ModelConfigManager:
    def __init__(self, model_cfg: dict):
        self.model_cfg = model_cfg
        self.serve_cfg = model_cfg.get("serve_config", {})

    def get_field(self, field_name: str, default=None):
        return self.model_cfg.get(field_name, default)

    def calc_required_gpus(self) -> int:
        return cal_gpu_count(self.model_cfg)

    def prepare_serve_cmd(
        self,
        host: str | None,
        port: int,
        mp_config: dict | None = None,
        force_backend: str | None = None,
    ) -> list[str]:
        # Prepare command
        serve_config = self.model_cfg.get("serve_config", {})
        # Backend selection:
        # - By default we use the model config value, falling back to 'mp'.
        # - In mp multi-node mode (mp_config is not None) we force backend to 'mp'.
        distributed_executor_backend = serve_config.get("distributed_executor_backend", "mp")
        if force_backend is not None:
            distributed_executor_backend = force_backend
        if mp_config is not None:
            distributed_executor_backend = "mp"

        cmd = [
            "vllm",
            "serve",
            self.model_cfg["model_path"],
            "--host",
            host if host is not None else "localhost",
            "--port",
            str(port),
            "-tp",
            str(serve_config.get("tp", 1)),
            "-pp",
            str(serve_config.get("pp", 1)),
            "-dp",
            str(serve_config.get("dp", 1)),
            "--trust-remote-code",
            "--gpu-memory-utilization",
            str(serve_config.get("gpu_memory_utilization", 0.8)),
            "--max-model-len",
            str(serve_config.get("max_model_len", 4096)),
            "--distributed-executor-backend",
            distributed_executor_backend,
        ]

        # Task type for embedding models (Qwen3-VL-Embedding etc.).
        # This vLLM-metax fork selects the model runner via --runner instead
        # of upstream's --task flag (--task embed is rejected by the CLI).
        task = serve_config.get("task")
        if task == "embed":
            cmd += ["--runner", "pooling"]

        # mp multi-node args (only when launching multi-node ranks)
        if mp_config is not None:
            # Expected keys: nnodes, node_rank, master_addr, master_port, headless(optional)
            cmd += [
                "--nnodes",
                str(mp_config["nnodes"]),
                "--node-rank",
                str(mp_config["node_rank"]),
                "--master-addr",
                str(mp_config["master_addr"]),
                "--master-port",
                str(mp_config["master_port"]),
            ]
            if mp_config.get("headless"):
                cmd.append("--headless")

        extra_args = serve_config.get("extra_args")
        if extra_args:
            if isinstance(extra_args, dict):
                for key, value in extra_args.items():
                    cmd.append(str(key))
                    if value is not None:
                        cmd.append(str(value))
            elif isinstance(extra_args, list):
                for item in extra_args:
                    cmd.append(item)

        return cmd

    def prepare_bench_cmd(self, host: str | None, port: int) -> list[str]:
        bench_cfg = self.model_cfg.get("benchmark", {})
        bench_cmd = [
            "vllm",
            "bench",
            "serve",
            "--model",
            self.model_cfg["model_path"],
            "--host",
            host if host is not None else "localhost",
            "--port",
            str(port),
            "--dataset-name",
            bench_cfg.get("dataset_name", "random"),
            "--trust-remote-code",
            "--ready-check-timeout-sec",
            "6000",
        ]
        if bench_cfg.get("ignore_eos"):
            bench_cmd.append("--ignore-eos")
        return bench_cmd

    def prepare_sweep_cmd(
        self,
        host: str | None,
        port: int,
        output_dir: str,
        mp_config: dict | None = None,
        force_backend: str | None = None,
    ) -> list[str]:
        # Prepare sweep command
        bench_cfg = self.model_cfg.get("benchmark", {})

        serve_cmd = self.prepare_serve_cmd(host, port, mp_config=mp_config, force_backend=force_backend)
        bench_cmd = self.prepare_bench_cmd(host, port)
        param_file = bench_cfg.get("bench_param")

        assert serve_cmd is not None, "Serve command is not prepared."
        assert bench_cmd is not None, "Benchmark command is not prepared."
        assert os.path.exists(os.path.abspath(param_file)), (
            f"Benchmark parameters file {param_file} does not exist."
        )

        sweep_cmd = [
            "vllm",
            "bench",
            "sweep",
            "serve",
            "--server-ready-timeout",
            "3600",
            "--serve-cmd",
            shlex.join(serve_cmd),
            "--bench-cmd",
            shlex.join(bench_cmd),
            "--bench-params",
            param_file,
            "--output-dir",
            output_dir,
            "--num-runs",
            str(bench_cfg.get("sweep_num_runs", "3")),
            "--show-stdout",
        ]

        return sweep_cmd, shlex.join(serve_cmd)

    def prepare_extra_env(self, occupied_gpus: list[int] | None) -> dict:
        # Prepare environment variables
        run_env = {}

        if occupied_gpus is not None:
            run_env["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(idx) for idx in occupied_gpus
            )

        extra_env = self.model_cfg.get("extra_env")
        if extra_env:
            if isinstance(extra_env, dict):
                # transfer all key-value pairs to string
                run_env.update({str(k): str(v) for k, v in extra_env.items()})
            elif isinstance(extra_env, list):
                for item in extra_env:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            run_env[str(k)] = str(v)

        return run_env

    def get_infer_type(self) -> list[str]:
        return self.model_cfg.get("infer_type", [])


class InferWorker(Worker):
    class InferenceStatus(Enum):
        INIT = auto()
        STARTING_SERVER = auto()
        INFERENCING = auto()
        NORMAL_END = auto()

    def __init__(
        self,
        text_case: str,
        image_case: str,
        model_cfg: dict,
        work_dir: str,
        long_text_case: str | None = None,
        embedding_case: str | None = None,
        last_resume: str | None = None,
        gpu_manager: MPClusterManager | GPUManager = None,
    ):
        super().__init__(
            work_dir=work_dir, model_cfg=model_cfg, gpu_manager=gpu_manager
        )

        self.text_case = text_case
        self.image_case = image_case
        self.long_text_case = long_text_case
        self.embedding_case = embedding_case
        self.api_serve_process = None
        self.serve_log_file = None
        self.status = self.InferenceStatus.INIT
        self.serve_cfg = model_cfg.get("serve_config", {})
        self.model_tag = f"{model_cfg['name']}[tp{self.serve_cfg.get('tp', 1)}pp{self.serve_cfg.get('pp', 1)}dp{self.serve_cfg.get('dp', 1)}]"

        self.csv_resumer = None
        if last_resume is not None:
            self.csv_resumer = pd.read_csv(last_resume)

    def _need_run(self) -> bool:
        if self.csv_resumer is None:
            return True

        # Check if case exists in resumer
        model_results = self.csv_resumer[
            self.csv_resumer["Model"].str.strip() == self.model_tag
        ]

        if model_results.empty:
            return True

        failed_cases = model_results[
            model_results["Stage"].str.strip() != self.InferenceStatus.NORMAL_END.name
        ]

        if failed_cases.empty:
            # All passed
            return False

        return True

    def _load_cases(self, case_file: str) -> list[dict]:
        import yaml

        with open(case_file, "r", encoding="utf-8") as f:
            test_cases = yaml.safe_load(f)
        return test_cases

    def _check_critical_words(
        self, content: str, blacklist: list[str] = CRITICAL_WORDS
    ) -> str | None:
        for _, item in enumerate(blacklist):
            if item in content:
                return item
        return None

    def _run_text_cases(self, cases: list[dict], log_file: str,
                         default_max_tokens: int = 256) -> int:
        """Run a list of text cases, return count of correct responses.

        Each case dict may have an optional ``max_tokens`` field; falls back
        to *default_max_tokens*.  Cases are grouped by max_tokens and sent in
        one batch per distinct value to minimise round-trips.
        """
        from collections import defaultdict
        client = ChatCompletionClient(host="localhost", port=self.port)

        # Group cases by their max_tokens value
        groups: dict[int, list[tuple[int, dict]]] = defaultdict(list)
        for idx, case in enumerate(cases):
            mt = case.get("max_tokens", default_max_tokens)
            groups[mt].append((idx, case))

        # Collect responses in original order
        responses: dict[int, str] = {}
        for mt, indexed_cases in groups.items():
            indices = [ic[0] for ic in indexed_cases]
            questions = [ic[1]["question"] for ic in indexed_cases]
            content_gen = client.run_text_only(
                questions=questions, max_completion_tokens=mt
            )
            for i, content in zip(indices, content_gen):
                responses[i] = content

        # Score
        corrected = 0
        for idx, case in enumerate(cases):
            content = responses.get(idx, "")
            if death_indication := self._check_critical_words(content):
                raise RuntimeError(
                    f"client received: {death_indication}, "
                    "which indicate that vllm serve might crashed. Aborting..."
                )
            keywords = case.get("keywords", [])
            if any(str(k).lower() in content.lower() for k in keywords):
                corrected += 1

            with open(log_file, "a") as f:
                question_text = case["question"]
                # Truncate very long questions in the log for readability
                if len(question_text) > 2000:
                    question_text = question_text[:2000] + "\n... [truncated]"
                f.write(
                    f"[{self.model_cfg['name']}] Question: {question_text}\n"
                )
                f.write(f"[{self.model_cfg['name']}] Response:\n")
                f.write(content + "\n")
                f.write("-" * 40 + "\n")

        return corrected

    def _do_text_only_inference(self, log_file: str) -> float:
        text_cases = self._load_cases(self.text_case)
        corrected = self._run_text_cases(text_cases, log_file, default_max_tokens=256)
        return corrected / len(text_cases)

    def _do_single_image_inference(self, log_file: str) -> float:
        client = ChatCompletionClient(host="localhost", port=self.port)
        image_cases = self._load_cases(self.image_case)
        image_urls = [case["picture_url"] for case in image_cases]

        # Get generator for responses
        content_gen = client.run_single_image(
            image_urls=image_urls,
            max_completion_tokens=256,
        )

        corrected_responses = 0

        # Zip test cases with yielded responses to match them
        for test_case, content in zip(image_cases, content_gen):
            if death_indication := self._check_critical_words(content):
                raise RuntimeError(
                    f"client received: {death_indication}, "
                    "which indicate that vllm serve might crashed. Aborting..."
                )

            keywords = test_case.get("keywords", [])
            # Check if any keyword is in the content (case-insensitive)
            if any(str(k).lower() in content.lower() for k in keywords):
                corrected_responses += 1
            with open(log_file, "a") as f:
                f.write(
                    f"[{self.model_cfg['name']}] Image URL: {test_case['picture_url']}\n"
                )
                f.write(f"[{self.model_cfg['name']}] Response:\n")
                f.write(content + "\n")
                f.write("-" * 40 + "\n")

        return corrected_responses / len(image_cases)

    def _run_embedding_cases(self, cases: list[dict], log_file: str) -> int:
        """Run embedding cases and return the number of correctly-ranked cases.

        Each case: {"query": str, "positive": [str, ...], "negative": [str, ...]}.
        A case is correct iff the mean cosine similarity between the query and
        the positive texts is higher than that of the negative texts.
        """
        client = EmbeddingClient(host="localhost", port=self.port)
        corrected = 0
        with open(log_file, "a", encoding="utf-8") as f:
            for case in cases:
                query = case["query"]
                positives = case.get("positive", [])
                negatives = case.get("negative", [])
                texts = [query] + positives + negatives
                vectors = client.embed(texts)
                query_vec = vectors[0]
                pos_sims = [
                    cosine_similarity(query_vec, v)
                    for v in vectors[1 : 1 + len(positives)]
                ]
                neg_sims = [
                    cosine_similarity(query_vec, v)
                    for v in vectors[1 + len(positives) :]
                ]
                mean_pos = sum(pos_sims) / len(pos_sims) if pos_sims else 0.0
                mean_neg = sum(neg_sims) / len(neg_sims) if neg_sims else 0.0
                ok = bool(positives and negatives and mean_pos > mean_neg)
                if ok:
                    corrected += 1

                f.write(f"[{self.model_cfg['name']}] Query: {query}\n")
                f.write(
                    f"[{self.model_cfg['name']}] Dim={len(query_vec)} "
                    f"mean_pos_sim={mean_pos:.4f} mean_neg_sim={mean_neg:.4f} "
                    f"=> {'OK' if ok else 'FAIL'}\n"
                )
                f.write("-" * 40 + "\n")
        return corrected

    def _do_embedding_inference(self, log_file: str) -> float:
        if not self.embedding_case:
            raise RuntimeError("embedding_case must be set for embedding infer_type.")
        assert os.path.exists(self.embedding_case), (
            f"Embedding case file not found: {self.embedding_case}"
        )
        cases = self._load_cases(self.embedding_case)
        if len(cases) == 0:
            return 0.0
        corrected = self._run_embedding_cases(cases, log_file)
        return corrected / len(cases)

    def run(self, stop_event: threading.Event) -> dict:
        self.stop_event = stop_event
        try:
            if not self._need_run():
                return self._warp_skipped()

            # Step 1. alloc GPU
            self._wait_and_allocate_gpus()

            # Step 2. launch serve
            self._launch_vllm_serve()

            # Step 3. client testing
            return self._post_client_test()
        except Exception as e:
            print(f"[{self.model_cfg['name']}] Inference failed: {e}")
            traceback.print_exc()
            return self._warp_failure(str(e))
        finally:
            self._cleanup()

    def _post_client_test(self):
        # 服务就绪(权重加载完)等待时间: 默认 3600s, 大模型加载慢时可在 YAML 里写
        # timeout: <秒> 单独调, 或用环境变量 MODEL_READY_TIMEOUT 全局覆盖
        timeout = int(
            os.environ.get("MODEL_READY_TIMEOUT")
            or self.model_cfg.get("timeout", 3600)
        )
        self._check_api_service_ready(timeout=timeout, blocking=True)

        correct_ratio = self._chat_completion()
        return {
            "Model": self.model_tag,
            "Correct Ratio": str(correct_ratio * 100) + "%",
            "Stage": self.InferenceStatus.NORMAL_END.name,
            "Reason": "",
            "Model Path": self.model_cfg["model_path"],
        }

    def _warp_failure(self, e: str):
        # Sanitize: replace newlines and truncate to keep CSV rows valid
        reason = str(e).replace("\n", " | ").replace("\r", "")
        if len(reason) > 500:
            reason = reason[:500] + "..."
        return {
            "Model": self.model_tag,
            "Correct Ratio": "0%",
            "Stage": self.status.name,
            "Reason": reason,
            "Model Path": self.model_cfg["model_path"],
        }

    def _warp_skipped(self):
        print(
            f"[{self.model_tag}] All tests on this combination have been passed! Skiped."
        )
        return {
            "Model": self.model_tag,
            "Correct Ratio": "0%",
            "Stage": self.InferenceStatus.NORMAL_END.name,
            "Reason": f"Resumed from last result",
            "Model Path": self.model_cfg["model_path"],
        }

    def _launch_vllm_serve(self):
        # Asynchronously launch vLLM serve process
        self.status = self.InferenceStatus.STARTING_SERVER

        assert len(self.related_gpu_ids) > 0, (
            "No GPUs allocated for launching vLLM serve."
        )

        # Prepare logfile
        log_file = net_utils.prepare_dir(
            os.path.join(self.work_dir, f"{self.model_tag}_serve.log")
        )
        self.serve_log_file = log_file  # expose for health-check log scanning

        # Prepare command (built below; may include mp multi-node args)

        # Set environment variable (local rank0 by default)
        # NOTE: In cluster (multi-node) mode, `self.related_gpu_ids` represents NODE indices,
        # so we must NOT feed it into prepare_extra_env directly.
        extra_env = {}

        if isinstance(self.gpu_manager, MPClusterManager):
            # ---- mp multi-node cluster mode ----
            nodes_list = self.related_gpu_ids
            nnodes = len(nodes_list)

            # Always force mp backend in cluster-config mode.
            # Allocate a dedicated distributed port (master-port) to avoid conflicts.
            if self.dist_port is None:
                self.dist_port = self.port_manager.get_next_available_port(
                    start_port=29500, max_port=29650
                )

            master_addr = self.gpu_manager.get_node_hostname(nodes_list[0])

            required_gpus = self.config_manager.calc_required_gpus()
            gpus_per_node = self.gpu_manager.gpu_per_node

            # Plan CUDA_VISIBLE_DEVICES per rank (contiguous from 0).
            remaining = required_gpus
            per_rank_visible: list[list[int]] = []
            for _ in range(nnodes):
                use = min(gpus_per_node, remaining)
                per_rank_visible.append(list(range(use)) if use > 0 else [])
                remaining -= use

            # Rank0 command (local; provides HTTP service)
            mp_rank0 = {
                "nnodes": nnodes,
                "node_rank": 0,
                "master_addr": master_addr,
                "master_port": self.dist_port,
                "headless": False,
            }
            cmd = self.config_manager.prepare_serve_cmd(
                host=None, port=self.port, mp_config=mp_rank0, force_backend="mp"
            )
            extra_env = {
                **self.gpu_manager.get_base_env(nodes_list[0]),
                **self.config_manager.prepare_extra_env(per_rank_visible[0]),
            }

            # Log rank0 cmd/env
            with open(log_file, "a") as f:
                cmd_str = f"[{self.model_cfg['name']}] command(rank0): {shlex.join(cmd)}"
                f.write(cmd_str + "\\n" + "-" * 80 + "\\n")
                f.write(extra_env.__str__() + "\\n" + "-" * 80 + "\\n")
                f.flush()
                print(cmd_str)

            # Launch local rank0 first (it will wait for other ranks to join)
            self.api_serve_process = net_utils.run_cmd(
                cmd=cmd, log_file=log_file, env={**os.environ, **extra_env}
            )

            # Launch remote headless ranks (rank>0) via SSH
            for rank in range(1, nnodes):
                node_idx = nodes_list[rank]
                mp_rank = {
                    "nnodes": nnodes,
                    "node_rank": rank,
                    "master_addr": master_addr,
                    "master_port": self.dist_port,
                    "headless": True,
                }
                remote_cmd = self.config_manager.prepare_serve_cmd(
                    host=None, port=self.port, mp_config=mp_rank, force_backend="mp"
                )
                remote_env = self.config_manager.prepare_extra_env(per_rank_visible[rank])

                remote_log = f"/tmp/batched_test_{self.model_tag}_rank{rank}.log"
                print(
                    f"[{self.model_cfg['name']}] command(rank{rank} headless @ "
                    f"{self.gpu_manager.get_node_hostname(node_idx)}): {shlex.join(remote_cmd)}"
                )
                pid = self.gpu_manager.start_headless_rank(
                    node_idx=node_idx,
                    cmd=remote_cmd,
                    env=remote_env,
                    log_path=remote_log,
                )
                self.remote_rank_pids[node_idx] = pid

                with open(log_file, "a") as f:
                    f.write(
                        f"[{self.model_cfg['name']}] started remote rank{rank} on node {node_idx} "
                        f"(host={self.gpu_manager.get_node_hostname(node_idx)}) pid={pid} log={remote_log}\\n"
                    )
                    f.flush()

            # Done: in mp cluster mode we already started rank0 and rank>0 here.
            return

        # ---- single-node mode (local GPUManager) ----
        cmd = self.config_manager.prepare_serve_cmd(host=None, port=self.port)
        extra_env = self.config_manager.prepare_extra_env(self.related_gpu_ids)

        # Log the command and environment
        with open(log_file, "a") as f:
            cmd_str = f"[{self.model_cfg['name']}] command: {shlex.join(cmd)}"
            f.write(cmd_str + "\n" + "-" * 80 + "\n")
            f.write(extra_env.__str__() + "\n" + "-" * 80 + "\n")
            f.flush()
            print(cmd_str)

        # Launch the command
        self.api_serve_process = net_utils.run_cmd(
            cmd=cmd, log_file=log_file, env={**os.environ, **extra_env}
        )

    def _check_api_service_ready(self, blocking=True, timeout=3600):
        """Block until the API service is healthy, or raise on error/timeout.

        Mirrors the shell ``wait_for_server()`` pattern:
        1. Process-alive check  (poll)
        2. Log scan for Traceback / critical errors
        3. HTTP endpoint check  (/health)
        4. Wall-clock timeout
        """
        t0 = time.time()
        last_log_size = 0  # incremental scanning to avoid re-reading the whole log

        print(f"[{self.model_cfg['name']}] Waiting for service on port {self.port}...")
        while time.time() - t0 < timeout:
            # --- 0. external stop signal ---
            if self.stop_event.is_set():
                raise KeyboardInterrupt("Stop event set, terminating service check.")

            # --- 1. process-alive check ---
            if self.api_serve_process is not None:
                return_code = self.api_serve_process.poll()
                if return_code is not None:
                    tail = ""
                    if self.serve_log_file and os.path.exists(self.serve_log_file):
                        try:
                            with open(self.serve_log_file, "r", encoding="utf-8", errors="replace") as lf:
                                lines = lf.readlines()
                                tail = "".join(lines[-50:])
                        except Exception:
                            pass
                    # Print full tail to stderr; keep exception message short for CSV
                    if tail:
                        print(
                            f"[{self.model_cfg['name']}] Serve log tail:\n{tail}",
                            flush=True,
                        )
                    raise RuntimeError(
                        f"vLLM serve exited with code {return_code}"
                    )

            # --- 2. scan log for Traceback / critical errors ---
            if self.serve_log_file and os.path.exists(self.serve_log_file):
                try:
                    with open(self.serve_log_file, "r", encoding="utf-8", errors="replace") as lf:
                        lf.seek(last_log_size)
                        new_content = lf.read()
                        last_log_size = lf.tell()  # advance cursor
                except Exception:
                    new_content = ""

                if new_content:
                    if "Traceback" in new_content:
                        # Collect first + last line of the Traceback for a short summary.
                        # Full dump goes to stderr; only a one-liner goes into the
                        # exception (and therefore the CSV Reason column).
                        tb_lines: list[str] = []
                        in_tb = False
                        for line in new_content.splitlines():
                            if "Traceback" in line:
                                in_tb = True
                            if in_tb:
                                tb_lines.append(line)
                                if len(tb_lines) > 40:
                                    break
                        full_tb = "\n".join(tb_lines)
                        print(f"[{self.model_cfg['name']}] Traceback in serve log:\n{full_tb}", flush=True)
                        summary = tb_lines[0] if tb_lines else "Traceback"
                        last = tb_lines[-1] if len(tb_lines) > 1 else ""
                        raise RuntimeError(
                            f"Traceback in serve log: {summary} ... {last}"
                        )

                    for word in CRITICAL_WORDS:
                        if word in new_content:
                            raise RuntimeError(
                                f"[{self.model_cfg['name']}] Critical error in serve log: "
                                f"'{word}'"
                            )

            # --- 3. HTTP endpoint check (/health) ---
            try:
                url = f"http://localhost:{self.port}/health"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        elapsed = time.time() - t0
                        print(
                            f"[{self.model_cfg['name']}] Service is ready on port "
                            f"{self.port} (took {elapsed:.0f}s)."
                        )
                        return True
            except Exception:
                pass  # not ready yet

            if not blocking:
                return False

            time.sleep(2)

        # --- 4. timeout ---
        tail = ""
        if self.serve_log_file and os.path.exists(self.serve_log_file):
            try:
                with open(self.serve_log_file, "r", encoding="utf-8", errors="replace") as lf:
                    lines = lf.readlines()
                    tail = "".join(lines[-50:])
            except Exception:
                pass
        if tail:
            print(
                f"[{self.model_cfg['name']}] Serve log tail:\n{tail}",
                flush=True,
            )
        raise TimeoutError(
            f"Service did not start within {timeout}s"
        )

    def _chat_completion(self) -> float:
        infer_type = self.model_cfg.get("infer_type", [])
        assert len(infer_type) > 0, "infer_type must be specified in model_cfg."

        self.status = self.InferenceStatus.INFERENCING

        # Load test cases from YAML
        assert os.path.exists(self.text_case), (
            f"Case file {self.text_case} does not exist."
        )
        assert os.path.exists(self.image_case), (
            f"Case file {self.image_case} does not exist."
        )

        if "embedding" in infer_type:
            log_file = net_utils.prepare_dir(
                os.path.join(self.work_dir, f"{self.model_tag}_embedding_inference.log")
            )
            return self._do_embedding_inference(log_file)

        if "single-image" in infer_type:
            log_file = net_utils.prepare_dir(
                os.path.join(
                    self.work_dir, f"{self.model_tag}_single_image_inference.log"
                )
            )
            return self._do_single_image_inference(log_file)

        if "text-only" in infer_type:
            log_file = net_utils.prepare_dir(
                os.path.join(self.work_dir, f"{self.model_tag}_text_only_inference.log")
            )

            # Load short text cases
            cases = self._load_cases(self.text_case)

            # When --long-text-case is specified, pick one long case and mix it in
            # so every model run includes at least one long-context request.
            if self.long_text_case:
                assert os.path.exists(self.long_text_case), (
                    f"Long text case file not found: {self.long_text_case}"
                )
                long_cases = self._load_cases(self.long_text_case)
                # Pick a different long case per model (round-robin by port offset)
                idx = (self.port % len(long_cases)) if long_cases else 0
                picked = long_cases[idx]
                picked["max_tokens"] = picked.get("max_tokens", 1024)
                cases = list(cases) + [picked]
                print(
                    f"[{self.model_cfg['name']}] Mixed 1 long case "
                    f"(#{idx + 1}/{len(long_cases)}, max_tokens={picked['max_tokens']}) "
                    f"into {len(cases)} total cases."
                )

            corrected = self._run_text_cases(
                cases, log_file, default_max_tokens=256
            )

            if len(cases) == 0:
                return 0.0
            return corrected / len(cases)

    def _shutdown_process(self):
        serve_process = self.api_serve_process

        if serve_process is None:
            return

        try:
            parent = psutil.Process(serve_process.pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            parent.kill()
            parent.wait()
        except psutil.NoSuchProcess:
            pass

        if hasattr(self.gpu_manager, "get_gpu_process_pid"):
            try:
                worker_pid = self.gpu_manager.get_gpu_process_pid(self.related_gpu_ids)
            except Exception as e:
                print(f"[{self.model_cfg['name']}] get_gpu_process_pid failed: {e}")
                worker_pid = []

            for pid in worker_pid:
                if psutil.pid_exists(pid):
                    try:
                        p = psutil.Process(pid)
                        p.kill()
                    except Exception as e:
                        print(
                            f"[{self.model_cfg['name']}] Error killing GPU worker process {pid}: {e}"
                        )
        print(f"[{self.model_cfg['name']}] Serve cleaned up successfully.")

    def _cleanup(self):
        """Additional cleanup after serve is stopped."""
        if isinstance(self.gpu_manager, MPClusterManager) and self.related_gpu_ids:
            try:
                nodes_list = self.related_gpu_ids
                for i in nodes_list[1:]:
                    try:
                        self.gpu_manager.stop_headless_rank(i, master_port=self.dist_port)
                    except Exception as e:
                        print(f"[InferWorker] stop remote rank on node {i} failed: {e}")
            except Exception as e:
                print(f"[InferWorker] pre-stop remote ranks failed: {e}")
        self._shutdown_process()
        super()._cleanup()


class InferPerfWorker(InferWorker):
    """精度 + 性能合并跑: 一次模型加载, 两个 client 各跑一遍.

    为什么需要: `--infer` 和 `--perf` 分别由 InferWorker / BenchSweepWorker 执行, 各自起
    一次 vllm serve, 也就是各自把权重加载一遍(大模型单次 20+ 分钟, 纯浪费)。本 worker 在
    同一次 serve 生命周期里:
      1) 分配 GPU 并起一次 vllm serve(权重只加载这一次);
      2) 等 service ready 后跑精度用例(与 --infer 完全同一套用例与日志);
      3) 对**同一个 server** 直接跑 `vllm bench serve`(压测参数/目录结构同 vllm bench sweep);
      4) 停服务。

    产物与"分开跑"保持一致, 下游解析脚本(parse_all_tasks.py)不用改:
      <infer_work_dir>/<model_tag>_text_only_inference.log + launch.py 写的 inference_results.csv
      <perf_work_dir>/<bench_tag>_serve.log
      <perf_work_dir>/<bench_tag>/BENCH--<params>/run=<n>.json + summary.json
      <perf_work_dir>/<bench_tag>/summary.csv
    其中 bench_tag = <name>_tp<tp>_pp<pp>_dp<dp>(与 BenchSweepWorker 的目录名一致,
    parse_all_tasks.py 靠它解析模型名与 tp/pp/dp)。
    """

    def __init__(
        self,
        text_case: str,
        image_case: str,
        model_cfg: dict,
        work_dir: str,
        perf_work_dir: str,
        long_text_case: str | None = None,
        embedding_case: str | None = None,
        last_resume: str | None = None,
        gpu_manager: MPClusterManager | GPUManager = None,
    ):
        super().__init__(
            text_case=text_case,
            image_case=image_case,
            model_cfg=model_cfg,
            work_dir=work_dir,
            long_text_case=long_text_case,
            embedding_case=embedding_case,
            last_resume=last_resume,
            gpu_manager=gpu_manager,
        )
        self.perf_work_dir = perf_work_dir
        serve_cfg = model_cfg.get("serve_config", {})
        self.bench_tag = (
            f"{model_cfg['name']}"
            f"_tp{serve_cfg.get('tp', 1)}"
            f"_pp{serve_cfg.get('pp', 1)}"
            f"_dp{serve_cfg.get('dp', 1)}"
        )
        # 与 BenchSweepWorker 的日志路径一致: parse_all_tasks.py 用
        # log_dir.parent/<stem 去掉最后一段> 去找 summary.csv, 路径变了就找不到
        self.bench_log_file = net_utils.prepare_dir(
            os.path.join(self.perf_work_dir, f"{self.bench_tag}_serve.log")
        )
        self.bench_client_cmds: list[str] = []
        self.bench_elapsed: float = 0.0
        # 只用来打断"本模型"的压测: 别的模型超时(set stop_event)不该把本模型的压测也杀掉,
        # 否则一次超时会让整批并发模型的性能数据一起丢失
        self.abort_event = threading.Event()

    # ---------- 压测部分(同一 server, 不再重新加载权重) ----------
    def _bench_param_items(self) -> list[dict]:
        """读 benchmark.bench_param(与 vllm bench sweep 支持的两种 JSON 格式一致)。"""
        bench_cfg = self.model_cfg.get("benchmark", {})
        param_file = bench_cfg.get("bench_param")
        assert param_file, (
            f"[{self.model_cfg['name']}] benchmark.bench_param 未配置, 无法跑性能"
        )
        param_file = os.path.abspath(param_file)
        assert os.path.exists(param_file), (
            f"Benchmark parameters file {param_file} does not exist."
        )
        with open(param_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            items = [{"_benchmark_name": name, **params} for name, params in data.items()]
        else:
            items = [dict(item) for item in data]
        assert items, f"[{self.model_cfg['name']}] bench_param 里没有压测参数: {param_file}"
        return items

    @staticmethod
    def _bench_combo_name(params: dict) -> str:
        """组合目录名与 vllm bench sweep 保持一致: BENCH--input_len=...-max_concurrency=..."""
        text = "-".join(
            f"{key}={value}" for key, value in params.items() if key != "_benchmark_name"
        )
        return "BENCH--" + text

    def _bench_cmd(self, params: dict, result_dir: str, run_number: int) -> list[str]:
        """对已就绪的 server 跑一次 vllm bench serve(参数来自 bench_param 的一条组合)。"""
        cmd = self.config_manager.prepare_bench_cmd(host="localhost", port=self.port)
        for key, value in params.items():
            if key == "_benchmark_name":
                continue
            cmd.append("--" + str(key).replace("_", "-"))
            cmd.append(str(value))
        cmd += [
            "--percentile-metrics",
            "ttft,tpot,itl,e2el",
            "--save-result",
            "--result-dir",
            result_dir,
            "--result-filename",
            f"run={run_number}.json",
        ]
        return cmd

    def _client_env(self) -> dict:
        """压测子进程的环境(单机=本机可见卡; 多机=rank0 节点的 base env + 可见卡)。"""
        if isinstance(self.gpu_manager, MPClusterManager):
            nodes_list = self.related_gpu_ids
            required_gpus = self.config_manager.calc_required_gpus()
            use0 = min(self.gpu_manager.gpu_per_node, required_gpus)
            return {
                **os.environ,
                **self.gpu_manager.get_base_env(nodes_list[0]),
                **self.config_manager.prepare_extra_env(list(range(use0))),
            }
        return {
            **os.environ,
            **self.config_manager.prepare_extra_env(self.related_gpu_ids),
        }

    def _run_cmd_blocking(self, cmd: list[str], env: dict, log_file: str, desc: str) -> int:
        """前台等一个子命令结束; 本模型被超时/打断(abort_event)时杀掉它。"""
        proc = net_utils.run_cmd(cmd=cmd, env=env, log_file=log_file)
        while True:
            returncode = proc.poll()
            if returncode is not None:
                return returncode
            if self.abort_event.is_set():
                proc.kill()
                proc.wait()
                raise KeyboardInterrupt(f"Aborted, killing {desc}")
            time.sleep(2)

    def _run_bench_on_running_server(self) -> dict:
        """对刚跑完精度用例的那个 server 逐组合压测, 并写 sweep 同款产物。"""
        bench_cfg = self.model_cfg.get("benchmark", {})
        experiment_dir = os.path.join(self.perf_work_dir, self.bench_tag)
        os.makedirs(experiment_dir, exist_ok=True)
        num_runs = int(bench_cfg.get("sweep_num_runs", 1) or 1)
        env = self._client_env()

        records: list[dict] = []
        t0 = time.time()
        for params in self._bench_param_items():
            combo_dir = os.path.join(experiment_dir, self._bench_combo_name(params))
            os.makedirs(combo_dir, exist_ok=True)
            combo_records: list[dict] = []
            for run_number in range(num_runs):
                out_path = os.path.join(combo_dir, f"run={run_number}.json")
                if os.path.exists(out_path):
                    # 断点续跑: 已有结果不再压一遍(与 vllm bench sweep 的 skip 行为一致)
                    print(f"[{self.model_cfg['name']}] 已有压测结果, 跳过: {out_path}")
                    with open(out_path, "r", encoding="utf-8") as f:
                        combo_records.append(json.load(f))
                    continue

                cmd = self._bench_cmd(params, combo_dir, run_number)
                if run_number == 0:
                    # 每组合只记一条命令(与 BenchSweepWorker.get_client_cmd 一致),
                    # parse_all_tasks.py 用 input/output/bs 前缀把它解析成 dict
                    self.bench_client_cmds.append(self._client_cmd_label(params, cmd))
                print(
                    f"[{self.model_cfg['name']}] benchmark({run_number + 1}/{num_runs}): "
                    f"{shlex.join(cmd)}",
                    flush=True,
                )
                returncode = self._run_cmd_blocking(
                    cmd, env, self.bench_log_file, "vllm bench serve"
                )
                if returncode != 0:
                    raise RuntimeError(
                        f"vllm bench serve exited with code {returncode} "
                        f"(log: {self.bench_log_file})"
                    )
                if not os.path.exists(out_path):
                    raise RuntimeError(
                        f"vllm bench serve 未产出结果文件: {out_path}"
                    )

                with open(out_path, "r", encoding="utf-8") as f:
                    record = json.load(f)
                # 与 vllm bench sweep 一致: 把 run 序号和压测参数并进结果,
                # 下游 summary.csv 里要靠这些列(max_concurrency/input_len/output_len)
                record["run_number"] = run_number
                record.update(
                    {k: v for k, v in params.items() if k != "_benchmark_name"}
                )
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(record, f, indent=4)
                combo_records.append(record)

            with open(os.path.join(combo_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(combo_records, f, indent=4)
            records.extend(combo_records)

        self.bench_elapsed = time.time() - t0
        try:
            pd.DataFrame.from_records(records).to_csv(
                os.path.join(experiment_dir, "summary.csv")
            )
        except Exception as e:  # 汇总表写不出来不影响测试结论, 只是少一份统计
            print(f"[{self.model_cfg['name']}] WARN 写 summary.csv 失败: {e}")

        print(
            f"[{self.model_cfg['name']}] 性能完成: {len(records)} 条结果, "
            f"{self.bench_elapsed:.0f}s -> {experiment_dir}"
        )
        return {
            "task_name": self.bench_tag,
            "status": "success",
            "log_dir": self.bench_log_file,
            "error": None,
            "server_command": None,
            "client_command": list(self.bench_client_cmds),
            "env": {"type": "normal", "server_cmd_env": select_bench_env(env)},
        }

    @staticmethod
    def _client_cmd_label(params: dict, cmd: list[str]) -> str:
        """压测命令的可读标签: 前缀 input/output/bs 供 parse_all_tasks.py 解析。"""
        return (
            f"input={params.get('random_input_len', '')}; "
            f"output={params.get('random_output_len', '')}; "
            f"bs={params.get('max_concurrency', '')}; "
            f"{shlex.join(cmd)}"
        )

    def _warp_bench_failure(self, e: str) -> dict:
        reason = str(e).replace("\n", " | ").replace("\r", "")
        if len(reason) > 500:
            reason = reason[:500] + "..."
        return {
            "task_name": self.bench_tag,
            "status": "error",
            "log_dir": self.bench_log_file,
            "error": reason,
            "server_command": None,
            "client_command": list(self.bench_client_cmds) or None,
            "env": None,
        }

    def _warp_bench_skipped(self) -> dict:
        """清单里没配 benchmark 的模型只做精度校验, 性能侧写一行 skip 便于汇总区分。"""
        return {
            "task_name": self.bench_tag,
            "status": "skip",
            "log_dir": self.bench_log_file,
            "error": "benchmark 未配置(只跑精度)",
            "server_command": None,
            "client_command": None,
            "env": None,
        }

    def run(self, stop_event: threading.Event) -> dict:
        """一次加载跑完精度 + 性能; 返回的 dict 同时带两个 client 的字段.

        精度与性能互不牵连: 精度用例失败/被别的模型超时打断时, 只要服务还活着就继续压测,
        避免个别模型的精度问题把性能数据整行丢掉(版本升级看护最怕缺行)。
        """
        self.stop_event = stop_event
        infer_row: dict | None = None
        try:
            need_infer = self._need_run()
            if not need_infer:
                print(
                    f"[{self.model_tag}] 断点续跑: 精度用例已全部通过, 本次只跑性能"
                )

            # Step 1. alloc GPU
            self._wait_and_allocate_gpus()

            # Step 2. launch serve(精度和性能共用这一次加载)
            self._launch_vllm_serve()

            # Step 3. 精度 client(与 --infer 完全一致)。
            # 精度和性能互不牵连: 精度用例失败/被别的模型超时打断, 只要服务还活着,
            # 性能照样采样, 避免因为个别模型的精度问题把性能数据整行丢掉。
            try:
                if need_infer:
                    infer_row = self._post_client_test()
                else:
                    infer_row = self._warp_skipped()
            except (Exception, KeyboardInterrupt) as e:
                reason = f"精度阶段失败/中断: {type(e).__name__}: {e}"
                print(f"[{self.model_cfg['name']}] {reason}")
                if not isinstance(e, KeyboardInterrupt):
                    traceback.print_exc()
                infer_row = self._warp_failure(reason)

            # Step 4. 性能 client: 对同一个 server 压测, 不再加载模型
            # 清单里没配 benchmark 的模型(例如 Qwen3-VL-Embedding)只校验精度, 不压测:
            # vllm bench serve 默认打 /v1/completions, 对 pooling/embedding 服务必然 404。
            if not self.model_cfg.get("benchmark"):
                print(
                    f"[{self.model_cfg['name']}] 清单里未配置 benchmark, 跳过性能压测"
                )
                perf_row = self._warp_bench_skipped()
            else:
                if self.api_serve_process is not None:
                    serve_returncode = self.api_serve_process.poll()
                    if serve_returncode is not None:
                        # 服务已经退出: 压测只会白等 ready-check-timeout, 直接记失败
                        raise RuntimeError(
                            f"vllm serve 已退出(code={serve_returncode}), 跳过压测"
                        )
                perf_row = self._run_bench_on_running_server()

            return {**infer_row, **perf_row}
        except Exception as e:
            print(f"[{self.model_cfg['name']}] Infer+Perf failed: {e}")
            traceback.print_exc()
            if infer_row is None:
                infer_row = self._warp_failure(str(e))
            return {**infer_row, **self._warp_bench_failure(str(e))}
        finally:
            self._cleanup()


class BenchSweepWorker(Worker):
    def __init__(
        self,
        work_dir: str,
        model_cfg: dict,
        gpu_manager: MPClusterManager | GPUManager = None,
    ):
        super().__init__(
            work_dir=work_dir, model_cfg=model_cfg, gpu_manager=gpu_manager
        )
        self.sweep_process = None

        self.serve_cfg = model_cfg.get("serve_config", {})
        self.model_tag = f"{model_cfg['name']}_tp{self.serve_cfg.get('tp', 1)}_pp{self.serve_cfg.get('pp', 1)}_dp{self.serve_cfg.get('dp', 1)}"
        self.log_file = os.path.join(self.work_dir, f"{self.model_tag}_serve.log")

    def get_client_cmd(self, bench_cmd):
        client_cmds = []
        bench_cfg = self.model_cfg.get("benchmark", {})
        param_file = bench_cfg.get("bench_param")
        serve_config = self.model_cfg.get("serve_config", {})
        assert os.path.exists(os.path.abspath(param_file)), (
            f"Benchmark parameters file {param_file} does not exist."
        )
        with open(os.path.abspath(param_file), 'r', encoding='utf-8') as f:
            data_params =json.load(f)

        param_to_var = {
            'max-concurrency': 'bs',
            'random-input-len': 'input',
            'random-output-len': 'output'
        }
        
        for params in data_params:
            # tp = serve_config.get('tp', 1)
            # pp = serve_config.get('pp', 1)
            # dp = serve_config.get('dp', 1)
            input_val = params['random_input_len']
            output_val = params['random_output_len']
            bs_val = params['max_concurrency']

            var_prefix = (
                # f"tp={tp}; "
                # f"pp={pp}; "
                # f"dp={dp}; "
                f"input={input_val}; "
                f"output={output_val}; "
                f"bs={bs_val}; "
            )

            temp_cmd = bench_cmd

            for key, value in params.items():
                temp_cmd.append(f"--{str(key).replace('_','-')}")

                if key in param_to_var:
                    var_name = param_to_var[key]
                    temp_cmd.append(f"${{{var_name}}}")
                else:
                    temp_cmd.append(str(value))
            full_cmd = f"{var_prefix} {' '.join(temp_cmd)}"
            client_cmds.append(full_cmd)
        return client_cmds
            
    def select_envs(self, env):
        return select_bench_env(env)

    def run(self, stop_event: threading.Event, alloc_time_out: int = 14400):
        self.stop_event = stop_event

        result = {
            "task_name": self.model_tag,
            "log_dir": None,
            "status": "unknown",
            "error": None,
            "server_command": None,
            "client_command": None,
            "env": None,
        }
        try:
            self._wait_and_allocate_gpus(timeout=alloc_time_out)
            serve_cmd, bench_cmd, env = self._launch_bench_sweep()
            print(f"[{self.model_cfg['name']}] {'Completed!'.center(90, '-')}")

            result["log_dir"] = self.log_file
            result["status"] = "success"
            result["server_command"] = {
                "type": "normal",
                "command": serve_cmd
            }
            result["client_command"] = self.get_client_cmd(bench_cmd)
            result["env"] = {
                "type": "normal",
                "server_cmd_env": self.select_envs(env)
            }

        except RuntimeError as e:
            self.warp_failure(str(e))
            result["status"] = "error"
            result["error"] = f"{type(e)}: {str(e)}"
            result["log_dir"] = self.log_file
        
        except Exception as e:
            self.warp_failure(str(e))
            result["status"] = "error"
            result["error"] = f"{type(e)}: {str(e)}"

        finally:
            self._cleanup()
            return result

    def _launch_bench_sweep(self):
        result_dir = os.path.join(self.work_dir, self.model_tag)

        # NOTE: In cluster (multi-node) mode, `self.related_gpu_ids` represents NODE indices.
        extra_env = {}

        mp_rank0_cfg: dict | None = None
        if isinstance(self.gpu_manager, MPClusterManager):
            nodes_list = self.related_gpu_ids
            nnodes = len(nodes_list)

            if self.dist_port is None:
                self.dist_port = self.port_manager.get_next_available_port(
                    start_port=29500, max_port=29650
                )
            master_addr = self.gpu_manager.get_node_hostname(nodes_list[0])

            # Rank0 serve-cmd will be started *inside* the sweep process.
            mp_rank0_cfg = {
                "nnodes": nnodes,
                "node_rank": 0,
                "master_addr": master_addr,
                "master_port": self.dist_port,
                "headless": False,
            }

            # Make sweep process inherit base env + CUDA_VISIBLE_DEVICES for rank0.
            required_gpus = self.config_manager.calc_required_gpus()
            use0 = min(self.gpu_manager.gpu_per_node, required_gpus)
            extra_env = {
                **self.gpu_manager.get_base_env(nodes_list[0]),
                **self.config_manager.prepare_extra_env(list(range(use0))),
            }

        # For local sweep process env we still set CUDA_VISIBLE_DEVICES in single-node mode.
        if not isinstance(self.gpu_manager, MPClusterManager):
            extra_env = self.config_manager.prepare_extra_env(self.related_gpu_ids)

        sweep_cmd, serve_cmd = self.config_manager.prepare_sweep_cmd(
            host=None,
            port=self.port,
            output_dir=result_dir,
            mp_config=mp_rank0_cfg,
            force_backend="mp" if mp_rank0_cfg is not None else None,
        )

        bench_cmd = self.config_manager.prepare_bench_cmd(host=None, port=self.port)
 
        # Log the process output
        log_file = net_utils.prepare_dir(self.log_file)

        with open(log_file, "a") as f:
            f.write(self.model_cfg["name"])
            f.write("\n" + "-" * 80 + "\n")
            f.write(" ".join(sweep_cmd))
            f.write("\n" + "-" * 80 + "\n")
            f.write(extra_env.__str__())
            f.write("\n" + "-" * 80 + "\n")
            f.flush()

        self.sweep_process = net_utils.run_cmd(
            cmd=sweep_cmd, env={**os.environ, **extra_env}, log_file=log_file
        )

        # In mp cluster mode, rank0 serve will wait for other ranks.
        # Start remote headless ranks immediately after launching the sweep process.
        if mp_rank0_cfg is not None and isinstance(self.gpu_manager, MPClusterManager):
            nodes_list = self.related_gpu_ids
            nnodes = len(nodes_list)
            required_gpus = self.config_manager.calc_required_gpus()
            gpus_per_node = self.gpu_manager.gpu_per_node

            remaining = required_gpus
            per_rank_visible: list[list[int]] = []
            for _ in range(nnodes):
                use = min(gpus_per_node, remaining)
                per_rank_visible.append(list(range(use)) if use > 0 else [])
                remaining -= use

            # small delay to increase the chance rank0 has bound the master-port
            time.sleep(2)

            for rank in range(1, nnodes):
                node_idx = nodes_list[rank]
                mp_rank = {
                    "nnodes": nnodes,
                    "node_rank": rank,
                    "master_addr": mp_rank0_cfg["master_addr"],
                    "master_port": mp_rank0_cfg["master_port"],
                    "headless": True,
                }
                remote_cmd = self.config_manager.prepare_serve_cmd(
                    host=None, port=self.port, mp_config=mp_rank, force_backend="mp"
                )
                remote_env = self.config_manager.prepare_extra_env(per_rank_visible[rank])

                remote_log = f"/tmp/batched_test_{self.model_tag}_rank{rank}.log"
                print(
                    f"[{self.model_cfg['name']}] command(rank{rank} headless @ "
                    f"{self.gpu_manager.get_node_hostname(node_idx)}): {shlex.join(remote_cmd)}"
                )
                pid = self.gpu_manager.start_headless_rank(
                    node_idx=node_idx,
                    cmd=remote_cmd,
                    env=remote_env,
                    log_path=remote_log,
                )
                self.remote_rank_pids[node_idx] = pid

        returncode = self.sweep_process.wait()
        if returncode != 0:
            raise RuntimeError(f"[{self.model_cfg['name']}] vllm bench sweep serve encounter an error, return code {returncode}. Please check the log: {log_file}")

        return serve_cmd, bench_cmd, {**os.environ, **extra_env}
    
    def warp_failure(self, e: str):
        # Implement failure handling for performance testing here
        print(f"[{self.model_cfg['name']}] {'Benchmark failed:'.center(100, '-')}")
        print(f"[{self.model_cfg['name']}] {e.center(100)}")

    def _cleanup(self):
        super()._cleanup()

    
    def _print_status(self):
        pass
