# SPDX-License-Identifier: Apache-2.0
# This script is used for model auto testing

import argparse
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import ClassVar
import os
import pandas as pd
import yaml
import csv
import time as time_module
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from pprint import pprint
import net_utils

from tqdm import tqdm
from mp_manager import MPClusterManager
from gpu_manager import GPUManager
from utils import cal_gpu_count


@dataclass(kw_only=True)
class SchedularArgs:
    work_dir: str
    model_config: str

    text_case: str
    image_case: str
    long_text_case: str | None = None
    resume_csv: str | None = None

    cluster_config: str | None = None
    infer: bool = False
    perf: bool = False
    concurrency: int | None = None  # Max concurrent models. None=default(GPU count local, 1 on cluster)
    model_timeout: int = 3600  # Hard per-model timeout in seconds (default: 1 hour)

    gpus: str = None  # comma-separated GPU counts to run (e.g., '1,2,4,8')
    tag: str | None = (
        None  # comma-separated tags to select (e.g., 'moe' or 'dense' or 'moe,vl'); default: all
    )
    dump_selected: str = None  # dump selected model configs to yaml
    dry_run: bool = False  # print selected models then exit

    parser_name: ClassVar[str] = "schedular"
    parser_help: ClassVar[str] = "Model Auto Testing Scheduler"

    def __post_init__(self):
        pass

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "SchedularArgs":
        if args.resume_csv is not None:
            assert os.path.exists(args.resume_csv)
            # resume_csv = os.path.join(os.path.abspath)
        return cls(
            work_dir=args.work_dir,
            model_config=args.model_config,
            cluster_config=args.cluster_config,
            text_case=args.text_case,
            image_case=args.image_case,
            long_text_case=args.long_text_case,
            resume_csv=args.resume_csv,
            infer=args.infer,
            perf=args.perf,
            concurrency=args.concurrency,
            model_timeout=args.model_timeout,
            gpus=args.gpus,
            tag=args.tag,
            dump_selected=args.dump_selected,
            dry_run=args.dry_run,
        )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--work-dir",
            type=str,
            default="/workspace/model_test",
            help="Save result for all kind of tests. Default to: </workspace/model_test>",
        )

        parser.add_argument(
            "--model-config",
            metavar="CONFIG_YAML_FILE",
            type=str,
            default=os.path.join(os.path.dirname(__file__), "configs", "model.yaml"),
            help="Model config file path. Default to: <configs/model.yaml>",
        )

        parser.add_argument(
            "--cluster-config",
            metavar="CONFIG_YAML_FILE",
            type=str,
            help="Cluster config file path.",
        )

        parser.add_argument(
            "--infer",
            action="store_true",
            help="Specify this to run inference test.",
        )

        parser.add_argument(
            "--text-case",
            metavar="LM_CASE_FILE",
            type=str,
            default=os.path.join(
                os.path.dirname(__file__), "configs", "inference", "text_case.yaml"
            ),
            help="Cases used for inference test. Default to: <configs/inference/text_case.yaml>",
        )

        parser.add_argument(
            "--image-case",
            metavar="IMAGE_CASE_FILE",
            type=str,
            default=os.path.join(
                os.path.dirname(__file__), "configs", "inference", "image_case.yaml"
            ),
            help="Cases used for inference test. Default to: <configs/inference/image_case.yaml>",
        )

        parser.add_argument(
            "--long-text-case",
            metavar="LONG_TEXT_CASE_FILE",
            type=str,
            default=None,
            help="Optional long-context text cases (YAML). When specified, these cases are "
            "run in addition to the short text cases. Each case may specify 'max_tokens' "
            "(default 512). Example: configs/inference/long_text_case.yaml",
        )

        parser.add_argument(
            "--resume-csv",
            metavar="RESUME_CSV",
            type=str,
            help="Resume from the failed case in specified inference_result.csv",
        )

        # Model selection / filtering
        parser.add_argument(
            "--gpus",
            type=str,
            default=None,
            help=(
                "Only run models that require the given number(s) of GPUs (tp*pp*dp). Comma-separated, e.g. '1,2,4,8'. "
                "If not set, default to '1,2,4,8'."
            ),
        )

        parser.add_argument(
            "--tag",
            type=str,
            default=None,
            help=(
                "Only run models matching the given tag(s). Comma-separated, e.g. 'moe' or 'dense' or 'moe,vl'. "
                "If not set, run all models. Models without 'tags' in model.yaml are treated as tag 'dense'."
            ),
        )

        parser.add_argument(
            "--dump-selected",
            type=str,
            default=None,
            help="Dump the selected model subset to a yaml file and continue running.",
        )

        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print the selected model list (name / gpu_count / moe) then exit.",
        )

        parser.add_argument(
            "--perf",
            action="store_true",
            help="Specify this to run performance benchmark.",
        )

        parser.add_argument(
            "--concurrency",
            type=int,
            default=None,
            help="Max number of models to run concurrently. Default: GPU count on local machine; 1 on cluster. Use 1 for serial.",
        )

        parser.add_argument(
            "--model-timeout",
            type=int,
            default=3600,
            help="Hard per-model timeout in seconds. If a model exceeds this, it is killed and marked as TIMEOUT. Default: 3600 (1 hour).",
        )

stop_event = threading.Event()


class Scheduler:
    def __init__(self, args: SchedularArgs):
        self.args = args
        self.model_list = self._load_yaml_config(args.model_config)
        self.model_list = self._filter_model_list(self.model_list)
        self.work_dir = os.path.join(args.work_dir, net_utils.current_dt())
        if args.cluster_config:
            cluster_nodes_config = self._load_yaml_config(args.cluster_config)
            self.gpu_manager = MPClusterManager(cluster_nodes_config)
            # TODO(hank) not allow concurrency on cluster mode now
            max_workers = 1
        else:
            self.gpu_manager = GPUManager()
            max_workers = self.gpu_manager.get_gpu_count()

        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Concurrency gate: controls how many model runs can be in-flight simultaneously,
        # without changing ThreadPoolExecutor worker count (to avoid side effects).
        limit = args.concurrency if args.concurrency is not None else max_workers
        if limit < 1:
            raise ValueError("--concurrency must be >= 1")
        self._gate = threading.BoundedSemaphore(limit)

    def _run_with_gate(self, fn, *args, **kwargs):
        with self._gate:
            return fn(*args, **kwargs)

    def _load_yaml_config(self, config_yaml: str) -> list[dict]:
        with open(config_yaml, "r") as f:
            config = yaml.safe_load(f)
        
        def sort_with_gpu(x):
            return cal_gpu_count(x)
        # sort with tp size. Grab resources via a greedy strategy.
        config = sorted(config, key=sort_with_gpu, reverse=True)
        return config

    def _required_gpus(self, model_cfg: dict) -> int:
        return cal_gpu_count(model_cfg)

    def _parse_gpus_filter(self) -> set[int] | None:
        """Parse --gpus like '1,2,4' into a set of ints.

        Default behavior:
          - If --gpus is not provided, run models requiring {1,2,4,8} GPUs by default.
          - If --gpus is provided, use the user-specified set.
        """
        if not self.args.gpus:
            return {1, 2, 4, 8}
        
        # If user provided empty string (rare), treat as no filter or default? Here we treat as default too.
        if str(self.args.gpus).strip() == "":
            return {1, 2, 4, 8}

        out: set[int] = set()
        for part in str(self.args.gpus).split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.add(int(part))
            except ValueError as e:
                raise ValueError(
                    f"Invalid --gpus value '{part}'. Expected comma-separated integers."
                ) from e
        return out or {1, 2, 4, 8}

    def _get_tags(self, model_cfg: dict) -> set[str]:
        """Return normalized tags for a model.

        Design:
        - If model_cfg has no 'tags' (or it's empty), treat it as {'dense'}.
        - If model_cfg.tags is a string, treat it as a single tag.
        - Tags are lower-cased strings.
        """
        tags = model_cfg.get("tags")
        if not tags:
            return {"dense"}
        if isinstance(tags, str):
            tags = [tags]
        out = {str(t).strip().lower() for t in tags if str(t).strip()}
        return out or {"dense"}

    def _parse_tag_filter(self) -> set[str] | None:
        """Parse --tag like 'moe,dense' into a set of tags (OR semantics)."""
        if not self.args.tag:
            return None
        out: set[str] = set()
        for part in str(self.args.tag).split(","):
            part = part.strip().lower()
            if part:
                out.add(part)
        return out or None

    def _filter_model_list(self, models: list[dict]) -> list[dict]:
        """Filter models by --gpus and --tag.

        Notes:
        - GPU count is computed as tp*pp*dp.
        - Tag filter uses OR semantics (match any tag).
        - Models without 'tags' are treated as tag 'dense'.
        - We attach derived fields prefixed with '_' for logging/debugging.
        """
        gpus_filter = self._parse_gpus_filter()
        tag_filter = self._parse_tag_filter()

        selected: list[dict] = []
        for m in models:
            req = self._required_gpus(m)
            if gpus_filter is not None and req not in gpus_filter:
                continue

            tags = self._get_tags(m)
            if tag_filter is not None and tags.isdisjoint(tag_filter):
                continue

            mm = dict(m)  # avoid mutating original
            mm["_required_gpus"] = req
            mm["_tags"] = sorted(tags)
            mm["_is_moe"] = "moe" in tags
            selected.append(mm)

        # Optional: dump selected subset for reproducibility
        if self.args.dump_selected:
            self._dump_selected_models(selected)

        return selected

    def _resolve_dump_selected_path(self, path: str) -> str:
        """Resolve dump path.

        If the user passed a bare filename (no directory component), place it under
        the same configs directory used by --model-config default (./configs).
        """
        if os.path.dirname(path) == "":
            base_dir = os.path.join(os.path.dirname(__file__), "configs")
            return os.path.join(base_dir, path)
        return path

    def _dump_selected_models(self, selected: list[dict]) -> None:
        dump_path = os.path.abspath(
            self._resolve_dump_selected_path(self.args.dump_selected)
        )
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)

        # Strip derived keys before dumping
        dump_models: list[dict] = []
        for m in selected:
            mm = {k: v for k, v in m.items() if not str(k).startswith("_")}
            dump_models.append(mm)

        # Write one model per list-item, with a blank line between models for readability
        with open(dump_path, "w", encoding="utf-8") as f:
            for i, m in enumerate(dump_models):
                if i > 0:
                    f.write("\n")
                yaml.safe_dump(
                    [m],
                    f,
                    sort_keys=False,
                    allow_unicode=True,
                    default_flow_style=False,
                )

        print(f"[Scheduler] Dumped selected models to: {dump_path}")

    def _print_selected_models(self) -> None:
        rows = []
        for m in self.model_list:
            name = m.get("name", "<unknown>")
            g = m.get("_required_gpus", "?")
            tags = ",".join(m.get("_tags") or [])
            rows.append((str(name), g, tags))
        rows.sort(key=lambda x: (int(x[1]) if str(x[1]).isdigit() else 10**9, x[0]))

        gpus_filter = self._parse_gpus_filter()
        tag_filter = self._parse_tag_filter()
        if gpus_filter is not None:
            print(f"[Scheduler] GPU count filter: {sorted(gpus_filter)}")
        if tag_filter is not None:
            print(f"[Scheduler] Tag filter (OR): {sorted(tag_filter)}")

        print(f"[Scheduler] Selected {len(rows)} model(s):")
        for name, g, tags in rows:
            print(f"  - {name} | gpus={g} | tags={tags}")

    def record_environment(self):
        log_file = os.path.join(self.work_dir, "collect_env.txt")
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        import collect_env

        with open(log_file, "w") as f:
            env_info = collect_env.get_pretty_env_info()
            f.write(env_info)

    def run_inference(self):
        """Run inference tests for all selected models with per-model timeout.

        Each model gets at most ``--model-timeout`` seconds from submission to
        completion.  Models that exceed the deadline are recorded as TIMEOUT,
        their worker is signalled to stop, and the scheduler moves on.
        """
        all_results = []
        # future -> model_cfg (so we can build error rows for timeouts)
        future_map: dict[concurrent.futures.Future, dict] = {}
        # Track submission time per future for timeout detection
        future_submit_time: dict[concurrent.futures.Future, float] = {}

        assert os.path.exists(self.args.text_case), (
            f"Case file not found: {self.args.text_case}"
        )
        assert os.path.exists(self.args.image_case), (
            f"Case file not found: {self.args.image_case}"
        )

        infer_work_dir = os.path.join(self.work_dir, "inference")
        csv_file_path = net_utils.prepare_dir(
            os.path.join(infer_work_dir, "inference_results.csv")
        )

        from model_worker import InferWorker

        for cfg in self.model_list:
            worker = InferWorker(
                work_dir=infer_work_dir,
                model_cfg=cfg,
                text_case=self.args.text_case,
                image_case=self.args.image_case,
                long_text_case=self.args.long_text_case,
                last_resume=self.args.resume_csv,
                gpu_manager=self.gpu_manager,
            )
            future = self.executor.submit(
                self._run_with_gate, worker.run, stop_event
            )
            future_map[future] = cfg
            future_submit_time[future] = time_module.time()

        # --- helper: build a result row for a model that never returned ---
        def _make_timeout_row(cfg: dict, elapsed: float) -> dict:
            serve_cfg = cfg.get("serve_config", {})
            tag = (
                f"{cfg.get('name', '?')}"
                f"[tp{serve_cfg.get('tp', 1)}"
                f"pp{serve_cfg.get('pp', 1)}"
                f"dp{serve_cfg.get('dp', 1)}]"
            )
            return {
                "Model": tag,
                "Correct Ratio": "0%",
                "Stage": "TIMEOUT",
                "Reason": (
                    f"Exceeded per-model timeout "
                    f"({self.args.model_timeout}s, elapsed {elapsed:.0f}s)"
                ),
                "Model Path": cfg.get("model_path", ""),
            }

        POLL_SECONDS = 30  # check for timeouts every N seconds
        remaining = set(future_map.keys())

        with open(csv_file_path, mode="w", newline="", encoding="utf-8") as f_csv:
            csv_writer = csv.DictWriter(
                f_csv,
                fieldnames=["Model", "Correct Ratio", "Stage", "Reason", "Model Path"],
                restval="",
            )
            csv_writer.writeheader()

            with tqdm(
                total=len(future_map),
                desc="Inference",
                unit="model",
                mininterval=0.5,
                maxinterval=2.0,
            ) as pbar:
                pbar.refresh()
                refresh_stop = threading.Event()

                def _refresher():
                    while not refresh_stop.wait(5.0):
                        pbar.refresh()

                refresher_thread = threading.Thread(target=_refresher, daemon=True)
                refresher_thread.start()

                try:
                    while remaining:
                        # Wait for at least one future to finish, or poll for
                        # timeouts every POLL_SECONDS.
                        done, remaining = concurrent.futures.wait(
                            remaining,
                            timeout=POLL_SECONDS,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )

                        # --- process completed futures ---
                        for f in done:
                            cfg = future_map[f]
                            name = cfg.get("name", "?")
                            try:
                                result = f.result()
                            except Exception as exc:
                                serve_cfg = cfg.get("serve_config", {})
                                tag = (
                                    f"{name}"
                                    f"[tp{serve_cfg.get('tp', 1)}"
                                    f"pp{serve_cfg.get('pp', 1)}"
                                    f"dp{serve_cfg.get('dp', 1)}]"
                                )
                                result = {
                                    "Model": tag,
                                    "Correct Ratio": "0%",
                                    "Stage": "CRASH",
                                    "Reason": f"{type(exc).__name__}: {exc}",
                                    "Model Path": cfg.get("model_path", ""),
                                }
                                print(f"\n[{name}] Worker crashed: {type(exc).__name__}: {exc}")

                            all_results.append(result)
                            csv_writer.writerow(result)
                            f_csv.flush()
                            pbar.update(1)

                        # --- kill overdue futures ---
                        now = time_module.time()
                        for f in list(remaining):
                            elapsed = now - future_submit_time[f]
                            if elapsed > self.args.model_timeout:
                                cfg = future_map[f]
                                name = cfg.get("name", "?")
                                print(
                                    f"\n[{name}] TIMEOUT after {elapsed:.0f}s "
                                    f"(limit: {self.args.model_timeout}s) — killing"
                                )
                                stop_event.set()    # signal worker to stop
                                f.cancel()          # best-effort for pending futures
                                remaining.discard(f)

                                row = _make_timeout_row(cfg, elapsed)
                                all_results.append(row)
                                csv_writer.writerow(row)
                                f_csv.flush()
                                pbar.update(1)

                                # Reset stop_event for subsequent models
                                stop_event.clear()

                finally:
                    refresh_stop.set()
                    refresher_thread.join()

        # --- summary ---
        self._print_inference_summary(all_results)
        pprint(all_results)

    @staticmethod
    def _print_inference_summary(results: list[dict]) -> None:
        """Print a compact summary of inference results."""
        if not results:
            return

        passed = [r for r in results if r.get("Stage") == "NORMAL_END"]
        failed = [r for r in results if r.get("Stage") not in ("NORMAL_END",)]

        print(f"\n{'='*60}")
        print(f"Inference Summary: {len(results)} total | "
              f"{len(passed)} PASS | {len(failed)} FAIL")
        print(f"{'='*60}")

        # Group by status
        by_stage: dict[str, list[dict]] = {}
        for r in failed:
            stage = r.get("Stage", "?")
            by_stage.setdefault(stage, []).append(r)

        for stage, items in by_stage.items():
            print(f"\n  [{stage}] ({len(items)}):")
            for r in items:
                reason = (r.get("Reason", "") or "")[:150]
                print(f"    - {r['Model']}: {reason}")

        # Print passed models compactly
        if passed:
            names = [r["Model"] for r in passed]
            print(f"\n  [PASSED] ({len(passed)}): {', '.join(names)}")

    def run_performance(self):
        """Run performance benchmarks for all selected models with per-model timeout."""
        all_results = []
        future_map: dict[concurrent.futures.Future, dict] = {}
        future_submit_time: dict[concurrent.futures.Future, float] = {}

        bench_work_dir = os.path.join(self.work_dir, "performance")
        from model_worker import BenchSweepWorker

        for cfg in self.model_list:
            worker = BenchSweepWorker(
                work_dir=bench_work_dir, model_cfg=cfg, gpu_manager=self.gpu_manager
            )
            # GPU allocation timeout: shorter for serial, generous for parallel
            alloc_timeout = 3600 if self.args.concurrency == 1 else 14400
            future = self.executor.submit(
                self._run_with_gate, worker.run, stop_event, alloc_timeout
            )
            future_map[future] = cfg
            future_submit_time[future] = time_module.time()

        POLL_SECONDS = 30
        remaining = set(future_map.keys())

        while remaining:
            done, remaining = concurrent.futures.wait(
                remaining,
                timeout=POLL_SECONDS,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            # Process completed futures
            for f in done:
                cfg = future_map[f]
                name = cfg.get("name", "?")
                try:
                    result = f.result()
                except Exception as exc:
                    result = {
                        "task_name": name,
                        "status": "error",
                        "log_dir": None,
                        "error": f"{type(exc).__name__}: {exc}",
                        "server_command": None,
                        "client_command": None,
                        "env": None,
                    }
                    print(f"\n[{name}] Benchmark worker crashed: {type(exc).__name__}: {exc}")
                all_results.append(result)

            # Kill overdue futures
            now = time_module.time()
            for f in list(remaining):
                elapsed = now - future_submit_time[f]
                if elapsed > self.args.model_timeout:
                    cfg = future_map[f]
                    name = cfg.get("name", "?")
                    print(
                        f"\n[{name}] TIMEOUT after {elapsed:.0f}s "
                        f"(limit: {self.args.model_timeout}s) — killing"
                    )
                    stop_event.set()
                    f.cancel()
                    remaining.discard(f)
                    all_results.append({
                        "task_name": name,
                        "status": "timeout",
                        "log_dir": None,
                        "error": f"Exceeded per-model timeout ({self.args.model_timeout}s)",
                        "server_command": None,
                        "client_command": None,
                        "env": None,
                    })
                    stop_event.clear()

        # deal with the result and create the bench_tasks_result.csv
        df = pd.DataFrame(all_results)
        df = df[
            [
                'task_name', 'status',
                'log_dir', 'error',
                'server_command', 'client_command', 'env'
            ]
        ]
        csv_path = Path(bench_work_dir) / "bench_tasks_result.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False, encoding='utf-8')

        # Print summary
        ok = sum(1 for r in all_results if r.get("status") == "success")
        err = sum(1 for r in all_results if r.get("status") == "error")
        to = sum(1 for r in all_results if r.get("status") == "timeout")
        print(f"\n{'='*60}")
        print(f"Performance Summary: {len(all_results)} total | "
              f"{ok} OK | {err} ERROR | {to} TIMEOUT")
        print(f"{'='*60}")
        for r in all_results:
            if r.get("status") != "success":
                print(f"  [{r['status'].upper()}] {r['task_name']}: "
                      f"{(r.get('error') or '')[:120]}")

    def run_all(self):
        self.record_environment()
        if self.args.dry_run:
            self._print_selected_models()
            return
        try:
            if self.args.infer:
                self.run_inference()

            if self.args.perf:
                self.run_performance()

        except KeyboardInterrupt:
            print("Ctrl-C detected, terminating all tests...")
            stop_event.set()

        except Exception as e:
            print(f"Script has unexpected exited: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=SchedularArgs.parser_help)
    SchedularArgs.add_cli_args(parser)

    args = parser.parse_args()

    sche = Scheduler(SchedularArgs.from_cli_args(args))
    sche.run_all()
