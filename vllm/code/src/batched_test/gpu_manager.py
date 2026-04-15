# SPDX-License-Identifier: Apache-2.0

from functools import wraps, cache
from typing_extensions import ParamSpec
from typing import Callable, TypeVar
import os 
import sys

import threading
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import pymxml as ml

from pprint import pprint

_P = ParamSpec("_P")
_R = TypeVar("_R")

ml_available = False
try:
    try:
        ml.nvmlInit()
        ml_available = True
    except Exception:
        # On Jetson, NVML is not supported.
        # try to nvmlInit
        import vllm.third_party.pynvml as ml
        ml.nvmlInit()
        ml_available = True
        print(f"[WARN] mx NVML is not available, but use nvidia NVML instead of it.")
finally:
    if ml_available:
        ml.nvmlShutdown()

assert ml_available, "mx NVML and nvidia NVML are both not available on this system."

# pprint(gpu_manager.get_gpu_count()) # initialize pymxml
# pprint(gpu_manager.get_gpu_memory_list()) # initialize pymxml
# pprint(gpu_manager.get_free_gpu_indices())
# pprint(gpu_manager.get_all_gpu_process_info())

def with_mxml_context(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        ml.nvmlInit()
        try:
            return fn(*args, **kwargs)
        finally:
            ml.nvmlShutdown()

    return wrapper


class GPUManager:
    _instance = None
    _lock = threading.Lock()  # class-level lock for singleton creation

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, max_idle_mem_mb=900):
        if hasattr(self, "_initialized"):
            return

        self._initialized = True

        # TODO(hank): auto-detect gpu count and gpu_ids
        self.max_idle_mem = max_idle_mem_mb
        self.global_mutex = threading.Lock()
        self.occupied_gpus = set()

    @cache
    @with_mxml_context
    def get_gpu_count(self) -> int:
        return ml.nvmlDeviceGetCount()

    @with_mxml_context
    def get_gpu_memory_list(self) -> list[dict]:
        """
        Get a list of dict, length equal to the number of GPUs, each dict contains used, free, and total memory (MiB).
        Uses pynvml to get memory usage.
        """
        gpu_count = self.get_gpu_count()
        mems_infos = []
        for i in range(gpu_count):
            handle = ml.nvmlDeviceGetHandleByIndex(i)
            info = ml.nvmlDeviceGetMemoryInfo(handle)
            mems_infos.append(
                {
                    "used": int(info.used) // (1024 * 1024),  # type: ignore
                    "free": int(info.free) // (1024 * 1024),  # type: ignore
                    "total": int(info.total) // (1024 * 1024),  # type: ignore
                }
            )
        return mems_infos

    @with_mxml_context
    def get_free_gpu_indices(self, used_threshold_mb: int | None = None) -> list[int]:
        """
        Get a list of GPU indices that have free memory above the given threshold (MiB).
        """
        if used_threshold_mb is None:
            used_threshold_mb = self.max_idle_mem

        mem_info = self.get_gpu_memory_list()
        free_list = [
            i
            for i, mem in enumerate(mem_info)
            if mem["used"] <= used_threshold_mb and i not in self.occupied_gpus
        ]
        return free_list

    @with_mxml_context
    def get_gpu_process_info(self, gpu_index: int) -> list[dict]:
        """
        Get memory info for a specific GPU index.
        Returns a dict with keys:
         - pid: Process ID using the GPU.
         - usedGpuMemory: Amount of GPU memory used by the process (MiB).
        """
        handle = ml.nvmlDeviceGetHandleByIndex(gpu_index)
        gpu_process_infos = []
        try:
            gpu_process_infos = ml.nvmlDeviceGetComputeRunningProcesses_v3(handle)
        except ml.NVMLError as e:
            print(f"[WARN] Failed to get processes for GPU {gpu_index}: {e}")
        except Exception as e:
            print(
                f"[ERROR] Unexpected error getting processes for GPU {gpu_index}: {e}"
            )
        return [
            {
                "pid": proc.pid,
                "usedGpuMemory": proc.usedGpuMemory // (1024 * 1024)
                if proc.usedGpuMemory is not None
                else None,
            }
            for proc in gpu_process_infos
        ]  # type: ignore

    @with_mxml_context
    def get_gpu_process_pid(self, gpu_index_list: list[int]) -> list[int]:
        """
        Get a list of PIDs for processes using the specified GPU indices.
        """
        pids = []
        for gpu_index in gpu_index_list:
            info = self.get_gpu_process_info(gpu_index)
            pids.extend([proc["pid"] for proc in info])
        return pids

    @with_mxml_context
    def get_all_gpu_process_info(self) -> dict[int, list[dict]]:
        """
        Get a dictionary mapping GPU index to a list of processes using that GPU.
        Each process info is a dict with keys:
         - pid: Process ID using the GPU.
         - usedGpuMemory: Amount of GPU memory used by the process (MiB).
        """
        gpu_count = self.get_gpu_count()
        gpu_process_infos: dict[int, list[dict]] = {}
        for i in range(gpu_count):
            info = self.get_gpu_process_info(i)
            gpu_process_infos[i] = info  # type: ignore
        return gpu_process_infos
    
    @with_mxml_context
    def get_gpu_name(self, gpu_index: int) -> str:
        handle = ml.nvmlDeviceGetHandleByIndex(gpu_index)
        try:
            name_bytes =  ml.nvmlDeviceGetName(handle)
            if isinstance(name_bytes, bytes):
               name = name_bytes.decode("utf-8")
            else:
               name = name_bytes
            return name
        except Exception as e:
            print(
                f"[ERROR] Unexpected error getting name for GPU {gpu_index}: {e}"
            )

    @with_mxml_context
    def get_all_gpu_names(self) -> dict[int, str]:
        gpu_count = self.get_gpu_count()
        gpu_names = {}
        for i in range(gpu_count):
            name = self.get_gpu_name(i)
            gpu_names[i] = name
        return gpu_names

    def allocate(self, num_required: int) -> list[int]:
        """
        !!!LOCKING!!!
        Allocate GPUs based on current free memory status.
        Returns a list of allocated GPU indices.
        """
        if num_required > self.get_gpu_count():
            raise ValueError("Requested more GPUs than available on the system.")

        with self.global_mutex:
            free_gpus = self.get_free_gpu_indices()
            if len(free_gpus) < num_required:
                return []
            allocated = free_gpus[:num_required]
            self.occupied_gpus.update(allocated)
            return allocated

    def release(self, gpu_indices: list[int]) -> None:
        """
        !!!LOCKING!!!
        Release previously allocated GPUs.
        """
        with self.global_mutex:
            for idx in gpu_indices:
                self.occupied_gpus.discard(idx)

