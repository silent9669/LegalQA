"""CUDA stage cleanup, VRAM snapshotting, and Trainer telemetry callback for V16."""

import gc
import logging
import time
from typing import Any, Dict, List, Optional, Sequence

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import TrainerCallback
except ImportError:
    class TrainerCallback:
        """Fallback base class when transformers is not installed."""
        pass

logger = logging.getLogger(__name__)


def cleanup_cuda_stage(*objects: Any, devices: Sequence[int] = (0, 1), empty_cache: bool = True) -> None:
    """Explicit stage boundary cleanup destroying references, running gc, and emptying CUDA cache."""
    # 1. Unbind / delete objects
    for obj in objects:
        del obj

    # 2. Python garbage collection
    gc.collect()

    # 3. CUDA cache release across target devices
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        for dev in devices:
            if dev < num_devices:
                try:
                    torch.cuda.set_device(dev)
                    if empty_cache:
                        torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
                except Exception as e:
                    logger.debug(f"Could not empty CUDA cache on device {dev}: {e}")
        gc.collect()


def snapshot_cuda_memory(label: str = "", devices: Sequence[int] = (0, 1)) -> Dict[str, Any]:
    """Capture snapshot of allocated, reserved, and free VRAM across target CUDA devices."""
    if torch is None or not hasattr(torch, "cuda") or not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "label": label,
            "timestamp": time.time(),
            "devices": {},
        }

    out_devices: Dict[int, Dict[str, float]] = {}
    num_devices = torch.cuda.device_count()

    for dev in devices:
        if dev < num_devices:
            try:
                allocated = torch.cuda.memory_allocated(dev) / (1024**2)
                reserved = torch.cuda.memory_reserved(dev) / (1024**2)
                max_allocated = torch.cuda.max_memory_allocated(dev) / (1024**2)

                free_mb = 0.0
                total_mb = 0.0
                if hasattr(torch.cuda, "mem_get_info"):
                    free_b, total_b = torch.cuda.mem_get_info(dev)
                    free_mb = free_b / (1024**2)
                    total_mb = total_b / (1024**2)
                elif hasattr(torch.cuda, "get_device_properties"):
                    props = torch.cuda.get_device_properties(dev)
                    total_mb = props.total_memory / (1024**2)
                    free_mb = total_mb - reserved

                out_devices[dev] = {
                    "allocated_mb": round(allocated, 2),
                    "reserved_mb": round(reserved, 2),
                    "max_allocated_mb": round(max_allocated, 2),
                    "free_mb": round(free_mb, 2),
                    "total_mb": round(total_mb, 2),
                }
            except Exception as e:
                logger.debug(f"Failed to read CUDA memory info for device {dev}: {e}")

    return {
        "cuda_available": True,
        "label": label,
        "timestamp": time.time(),
        "devices": out_devices,
    }


class TrainerMemoryCallback(TrainerCallback):
    """Callback logging and tracking VRAM usage during QLoRA training steps."""

    def __init__(
        self,
        log_every_n_steps: int = 50,
        empty_cache_every_n_steps: Optional[int] = None,
        target_devices: Sequence[int] = (0, 1),
    ):
        self.log_every_n_steps = max(1, log_every_n_steps)
        self.empty_cache_every_n_steps = empty_cache_every_n_steps
        self.target_devices = target_devices
        self.history: List[Dict[str, Any]] = []

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        step = getattr(state, "global_step", 0)

        # Log memory telemetry every N steps
        if step > 0 and step % self.log_every_n_steps == 0:
            snap = snapshot_cuda_memory(
                label=f"step_{step}",
                devices=self.target_devices,
            )
            entry = {
                "step": step,
                "snapshot": snap,
            }
            self.history.append(entry)

            if snap.get("cuda_available") and 0 in snap.get("devices", {}):
                d0 = snap["devices"][0]
                print(
                    f"[VRAM Telemetry @ Step {step}] GPU 0: "
                    f"allocated={d0['allocated_mb']:.1f} MB, "
                    f"reserved={d0['reserved_mb']:.1f} MB, "
                    f"free={d0['free_mb']:.1f} MB, "
                    f"peak={d0['max_allocated_mb']:.1f} MB"
                )

        # Optional fragmentation guard
        if (
            self.empty_cache_every_n_steps is not None
            and step > 0
            and step % self.empty_cache_every_n_steps == 0
        ):
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
