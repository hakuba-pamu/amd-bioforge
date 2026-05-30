"""ROCm utilities for AMD-BioForge."""

import torch


def get_amd_gpu_info() -> dict:
    """Get AMD GPU information."""
    if not torch.cuda.is_available():
        return {"detected": False, "name": "None", "vram_gb": 0, "fp64_tflops": 0}

    props = torch.cuda.get_device_properties(0)
    name = props.name.lower()

    is_amd = any(x in name for x in ["amd", "radeon", "instinct", "gfx"])

    if not is_amd:
        return {
            "detected": False,
            "name": props.name,
            "vram_gb": round(props.total_memory / 1024**3, 1),
            "warning": "NVIDIA GPU detected. FP64 performance may be limited.",
        }

    return {
        "detected": True,
        "name": props.name,
        "vram_gb": round(props.total_memory / 1024**3, 1),
        "fp64_tflops": estimate_fp64_tflops(name, props),
    }


def estimate_fp64_tflops(name: str, props) -> float:
    """Estimate FP64 TFLOPS based on GPU architecture."""
    if "mi250" in name:
        return 47.9
    elif "mi210" in name:
        return 22.6
    elif "mi100" in name:
        return 11.5
    elif "mi300" in name:
        return 163.4
    elif "w7900" in name or "7900" in name:
        return 1.9
    elif "6900" in name or "6800" in name:
        return 1.4
    else:
        return round(props.multi_processor_count * 64 * 2 * props.clock_rate / 1e12 * 2, 1)


def optimize_for_amd(model) -> None:
    """Apply AMD GPU-specific optimizations."""
    if not torch.cuda.is_available():
        return

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
