
import os
import torch
import contextlib
import time
from megatron.core import parallel_state

PHASE_LOGGER = True
PHASE_LAYER_LOGGER = False

def _is_rank0():
    try:
        if parallel_state.is_unitialized() if hasattr(parallel_state, "is_unitialized") else False:
            return int(os.environ.get("RANK", "0")) == 0
        # Prefer Megatron’s notion of data-parallel rank if available
        if hasattr(parallel_state, "get_data_parallel_rank"):
            return parallel_state.get_data_parallel_rank() == 0
    except Exception:
        pass
    return int(os.environ.get("RANK", "0")) == 0

def _bytes(x: int) -> str:
    x = float(x)
    for u in ["B","KB","MB","GB","TB","PB"]:
        if x < 1024: return f"{x:.2f}{u}"
        x /= 1024
    return f"{x:.2f}EB"

def _mem_stats(device=None):
    if device is None:
        device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    alloc    = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak_a   = torch.cuda.max_memory_allocated(device)
    peak_r   = torch.cuda.max_memory_reserved(device)
    return {
        "allocated": _bytes(alloc),
        "reserved":  _bytes(reserved),
        "peak_allocated": _bytes(peak_a),
        "peak_reserved":  _bytes(peak_r),
    }

def _barrier():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass

@contextlib.contextmanager
def mem_phase(name: str, do_barrier: bool = False):
    """Emit per-phase CUDA memory stats (allocated/reserved + peaks)."""
    if not PHASE_LOGGER or not torch.cuda.is_available():
        yield
        return
    device = torch.cuda.current_device()
    if do_barrier:
        _barrier()
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    try:
        yield
    finally:
        torch.cuda.synchronize(device)
        dt = time.time() - t0
        stats = _mem_stats(device)
        world_rank = torch.distributed.get_rank() 
        print(
            f"[Rank {world_rank}] [MEM][{name}] dt={dt:.3f}s | "
            f"alloc={stats['allocated']} res={stats['reserved']} "
            f"peak_alloc={stats['peak_allocated']} peak_res={stats['peak_reserved']}",
            flush=True,
        )

def attach_module_peaks(module: torch.nn.Module, device=None):
    """Optionally print per-leaf-module forward peaks on rank 0."""
    if not (PHASE_LOGGER and PHASE_LAYER_LOGGER and torch.cuda.is_available() and _is_rank0()):
        return
    if device is None:
        device = torch.cuda.current_device()

    for name, m in module.named_modules():
        # Skip container modules; focus on compute layers
        if any(True for _ in m.children()):
            continue

        def pre_hook(mod, inp, n=name):
            torch.cuda.reset_peak_memory_stats(device)

        def post_hook(mod, out, n=name):
            pa = torch.cuda.max_memory_allocated(device)
            pr = torch.cuda.max_memory_reserved(device)
            print(f"[LAYER][{n}] peak_alloc={_bytes(pa)} peak_res={_bytes(pr)}", flush=True)

        m.register_forward_pre_hook(pre_hook)
        m.register_forward_hook(post_hook)

# NVTX helpers (nice in Chrome/Nsight traces)
try:
    import torch.cuda.nvtx as nvtx
    def nvtx_range(name):
        return contextlib.ExitStack().__enter__() if not torch.cuda.is_available() else _NVTX(name)
    class _NVTX:
        def __init__(self, name): self.name=name
        def __enter__(self): 
            try: nvtx.range_push(self.name)
            except Exception: pass
        def __exit__(self, exc_type, exc, tb):
            try: nvtx.range_pop()
            except Exception: pass
except Exception:
    def nvtx_range(name):  # fallback no-op
        return contextlib.nullcontext()

# Wrapper to attach per-layer hooks as soon as the model is created (works with pretrain())
def _model_provider_with_phase(gbuilder, *args, **kwargs):
    mdl = model_provider(gbuilder, *args, **kwargs)
    if PHASE_LOGGER and PHASE_LAYER_LOGGER:
        try:
            if not hasattr(mdl, "_phase_layer_hooks_attached"):
                attach_module_peaks(mdl)
                mdl._phase_layer_hooks_attached = True
        except Exception:
            pass
    return mdl
