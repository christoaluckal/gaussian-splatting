# Gaussian Splatting Metrics

## Purpose

This note explains how to replace misleading GPU memory reporting with metrics that are useful for comparing a baseline Gaussian Splatting pipeline against a proposed initialization method.

## Problem

Using `torch.cuda.memory_reserved()` alone is misleading.

It reports allocator-reserved memory, not the true live memory footprint of the model or the true peak live memory used during training. It is affected by caching, fragmentation, and allocator behavior.

## Recommended Memory Metrics

Use three categories of metrics:

1. representation size
2. peak training memory
3. quality-efficiency tradeoff

## 1. GPU Memory Stats

Use a helper like this:

```python
def _get_gpu_memory_stats_mb():
    if not torch.cuda.is_available():
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "max_allocated_mb": 0.0,
            "max_reserved_mb": 0.0,
        }

    device = torch.cuda.current_device()
    bytes_per_mb = 1024.0 * 1024.0
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / bytes_per_mb,
        "reserved_mb": torch.cuda.memory_reserved(device) / bytes_per_mb,
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / bytes_per_mb,
        "max_reserved_mb": torch.cuda.max_memory_reserved(device) / bytes_per_mb,
    }
```

Meaning:

- `allocated_mb`: current live tensor memory
- `reserved_mb`: memory reserved by the PyTorch allocator
- `max_allocated_mb`: peak live tensor memory since the last reset
- `max_reserved_mb`: peak reserved memory since the last reset

Reset peak stats before the measurement window you care about:

```python
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
```

Placement:

- before scene/model creation: full-run peak memory
- after scene/model creation but before the loop: training-only peak memory

For most comparisons, `max_allocated_mb` should be the main GPU memory metric.

## 2. Gaussian Model Memory

To estimate the representation size directly:

```python
def _tensor_nbytes(t):
    if t is None:
        return 0
    return t.numel() * t.element_size()


def _gaussian_model_memory_mb(gaussians):
    total_bytes = 0
    seen = set()

    for name, value in gaussians.__dict__.items():
        if torch.is_tensor(value):
            ptr = value.data_ptr()
            if ptr != 0 and ptr not in seen:
                total_bytes += _tensor_nbytes(value)
                seen.add(ptr)
        elif isinstance(value, (list, tuple)):
            for v in value:
                if torch.is_tensor(v):
                    ptr = v.data_ptr()
                    if ptr != 0 and ptr not in seen:
                        total_bytes += _tensor_nbytes(v)
                        seen.add(ptr)

    return total_bytes / (1024.0 * 1024.0)
```

Why this matters:

- it approximates the true memory footprint of the Gaussian representation
- it separates model state from activations, gradients, optimizer state, and allocator cache

If `GaussianModel` stores tensors in nested containers or custom objects, tailor the helper to that structure.

## 3. Runtime CSV Fields

Prefer runtime fields like:

```python
runtime_csv_fields = [
    "event",
    "iteration",
    "gpu_allocated_mb",
    "gpu_reserved_mb",
    "gpu_peak_allocated_mb",
    "gpu_peak_reserved_mb",
    "gaussian_model_mb",
    "scene_load_time_sec",
    "total_training_time_sec",
]
```

## 4. Scene-Load Metrics

At scene-load time:

```python
scene_mem = _get_gpu_memory_stats_mb()
scene_model_mb = _gaussian_model_memory_mb(gaussians)

_append_csv_row(
    runtime_metrics_csv,
    runtime_csv_fields,
    {
        "event": "scene_load",
        "iteration": 0,
        "gpu_allocated_mb": scene_mem["allocated_mb"],
        "gpu_reserved_mb": scene_mem["reserved_mb"],
        "gpu_peak_allocated_mb": scene_mem["max_allocated_mb"],
        "gpu_peak_reserved_mb": scene_mem["max_reserved_mb"],
        "gaussian_model_mb": scene_model_mb,
        "scene_load_time_sec": scene_load_time_sec,
        "total_training_time_sec": "",
    },
)
```

## 5. Per-Iteration Metrics

Inside the training loop:

```python
mem_stats = _get_gpu_memory_stats_mb()
gaussian_model_mb = _gaussian_model_memory_mb(gaussians)

_append_csv_row(
    runtime_metrics_csv,
    runtime_csv_fields,
    {
        "event": "iteration",
        "iteration": iteration,
        "gpu_allocated_mb": mem_stats["allocated_mb"],
        "gpu_reserved_mb": mem_stats["reserved_mb"],
        "gpu_peak_allocated_mb": mem_stats["max_allocated_mb"],
        "gpu_peak_reserved_mb": mem_stats["max_reserved_mb"],
        "gaussian_model_mb": gaussian_model_mb,
        "scene_load_time_sec": "",
        "total_training_time_sec": "",
    },
)
```

## 6. Priority of Metrics

Primary metrics:

- final PSNR, SSIM, LPIPS
- final number of Gaussians
- Gaussian model memory in MB
- peak allocated GPU memory in MB
- total training time
- iteration time or throughput

Secondary metrics:

- reserved GPU memory
- peak reserved GPU memory

Reserved-memory metrics are useful context, but they should not be the main efficiency claim.

## 7. Matched-Budget Comparisons

A likely reviewer objection is that a proposed method may do better simply because it uses more Gaussians.

To address that, compare methods under matched budgets:

- same number of Gaussians
- same Gaussian model memory
- same peak allocated GPU memory
- same training time

Then compare PSNR, SSIM, and LPIPS.

## 8. Derived Efficiency Metrics

Useful post-run summaries:

```python
psnr_gain = proposed_psnr - baseline_psnr
gaussian_ratio = proposed_num_gaussians / baseline_num_gaussians
peak_mem_ratio = proposed_peak_allocated_mb / baseline_peak_allocated_mb
```

Also useful:

```python
psnr_per_100k_gaussians = psnr / (num_gaussians / 100000.0)
psnr_per_gb_peak = psnr / (peak_allocated_mb / 1024.0)
```

These do not replace raw metrics, but they help summarize efficiency.

## 9. Reporting Table

Use a table like this:

| Method | Final PSNR | Final #Gaussians | Gaussian State MB | Peak Allocated MB | Total Train Time |
|---|---:|---:|---:|---:|---:|
| Baseline | 22.5 | 530k | X | A | T1 |
| Proposed | 24.4 | 687k | Y | B | T2 |

Then report deltas for:

- PSNR
- number of Gaussians
- Gaussian state MB
- peak allocated MB
- training time

## 10. Recommended Interpretation Language

If the proposed method still looks favorable under proper metrics:

> The proposed initialization increases representation size and training memory moderately, but yields a disproportionately larger gain in reconstruction quality. This suggests that the added primitives are informative rather than redundant.

If matched-budget experiments also favor the proposed method:

> At matched model size or memory budget, the proposed initialization still improves PSNR, indicating that the gain comes from better initialization rather than simply using more Gaussians.

## Bottom Line

Do not use `torch.cuda.memory_reserved()` alone as the main evidence for efficiency.

Use:

- `torch.cuda.max_memory_allocated()` for peak training memory
- direct tensor-size accounting for Gaussian model size
- matched-budget quality comparisons to make the argument convincing
