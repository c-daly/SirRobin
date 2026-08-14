# Milestone 0 starting performance envelope

**Date:** 2026-08-09
**Branch:** `restart/original-baseline`
**Measured parent:** `0f472b2`
**Surface:** `tools/run_world.py`, cheap 0.1-second composed-world fixture
**Status:** exploratory operational baseline, not a stable benchmark or scientific result

## Result

The committed baseline world runs headlessly and closes its exact nutrient books at
8, 128, and 5,000 cloned donor bodies on both CPU and CUDA. It is not remotely fast
enough for the intended ecological loop: at 128 bodies the CPU advances only 1.20
simulated seconds per wall second, versus the provisional future target of 300 for
the reduced loop. This is a starting measurement, not a failure of that future gate.

The cheap fixture uses 12 mechanics substeps per 0.1-second economy interval. The
shipped configuration requires 1,036,800 mechanics substeps per 8,640-second economy
interval. Milestone 1 therefore remains necessary before useful ecological runs.

## Environment

- WSL2 kernel `6.6.114.1-microsoft-standard-WSL2`
- Python 3.12.13
- PyTorch `2.13.0+cu130`
- CPU: AMD Ryzen 7 8700F, 8 cores / 16 logical CPUs
- GPU: NVIDIA GeForce RTX 5070, 12,227 MiB; Windows driver 596.36; reported CUDA 13.2
- Python environment and caches were on WSL-native `/tmp` storage.
- The managed command sandbox hides WSL's `/dev/dxg`. CUDA commands were therefore
  run outside that sandbox after `nvidia-smi` and `torch.cuda.is_available()` both
  confirmed the RTX 5070.

## Commands

From the `/mnt/c` worktree:

```bash
for body_count in 8 128 5000; do
  PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
    /tmp/sirrobin-original-venv/bin/python tools/run_world.py \
    --seconds 0.1 --bodies "$body_count" --device cpu
done
```

The same command used `--device cuda` for the GPU measurements. Each row below is
one cold process and one economy interval. CUDA was synchronized after construction
and after advancement so asynchronous setup work could not leak into the step timer.

## Starting envelope

| Bodies | Device | Setup wall s | Advance wall s | Sim s / wall s | Exact closure |
|---:|:---:|---:|---:|---:|:---:|
| 8 | CPU | 0.127255 | 0.061468 | 1.626875 | yes |
| 128 | CPU | 0.375746 | 0.083668 | 1.195204 | yes |
| 5,000 | CPU | 7.808954 | 1.205801 | 0.082932 | yes |
| 8 | CUDA | 0.916331 | 0.589327 | 0.169685 | yes |
| 128 | CUDA | 0.964591 | 0.581704 | 0.171909 | yes |
| 5,000 | CUDA | 7.400966 | 0.596838 | 0.167550 | yes |

CUDA launch overhead dominates the small batches. At 5,000 bodies CUDA is about
2.0 times faster than CPU during advancement, but neither backend is useful at the
present full-mechanics cadence. No backend-specific model change is justified by
this one exploratory measurement.

## `/mnt/c` versus WSL-native source

The 128-body CPU smoke was repeated once from a shared clone at
`/tmp/sirrobin-original-native-0f472b2`, using the same WSL-native Python environment.

| Source location | Setup wall s | Advance wall s | Sim s / wall s |
|:---|---:|---:|---:|
| `/mnt/c` worktree | 0.375746 | 0.083668 | 1.195204 |
| WSL-native `/tmp` clone | 0.288039 | 0.080324 | 1.244954 |

The native source tree improved advancement throughput by about 4.2 percent in this
single bounded comparison. Setup improved more, but setup is not the dominant cost of
a long simulation. That is not enough evidence to move the active worktree or build
filesystem-mirroring infrastructure. Keep environments, caches, and artifacts on WSL
storage and leave source placement alone for now.
