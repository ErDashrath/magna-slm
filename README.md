# magna-slm

## First check

Run the hardware probe without installing model dependencies:

```bash
PYTHONPATH=src python3 -m magna_slm
```

It reports system RAM, CPU count, NVIDIA VRAM when `nvidia-smi` is available, and free space on the current disk.
