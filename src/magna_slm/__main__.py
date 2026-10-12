from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from .config import COLIBRI_PROFILE


def _nvidia_vram_mb() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            check=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    values = [int(line) for line in result.stdout.splitlines() if line.strip()]
    return max(values) if values else None


def hardware_report() -> dict[str, object]:
    memory_kb = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            memory_kb = int(line.split()[1])
            break

    vram_mb = _nvidia_vram_mb()
    disk = shutil.disk_usage(Path.cwd())
    ram_gb = round(memory_kb / 1024 / 1024, 2)
    vram_gb = vram_mb / 1024 if vram_mb is not None else 0.0
    return {
        "cpu_count": os.cpu_count(),
        "ram_gb": ram_gb,
        "nvidia_vram_mb": vram_mb,
        "disk_free_gb": round(disk.free / 1024**3, 2),
        "profile": COLIBRI_PROFILE.name,
        "profile_compatible": ram_gb >= COLIBRI_PROFILE.ram_gb * 0.9
        and vram_gb >= COLIBRI_PROFILE.vram_gb * 0.9,
    }


if __name__ == "__main__":
    print(json.dumps(hardware_report(), indent=2))