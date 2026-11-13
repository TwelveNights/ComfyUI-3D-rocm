"""
Run by ComfyUI-Manager after it installs this node pack (or by hand: ``<ComfyUI's python> install.py``).

Installs the Python packages TripoSG and the Hunyuan3D-2.1 painter need, with the installed torch / torchvision /
torchaudio / triton / rocm packages pinned to their current versions so pip never swaps them for other builds.
Nothing is compiled.
"""

import importlib.metadata as metadata
import os
import re
import subprocess
import sys
import tempfile

PACKAGES = [
    "diffusers>=0.32",
    "transformers>=4.46",
    "huggingface_hub",
    "einops",
    "omegaconf",
    "jaxtyping",
    "typeguard",
    "scikit-image",
    "trimesh",
    "pymeshlab",
    "xatlas",
    "opencv-python-headless",
    "realesrgan",
    "basicsr",
    "peft",
]
PINNED = re.compile(r"(torch|torchvision|torchaudio|triton|pytorch-triton|rocm)", re.IGNORECASE)


def log(msg: str) -> None:
    print(f"[ComfyUI-3D-rocm] {msg}", flush=True)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        constraints = os.path.join(tmp, "constraints.txt")
        with open(constraints, "w") as f:
            for dist in metadata.distributions():
                name = dist.metadata["Name"] or ""
                if PINNED.match(name):
                    f.write(f"{name}=={dist.version}\n")
        cmd = [sys.executable, "-s", "-m", "pip", "install", "-c", constraints, *PACKAGES]
        log(" ".join(cmd))
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            log(f"install failed ({e}); see https://github.com/TwelveNights/ComfyUI-3D-rocm#install")
            return 1
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
