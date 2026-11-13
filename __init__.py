"""ComfyUI nodes for TripoSG and Hunyuan3D-2.1 PBR texturing that run on AMD GPUs (ROCm), Linux and Windows."""

import os
import sys

# MIOpen's default first-run search tunes every convolution shape for many minutes; FAST picks one right away
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

# TripoSG's code imports itself as `triposg`
_vendor = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "triposg_pkg")
if _vendor not in sys.path:
    sys.path.append(_vendor)

from comfy_api.latest import ComfyExtension  # noqa: E402

from .nodes import NODES  # noqa: E402


class ThreeDROCmExtension(ComfyExtension):
    async def get_node_list(self):
        return NODES


async def comfy_entrypoint() -> ComfyExtension:
    return ThreeDROCmExtension()
