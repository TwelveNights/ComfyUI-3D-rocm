"""
TripoSG (image -> mesh) and Hunyuan3D-2.1 Paint (mesh + image -> PBR-textured mesh) as ComfyUI nodes, without CUDA or
C++ extensions, so they run on AMD GPUs (ROCm, Linux and Windows) as well as NVIDIA. Meshes are ComfyUI MESH objects,
so ComfyUI's own nodes do background removal, Hunyuan3D-2.1 shape generation, decimation and GLB export.
"""

import contextlib
import logging
import os
import tempfile
import time

import comfy.model_management as mm
import folder_paths
import numpy as np
import torch
from comfy_api.latest import IO, Types
from PIL import Image

from .rocm_attention import use_sage_attention

log = logging.getLogger(__name__)

TRIPOSG_DIR = os.path.join(folder_paths.models_dir, "triposg", "TripoSG")
HY3D_DIR = os.path.join(folder_paths.models_dir, "hunyuan3d")
REALESRGAN = "RealESRGAN_x4plus.pth"
REALESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

_cache = {}

ATTENTION = ["auto", "sage", "pytorch"]
ATTENTION_TOOLTIP = "auto: SageAttention when installed (much faster on AMD), else PyTorch's attention"


@contextlib.contextmanager
def backends(mode):
    """While the node runs: SageAttention for torch's scaled_dot_product_attention, and on ROCm MIOpen convolutions,
    which ComfyUI turns off (the painter's convolutions are 2.5x slower without them on Windows). Restored after."""
    import torch.nn.functional as F

    saved_sdpa, saved_cudnn = F.scaled_dot_product_attention, torch.backends.cudnn.enabled
    if mode != "pytorch" and not use_sage_attention() and mode == "sage":
        raise RuntimeError("attention=sage, but SageAttention is not installed (pip install sageattention-rocm on "
                           "AMD GPUs, sageattention on NVIDIA)")
    if torch.version.hip:
        torch.backends.cudnn.enabled = True
    try:
        yield
    finally:
        F.scaled_dot_product_attention = saved_sdpa
        torch.backends.cudnn.enabled = saved_cudnn


def free_gpu():
    mm.unload_all_models()
    mm.soft_empty_cache()


def foreground(image, mask=None):
    """(H, W, 3) image and alpha in [0, 1]: the given mask, else everything that differs from the corner color."""
    img = image.float().clamp(0, 1)
    if mask is not None and mask.max() - mask.min() < 0.5:  # LoadImage's mask of an image without alpha
        mask = None
    if mask is not None:
        alpha = mask.float()
        if alpha.shape != img.shape[:2]:
            alpha = torch.nn.functional.interpolate(alpha[None, None], size=img.shape[:2], mode="bilinear")[0, 0]
        if alpha.mean() > 0.5 and alpha[0, 0] > 0.5:  # ComfyUI masks are often "1 = background"
            alpha = 1.0 - alpha
    else:
        corners = torch.stack([img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]]).median(0).values
        alpha = ((img - corners).abs().amax(-1) > 0.06).float()
    if alpha.max() <= 0:
        raise ValueError("no foreground found in the image; connect a mask (e.g. from Remove Background)")
    return img, alpha.clamp(0, 1)


def crop_on_white(img, alpha, padding_ratio=0.1):
    """TripoSG's preprocessing: composite on white, crop to the object and pad to a square."""
    rgb = img * alpha[..., None] + (1 - alpha[..., None])
    ys, xs = torch.nonzero(alpha > 0.5, as_tuple=True)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    crop = rgb[y0:y1, x0:x1].permute(2, 0, 1)
    h, w = crop.shape[1:]
    if w > h:
        pw = int(w * padding_ratio)
        ph = int(pw + (w - h) / 2)
    else:
        ph = int(h * padding_ratio)
        pw = int(ph + (h - w) / 2)
    crop = torch.nn.functional.pad(crop, (pw, pw, ph, ph), value=1.0)
    return Image.fromarray((crop.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))


def mesh_items(mesh):
    for i in range(mesh.vertices.shape[0]):
        v, f = mesh.vertices[i], mesh.faces[i]
        if mesh.vertex_counts is not None:
            v, f = v[: int(mesh.vertex_counts[i])], f[: int(mesh.face_counts[i])]
        yield v.float().cpu().numpy(), f.long().cpu().numpy()


# --------------------------------------------------------------------------------------------------- TripoSG


class TripoSGLoader(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoSGLoaderROCm",
            display_name="TripoSG Loader (ROCm)",
            category="3d/TripoSG",
            description="Loads TripoSG (downloaded to models/triposg on first use, 7.5 GB).",
            inputs=[IO.Combo.Input("precision", options=["fp16", "bf16", "fp32"], default="fp16")],
            outputs=[IO.Custom("TRIPOSG_MODEL").Output(display_name="model")],
        )

    @classmethod
    def execute(cls, precision):
        from huggingface_hub import snapshot_download
        from triposg.pipelines.pipeline_triposg import TripoSGPipeline

        if not os.path.exists(os.path.join(TRIPOSG_DIR, "model_index.json")):
            snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=TRIPOSG_DIR)
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
        key = ("triposg", precision)
        if key not in _cache:
            _cache[key] = TripoSGPipeline.from_pretrained(TRIPOSG_DIR).to(dtype=dtype)
        return IO.NodeOutput({"pipe": _cache[key], "dtype": dtype})


class TripoSGImageToMesh(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoSGImageToMeshROCm",
            display_name="TripoSG Image to Mesh (ROCm)",
            category="3d/TripoSG",
            description="Image of a single object -> untextured mesh. Connect the object's mask (e.g. from Remove "
                        "Background) unless the background is plain.",
            inputs=[
                IO.Custom("TRIPOSG_MODEL").Input("model"),
                IO.Image.Input("image"),
                IO.Mask.Input("mask", optional=True),
                IO.Int.Input("steps", default=25, min=1, max=100, tooltip="50 = upstream default; 25 is as good for "
                             "most objects"),
                IO.Float.Input("guidance", default=7.0, min=0.0, max=20.0, step=0.1),
                IO.Int.Input("seed", default=42, min=0, max=0xFFFFFFFF, control_after_generate=True),
                IO.Int.Input("octree_depth", default=8, min=6, max=9, tooltip="Surface resolution: 8 = 256³ "
                             "(~300K faces, plenty for game assets), 9 = 512³ (~1.2M faces, upstream default)"),
                IO.Combo.Input("attention", options=ATTENTION, default="auto", tooltip=ATTENTION_TOOLTIP),
                IO.Boolean.Input("keep_loaded", default=False, tooltip="Keep the model on the GPU between runs "
                                 "(otherwise it waits in CPU memory)"),
            ],
            outputs=[IO.Mesh.Output()],
        )

    @classmethod
    def execute(cls, model, image, steps, guidance, seed, octree_depth, attention="auto", keep_loaded=False,
                mask=None):
        pipe = model["pipe"]
        device = mm.get_torch_device()
        free_gpu()
        t0 = time.perf_counter()
        pipe.to(device)
        t1 = time.perf_counter()
        meshes = []
        try:
            with backends(attention):
                for b in range(image.shape[0]):
                    img, alpha = foreground(image[b], mask[min(b, mask.shape[0] - 1)] if mask is not None else None)
                    out = pipe(image=crop_on_white(img, alpha), num_inference_steps=steps, guidance_scale=guidance,
                               generator=torch.Generator(device=device).manual_seed(seed + b),
                               flash_octree_depth=octree_depth)
                    v, f = out.samples[0]
                    if v is None:
                        raise RuntimeError("TripoSG produced no surface for this image")
                    meshes.append((torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)),
                                   torch.from_numpy(np.ascontiguousarray(f).astype(np.int64))))
        finally:
            if not keep_loaded:
                pipe.to("cpu")
                mm.soft_empty_cache()
        log.info("TripoSG: model to GPU %.1f s, %d mesh(es) %.1f s", t1 - t0, len(meshes), time.perf_counter() - t1)
        return IO.NodeOutput(_pack(meshes))


def _pack(meshes):
    if len(meshes) == 1:
        v, f = meshes[0]
        return Types.MESH(vertices=v[None], faces=f[None])
    from comfy_extras.nodes_save_3d import pack_variable_mesh_batch

    return pack_variable_mesh_batch([m[0] for m in meshes], [m[1] for m in meshes])


# ------------------------------------------------------------------------------------------- Hunyuan3D-2.1 Paint


def _download_paint_models():
    from huggingface_hub import snapshot_download

    paint_dir = os.path.join(HY3D_DIR, "Hunyuan3D-2.1")
    if not os.path.exists(os.path.join(paint_dir, "hunyuan3d-paintpbr-v2-1", "model_index.json")):
        snapshot_download(repo_id="tencent/Hunyuan3D-2.1", allow_patterns=["hunyuan3d-paintpbr-v2-1/*"],
                          local_dir=paint_dir)
    dino_dir = os.path.join(HY3D_DIR, "dinov2-giant")
    if not os.path.exists(os.path.join(dino_dir, "config.json")):
        snapshot_download(repo_id="facebook/dinov2-giant", local_dir=dino_dir,
                          allow_patterns=["*.json", "*.safetensors", "*.txt"])
    upscale_dir = folder_paths.get_folder_paths("upscale_models")[0]
    esrgan = folder_paths.get_full_path("upscale_models", REALESRGAN) or os.path.join(upscale_dir, REALESRGAN)
    if not os.path.exists(esrgan):
        os.makedirs(upscale_dir, exist_ok=True)
        torch.hub.download_url_to_file(REALESRGAN_URL, esrgan)
    return paint_dir, dino_dir, esrgan


class Hunyuan3DPaintLoader(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Hunyuan3DPaintLoaderROCm",
            display_name="Hunyuan3D-2.1 Paint Loader (ROCm)",
            category="3d/Hunyuan3D",
            description="Settings for Hunyuan3D-2.1's PBR texture painter. Models are downloaded on first use to "
                        "models/hunyuan3d (11 GB) and models/upscale_models.",
            inputs=[
                IO.Int.Input("views", default=6, min=6, max=9, tooltip="Generated views (6-9)"),
                IO.Combo.Input("view_resolution", options=["512", "768"], default="512"),
                IO.Combo.Input("texture_size", options=["1024", "2048", "4096"], default="2048",
                               tooltip="Size of the baked texture maps (upstream: 4096)"),
            ],
            outputs=[IO.Custom("HY3D_PAINT").Output(display_name="paint")],
        )

    @classmethod
    def execute(cls, views, view_resolution, texture_size):
        return IO.NodeOutput({"views": views, "resolution": int(view_resolution), "texture_size": int(texture_size)})


def _paint_pipeline(settings):
    key = ("hy3d_paint", settings["views"], settings["resolution"], settings["texture_size"])
    if key in _cache:
        return _cache[key]
    for k in [k for k in _cache if k[0] == "hy3d_paint"]:
        del _cache[k]
    paint_dir, dino_dir, esrgan = _download_paint_models()
    from .vendor.hy3dpaint.textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "hy3dpaint")
    conf = Hunyuan3DPaintConfig(settings["views"], settings["resolution"])
    conf.device = str(mm.get_torch_device())
    conf.multiview_pretrained_path = paint_dir
    conf.dino_ckpt_path = dino_dir
    conf.realesrgan_ckpt_path = esrgan
    conf.multiview_cfg_path = os.path.join(here, "cfgs", "hunyuan-paint-pbr.yaml")
    conf.custom_pipeline = os.path.join(here, "hunyuanpaintpbr")
    conf.texture_size = settings["texture_size"]
    _cache[key] = Hunyuan3DPaintPipeline(conf)
    return _cache[key]


def _paint_to(pipe, device):
    """Moves the painter's diffusion model and DINOv2 between the GPU and CPU memory (RealESRGAN is small)."""
    mv = pipe.models["multiview_model"]
    mv.pipeline.to(device)
    if hasattr(mv, "dino_v2"):
        mv.dino_v2.to(device)


class Hunyuan3DPaint(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Hunyuan3DPaintROCm",
            display_name="Hunyuan3D-2.1 Paint (ROCm)",
            category="3d/Hunyuan3D",
            description="Paints PBR textures (base color, metallic, roughness) onto any mesh from a reference image: "
                        "meshes from Hunyuan3D-2.1, TripoSG, TRELLIS.2, ... The mesh is remeshed to ~40K faces and "
                        "UV-unwrapped.",
            inputs=[
                IO.Custom("HY3D_PAINT").Input("paint"),
                IO.Mesh.Input("mesh"),
                IO.Image.Input("image", tooltip="Reference image of the object"),
                IO.Mask.Input("mask", optional=True),
                IO.Combo.Input("attention", options=ATTENTION, default="auto", tooltip=ATTENTION_TOOLTIP),
                IO.Boolean.Input("keep_loaded", default=False, tooltip="Keep the painter on the GPU between runs "
                                 "(otherwise it waits in CPU memory)"),
            ],
            outputs=[IO.Mesh.Output()],
        )

    @classmethod
    def execute(cls, paint, mesh, image, attention="auto", keep_loaded=False, mask=None):
        import trimesh

        if mesh.vertices.shape[0] > 1:
            raise ValueError("painting a batch of meshes is not supported: connect one mesh at a time")
        free_gpu()
        pipe = None
        try:
            with backends(attention):
                t0 = time.perf_counter()
                pipe = _paint_pipeline(paint)
                _paint_to(pipe, mm.get_torch_device())
                t1 = time.perf_counter()
                v, f = next(mesh_items(mesh))
                img, alpha = foreground(image[0], mask[0] if mask is not None else None)
                rgba = torch.cat([img, alpha[..., None]], -1)
                ref = Image.fromarray((rgba.cpu().numpy() * 255).round().astype(np.uint8), "RGBA")
                # ignore_cleanup_errors: on Windows a file still held open would otherwise hide the real error
                with tempfile.TemporaryDirectory(prefix="hy3d_paint_", ignore_cleanup_errors=True) as tmp:
                    trimesh.Trimesh(v, f, process=False).export(os.path.join(tmp, "mesh.glb"))
                    out = pipe(mesh_path=os.path.join(tmp, "mesh.glb"), image_path=ref,
                               output_mesh_path=os.path.join(tmp, "textured.obj"), save_glb=False)
                    result = _load_textured(out)
                log.info("Hunyuan3D-2.1 Paint: load %.1f s, paint %.1f s", t1 - t0, time.perf_counter() - t1)
        finally:
            if pipe is not None and not keep_loaded:
                _paint_to(pipe, "cpu")
                mm.soft_empty_cache()
        return IO.NodeOutput(result)


def _load_textured(obj_path):
    """The painter's OBJ + maps -> MESH with glTF-convention UVs, base color and metallic-roughness textures."""
    import trimesh

    base = os.path.splitext(obj_path)[0]
    tm = trimesh.load(obj_path, process=False, force="mesh")
    uv = np.asarray(tm.visual.uv, dtype=np.float32).copy()
    uv[:, 1] = 1.0 - uv[:, 1]  # OBJ (origin bottom-left) -> glTF (origin top-left)

    def img(path, mode="RGB"):
        return np.asarray(Image.open(path).convert(mode), dtype=np.float32) / 255.0 if os.path.exists(path) else None

    color = img(base + ".jpg")
    metallic, roughness = img(base + "_metallic.jpg", "L"), img(base + "_roughness.jpg", "L")
    mr = None
    if metallic is not None or roughness is not None:
        shape = (metallic if metallic is not None else roughness).shape
        r = roughness if roughness is not None else np.ones(shape, np.float32)
        m = metallic if metallic is not None else np.zeros(shape, np.float32)
        mr = torch.from_numpy(np.stack([np.zeros_like(r), r, m], -1))[None]
    return Types.MESH(
        vertices=torch.from_numpy(np.asarray(tm.vertices, np.float32))[None],
        faces=torch.from_numpy(np.asarray(tm.faces, np.int64))[None],
        uvs=torch.from_numpy(uv)[None],
        texture=torch.from_numpy(color)[None] if color is not None else None,
        metallic_roughness=mr,
    )


NODES = [TripoSGLoader, TripoSGImageToMesh, Hunyuan3DPaintLoader, Hunyuan3DPaint]
