import runpod

import math
import base64
import io
import sys
import gc
import traceback

from PIL import Image

from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline


# -----------------------------
# Config (v2)
# -----------------------------
# Guardrail to avoid rare killer requests (adjust if you need bigger)
MAX_PIXELS = 1024 * 1024  # 1024x1024
# Optional guardrail for very large base64 payloads (string length)
MAX_B64_LEN = 15_000_000


def log(msg: str):
    print(msg)
    sys.stdout.flush()


def cleanup_on_cuda_error(e: Exception):
    """
    Only clean up on CUDA-ish failures to avoid slowing down normal requests.
    """
    msg = str(e).lower()
    if ("cuda" in msg) or ("out of memory" in msg) or ("cublas" in msg):
        log("CUDA-related error detected -> running gc + empty_cache()")
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


# -----------------------------------------------
# -- Load Model Qwen (v2)
# -----------------------------------------------
def load_model_qwen():
    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),  # We use shift=3 in distillation
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),  # We use shift=3 in distillation
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,  # set shift_terminal to None
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

    model_path_nunchaku_transformer_safetensor_file = (
        "nunchaku-tech/nunchaku-qwen-image-edit-2509/"
        "svdq-int4_r32-qwen-image-edit-2509-lightningv2.0-4steps.safetensors"
    )
    model_path_qwen = "Qwen/Qwen-Image-Edit-2509"

    log(f"Load transformer : {model_path_nunchaku_transformer_safetensor_file}")
    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
        model_path_nunchaku_transformer_safetensor_file,
        local_files_only=True,
    )

    # Load the model pipeline Qwen
    log(f"Load pipeline : {model_path_qwen}")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        model_path_qwen,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=torch.float16,  # ✅ v2: force FP16 (avoid BF16 cuBLAS issues)
        local_files_only=False,
    )

    # Put pipeline on GPU if available (safe even if CPU offload is used later)
    if torch.cuda.is_available():
        pipeline = pipeline.to("cuda")

    # ✅ v2: Do NOT enable CPU offload on 24GB GPUs (4090/L4).
    # Only use offloading for genuinely low VRAM GPUs.
    vram_gb = get_gpu_memory()
    log(f"Detected VRAM: {vram_gb:.1f} GB")

    if vram_gb <= 18:
        log("Low VRAM mode: enabling sequential CPU offload")
        transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
        pipeline._exclude_from_cpu_offload.append("transformer")
        pipeline.enable_sequential_cpu_offload()
    else:
        log("High VRAM mode: keep model on GPU (no CPU offload)")

    return pipeline


# -----------------------------------------------
# -- Inference helpers
# -----------------------------------------------
def generate_image(
    pipeline,
    input_image,
    input_jersey,
    input_prompt: str,
    input_negative_prompt: str,
    input_width: int,
    input_height: int,
    input_cfg_scale: float,
):
    inputs = {
        "image": [input_image, input_jersey],
        "prompt": input_prompt,
        "negative_prompt": input_negative_prompt,
        "width": input_width,
        "height": input_height,
        "true_cfg_scale": input_cfg_scale,
        "num_inference_steps": 4,
    }

    return pipeline(**inputs).images[0]


def convert_image_to_base64(input_image: Image.Image) -> str:
    buffered = io.BytesIO()
    input_image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64


def decode_b64_image(b64_str: str) -> Image.Image:
    if len(b64_str) > MAX_B64_LEN:
        raise ValueError("base64 payload too large")
    image_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


# -----------------------------------------------
# -- Initialize model (global, once per worker)
# -----------------------------------------------
pipeline = load_model_qwen()


# -----------------------------------------------
# -- Handler
# -----------------------------------------------
def handler(event):
    if "input" not in event:
        return {"error": "Payload must contain 'input' key"}

    input_data = event.get("input", {})

    # Selfie
    if "image_base64" in input_data:
        try:
            image = decode_b64_image(input_data["image_base64"])
        except Exception as e:
            return {"error": f"Invalid base64 image: {str(e)}"}
    elif "image_path" in input_data:
        try:
            image = Image.open(input_data["image_path"]).convert("RGB")
        except Exception as e:
            return {"error": f"Cannot open image at path: {str(e)}"}
    else:
        return {"error": "No image provided. Use 'image_base64' or 'image_path'."}

    # Jersey
    if "jersey_base64" in input_data:
        try:
            jersey = decode_b64_image(input_data["jersey_base64"])
        except Exception as e:
            return {"error": f"Invalid base64 jersey image: {str(e)}"}
    elif "jersey_path" in input_data:
        try:
            jersey = Image.open(input_data["jersey_path"]).convert("RGB")
        except Exception as e:
            return {"error": f"Cannot open jersey image at path: {str(e)}"}
    else:
        return {"error": "No jersey image provided. Use 'jersey_base64' or 'jersey_path'."}

    prompt = input_data.get("prompt")
    if prompt is None:
        return {"error": "Missing Prompt into input"}

    negative_prompt = input_data.get("negative_prompt", "")
    width = int(input_data.get("width", 1024))
    height = int(input_data.get("height", 1024))
    true_cfg_scale = float(input_data.get("true_cfg_scale", 1.0))

    # Guardrail for OOM spikes (v2)
    if width <= 0 or height <= 0:
        return {"error": "Invalid width/height"}
    if width * height > MAX_PIXELS:
        return {"error": f"Resolution too large: {width}x{height}"}

    try:
        image_generated = generate_image(
            pipeline=pipeline,
            input_image=image,
            input_jersey=jersey,
            input_prompt=prompt,
            input_negative_prompt=negative_prompt,
            input_width=width,
            input_height=height,
            input_cfg_scale=true_cfg_scale,
        )

        img_base64 = convert_image_to_base64(input_image=image_generated)
        return {"image_base64": img_base64}

    except Exception as e:
        cleanup_on_cuda_error(e)
        tb = traceback.format_exc()
        log(tb)
        return {"error": f"error creating image: {str(e)}", "trace": tb[:1500]}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
