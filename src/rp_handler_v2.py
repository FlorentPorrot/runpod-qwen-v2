import runpod
import math
import base64
import io
import sys
import gc
import os

# --- IMPROVEMENT 1: Fix Memory Fragmentation ---
# This helps when VRAM is nearly full (prevents "fake" OOMs)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import traceback
from PIL import Image
from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
MAX_PIXELS = 1024 * 1024  # 1MP limit
MAX_B64_LEN = 15_000_000  # ~15MB base64 limit

def log(msg: str):
    print(msg)
    sys.stdout.flush()

def cleanup_on_cuda_error(e: Exception):
    msg = str(e).lower()
    if ("cuda" in msg) or ("out of memory" in msg) or ("cublas" in msg):
        log("CUDA-related error detected -> running gc + empty_cache()")
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

# ----------------------------------------------------------------------------
# Load Model (Fixed for 4090 OOM & Dtype Crash)
# ----------------------------------------------------------------------------
def load_model_qwen():
    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

    model_path_nunchaku = (
        "nunchaku-tech/nunchaku-qwen-image-edit-2509/"
        "svdq-int4_r32-qwen-image-edit-2509-lightningv2.0-4steps.safetensors"
    )
    model_path_qwen = "Qwen/Qwen-Image-Edit-2509"

    log(f"Load transformer: {model_path_nunchaku}")
    
    # --- CRITICAL FIX: Force Transformer to Float16 immediately ---
    # This prevents the "RuntimeError: mat1 and mat2 must have the same dtype"
    # because the Nunchaku weights default to BFloat16, but our pipeline is Float16.
    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
        model_path_nunchaku, 
        local_files_only=True
    ).to(dtype=torch.float16)

    log(f"Load pipeline: {model_path_qwen}")
    # Initialize pipeline. Note: We do NOT call .to("cuda") here.
    # We let the offload logic handle device placement to avoid instant OOM.
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        model_path_qwen,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=torch.float16,  # Force pipeline to FP16
        local_files_only=False,
    )

    # --- MEMORY LOGIC: Restore CPU Offload for 24GB cards ---
    vram_gb = get_gpu_memory()
    log(f"Detected VRAM: {vram_gb:.1f} GB")

    if vram_gb <= 18:
        # L4 logic (Low VRAM)
        log("Low VRAM mode: enabling sequential CPU offload")
        transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
        pipeline._exclude_from_cpu_offload.append("transformer")
        pipeline.enable_sequential_cpu_offload()
    else:
        # 4090 logic (High VRAM)
        # Even with 24GB, Qwen + activations is too big for full GPU residency.
        # We enable model CPU offload to swap modules in/out as needed.
        log("High VRAM mode: enabling model CPU offload")
        pipeline.enable_model_cpu_offload()

    return pipeline

# ----------------------------------------------------------------------------
# Inference Helpers
# ----------------------------------------------------------------------------
def generate_image(pipeline, input_image, input_jersey, prompt, negative_prompt, width, height, cfg):
    inputs = {
        "image": [input_image, input_jersey],
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "true_cfg_scale": cfg,
        "num_inference_steps": 4,
    }
    return pipeline(**inputs).images[0]

def convert_image_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def decode_b64_image(b64_str: str) -> Image.Image:
    if len(b64_str) > MAX_B64_LEN:
        raise ValueError("base64 payload too large")
    image_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

# ----------------------------------------------------------------------------
# Init
# ----------------------------------------------------------------------------
log("BOOT: starting rp_handler_v2")
try:
    pipeline = load_model_qwen()
    log("BOOT: Model loaded successfully")
except Exception as e:
    log(f"BOOT CRASH: {e}")
    traceback.print_exc()
    sys.stdout.flush()
    sys.exit(1)

# ----------------------------------------------------------------------------
# Handler
# ----------------------------------------------------------------------------
def handler(event):
    if "input" not in event:
        return {"error": "Payload must contain 'input' key"}
        
    input_data = event.get("input", {})

    # Decode Images
    try:
        if "image_base64" in input_data:
            image = decode_b64_image(input_data["image_base64"])
        elif "image_path" in input_data:
            image = Image.open(input_data["image_path"]).convert("RGB")
        else:
            return {"error": "No image provided. Use 'image_base64' or 'image_path'."}

        if "jersey_base64" in input_data:
            jersey = decode_b64_image(input_data["jersey_base64"])
        elif "jersey_path" in input_data:
            jersey = Image.open(input_data["jersey_path"]).convert("RGB")
        else:
            return {"error": "No jersey image provided. Use 'jersey_base64' or 'jersey_path'."}
            
    except Exception as e:
         return {"error": f"Image decode error: {str(e)}"}

    prompt = input_data.get("prompt")
    if not prompt:
        return {"error": "Missing Prompt into input"}
        
    negative_prompt = input_data.get("negative_prompt", "")
    width = int(input_data.get("width", 1024))
    height = int(input_data.get("height", 1024))
    true_cfg_scale = float(input_data.get("true_cfg_scale", 1.0))

    # Guardrails
    if width <= 0 or height <= 0:
        return {"error": "Invalid width/height"}
    if width * height > MAX_PIXELS:
        return {"error": f"Resolution too large: {width}x{height}"}

    try:
        log(f"Request: {width}x{height} cfg={true_cfg_scale}")
        out_img = generate_image(
            pipeline=pipeline,
            input_image=image,
            input_jersey=jersey,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            cfg=true_cfg_scale,
        )
        return {"image_base64": convert_image_to_base64(out_img)}
        
    except Exception as e:
        cleanup_on_cuda_error(e)
        tb = traceback.format_exc()
        log(tb)
        return {"error": f"Error creating image: {str(e)}", "trace": tb[:1500]}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
