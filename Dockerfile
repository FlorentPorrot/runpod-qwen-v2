FROM python:3.10-slim AS nunchaku-model
ENV HF_HOME=/models
RUN pip install --no-cache-dir huggingface_hub
RUN hf download nunchaku-tech/nunchaku-qwen-image-edit-2509 svdq-int4_r32-qwen-image-edit-2509-lightningv2.0-4steps.safetensors

FROM python:3.10-slim AS final
WORKDIR /app
ENV HF_HOME=/models

COPY --from=nunchaku-model /models /models

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install uv

COPY src/* /app
COPY pyproject.toml /app
RUN uv sync

ENTRYPOINT [ "uv" ]
CMD ["run", "rp_handler_v2.py", "--rp_api_concurrency", "1"]
