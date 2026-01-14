
FROM python:3.10-slim AS final

WORKDIR /app
ENV HF_HOME=/models

# Copier uniquement les fichiers nécessaires depuis l'image builder
COPY --from=nunchaku-model:latest /models /models

# Installer git pour permettre les installations depuis des dépôts Git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install uv

# Copie des Fichiers
COPY src/* /app
COPY pyproject.toml /app

RUN uv sync

# Start the container
ENTRYPOINT [ "uv" ]
CMD ["run", "rp_handler.py", "--rp_api_concurrency 1"]
