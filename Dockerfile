# filepath: /home/victor/chipiron/Dockerfile
FROM ubuntu:24.04

WORKDIR /app

ENV PYTHONPATH="${PYTHONPATH}:/app"

# 1. Installer Python + pip et dépendances
RUN set -xe \
    && apt-get update \
    && apt-get install -y python3-full python3-pip libegl1 python3-opencv python3-tk wget curl \
    libxcb-cursor0 libxcb-xinerama0 '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev

# 2. Supprimer EXTERNALLY-MANAGED si besoin
RUN rm -f /usr/lib/python*/EXTERNALLY-MANAGED

# 3. Créer l'environnement virtuel
RUN python3 -m venv .venv

# 4. Activer l'environnement virtuel dans le PATH
ENV PATH="/app/.venv/bin:$PATH"



# Install required Python packages
RUN .venv/bin/pip install opencv-python-headless



# Upgrade pip and install gdown
RUN .venv/bin/pip install --upgrade pip setuptools wheel gdown

# Install torch first (before installing chipiron!)
RUN .venv/bin/pip install torch==2.2.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install numpy (force-reinstall to overwrite system numpy)
RUN .venv/bin/pip install --upgrade --force-reinstall numpy==1.26.0


# Copy only project metadata to install deps
COPY pyproject.toml .

# Install chipiron with dev dependencies
RUN .venv/bin/pip install ".[dev]"

# Now copy full codebase
COPY . .

# Install project in editable mode
RUN .venv/bin/pip install -e .

# Get Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN make chipiron/syzygy-tables chipiron/data

RUN chmod -R 777 chipiron
RUN chmod -R 777 tests

RUN addgroup --system test
RUN adduser --system testuser --ingroup test

USER testuser:test

CMD ["/app/.venv/bin/python", "/app/chipiron/scripts/main_chipiron.py"]