# filepath: /home/victor/chipiron/Dockerfile
FROM ubuntu:24.04

WORKDIR /app

ENV PYTHONPATH="${PYTHONPATH}:/app"

RUN set -xe \
    && apt-get update \
    && apt-get install -y python3-full python3-pip libegl1 python3-opencv python3-tk wget curl \
    libxcb-cursor0 libxcb-xinerama0 '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev

RUN python3 -m venv .venv

# Remove EXTERNALLY-MANAGED if present
RUN rm -f /usr/lib/python*/EXTERNALLY-MANAGED

# Upgrade pip and install gdown
RUN .venv/bin/pip install --upgrade pip setuptools wheel gdown

# Copy project files
COPY . .

# Remove system numpy and install required version
RUN apt-get remove -y python3-numpy
RUN .venv/bin/pip install numpy==1.26.4


# After copying .venv setup
COPY pyproject.toml .
COPY README.md .
RUN .venv/bin/pip install ".[dev]"
COPY . .


# Install project in editable mode
RUN .venv/bin/pip install -e .

# Install torch (CPU version)
RUN python3 -m pip install torch==2.7.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Get Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN make init

RUN python3 -m pip uninstall -y opencv-python
RUN python3 -m pip install opencv-python-headless

RUN chmod -R 777 chipiron
RUN chmod -R 777 tests

RUN addgroup --system test
RUN adduser --system testuser --ingroup test

USER testuser:test

CMD ["python3", "chipiron/scripts/main_chipiron.py"]