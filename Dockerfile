# Use official Python 3.12 slim image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/back-end
WORKDIR $APP_HOME



RUN apt-get update && apt-get install -y build-essential
RUN apt-get install -y python3-dev
RUN apt-get install -y libffi-dev
RUN apt-get install -y libssl-dev
RUN apt-get install -y cargo
RUN apt-get install -y gcc
RUN apt-get install -y g++
RUN apt-get install -y make
RUN apt-get install -y    curl
RUN apt-get install -y    pkg-config

RUN pip install --upgrade pip setuptools wheel build


# Copy application code
COPY requirements/ requirements/


# Install Rust for building Rust-based Python packages
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

RUN cargo install cargo-update
RUN cargo search serde || true  # Trigger index fetch early, avoids hanging during `pip install`

RUN apt install libxcb-cursor0  libxcb-xinerama0  '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev -y
RUN apt-get update && apt-get install -y libgl1-mesa-glx libegl1-mesa


# Install Python packages
RUN pip install --upgrade pip
RUN pip install -r requirements/base.txt --progress-bar=off -v
RUN pip install -r requirements/flask.txt --progress-bar=off -v
RUN pip install torch==2.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Add and run Makefile
COPY MakefileForTest.mk Makefile

# Copy application code
COPY . .

# Fix permissions (Cloud Run uses rootless container runtime)
RUN chmod -R 777 tests chipiron
RUN chmod -R 777 flaskapp


# Add non-root user
RUN addgroup --system test && \
    adduser --system testuser --ingroup test
USER testuser:test

# Cloud Run will inject PORT, so we bind to it
ENV HOME=/root
CMD exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 8 --timeout 0 flaskapp.flask_app:app
