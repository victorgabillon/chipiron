# Dockerfile
FROM ubuntu:24.04

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONPATH "${PYTHONPATH}:."
# Copy local code to the container image.
WORKDIR .


RUN set -xe \
    && apt-get update \
    && apt-get install python3-full -y \
    && apt-get install python3-pip -y


RUN apt install libegl1 -y
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install python3-opencv -y
RUN apt-get install python3-tk -y
RUN apt-get install wget
RUN apt-get install curl
RUN apt install libxcb-cursor0  libxcb-xinerama0  '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev -y


RUN python3 -m venv .venv
RUN . .venv/bin/activate
RUN rm /usr/lib/python*/EXTERNALLY-MANAGED && \
    pip install gdown --break-system-packages\
    && pip install --upgrade gdown


ADD ./requirements_dev.txt requirements_dev.txt
ADD ./pyproject.toml pyproject.toml
ADD ./Makefile Makefile


RUN .venv/bin/pip install --upgrade pip


# Install Python packages
RUN apt-get remove -y python3-numpy
RUN pip install numpy==1.26.4

RUN pip install -e .
RUN python3 -m pip install torch==2.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html




# Get Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN make init

RUN python3 -m pip uninstall opencv-python
RUN python3 -m pip install opencv-python-headless

ADD . .

RUN chmod -R 777 chipiron
RUN chmod -R 777 tests

RUN addgroup --system test
RUN adduser --system testuser --ingroup test

USER testuser:test


# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.


CMD ["python3", "chipiron/scripts/main_chipiron.py"]
