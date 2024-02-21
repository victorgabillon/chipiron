# Dockerfile
FROM ubuntu:22.04

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONPATH "${PYTHONPATH}:."
# Copy local code to the container image.
WORKDIR .


RUN set -xe \
    && apt-get update \
    && apt-get install python3.11 -y \
    && apt-get install python3-pip -y


RUN rm -rf /usr/share/dotnet
RUN rm -rf /opt/ghc
RUN rm -rf "/usr/local/share/boost"
RUN rm -rf "$AGENT_TOOLSDIRECTORY"

RUN apt install libegl1 -y
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install python3-opencv -y
RUN pip install gdown

RUN pip install --no-cache-dir --upgrade pip
RUN apt-get install python3-tk -y

RUN apt-get install wget
RUN apt-get install curl


ADD ./requirements.txt requirements.txt
ADD ./Makefile Makefile

RUN make init

ADD . .

RUN chmod -R 777 scripts

RUN apt install libxcb-cursor0  libxcb-xinerama0  '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev -y

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.


CMD ["main_chipiron_.py"]
ENTRYPOINT ["python3"]