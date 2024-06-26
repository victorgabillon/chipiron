# Dockerfile
FROM ubuntu:22.04

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONPATH "${PYTHONPATH}:."
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /back-end
WORKDIR $APP_HOME

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
RUN apt install libxcb-cursor0  libxcb-xinerama0  '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev -y
RUN pip3 install torch==2.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

ADD ./requirements_light.txt requirements.txt
ADD ./MakefileForTest.mk Makefile

RUN make init

RUN pip uninstall opencv-python
RUN pip install opencv-python-headless

ADD . .

RUN chmod -R 777 tests
RUN chmod -R 777 chipiron

RUN addgroup --system test
RUN adduser --system testuser --ingroup test

USER testuser:test


# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.


# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD HOME=/root exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 flaskapp.flask_app:app
