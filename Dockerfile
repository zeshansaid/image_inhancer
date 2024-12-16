FROM runpod/base:0.4.2-cuda11.8.0


SHELL ["/bin/bash", "-o", "pipefail", "-c"]

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

# Python dependencies
COPY requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

ADD . .


CMD [ "python3.11", "-u", "main.py" ]
