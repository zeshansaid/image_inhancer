FROM runpod/base:0.4.2-cuda11.8.0

# Python dependencies
COPY requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

ADD . .


CMD [ "python3.11", "-u", "main.py" ]
