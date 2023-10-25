FROM ghcr.io/opennmt/ctranslate2:3.20.0-ubuntu20.04-cuda11.2

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
WORKDIR /ctllama2
ENTRYPOINT [ "python3", "ct2_llama2.py" ]