FROM nvcr.io/nvidia/tritonserver:23.09-pyt-python-py3
# install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg
RUN pip install -U audiocraft
WORKDIR /musicgen
ENTRYPOINT [ "python -m demos.musicgen_app" ]