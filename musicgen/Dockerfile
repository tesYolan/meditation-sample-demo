FROM nvcr.io/nvidia/tritonserver:23.09-pyt-python-py3
# install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg
RUN pip install -U audiocraft
WORKDIR /
RUN git clone https://github.com/facebookresearch/audiocraft.git
WORKDIR /audiocraft
# run python3 -m demos.musicgen_app in /audiocraft to run the app
ENTRYPOINT [ "python3", "-m", "demos.musicgen_app", "--listen", "0.0.0.0"]