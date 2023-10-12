import requests
from gradio_client import Client
import gradio as gr
import warnings
import time
import subprocess
from tempfile import NamedTemporaryFile
import string, random

import typing as tp
import numpy as np
from pathlib import Path
import wave
import base64
from PIL import Image
import io

affirmation_gen = "http://100.76.16.85:6888/affirmation_gen"
client = Client("http://100.76.16.85:6889/")
tts_api = "http://100.76.16.85:5002/api/tts"
txt_to_img = "http://100.76.16.85:7860/sdapi/v1/txt2img"

def resize_video(input_path, output_path, target_width, target_height):
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-i', input_path,
        '-vf', f'scale={target_width}:{target_height}',
        '-c:a', 'copy',
        output_path
    ]
    subprocess.run(ffmpeg_cmd)


def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def make_affirmation_request(prompt):
    # make prompt better
    
    text = "Make a self affirmation message about " + prompt + "."
    response = requests.post(affirmation_gen, json={"prompt": text})

    if response.status_code == 200:
        return response.json()['response']
    else:
        return "Error on the backend"

def make_audio_beat_request(prompt):
    # make prompt better

    result = client.predict("melody", prompt, "", 10, 250, 0, 1, 3, fn_index=1)

    # this would result something like 
    # /private/var/folders/my/hg4vt5993352zz0kd4vhg16m0000gn/T/gradio/b9da0a6bb161d10f49ea619b3e5460453eaa6c97/TlkSCEN8wMCz.mp4

    return result

def make_waveform(*args, **kwargs):
    # Further remove some warnings.
    be = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        waveform_video = gr.make_waveform(*args, **kwargs)
        out = f"{generate_random_string(12)}.mp4"
        resize_video(waveform_video, out, 900, 300)
        print("Make a video took", time.time() - be)
        return out

def make_tts_audio_request(text, speaker_id="p336"):
    headers = {"Content-Type": "application/x-www-form-urlencoded","text":text, "speaker_id":speaker_id, "style_wav":"", "language_id":""}

    response = requests.get(tts_api, params=headers)

    file_name = generate_random_string(12) + ".wav"

    if response.status_code == 200:
        # save wav file to disk
        with wave.open(file_name, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            wav_file.writeframes(response.content)
        return file_name

def make_txt_to_img_request(prompt):
    headers = {"Content-Type": "application/json", "accept":"application/json"}

    data = {"prompt":prompt}

    response = requests.post(txt_to_img, json=data, headers=headers)
    file_name = generate_random_string(12) + ".png"
    if response.status_code == 200:
        # save the image and return path
        image = Image.open(io.BytesIO(base64.b64decode(response.json()['images'][0])))
        image.save(file_name)

        return file_name
    else:
        return "Error on the backend"
