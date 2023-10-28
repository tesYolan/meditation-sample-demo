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
from pydub import AudioSegment

affirmation_gen = "http://0.0.0.0:6888/affirmation_gen"
client = Client("http://158.132.58.150:8000/")
tts_api = "http://158.132.58.150:5002/api/tts"
txt_to_img = "http://158.132.58.150:7860/sdapi/v1/txt2img"

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

def extract_audio(input_file, output_file):
    # this function would take the input file and extract the audio from it. 
    command = ["ffmpeg", "-i", input_file, "-vn", "-acodec", "copy", output_file]
    result = subprocess.run(command)
    if result.returncode == 0:
        return output_file
    else:
        print("Error in extracting")
        return ''

def generate_thumbnail(audio_gr, generated_video, img_result):
    # this one we would take the image and the audio and the video and combine them into one.
    # take the wav audio of the generated_video first
    file_name = generate_random_string(12) + ".wav"
    extract_audio(generated_video, file_name)

    # now merge this audio and background audio. there is some package that does that. 

    base_audio = AudioSegment.from_file(audio_gr) # this should be tts. 
    background_audio = AudioSegment.from_file(file_name) # extracted audio

    while len(background_audio) < len(base_audio):
        background_audio += background_audio
    
    background_audio = background_audio[:len(base_audio)]

    background_audio -= 10

    output_audio = base_audio.overlay(background_audio)

    output_file = generate_random_string(12) + ".wav"
    output_audio.export("wavs/"+output_file, format="wav")

    return output_file

def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def make_affirmation_request(prompt, dialog):
    # make prompt better
    
    response = requests.post(affirmation_gen, json={"prompt": prompt, "dialog":dialog})

    if response.status_code == 200:
        return response.json()['response'], response.json()['dialog']
    else:
        return "Error on the backend"

def make_audio_beat_request(prompt):
    # make prompt better

    result = client.predict("facebook/musicgen-medium", "MultiBand_Diffusion", prompt, "", 4, 250, 0, 1, 3, fn_index=2)

    # this would result something like 
    # /private/var/folders/my/hg4vt5993352zz0kd4vhg16m0000gn/T/gradio/b9da0a6bb161d10f49ea619b3e5460453eaa6c97/TlkSCEN8wMCz.mp4

    return result[0]

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
    file_name = "imgs/" + generate_random_string(12) + ".png"
    if response.status_code == 200:
        # save the image and return path
        image = Image.open(io.BytesIO(base64.b64decode(response.json()['images'][0])))
        image.save(file_name)

        return file_name
    else:
        return "Error on the backend"
