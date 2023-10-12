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
import torch
import torchaudio
import wave
import base64
from PIL import Image

affirmation_gen = "http://100.76.16.85:6888/affirmation_gen"
client = Client("http://100.76.16.85:6889/")
tts_api = "http://100.76.16.85:5002/api/tts"
txt_to_img = "http://100.76.16.85:7860/sdapi/v1/txt2img"

def normalize_audio(wav: torch.Tensor, normalize: bool = True,
                    strategy: str = 'peak', peak_clip_headroom_db: float = 1,
                    rms_headroom_db: float = 18, loudness_headroom_db: float = 14,
                    loudness_compressor: bool = False, log_clipping: bool = False,
                    sample_rate: tp.Optional[int] = None,
                    stem_name: tp.Optional[str] = None) -> torch.Tensor:
    """Normalize the audio according to the prescribed strategy (see after).

    Args:
        wav (torch.Tensor): Audio data.
        normalize (bool): if `True` (default), normalizes according to the prescribed
            strategy (see after). If `False`, the strategy is only used in case clipping
            would happen.
        strategy (str): Can be either 'clip', 'peak', or 'rms'. Default is 'peak',
            i.e. audio is normalized by its largest value. RMS normalizes by root-mean-square
            with extra headroom to avoid clipping. 'clip' just clips.
        peak_clip_headroom_db (float): Headroom in dB when doing 'peak' or 'clip' strategy.
        rms_headroom_db (float): Headroom in dB when doing 'rms' strategy. This must be much larger
            than the `peak_clip` one to avoid further clipping.
        loudness_headroom_db (float): Target loudness for loudness normalization.
        loudness_compressor (bool): If True, uses tanh based soft clipping.
        log_clipping (bool): If True, basic logging on stderr when clipping still
            occurs despite strategy (only for 'rms').
        sample_rate (int): Sample rate for the audio data (required for loudness).
        stem_name (Optional[str]): Stem name for clipping logging.
    Returns:
        torch.Tensor: Normalized audio.
    """
    scale_peak = 10 ** (-peak_clip_headroom_db / 20)
    scale_rms = 10 ** (-rms_headroom_db / 20)
    if strategy == 'peak':
        rescaling = (scale_peak / wav.abs().max())
        if normalize or rescaling < 1:
            wav = wav * rescaling
    elif strategy == 'clip':
        wav = wav.clamp(-scale_peak, scale_peak)
    elif strategy == 'rms':
        mono = wav.mean(dim=0)
        rescaling = scale_rms / mono.pow(2).mean().sqrt()
        if normalize or rescaling < 1:
            wav = wav * rescaling
        _clip_wav(wav, log_clipping=log_clipping, stem_name=stem_name)
    elif strategy == 'loudness':
        assert sample_rate is not None, "Loudness normalization requires sample rate."
        wav = normalize_loudness(wav, sample_rate, loudness_headroom_db, loudness_compressor)
        _clip_wav(wav, log_clipping=log_clipping, stem_name=stem_name)
    else:
        assert wav.abs().max() < 1
        assert strategy == '' or strategy == 'none', f"Unexpected strategy: '{strategy}'"
    return wav

def convert_audio_channels(wav: torch.Tensor, channels: int = 2) -> torch.Tensor:
    """Convert audio to the given number of channels.

    Args:
        wav (torch.Tensor): Audio wave of shape [B, C, T].
        channels (int): Expected number of channels as output.
    Returns:
        torch.Tensor: Downmixed or unchanged audio wave [B, C, T].
    """
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, and the stream has multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file has
        # a single channel, replicate the audio over all channels.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file has
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav


def convert_audio(wav: torch.Tensor, from_rate: float,
                  to_rate: float, to_channels: int) -> torch.Tensor:
    """Convert audio to new sample rate and number of audio channels.
    """
    wav = julius.resample_frac(wav, int(from_rate), int(to_rate))
    wav = convert_audio_channels(wav, to_channels)
    return wav


def normalize_loudness(wav: torch.Tensor, sample_rate: int, loudness_headroom_db: float = 14,
                       loudness_compressor: bool = False, energy_floor: float = 2e-3):
    """Normalize an input signal to a user loudness in dB LKFS.
    Audio loudness is defined according to the ITU-R BS.1770-4 recommendation.

    Args:
        wav (torch.Tensor): Input multichannel audio data.
        sample_rate (int): Sample rate.
        loudness_headroom_db (float): Target loudness of the output in dB LUFS.
        loudness_compressor (bool): Uses tanh for soft clipping.
        energy_floor (float): anything below that RMS level will not be rescaled.
    Returns:
        output (torch.Tensor): Loudness normalized output data.
    """
    energy = wav.pow(2).mean().sqrt().item()
    if energy < energy_floor:
        return wav
    transform = torchaudio.transforms.Loudness(sample_rate)
    input_loudness_db = transform(wav).item()
    # calculate the gain needed to scale to the desired loudness level
    delta_loudness = -loudness_headroom_db - input_loudness_db
    gain = 10.0 ** (delta_loudness / 20.0)
    output = gain * wav
    if loudness_compressor:
        output = torch.tanh(output)
    assert output.isfinite().all(), (input_loudness_db, wav.pow(2).mean().sqrt())
    return output


def _clip_wav(wav: torch.Tensor, log_clipping: bool = False, stem_name: tp.Optional[str] = None) -> None:
    """Utility function to clip the audio with logging if specified."""
    max_scale = wav.abs().max()
    if log_clipping and max_scale > 1:
        clamp_prob = (wav.abs() > 1).float().mean().item()
        print(f"CLIPPING {stem_name or ''} happening with proba (a bit of clipping is okay):",
              clamp_prob, "maximum scale: ", max_scale.item(), file=sys.stderr)
    wav.clamp_(-1, 1)

def f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to float 32 bits PCM format.
    """
    if wav.dtype.is_floating_point:
        return wav
    else:
        assert wav.dtype == torch.int16
        return wav.float() / 2**15


def i16_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to int 16 bits PCM format.

    ..Warning:: There exist many formula for doing this convertion. None are perfect
    due to the asymetry of the int16 range. One either have possible clipping, DC offset,
    or inconsistancies with f32_pcm. If the given wav doesn't have enough headroom,
    it is possible that `i16_pcm(f32_pcm)) != Identity`.
    """
    if wav.dtype.is_floating_point:
        assert wav.abs().max() <= 1
        candidate = (wav * 2 ** 15).round()
        if candidate.max() >= 2 ** 15:  # clipping would occur
            candidate = (wav * (2 ** 15 - 1)).round()
        return candidate.short()
    else:
        assert wav.dtype == torch.int16
        return wav
    

def audio_write(stem_name: tp.Union[str, Path],
                wav: torch.Tensor, sample_rate: int,
                format: str = 'wav', mp3_rate: int = 320, normalize: bool = True,
                strategy: str = 'peak', peak_clip_headroom_db: float = 1,
                rms_headroom_db: float = 18, loudness_headroom_db: float = 14,
                loudness_compressor: bool = False,
                log_clipping: bool = True, make_parent_dir: bool = True,
                add_suffix: bool = True) -> Path:
    """Convenience function for saving audio to disk. Returns the filename the audio was written to.

    Args:
        stem_name (str or Path): Filename without extension which will be added automatically.
        format (str): Either "wav" or "mp3".
        mp3_rate (int): kbps when using mp3s.
        normalize (bool): if `True` (default), normalizes according to the prescribed
            strategy (see after). If `False`, the strategy is only used in case clipping
            would happen.
        strategy (str): Can be either 'clip', 'peak', or 'rms'. Default is 'peak',
            i.e. audio is normalized by its largest value. RMS normalizes by root-mean-square
            with extra headroom to avoid clipping. 'clip' just clips.
        peak_clip_headroom_db (float): Headroom in dB when doing 'peak' or 'clip' strategy.
        rms_headroom_db (float): Headroom in dB when doing 'rms' strategy. This must be much larger
            than the `peak_clip` one to avoid further clipping.
        loudness_headroom_db (float): Target loudness for loudness normalization.
        loudness_compressor (bool): Uses tanh for soft clipping when strategy is 'loudness'.
         when strategy is 'loudness'log_clipping (bool): If True, basic logging on stderr when clipping still
            occurs despite strategy (only for 'rms').
        make_parent_dir (bool): Make parent directory if it doesn't exist.
    Returns:
        Path: Path of the saved audio.
    """
    assert wav.dtype.is_floating_point, "wav is not floating point"
    if wav.dim() == 1:
        wav = wav[None]
    elif wav.dim() > 2:
        raise ValueError("Input wav should be at most 2 dimension.")
    assert wav.isfinite().all()
    wav = normalize_audio(wav, normalize, strategy, peak_clip_headroom_db,
                          rms_headroom_db, loudness_headroom_db, log_clipping=log_clipping,
                          sample_rate=sample_rate, stem_name=str(stem_name))
    kwargs: dict = {}
    if format == 'mp3':
        suffix = '.mp3'
        kwargs.update({"compression": mp3_rate})
    elif format == 'wav':
        wav = i16_pcm(wav)
        suffix = '.wav'
        kwargs.update({"encoding": "PCM_S", "bits_per_sample": 16})
    else:
        raise RuntimeError(f"Invalid format {format}. Only wav or mp3 are supported.")
    if not add_suffix:
        suffix = ''
    path = Path(str(stem_name) + suffix)
    if make_parent_dir:
        path.parent.mkdir(exist_ok=True, parents=True)
    try:
        ta.save(path, wav, sample_rate, **kwargs)
    except Exception:
        if path.exists():
            # we do not want to leave half written files around.
            path.unlink()
        raise
    return path

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
        image = Image.open(io.BytesIO(base64.b64decode(r['images'][0])))
