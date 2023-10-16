# Description

This is a wrapper around a couple of ML models that provide functionalities. 

* A language model served using fastapi using [ctranslate2](https://github.com/OpenNMT/CTranslate2). Currently llama2 compiled.  The caller is ct2_llama.py file. 
* A MusicGen called from [Musicgen](https://github.com/camenduru/audiocraft/tree/v1.0) to generate the Audio. 
* A TTS using [CoQUI](https://tts.readthedocs.io/en/latest/docker_images.html) to generate the Text to Audio model from the LLM output. 
* A calling endpoint to stable-difussion-webui from Automatic1111 end point running locally. 

## TODO 
- Need to implement a merger that combines all of them together. 