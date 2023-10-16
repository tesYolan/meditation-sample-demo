# create a tmux session and run the server

tmux new-session -d -s audiov1

tmux new-window -t audiov1:2 -n 'melody_server' "bash -c 'cd ~/repos/audiov1/; conda activate pytorch2.1; python app.py --listen 0.0.0.0 --server_port 6889' "

tmux new-window -t audiov1:3 -n 'ctranslate2' "bash -c 'cd ~/repos/meditation-sample-demo; conda activate pytorch2.1; export KMP_DUPLICATE_LIB_OK=TRUE; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/ctranslate2/lib; python ct2_llama2.py' "

docker stop coqui_tts
tmux new-window -t audiov1:4 -n 'tts' "bash -c 'docker run --rm -it -p 5002:5002 --name coqui_tts --gpus device=1 --entrypoint python3 --volume tts_models:/root/.local/share/tts ghcr.io/coqui-ai/tts TTS/server/server.py --model_name tts_models/en/vctk/vits --use_cuda true' "

tmux new-window -t audiov1:5 -n 'sd-webui' "bash -c 'export CUDA_VISIBLE_DEVICES=1; cd /home/ssd_2tb/stable-diffusion-webui; ./webui.sh   --xformers --api --enable-insecure-extension-access --listen  --no-half-vae' "
#tmux new-window -t audiov1:6 -n 'meditation' "bash -c 'cd ~/repos/meditation-sample-demo/; gradio meditation.py' "

tmux attach -t audiov1

