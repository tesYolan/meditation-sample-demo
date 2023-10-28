# this is a gradio application. Gradio is a way to prototype and present ML models in a web app.

# this function would be an interface to five models
# a stable diffusion model. 
# a LLM with English and Chinese text support. 
# a MusicGen Model to generate model. 
# a TTS system that generates for us a voice. 
# A clip interrogator. 

# an ffmpeg interface to combine all of this into one. 
# 

import gradio as gr
import subprocess
from request_handler import make_affirmation_request, make_audio_beat_request, make_tts_audio_request, make_txt_to_img_request, generate_thumbnail


def generate_affirmation(text, prompt_state):

    dialog = prompt_state.get("dialog", [])

    system_prompt = """Return just only the answer while satisfying the user request. For example. Give me a short affirmation 
    messagea about being a good person. The system returns - "I am a good person. I am good mannered." """
    # if dialog is empty
    if not dialog:
        dialog.append({"role": "system", "content": system_prompt})
    response, dialog = make_affirmation_request(text, dialog)
    prompt_state["dialog"] = dialog

    # this is the meditaiton one. so the dialog is empty. 
    return gr.Textbox(lines=1, label="Generated Text", value=response, interactive=True), prompt_state, prompt_state

def generate_melody(text):

    response = make_audio_beat_request(text)

    return response

def generate_tts_fn(text, dropdown):
    response = make_tts_audio_request(text, dropdown)

    return response

def txt_to_img(text):
    response = make_txt_to_img_request(text)

    return gr.Image(response)

def generate_thumbnail_rq(audio_gr, generated_video, img_result):
    response = generate_thumbnail(audio_gr, generated_video, img_result)

    # currently returning the file name
    return response

with gr.Blocks(theme="gradio/monochrome") as demo:
    # later on i want it to look like a mobile app with entire application centred. 
    gr.Markdown('<h1 style="text-align: center;">Self Affirmation App: An Illustration of AI in Product Design</h1>')

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(lines=1, label="Self Affirmation Focus Area", placeholder="Being Good Person")

            generate = gr.Button("Create a Self Affirmation Message", label="Create Self Affirmation Message")

            prompt_state = gr.State(value=dict())


            # generate_zh = gr.Button("Create a Self Affirmation Message- Chinese", label="Create Self Affirmation Message - Chinese")

            with gr.Row():
                text_gen = gr.Textbox(label="Generated Text", lines=1, placeholder="I am a good person. I am good mannered.", interactive=False)
                table = gr.JSON()
            dropdown = gr.Dropdown(label="Speaker style", choices=["p336", "p339", "p326"])
            generate.click(fn=generate_affirmation, inputs=[text, prompt_state], outputs=[text_gen, prompt_state, table])

            audio_gr = gr.Audio(label="Generated Audio", type="filepath")
            generate_tts = gr.Button("Generate TTS Audio", label="Generate TTS Audio")
            generate_tts.click(fn=generate_tts_fn, inputs=[text_gen, dropdown], outputs=[audio_gr])

            # drop down with five options. 
            # tts_style = gr.Dropdown(label="Audio style", choices=["Voice 1", "Voice 2"])
            
            text_audio_style = gr.Textbox(label="Background Music Style", lines=1, placeholder="lofi hip hop style")
            generated_video = gr.Video(label="Generated Music")
            generate_audio = gr.Button("Generate Background Audio", label="Play Generated Audio")

            generate_audio.click(fn=generate_melody, inputs=[text_audio_style], outputs=[generated_video])

            image_generation = gr.Textbox(label="Image Generation", lines=1, placeholder="Serene image of a beach.")
            image_generation_button = gr.Button("Generate Image", label="Generate Image")

            img_result = gr.Image()

            image_generation_button.click(fn=txt_to_img, inputs=[image_generation], outputs=[img_result])

            combine_all = gr.Button("Combine All", label="Combine All Not Implemented")

            merged_audio = gr.Audio(label="Merged Audio")

            combine_all.click(fn=generate_thumbnail_rq, inputs=[audio_gr, generated_video, img_result], outputs=[merged_audio])

if __name__ == "__main__":
    print("Starting Gradio Server")
    demo.queue(concurrency_count=3, api_open=False).launch(server_port=9000, debug=True, server_name="0.0.0.0")            


