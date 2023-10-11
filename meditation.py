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


with gr.Blocks(theme="gradio/monochrome") as demo:
    # later on i want it to look like a mobile app with entire application centred. 
    gr.Markdown('<h1 style="text-align: center;">Self Affirmation App: An Illustration of AI in Product Design</h1>')

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(lines=1, label="Self Affirmation Focus Area", placeholder="Being Good Person")

            generate = gr.Button("Create a Self Affirmation Message", label="Create Self Affirmation Message")
            generate_zh = gr.Button("Create a Self Affirmation Message- Chinese", label="Create Self Affirmation Message - Chinese")

            text_gen = gr.Textbox(label="Generated Text", lines=1, placeholder="I am a good person. I am good mannered.", interactive=False)


            generate_text = gr.Button("Play Generated Text", label="Play Generated Text")

            # drop down with five options. 
            tts_style = gr.Dropdown(label="Audio style", choices=["Voice 1", "Voice 2"])
            
            text_audio_style = gr.Textbox(label="Background Music Style", lines=1, placeholder="lofi hip hop style")

            generate_audio = gr.Button("Play Generated Audio", label="Play Generated Audio")

            image_generation = gr.Textbox(label="Image Generation", lines=1, placeholder="Serene image of a beach.")
            image_generation_button = gr.Button("Generate Image", label="Generate Image")


            combine_all = gr.Button("Combine All", label="Combine All")




if __name__ == "__main__":
    demo.launch(server_port=8443, debug=True, server_name="0.0.0.0")            

