import ctranslate2
import transformers
import sentencepiece as spm
import os
from fastapi import FastAPI

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
class llm:
    def __init__(self, model_location="/home/dlpc/llama-2-7b-chat-ct2", system_prompt="Return just only the answer while answering requests") -> None:
        print("Loading Generator")
        self.generator = ctranslate2.Generator(model_location, device="cuda")

        self.context_length = 4096

        self.max_generation_length = 512
        self.sampling_temperature = 0.6
        self.sampling_topk = 20
        self.sampling_topp = 1

        self.max_prompt_length = self.context_length - self.max_generation_length

        self.system_prompt = True


        print("Loading Tokenizer")
        self.sp = spm.SentencePieceProcessor(os.path.join(model_location, "tokenizer.model"))

        print("Finished loading")

    
    def append_system_prompt(self, prompt: str):
        "Append a system prompt to the dialog"
        self.dialog.append({"role": "system", "content": prompt})

    def predict(self, prompt: str, dialog: list):
        "Generate text give a prompt"

        dialog.append({"role": "user", "content": prompt})

        prompt_tokens = self.build_prompt(self.sp, dialog)

        if len(prompt_tokens) > self.max_prompt_length:
            if self.system_prompt:
                dialog = [dialog[0]] + dialog[3:]
            else:
                dialog = dialog[2:]
        
        step_results = self.generator.generate_tokens(prompt_tokens, 
                                                      max_length=self.max_generation_length, 
                                                      sampling_temperature=0.6,
                                                      sampling_topk=20, 
                                                      sampling_topp=1
                                                      )
        text_output = ""

        for word in self.generate_words(self.sp, step_results):
            text_output += word + " "
        
        dialog.append({"role": "assistant", "content": text_output})
        return text_output, dialog

    def generate_words(self, sp, step_results):
        tokens_buffer = []

        for step_result in step_results:
            is_new_word = step_result.token.startswith("▁")

            if is_new_word and tokens_buffer:
                word = sp.decode(tokens_buffer)
                if word:
                    yield word
                tokens_buffer = []

            tokens_buffer.append(step_result.token_id)

        if tokens_buffer:
            word = sp.decode(tokens_buffer)
            if word:
                yield word
        

    
    def build_prompt(self,sp, dialog):
        if dialog[0]["role"] == "system":
            dialog = [
                {
                    "role": dialog[1]["role"],
                    "content": B_SYS + dialog[0]["content"] + E_SYS + dialog[1]["content"],
                }
            ] + dialog[2:]

        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )

        dialog_tokens = sum(
            [
                ["<s>"]
                + sp.encode_as_pieces(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} "
                )
                + ["</s>"]
                for prompt, answer in zip(
                    dialog[::2],
                    dialog[1::2],
                )
            ],
            [],
        )

        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"

        dialog_tokens += ["<s>"] + sp.encode_as_pieces(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"
        )

        return dialog_tokens

app = FastAPI()
llm_model = llm()

@app.post("/affirmation_gen")

async def predict(prompt: dict):

    # change from json to text
    response, dialog = llm_model.predict(prompt['prompt'], prompt['dialog'])

    return {'response':response, 'dialog':dialog}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6888)