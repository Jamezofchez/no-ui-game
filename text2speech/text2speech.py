from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, set_seed
import soundfile as sf

class Talker:
    def __init__(self, model_name="parler-tts/parler_tts_mini_v0.1", narrator_model_name="parler-tts/parler-tts-mini-expresso", device="cpu"):
        self.device = device
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Narrator "consistent model" (Thomas)
        self.narrator_model = ParlerTTSForConditionalGeneration.from_pretrained(narrator_model_name).to(device)
        self.narrator_tokenizer = AutoTokenizer.from_pretrained(narrator_model_name)
        

    def generate_speech(self, text, description, output_file="output.wav"):
        input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to(self.device)
        prompt_input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

        generation = self.model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write(output_file, audio_arr, self.model.config.sampling_rate)
        print(f"Audio saved to {output_file}")

    def narrator_generate_speech(self, text, output_file="output.wav"):
        description = (
            "Jerry speaks moderately slowly in a neutral tone and high quality audio."
        )
        input_ids = self.narrator_tokenizer(description, return_tensors="pt").input_ids.to(self.device)
        prompt_input_ids = self.narrator_tokenizer(text, return_tensors="pt").input_ids.to(self.device)

        set_seed(42)
        generation = self.narrator_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write(output_file, audio_arr, self.narrator_model.config.sampling_rate)
        print(f"Audio saved to {output_file}")

if __name__ == "__main__":
    talker = Talker()
    prompt = "Once upon a time in a land far away"
    description = "A calm and soothing narration voice"
    talker.generate_speech(prompt, description, "story.wav")