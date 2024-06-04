from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import requests
import torch

model_id = "PaliGemma"

def infer_gemma(prompt, image):

    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        token="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"
    ).eval()

    processor = AutoProcessor.from_pretrained(
        model_id, 
        token="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"
    )

    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=4000, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)

    return decoded

# if __name__ == "__main__":
#     prompt = "caption in english"
#     url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
#     image = Image.open(requests.get(url, stream=True).raw)

#     print("\n\nCaption: " + infer_gemma(prompt, image) + "\n\n")
