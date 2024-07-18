from transformers import AutoModel, AutoTokenizer
import torch
import warnings
warnings.filterwarnings("ignore")

model_id = "MiniCPM_llama3_vision_2_5/4bit" #"MiniCPM_llama3_vision_2_5/16bit", #'openbmb/MiniCPM-Llama3-V-2_5', #'openbmb/MiniCPM-Llama3-V-2_5-int4'

@torch.inference_mode()
def scrape_Image(image, question):

    with torch.no_grad():
        # -----Set CUDA
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16)

        if model_id == "MiniCPM_llama3_vision_2_5/16bit":
            model = model.to(device=device)

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model.eval()

        # image = Image.open('samples/Desktop Screenshot 2024.06.19 - 18.48.02.92.png').convert('RGB')
        # question = 'Find all Instagram accounts usernames present in this image.'

        msgs = [{'role': 'user', 'content': question}]

        system_prompt="""You are an AI model specialized in extracting text from images. Your task is to identify and extract all Instagram usernames present in the image. Instagram usernames typically follow a specific pattern: 
        they start with '@' followed by alphanumeric characters, underscores, or periods. Carefully scan the image and list all detected usernames. Ensure accuracy and completeness in your response."""

        res = model.chat(
            image=image,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True, # if sampling=False, beam_search will be used by default
            temperature=0.7,
            system_prompt=system_prompt, # pass system_prompt if needed
        )

    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    return res