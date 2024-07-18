# import torch
# from PIL import Image
# from transformers import AutoModelForCausalLM, LlamaTokenizer
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--quant", choices=[4], type=int, default=None, help='quantization bits')
# parser.add_argument("--from_pretrained", type=str, default="THUDM/cogagent-chat-hf", help='pretrained ckpt')
# parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
# parser.add_argument("--fp16", action="store_true")
# parser.add_argument("--bf16", action="store_true")

# args = parser.parse_args()
# MODEL_PATH = args.from_pretrained
# TOKENIZER_PATH = args.local_tokenizer
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
# if args.bf16:
#     torch_type = torch.bfloat16
# else:
#     torch_type = torch.float16

# print("========Use torch type as:{} with device:{}========\n\n".format(torch_type, DEVICE))

# if args.quant:
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_PATH,
#         torch_dtype=torch_type,
#         low_cpu_mem_usage=True,
#         load_in_4bit=True,
#         trust_remote_code=True
#     ).eval()
# else:
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_PATH,
#         torch_dtype=torch_type,
#         low_cpu_mem_usage=True,
#         load_in_4bit=args.quant is not None,
#         trust_remote_code=True
#     ).to(DEVICE).eval()

# while True:
#     image_path = input("image path >>>>> ")
#     if image_path == "stop":
#         break

#     image = Image.open(image_path).convert('RGB')
#     history = []
#     while True:
#         query = input("Human:")
#         if query == "clear":
#             break
#         input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=history, images=[image])
#         inputs = {
#             'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
#             'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
#             'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
#             'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]],
#         }
#         if 'cross_images' in input_by_model and input_by_model['cross_images']:
#             inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(torch_type)]]

#         # add any transformers params here.
#         gen_kwargs = {"max_length": 2048,
#                       "temperature": 0.8,
#                       "do_sample": False}
#         with torch.no_grad():
#             outputs = model.generate(**inputs, **gen_kwargs)
#             outputs = outputs[:, inputs['input_ids'].shape[1]:]
#             response = tokenizer.decode(outputs[0])
#             response = response.split("</s>")[0]
#             print("\nCog:", response)
#         history.append((query, response))

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
import warnings

warnings.filterwarnings("ignore")

def scrape_Image_cogvlm(image, question, bf16=True):

    model_id="cogvlm" #"THUDM/cogagent-chat-hf"
    local_tokenizer="cogvlm" #"lmsys/vicuna-7b-v1.5"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = LlamaTokenizer.from_pretrained(local_tokenizer)

    torch_type = torch.bfloat16 if bf16 else torch.float16

    print(f"========Use torch type as:{torch_type} with device:{device}========\n\n")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        trust_remote_code=True
    ).eval()

    history = []

    input_by_model = model.build_conversation_input_ids(tokenizer, query=question, history=history, images=[image])
    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(device),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(device),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(device),
        'images': [[input_by_model['images'][0].to(device).to(torch_type)]],
    }
    if 'cross_images' in input_by_model and input_by_model['cross_images']:
        inputs['cross_images'] = [[input_by_model['cross_images'][0].to(device).to(torch_type)]]

    gen_kwargs = {"max_length": 2048, "temperature": 0.8, "do_sample": False} # , "top_p": 0.40, "top_k": 1

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0])
        response = response.split("</s>")[0]

    del model
    del tokenizer
    torch.cuda.empty_cache()

    return response

# if __name__ == '__main__':
#     # Example usage
#     image_path = 'samples/Desktop Screenshot 2024.06.19 - 18.48.02.92.png'
#     question = 'Find all Instagram accounts usernames present in this image.'
#     result = scrape_Image_cogvlm(image_path, question)
#     print("\nResponse:", result)

