import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

model_path = "MiniCPM_V2" # openbmb/MiniCPM-V-2

msgs = [{'role': 'user', 'content': 'Hello!'}]
msgs.append({"role": "assistant", "content": 'Hi!'})

@torch.inference_mode()
def parse_Image(image, question):

    # image = Image.open(image_path).convert('RGB')

    with torch.no_grad():
        # -----Set CUDA
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        model = model.to(device=device, dtype=torch.bfloat16)
        model.eval()
        
        msgs.append({"role": "user", "content": question})

        response, context, _ = model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.7,
        )

    # pass history context of multi-turn conversation
    # msgs.append({"role": "assistant", "content": response})

    del model
    del tokenizer
    torch.cuda.empty_cache()

    return response


# if __name__ == '__main__':
#     image_path = 'samples/persons.jpeg'

#     while True:
#         question = input('\n\nQuestion: ')
#         if question == 'q':
#             break
#         else:
#             response = parse_Image(image_path, question)
#             print(f"Answer: {response}")