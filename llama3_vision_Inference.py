import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

model_id = "llama3_vision" #"qresearch/llama-3-vision-alpha-hf"

memory = []  # Initialize an empty list to store the memory

def inference(image, question):
    # image = Image.open(image_path)

    # -----Set CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    model.to(device)
    model.eval()
    
    # # Prepare the context based on the last 5 questions and answers
    # context = ""
    # for entry in memory[-5:]:
    #     context += f"Question: {entry['question']}\nAnswer: {entry['answer']}\n\n"
    
    # # Add the current question to the context
    # context += f"Question: {question}"
    
    # Generate the answer using the context
    answer = tokenizer.decode(model.answer_question(image, question, tokenizer), skip_special_tokens=True)
    
    # Store the question and answer in memory
    # memory.append({"question": question, "answer": answer})

    # Release GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return answer


# if __name__ == "__main__":
#     image_path = "samples/persons.jpeg"

#     while True:
#         question = input("\n\nQuestion: ")
#         if question == "q":
#             break
#         else:
#             response = inference(image_path, question)
#             print(f"Answer: {response}")
