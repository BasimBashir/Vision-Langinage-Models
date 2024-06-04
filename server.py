from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from PIL import Image
import warnings
import torch

# To suppress all warnings globally
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    # Release GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:
        image = request.files['image']

    image = Image.open(image).convert('RGB')

    query = request.form['query']
    selected_model = request.form['model']

    # You can process the query and selected model here and prepare a response
    if selected_model == 'MiniCPM_v2':
        from MiniCPM_V2_Inference import parse_Image
        response = parse_Image(image, query)
        # Release GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif selected_model == 'Llama3_Vision':
        from llama3_vision_Inference import inference
        response = inference(image, query)
        # Release GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif selected_model == 'PaliGemma':
        from PaliGemma import infer_gemma
        response = infer_gemma(query, image)
        # Release GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        response = "Invalid model selected"
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5555)
