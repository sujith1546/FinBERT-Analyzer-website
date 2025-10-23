from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
import os
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Model config
MODEL_NAME = "chintu1546/finbert-indian-finance"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}

tokenizer = None
model = None

def load_model():
    """Load the Hugging Face model with retry logic"""
    global tokenizer, model

    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            print(f"Loading model from Hugging Face... (Attempt {attempt + 1}/{max_retries})")
            
            # Login with Hugging Face token
            hf_token = os.environ.get('HUGGINGFACE_HUB_TOKEN')
            if hf_token:
                login(token=hf_token)
                print("✅ Logged into Hugging Face Hub")

            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, trust_remote_code=True)
            model.to(device)
            model.eval()

            print(f"✅ Model loaded successfully on {device}!")
            return True

        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("❌ All attempts to load model failed")
                return False

# Load model on startup
model_loaded = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    if tokenizer is None or model is None:
        return jsonify({'error': 'Model not loaded', 'status': 'model_error'}), 503

    data = request.json
    if not data or not data.get('text'):
        return jsonify({'error': 'No text provided'}), 400

    text = data['text'].strip()
    if len(text) == 0:
        return jsonify({'error': 'Text is empty'}), 400

    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
        probs = {label: round(predictions[0][i].item(), 4) for i, label in id2label.items()}

        return jsonify({
            'sentiment': id2label[predicted_class],
            'confidence': round(confidence, 4),
            'probabilities': probs,
            'status': 'success',
            'model': MODEL_NAME
        })
    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'message': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    status = 'healthy' if (tokenizer is not None and model is not None) else 'unhealthy'
    return jsonify({
        'status': status,
        'model_loaded': model is not None,
        'device': str(device),
        'service': 'FinBERT Indian Finance Analysis',
        'model_path': MODEL_NAME
    })

@app.route('/reload-model', methods=['POST'])
def reload_model():
    global tokenizer, model
    try:
        if model is not None:
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        tokenizer = None
        model = None

        success = load_model()
        if success:
            return jsonify({'status': 'success', 'message': 'Model reloaded successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to reload model'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'FinBERT Indian Finance Sentiment Analysis API',
        'model': MODEL_NAME,
        'endpoints': {
            'health': '/health (GET)',
            'predict': '/predict (POST)',
            'reload': '/reload-model (POST)'
        },
        'usage': {
            'predict': 'POST JSON: {"text": "your financial text here"}'
        }
    })

if __name__ == '__main__':
    # Use Render's PORT environment variable if available
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'

    print(f"🚀 Starting Flask server on {host}:{port}")
    print(f"📊 Model status: {'✅ Loaded' if model_loaded else '❌ Failed'}")
    print(f"🔧 Device: {device}")
    print(f"📍 Model path: {MODEL_NAME}")

    # Debug=False for production
    app.run(host=host, port=port, debug=False)
