from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
import os
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Correct path to your model files within the repository
MODEL_NAME = "chintu1546/finbert-indian-finance/finbert_indian_finance"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Initializing FinBERT Analyzer...")

# Initialize model and tokenizer as None initially
tokenizer = None
model = None
id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}

def load_model():
    """Load the model with retry logic and error handling"""
    global tokenizer, model
    
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            print(f"Loading model from Hugging Face... (Attempt {attempt + 1}/{max_retries})")
            print(f"Model path: {MODEL_NAME}")
            
            # Login with Hugging Face token
            hf_token = os.environ.get('HUGGINGFACE_HUB_TOKEN')
            if hf_token:
                login(token=hf_token)
                print("✅ Logged into Hugging Face Hub")
            
            # Load tokenizer and model from the correct subfolder
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME, 
                trust_remote_code=True
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True
            )
            
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
    try:
        # Check if model is loaded
        if tokenizer is None or model is None:
            return jsonify({
                'error': 'Model not loaded. Please try again later.',
                'status': 'model_error'
            }), 503

        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        if len(text.strip()) == 0:
            return jsonify({'error': 'Text is empty'}), 400

        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)

        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Get predicted class and confidence
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()

        # Get all probabilities
        probs = {
            'negative': round(predictions[0][0].item(), 4),
            'neutral': round(predictions[0][1].item(), 4),
            'positive': round(predictions[0][2].item(), 4)
        }

        return jsonify({
            'sentiment': id2label[predicted_class],
            'confidence': round(confidence, 4),
            'probabilities': probs,
            'status': 'success',
            'model': 'your_finbert_indian_finance'
        })

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
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
    """Endpoint to reload the model if needed"""
    global tokenizer, model
    try:
        # Clear current model from memory
        if model is not None:
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        tokenizer = None
        model = None
        
        # Reload model
        success = load_model()
        
        if success:
            return jsonify({'status': 'success', 'message': 'Model reloaded successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to reload model'}), 500
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """Root endpoint with service information"""
    return jsonify({
        'message': 'FinBERT Indian Finance Sentiment Analysis API',
        'model': 'chintu1546/finbert-indian-finance/finbert_indian_finance',
        'endpoints': {
            'health': '/health (GET)',
            'predict': '/predict (POST)',
            'reload': '/reload-model (POST)'
        },
        'usage': {
            'predict': 'Send POST request with JSON: {"text": "your financial text here"}'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    print(f"🚀 Starting Flask server on {host}:{port}")
    print(f"📊 Model status: {'✅ Loaded' if model_loaded else '❌ Failed'}")
    print(f"🔧 Device: {device}")
    print(f"📍 Model path: {MODEL_NAME}")
    
    app.run(host=host, port=port, debug=False)
