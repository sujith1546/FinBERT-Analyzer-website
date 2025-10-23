from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Hugging Face model repo
MODEL_NAME = "chintu1546/finbert-indian-finance"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading model from Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()
print(f"Model loaded successfully on {device}!")

# Label mapping
id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

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
            'negative': predictions[0][0].item(),
            'neutral': predictions[0][1].item(),
            'positive': predictions[0][2].item()
        }

        return jsonify({
            'sentiment': id2label[predicted_class],
            'confidence': confidence,
            'probabilities': probs
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'FinBERT loaded from Hugging Face'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
