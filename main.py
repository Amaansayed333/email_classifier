from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
from utils import mask_pii
from models import load_models, predict_email_category

app = Flask(__name__)
run_with_ngrok(app)

# Load models for classification
clf, vectorizer, label_encoder = load_models()

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()

    # Validate input
    if not data or 'input_email_body' not in data:
        return jsonify({"error": "Missing 'input_email_body'"}), 400

    input_email_body = data['input_email_body']

    # Mask PII
    masked_email, masked_entities = mask_pii(input_email_body)

    # Predict category
    category = predict_email_category(masked_email, vectorizer, clf, label_encoder)

    # Return response
    response = {
        "input_email_body": input_email_body,
        "list_of_masked_entities": masked_entities,
        "masked_email": masked_email,
        "category_of_the_email": category
    }
    return jsonify(response), 200

if __name__ == "__main__":
    app.run()
