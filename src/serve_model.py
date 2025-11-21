"""
Flask API of the SMS Spam detection model model.
"""
import os

import joblib
from flasgger import Swagger
from flask import Flask, jsonify, request
from text_preprocessing import (  # noqa: F401
    _extract_message_len,
    _text_process,
    prepare,
)
import requests

MODEL_DIR = os.getenv("MODEL_DIR", "/app/output")
MODEL_PATH = os.path.join(MODEL_DIR, 'model.joblib')
DOWNLOAD_URL = "https://github.com/doda25-team6/model-service/releases/download/v0.1.2/preprocessor-v0.1.2.joblib"

# Global variable to hold the loaded model
clf = None

def dynamically_load_model():
    """
    Checks if the model exists in the volume mount path. If not, downloads it.
    Loads the model into the global 'clf' variable.
    """
    global clf

    if os.path.exists(MODEL_PATH):
        print(f"Loading model from volume mount: {MODEL_PATH}")
        clf = joblib.load(MODEL_PATH)
    else:
        print(f"Model not found at {MODEL_PATH}. Attempting to download from {DOWNLOAD_URL}...")
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            response = requests.get(DOWNLOAD_URL, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
            clf = joblib.load(MODEL_PATH)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading model: {e}")
            raise RuntimeError("Could not load or download the model file.")

app = Flask(__name__)
swagger = Swagger(app)


@app.route('/')
def home():
    return jsonify({
        "message": "SMS Spam Detection API",
        "endpoints": {
            "predict": "/predict (POST only)",
            "docs": "/apidocs/",
            "health": "/health"
        },
        "usage": "POST to /predict with JSON: {'sms': 'your message here'}"
    })


@app.route('/health')
def health():
    return jsonify({"status": "healthy", "service": "model-service"})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict whether an SMS is Spam.
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: message to be classified.
          required: True
          schema:
            type: object
            required: sms
            properties:
                sms:
                    type: string
                    example: This is an example of an SMS.
    responses:
      200:
        description: "The result of the classification: 'spam' or 'ham'."
    """
    input_data = request.get_json()
    sms = input_data.get('sms')
    processed_sms = prepare(sms)
    prediction = clf.predict(processed_sms)[0]

    res = {
        "result": prediction,
        "classifier": "decision tree",
        "sms": sms
    }
    print(res)
    return jsonify(res)


if __name__ == '__main__':
    # Load or download model before starting the app
    dynamically_load_model()
    port = int(os.getenv("SERVER_PORT", 8081))
    app.run(host="0.0.0.0", port=port, debug=True)
