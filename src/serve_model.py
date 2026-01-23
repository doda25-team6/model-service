"""
Flask API of the SMS Spam detection model.
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
MODEL_URL = os.getenv("MODEL_URL", "")
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_URL.split('/')[-1])
PREPROCESSOR_URL = os.getenv("PREPROCESSOR_URL", "")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, PREPROCESSOR_URL.split('/')[-1])

# DOWNLOAD_URL = "https://github.com/doda25-team6/model-service/releases/download/v0.1.2/preprocessor-v0.1.2.joblib"

# Global variable to hold the loaded model
clf = None

def dynamically_load_model_files():
    """
    Stores the loaded model and preprocessor into global variables.
    """
    global clf

    clf = dynamically_load_file(MODEL_PATH, MODEL_URL)
    # just need the preprocessor to be loaded in the path
    _ = dynamically_load_file(PREPROCESSOR_PATH, PREPROCESSOR_URL)

def dynamically_load_file(path, url):
    """
    Checks if the model exists in the volume mount path. If not, downloads it.
    Returns the loaded model.
    """
    if os.path.exists(path):
        print(f"Loading model from volume mount: {path}")
        return joblib.load(path)
    else:
        print(f"Model not found at {path}. Attempting to download from {url}...")

        if not url:
            raise RuntimeError(f"No URL is provided for {path}")

        try:
            print("step11")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print("step1")
            response = requests.get(url, stream=True)
            print("step2")
            response.raise_for_status()
            print("step3")
            with open(path, 'wb') as f:
                print("step4")
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    print("step5")
            print("step6")
            print("Download complete.")
            print("step7")
            return joblib.load(path)
            print("step8")
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
    
    # Get prediction probabilities for confidence score
    try:
        probabilities = clf.predict_proba(processed_sms)[0]
        # Confidence is the probability of the predicted class
        classes = clf.classes_
        predicted_class_idx = list(classes).index(prediction)
        confidence = float(probabilities[predicted_class_idx])
    except AttributeError:
        confidence = None

    res = {
        "result": prediction,
        "classifier": "decision tree",
        "sms": sms,
        "confidence": confidence
    }
    print(res)
    return jsonify(res)


if __name__ == '__main__':
    # Load or download model before starting the app
    dynamically_load_model_files()
    port = int(os.getenv("SERVER_PORT", 8081))
    app.run(host="0.0.0.0", port=port, debug=True)
