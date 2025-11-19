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
    model = joblib.load('output/model.joblib')
    prediction = model.predict(processed_sms)[0]

    res = {
        "result": prediction,
        "classifier": "decision tree",
        "sms": sms
    }
    print(res)
    return jsonify(res)


if __name__ == '__main__':
    # clf = joblib.load('output/model.joblib')
    port = int(os.getenv("SERVER_PORT", 8081))
    app.run(host="0.0.0.0", port=port, debug=True)
