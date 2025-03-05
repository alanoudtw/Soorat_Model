from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Load the trained CNN model
# Update the model path to be relative for deployment (assume it's in the current directory)
MODEL_PATH = "example_for_api_model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Make sure the file is uploaded to the Render environment.")

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    """Preprocess the uploaded image to match the model's input format."""
    image = image.resize((32, 32))  # Resize to (32,32) as required by model
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return model predictions."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')  # Ensure 3-channel image
        image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction)  # Get the class with highest probability
        confidence = float(np.max(prediction))  # Confidence score

        return jsonify({
            "predicted_class": int(predicted_class),
            "confidence": confidence,
            "raw_predictions": prediction.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Use the dynamic port provided by Render or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
