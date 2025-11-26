from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
from huggingface_hub import hf_hub_download

#using hugging face to upload my model and use it cause github dosnt allow 1gb+ archives
model_path = hf_hub_download(
    repo_id="jamirc/cat_dog_classifier",
    filename="model.h5")


app = Flask(__name__)

# Load the model model
model = load_model(model_path, compile= False)

NAMES = ['Cat', 'Dog']   # 0 = Cat, 1 = Dog


def load_image_from_bytes(file_bytes):
    """Load and prepare image from uploaded file"""
    img = Image.open(io.BytesIO(file_bytes)).resize((200, 200))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)


def predict_image(img_array):
    """Run the model prediction"""
    pred = model.predict(img_array)
    idx = np.argmax(pred)
    prob = float(np.max(pred))
    return NAMES[idx], prob


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['file']
    img_bytes = file.read()

    img_array = load_image_from_bytes(img_bytes)
    label, prob = predict_image(img_array)

    return jsonify({
        "prediction": label,
        "probability": prob
    })


if __name__ == '__main__':
    app.run(debug=True)

