from flask import Flask, request, jsonify, render_template
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np
import io
import tensorflow.lite as tflite

app = Flask(__name__)

NAMES = ['Cat', 'Dog']

# Download model from HuggingFace
model_path = hf_hub_download(
    repo_id="jamirc/cat_dog_classifier",
    filename="model.tflite"
)

# Load TFLite model (super lightweight)
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def load_image_from_bytes(file_bytes):

    img = Image.open(io.BytesIO(file_bytes)).resize((200, 200))
    img = img.convert("RGB")

    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_image(img_array):

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    idx = int(np.argmax(output))
    prob = float(np.max(output))

    return NAMES[idx], prob


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict_route():

    if 'file' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_bytes = request.files['file'].read()
    img_array = load_image_from_bytes(img_bytes)

    label, prob = predict_image(img_array)

    return jsonify({
        "prediction": label,
        "probability": prob
    })


if __name__ == "__main__":
    app.run()

