from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io

from huggingface_hub import hf_hub_download
import tflite_runtime.interpreter as tflite


# Download model from my huggingface
model_path = hf_hub_download(
    repo_id="jamirc/cat_dog_classifier",
    filename="model.tflite"
)

app = Flask(__name__)

NAMES = ['Cat', 'Dog']

#loads the image
def load_image_from_bytes(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).resize((200, 200))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array.astype(np.float32), axis=0)

#calls the model and predicts the image
def predict_image(img_array):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    interpreter.set_tensor(input_index, img_array)
    interpreter.invoke()

    pred = interpreter.get_tensor(output_index)
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

    img_bytes = request.files['file'].read()
    img_array = load_image_from_bytes(img_bytes)

    label, prob = predict_image(img_array)

    return jsonify({"prediction": label, "probability": prob})


if __name__ == '__main__':
    app.run(debug=True)

