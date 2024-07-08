import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)
app.config["ALLOWED_EXTENSIONS"] = set(['png', 'jpg', 'jpeg'])
app.config["UPLOAD_FOLDER"] = "static/uploads/"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


model = load_model("model.h5", compile=False)


with open("labels.txt", "r") as file:
    labels = file.read().splitlines()


class_names = ["Cardboard", "Glass", "Metal", "Paper", "Plastic"]

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        if 'image' not in request.files:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "No image part"
                },
                "data": None
            }), 400

        image = request.files['image']

        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            image.save(image_path)

            img = Image.open(image_path).convert("RGB")
            img = img.resize((224, 224))
            img_array = np.asarray(img)
            img_array = np.expand_dims(img_array, axis=0)
            normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array

            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = float(prediction[0][index])

            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Success predicting"
                },
                "data": {
                    "sampah_types_prediction": class_name,
                    "confidence": confidence_score
                }
            }), 200

        else:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Invalid image format"
                },
                "data": None
            }), 400
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed"
            },
            "data": None
        }), 405

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
