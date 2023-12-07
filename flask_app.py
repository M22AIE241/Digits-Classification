import pickle
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import json

app = Flask("quiz4")

with open("saved_model/mnist_classification_svm.pkl", "rb") as f:
    model = pickle.load(f)

def preprocess(image_file):
    with Image.open(image_file) as img:
        img = img.convert('L')
        img = img.resize((8, 8), Image.Resampling.LANCZOS)
        image_array = np.array(img, dtype=np.float64)
        image_array = image_array.reshape(1, -1)
        image_array *= (16 / 255) 
    return image_array

def predict_digit(image):
    prediction = model.predict(image)
    return prediction

@app.route('/test', methods=['GET', 'POST'])
def test():
    print("test")
    return jsonify({'test': 'hit'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image = request.files['image']
        digit = predict_digit(preprocess(image))[0]

        return json.dumps({'prediction': int(digit)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)