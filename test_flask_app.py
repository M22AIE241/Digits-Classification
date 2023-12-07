import unittest
from flask import Flask
from flask_app import app  # Import your Flask app instance
import json
import requests
import io
import logging
import tensorflow as tf
from PIL import Image


app = app.test_client()

def convert_to_image(array):
    image = Image.fromarray((array * 255).astype('uint8'), 'L')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def test_post_predict():

    mnist = tf.keras.datasets.mnist
    (_, _), (x_test, y_test) = mnist.load_data()

    for digit in range(10):
        index = list(y_test).index(digit)
        sample_data = x_test[index]

        image_data = convert_to_image(sample_data)

        response = app.post("/predict", data={'image': (io.BytesIO(image_data), f'{digit}.png')})
        predicted_digit = json.loads(response.text)['prediction']

        assert response.status_code == 200
        try:
            assert predicted_digit == digit
        except Exception as e:
            print(f"predicted_digit: {predicted_digit}\t digit: {digit}")