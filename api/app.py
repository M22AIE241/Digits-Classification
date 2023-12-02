from joblib import load
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from flask import Flask, request, jsonify
import numpy as np

# Add the load_model function here
def load_model(model_type):
    model_path = f"models/M22AIE241_lr_lbfgs.joblib"
    loaded_model = load(model_path)
    return loaded_model

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello World!</p>"

@app.route('/user/<username>')
def profile(username):
    return f'{username}\'s profile'


@app.route("/sum/<x>/<y>")
def sum_two_numbers(x,y):
    res = int(x) + int(y)
    return f"sum of {x} and {y} is {res}"


@app.route('/predict/', methods=['POST'])
def predict(model_type):
    # Load the model
    model = load_model(model_type)

    # Get the input data from the request
    data = request.get_json(force=True)
    # Assuming the input data is a list of feature values
    features = np.array(data['features']).reshape(1, -1)

    # Make predictions
    prediction = model.predict(features)

    # Prepare the response
    response = {'prediction': int(prediction[0])}

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)