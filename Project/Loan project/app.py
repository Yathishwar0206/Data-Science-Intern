from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.datasets import load_iris

app = Flask(__name__)

model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

iris = load_iris()
class_names = iris.target_names

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    sepal_length = float(request.form['sepal_length'])
    sepal_width  = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width  = float(request.form['petal_width'])

    # Correct input array
    input_data = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
    
    # Correct transform
    scaled_input = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(scaled_input)[0]
    predicted_species = class_names[prediction].capitalize()

    return render_template('index.html',
                           prediction_text=f"🌸 Predicted Iris Species: {predicted_species}"
                          )

if __name__ == '__main__':
    app.run(debug=True)

   