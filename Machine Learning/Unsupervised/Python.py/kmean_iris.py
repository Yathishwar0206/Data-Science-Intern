import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_iris

# Load and save model
model = joblib.load("kmeans.pkl")
scaler = joblib.load("scaler.pkl")

# Load the cancer dataset
df = load_iris()
class_names = df.target_names

# Streamlit APP UI
st.title('Iris flower prediction using K-Mean')
st.write("Enter the measurment below to Predict")

# Side bar info
st.sidebar.header("About the Project")
st.sidebar.info(
    "This app uses a K-Mean model trained on the Iris dataset. "
    "It predicts whether the flower is Setosa, Versicolor, or Virginica."
)
st.sidebar.write("Make your prediction below !")

# User input fields
sepal_length = st.number_input("Sepal length (cm)",min_value=0.0, max_value=10.0, value=5.1)
sepal_width  = st.number_input("Sepal width (cm)",min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal length (cm)",min_value=0.0, max_value=10.0, value=1.4)
petal_width  = st.number_input("Petal width (cm)",min_value=0.0, max_value=10.0, value=0.2)

# Prediction
if st.button('Predict'):
    input_data = np.array[[sepal_length, sepal_width, petal_length, petal_width]]
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    st.success(f"Predicted iris species:**{class_names[prediction].capitalize()}**")
    