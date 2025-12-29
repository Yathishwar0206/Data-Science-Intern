import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer

# Load save model and scaler
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load the cancer dataset
cancer = load_breast_cancer()
class_names = cancer.target_names

# Streamlit APP UI
st.title('Load breast cancer using SVM')
st.write('Enter the measurment below to preict it their is cancer or not')

# Side bar info
st.sidebar.header("About the Project")
st.sidebar.info("This app is very usefull to identify the person was affcted by the cancer. " \
"It predicts the Benign(Non-Cancerous) or Malignant(Cancerous).")
st.sidebar.write("Make your prediction below")

# User input fields

mean_radius = st.number_input("Mean_radius", min_value=0.0, max_value=20.0, value=5.1)
mean_texture = st.number_input("Mean_texture", min_value=0.0, max_value=20.0, value=3.0)
mean_perimeter  = st.number_input("Mean_perimeter", min_value=0.0, max_value=150.0, value=1.4)
mean_area = st.number_input("Mean_area", min_value=0.0, max_value=1500.0, value=0.2)
mean_smoothness = st.number_input("Mean_smoothness", min_value=0.0, max_value=1.0, value=0.2)
mean_compactness = st.number_input("Mean_compactness", min_value=0.0, max_value=1.0, value=0.1)
mean_concavity = st.number_input("Mean_concavity", min_value=0.0, max_value=1.0, value=0.2)
mean_concave_points = st.number_input("Mean_concave_points", min_value=0.0, max_value=1.0, value=0.4)
mean_symmetry = st.number_input("Mean_symmetry", min_value=0.0, max_value=1.0, value=0.1)
mean_fractal_dimension = st.number_input("Mean_fractal_dimension", min_value=0.0, max_value=1.0, value=0.5)
radius_error = st.number_input("Radius_error", min_value=0.0, max_value=10.0, value=1.6)
texture_error = st.number_input("Texture_error", min_value=0.0, max_value=10.0, value=4.5)
perimeter_error = st.number_input("Perimeter_error", min_value=0.0, max_value=10.0, value=1.6)
area_error = st.number_input("area_error", min_value=0.0, max_value=10.0, value=3.2)
smoothness_error = st.number_input("Smoothness_error", min_value=0.0, max_value=10.0, value=3.6)
compactness_error = st.number_input("Compactness_error", min_value=0.0, max_value=10.0, value=5.4)
concavity_error = st.number_input("Concavity_error", min_value=0.0, max_value=10.0, value=2.5)
concave_points_error = st.number_input("Concave_points_error", min_value=0.0, max_value=10.0, value=0.9)
symmetry_error = st.number_input("Symmetry_error", min_value=0.0, max_value=10.0, value=5.7)
fractual_dimesnsion_error = st.number_input("Fractual_dimension_error", min_value=0.0, max_value=10.0, value=0.2)
worst_texture  = st.number_input("Worst_texture", min_value=0.0, max_value=30.0, value=3.8) 
worst_perimeter = st.number_input("Worst_texture", min_value=0.0, max_value=200.0, value=1.2)  
worst_area = st.number_input("Worst_area", min_value=0.0, max_value=2500.0, value=4.4)
worst_smoothness  = st.number_input("Worst_texture", min_value=0.0, max_value=1.0, value=0.9)
worst_compactness  = st.number_input("Worst_compactness", min_value=0.0, max_value=1.0, value=0.4)
worst_concavity  = st.number_input("Worst_concavity", min_value=0.0, max_value=1.0, value=0.2) 
worst_concave_points = st.number_input("Worst_conave_points", min_value=0.0, max_value=1.0, value=0.5)
worst_radius = st.number_input('Worst_radius',min_value=0.0, max_value=1.0, value=0.7)
worst_symmetry = st.number_input('Worst_symmetry',min_value=0.0, max_value=1.0, value=0.4)
worst_fractal_dimesnsion = st.number_input('Worst_fractual_dimension',min_value=0.0, max_value=1.0, value=0.3)
# Prediction
if st.button('Predict'):
    input_data = np.array([[mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness,mean_compactness,
                           mean_concavity,mean_concave_points,mean_symmetry,mean_fractal_dimension,
                           radius_error,texture_error,perimeter_error,area_error,smoothness_error,
                           compactness_error,concavity_error,concave_points_error,symmetry_error,
                           fractual_dimesnsion_error,worst_texture,worst_perimeter,worst_area,
                           worst_smoothness,worst_compactness,worst_concavity,worst_concave_points,worst_radius,
                            worst_symmetry, worst_fractal_dimesnsion ]])  # total 30
    scaler_input = scaler.transform(input_data)
    prediction = model.predict(scaler_input)[0]
    st.success(f'Predicted cancer Banign(Non-Cancerous) or Malignant(Cancerous) :**{class_names[prediction].capitalize()}**')
