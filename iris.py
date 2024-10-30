import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris_dataset = load_iris()
X = iris_dataset['data'][:, [0, 1]]  # Sepal length and width
y = iris_dataset['target']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train the Random Forest model
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)

# Predict on the test data
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Function to predict flower species
def predict_flower(sepal_length, sepal_width):
    input_data = np.array([sepal_length, sepal_width]).reshape(1, -1)
    prediction = rf.predict(input_data)
    return iris_dataset['target_names'][prediction[0]]

# Streamlit UI setup
st.title("Iris Flower Predictor")
st.write(f"Model Accuracy: {accuracy:.2f}")

# Set the minimum, maximum, and default values for the sliders
sl_min = X_train[:, 0].min().round(2)
sl_max = X_train[:, 0].max().round(2)
sl_default = X_train[:, 0].mean().round(2)

sw_min = X_train[:, 1].min().round(2)
sw_max = X_train[:, 1].max().round(2)
sw_default = X_train[:, 1].mean().round(2)

# Create sliders for user input
sepal_length = st.slider("Sepal Length (cm)", min_value=sl_min, max_value=sl_max, value=sl_default, step=0.1)
sepal_width = st.slider("Sepal Width (cm)", min_value=sw_min, max_value=sw_max, value=sw_default, step=0.1)

# Predict and display the result
if st.button("Predict Flower Species"):
    prediction = predict_flower(sepal_length, sepal_width)
    st.write(f"The predicted species is: **{prediction}**")
