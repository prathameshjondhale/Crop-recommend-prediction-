import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('crop_prediction_model_one.csv')

# Prepare features and label
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Split data for training (optional, for evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the newly trained model to 'model.pkl'
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Predict function using the freshly trained model
def predict_crop(n, p, k, temp, hum, ph_val, rain):
    input_data = np.array([[n, p, k, temp, hum, ph_val, rain]]).astype(np.float64)
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app
def main():
    st.title("Crop Prediction App")

    n = st.slider("Nitrogen", 0, 140)
    p = st.slider("Phosphorus", 5, 145)
    k = st.slider("Potassium", 5, 205)
    temp = st.slider("Temperature (°C)", 8.0, 45.0)
    hum = st.slider("Humidity (%)", 10.0, 100.0)
    ph_val = st.slider("pH Level", 3.0, 10.0)
    rain = st.slider("Rainfall (mm)", 20.0, 300.0)

    if st.button("Predict Crop"):
        crop = predict_crop(n, p, k, temp, hum, ph_val, rain)
        st.success(f"The best crop to grow is: **{crop.capitalize()}**")

if __name__ == '__main__':
    main()
