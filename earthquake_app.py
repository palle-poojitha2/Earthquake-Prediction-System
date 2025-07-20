import streamlit as st
import torch
import numpy as np
from model import LSTMCell
import os
from datetime import datetime, timedelta  # Import datetime

# Set parameters
N_CELLS_HOR = 200
N_CELLS_VER = 250
LEFT_BORDER = -1200
RIGHT_BORDER = 0
DOWN_BORDER = -300
UP_BORDER = 700
HEAVY_QUAKE_THRES = 3.5

# Load trained model and data
@st.cache_resource
def load_model_and_data():
    # Load frequency map
    freq_map = torch.load("freq_map_200x250")
    
    # Initialize RNN model
    RNN_cell = LSTMCell(freq_map, embedding_size=16, hidden_state_size=32,
                        n_cells_hor=N_CELLS_HOR, n_cells_ver=N_CELLS_VER, device='cpu')
    RNN_cell.load_state_dict(torch.load("Model/state_dict", map_location='cpu'))
    RNN_cell.eval()
    return RNN_cell

# Spherical to Cartesian transformation
def spherical_to_cartesian(longitude, latitude):
    ORIGIN_LATITUDE = 31.5
    ORIGIN_LONGITUDE = 84.5
    EARTH_RADIUS = 6372.33

    X = (longitude - ORIGIN_LONGITUDE) * np.pi / 180 * EARTH_RADIUS * np.cos(latitude * np.pi / 180)
    Y = (latitude - ORIGIN_LATITUDE) * np.pi / 180 * EARTH_RADIUS
    return X, Y

# Predict earthquake
def predict_earthquake(model, longitude, latitude):
    # Get current date and date 20 days later
    current_date = datetime.now()
    future_date = current_date + timedelta(days=20)

    # Convert to Cartesian coordinates
    x, y = spherical_to_cartesian(longitude, latitude)
    
    # Check if within grid bounds
    if not (LEFT_BORDER <= x <= RIGHT_BORDER and DOWN_BORDER <= y <= UP_BORDER):
        return f"Coordinates out of bounds. Current date: {current_date.strftime('%Y-%m-%d')}"

    # Normalize to grid cell
    cell_size_hor = (RIGHT_BORDER - LEFT_BORDER) / N_CELLS_HOR
    cell_size_ver = (UP_BORDER - DOWN_BORDER) / N_CELLS_VER
    x_cell = int((x - LEFT_BORDER) / cell_size_hor)
    y_cell = int((y - DOWN_BORDER) / cell_size_ver)
    
    # Prepare data tensor
    input_data = torch.zeros(1, 1, N_CELLS_HOR, N_CELLS_VER)
    input_data[0, 0, x_cell, y_cell] = 1.0
    
    # Perform prediction
    with torch.no_grad():
        hid_state = model.init_state(batch_size=1, device='cpu')
        _, output = model.forward(input_data, hid_state)
        earthquake_prob = output[0, 0, x_cell, y_cell].item()
    
    prediction = "Earthquake occurred" if earthquake_prob > HEAVY_QUAKE_THRES else "No earthquake detected"
    return (f"Current date: {current_date.strftime('%Y-%m-%d')}\n\n"
            f"Probable prediction: {future_date.strftime('%Y-%m-%d')}\n\n"
            f"Prediction: {prediction}\n"
            
            )

# Streamlit GUI
st.title("Earthquake Prediction System")
st.write("Enter the latitude and longitude.")

latitude = st.number_input("Latitude:", value=0.0, format="%.6f")
longitude = st.number_input("Longitude:", value=0.0, format="%.6f")

if st.button("Predict"):
    model = load_model_and_data()
    result = predict_earthquake(model, longitude, latitude)
    st.write(result)
