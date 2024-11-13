# House Rent Prediction Using Machine Learning

This project involves building a machine learning model to predict house rental prices based on a variety of features, such as location, size, furnishing status, and number of rooms. The aim is to provide a reliable rent prediction model that can aid in assessing rental trends and setting rental expectations.

## Project Overview

Rental price prediction is valuable in the real estate industry for property management, tenant expectations, and market analysis. By analyzing different property characteristics, this project uses machine learning to provide accurate rental price estimates. The workflow involves data processing, visualization, and regression modeling to create an effective prediction model.

## Dataset and File Structure

- **Dataset**: Contains features like BHK, size, area type, city, furnishing status, tenant preference, bathrooms, and rent.
- **File Structure**:
  - `house_rent_prediction.py`: Python script with code for data preprocessing, exploratory data analysis (EDA), model training, and prediction.
  - `House_Rent_Dataset.csv`: CSV file with data used for training and testing the model.
  - `README.md`: Project description and usage guide.

## Libraries and Tools Used

- **Data Processing**: `NumPy`, `Pandas`
- **Visualization**: `Plotly`
- **Machine Learning**: `Scikit-learn`

## Data Analysis and Preprocessing

1. **Exploratory Data Analysis (EDA)**: Provides insights into data distribution, identifies missing values, and analyzes key features affecting rental prices.
2. **Visualizations**:
   - **City-wise Rent Comparison**: Bar plot showing rent distribution across different cities.
   - **Rent by Area Type**: Visual comparison of rental prices by area type.
   - **Rent Distribution by Size and BHK**: Visualizes rent trends according to property size and number of rooms.
3. **Data Cleaning**: Encodes categorical data, manages missing values, and prepares data for model training.

## Model Training and Evaluation

1. **Model Selection**: Random Forest Regressor is chosen for its strength in capturing complex relationships in the data and providing reliable predictions.
2. **Training and Testing**: The data is split into training and testing sets, and model performance is evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
3. **Prediction Feature**: Users can input property details and receive predicted rental prices based on model outputs.

## How to Run the Project

1. **Install Required Libraries**:
   ```bash
   pip install numpy pandas plotly scikit-learn
   ```
2. **Run the Script**: Execute the `house_rent_prediction.py` file in your Python environment.
3. **Usage**: The script includes an interactive component where users can input property details to predict rental prices.

## Summary

This project uses machine learning to predict house rent based on features like BHK, size, area type, city, furnishing status, tenant type, and bathrooms. It preprocesses data, trains a Random Forest Regressor model, and evaluates performance using error metrics. Users can input property details to get rental predictions, with potential improvements to add location data and test additional models.
