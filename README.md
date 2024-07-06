# Laptop Price Prediction

## Project Overview

This project aims to predict laptop prices based on various features using machine learning models. The workflow includes data preprocessing, model building, and creating a GUI for real-time predictions.

## Table of Contents

1. [Introduction](#introduction)
2. [Libraries and Tools](#libraries-and-tools)
3. [Data Preprocessing](#data-preprocessing)
4. [Modeling](#modeling)
5. [GUI Implementation](#gui-implementation)
6. [Results](#results)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Conclusion](#conclusion)
10. [Contact](#contact)

## Introduction

Accurately predicting laptop prices is crucial for consumers and manufacturers. This project utilizes various machine learning techniques to forecast prices based on laptop specifications.

## Libraries and Tools

- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization
- **Scikit-learn**: Machine learning modeling and evaluation
- **XGBoost**: Extreme Gradient Boosting
- **Streamlit**: Interactive web applications

## Data Preprocessing

1. **Data Loading**:
   - Loaded the dataset using `pd.read_csv()`.

2. **Cleaning Data**:
   - Removed units from `Ram` and `Weight` columns and converted them to numerical values.
   - Created new features such as `Touchscreen` and `IPS` from `ScreenResolution`.
   - Derived `ppi` (pixels per inch) from `x_res` and `y_res`.

3. **Label Encoding**:
   - Applied `LabelEncoder` to convert categorical features to numerical values.

4. **Feature Engineering**:
   - Dropped irrelevant columns such as `ScreenResolution` and `Inches`.

## Modeling

1. **Random Forest Regressor**:
   - Configured with `n_estimators=3000`, `max_depth=500`.
   - Evaluated with training and test scores.

2. **AdaBoost Regressor**:
   - Used `DecisionTreeRegressor` as the base estimator.
   - Configured with `n_estimators=500` and `learning_rate=0.01`.

3. **XGBoost Regressor**:
   - Applied default parameters and trained the model.

## GUI Implementation

Developed a simple GUI using Streamlit for real-time laptop price prediction. The GUI allows users to input laptop specifications and obtain predicted prices instantly.

### Sample Code (Streamlit)

```python
import streamlit as st
import pickle

# Load the trained model
model = pickle.load(open("D:\Pycharm\model_pickle\laptob_price.pkl", "rb"))

# Streamlit interface
st.title("Laptop Price Predictor")

# User inputs
company = st.selectbox("Brand", data["Company"].unique())
type = st.selectbox("Type", data["TypeName"].unique())
ram = st.selectbox("Ram", [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input("Weight (kg)")
touchscreen = st.selectbox("TouchScreen", ["Yes", "No"])
ips = st.selectbox("IPS", ["Yes", "No"])
screen_size = st.number_input("Screen size (inches)")

# Predict button
if st.button("Predict Price"):
    input_features = np.array([company, type, ram, weight, touchscreen, ips, screen_size]).reshape(1, -1)
    prediction = model.predict(input_features)
    st.write(f"Predicted Price: ${prediction[0]:.2f}")
```

## Results

- **Random Forest Regressor**:
  - Training Score: 0.96
  - Test Score: 0.84

- **AdaBoost Regressor**:
  - Training Score: 0.92
  - Test Score: 0.84

- **XGBoost Regressor**:
  - Training Score: 0.99
  - Test Score: 0.86

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Osama-Abo-Bakr/laptop-price-prediction.git
   ```

2. Navigate to the project directory:
   ```bash
   cd laptop-price-prediction
   ```

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Prepare Data**:
   - Ensure the dataset is available at the specified path.

2. **Train Models**:
   - Run the provided script or Jupyter notebook to train models and evaluate performance.

3. **Predict Outcomes**:
   - Use the Streamlit interface to input specifications and predict laptop prices.

## Conclusion

This project demonstrates the effectiveness of machine learning in predicting laptop prices. The developed models and GUI provide an interactive and practical solution for real-time price estimation.

---

### Sample Code (for reference)

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

# Load data
data = pd.read_csv(r"D:\Courses language programming\6_Deep Learning\Deep Learning Project\Folder\laptop_data.csv")

# Data cleaning
data["Ram"] = data["Ram"].str.replace("GB", "").astype("float32")
data["Weight"] = data["Weight"].str.replace("kg", "").astype("float32")
data["Touchscreen"] = data["ScreenResolution"].apply(lambda x:1 if "Touchscreen" in x else 0)
data["IPS"] = data["ScreenResolution"].apply(lambda x:1 if "IPS" in x else 0)

new = data["ScreenResolution"].str.split("x", n=1, expand=True)
data["x_res"] = new[0].str.extract(r'(\d+)').astype("int32")
data["y_res"] = new[1].astype("int32")
data["ppi"] = ((data["x_res"]**2) + (data["y_res"]**2)) ** 0.5 / data["Inches"]

data = data.drop(columns=["ScreenResolution", "Inches"], axis=1)

# Label Encoding
La = LabelEncoder()
for col in data.select_dtypes(include=["object"]).columns:
    data[col] = La.fit_transform(data[col])

# Train-test split
x = data.drop(columns="Price", axis=1)
y= data["Price"]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)

# Random Forest Regressor
model_RF = RandomForestRegressor(n_estimators=3000, max_depth=500, min_samples_leaf=1, min_samples_split=4)
model_RF.fit(x_train, y_train)
print(f"Random Forest - Train Score: {model_RF.score(x_train, y_train)}")
print(f"Random Forest - Test Score: {model_RF.score(x_test, y_test)}")

# AdaBoost Regressor
model_AD = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=50, min_samples_leaf=10), n_estimators=500, learning_rate=0.01)
model_AD.fit(x_train, y_train)
print(f"AdaBoost - Train Score: {model_AD.score(x_train, y_train)}")
print(f"AdaBoost - Test Score: {model_AD.score(x_test, y_test)}")

# XGBoost Regressor
model_XGB = xgb.XGBRegressor()
model_XGB.fit(x_train, y_train)
print(f"XGBoost - Train Score: {model_XGB.score(x_train, y_train
