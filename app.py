import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# -----------------------------
# App Title
# -----------------------------
st.title("Hamilton County Property Value Predictor")

st.write(
"""
This app predicts **APPRAISED_VALUE** using Hamilton County assessor data.

Predictors used:
- LAND_VALUE
- BUILD_VALUE
- YARDITEMS_VALUE
- CALC_ACRES
"""
)


# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_excel("Housing_Hamilton_County.xlsx")


# -----------------------------
# Select Features + Target
# -----------------------------
X = data[["LAND_VALUE", "BUILD_VALUE", "YARDITEMS_VALUE", "CALC_ACRES"]]

y = data["APPRAISED_VALUE"]


# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# Model Selection
# -----------------------------
model_choice = st.sidebar.selectbox(
    "Choose Regression Model",
    ("Linear Regression", "Random Forest")
)

if model_choice == "Linear Regression":
    model = LinearRegression()
else:
    model = RandomForestRegressor()


# -----------------------------
# Train Model
# -----------------------------
model.fit(X_train, y_train)


# -----------------------------
# Model Performance
# -----------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")

st.write("Mean Absolute Error (MAE):", round(mae, 2))
st.write("R² Score:", round(r2, 3))


# -----------------------------
# Prediction Interface
# -----------------------------
st.subheader("Predict Property Value")

land_value = st.number_input("LAND_VALUE")
build_value = st.number_input("BUILD_VALUE")
yard_value = st.number_input("YARDITEMS_VALUE")
acres = st.number_input("CALC_ACRES")


# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Appraised Value"):

    prediction = model.predict([[land_value, build_value, yard_value, acres]])

    st.success(f"Predicted APPRAISED_VALUE: ${prediction[0]:,.2f}")


# -----------------------------
# Disclaimer
# -----------------------------
st.write("---")
st.write("Disclaimer: This tool is for educational purposes only.")
