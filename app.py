import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


DATA_FILE = "Housing_Hamilton_County.xlsx"
TARGET_COL = "APPRAISED_VALUE"
FEATURES = ["LAND_VALUE", "BUILD_VALUE", "YARDITEMS_VALUE", "CALC_ACRES"]


st.set_page_config(page_title="Property Value Predictor", layout="wide")
st.title("🏠 Hamilton County Property Value Predictor")

st.markdown(
"""
This interactive app estimates **property appraised value** using Hamilton County assessor data.

### Model Inputs
- LAND_VALUE
- BUILD_VALUE
- YARDITEMS_VALUE
- CALC_ACRES

Use the sidebar to choose a regression model and enter property characteristics below.
"""
)


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    # Basic cleaning (keeps app from crashing)
    keep = FEATURES + [TARGET_COL]
    df = df[keep].copy()

    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=keep)
    df = df[df[TARGET_COL] > 0]
    return df


@st.cache_resource(show_spinner=True)
def train_model(model_choice: str):
    df = load_data(DATA_FILE)

    X = df[FEATURES]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_choice == "Linear Regression":
        model = LinearRegression()
    else:
        # Faster RF settings for a web app
        model = RandomForestRegressor(
            n_estimators=200,      # lower = faster
            random_state=42,
            n_jobs=-1              # use all CPU cores
        )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mae, r2, len(df)


# Sidebar (changing this will retrain ONCE, then cached)
model_choice = st.sidebar.selectbox(
    "Select Model Type",
    ("Linear Regression", "Random Forest"),
    index=0
)

model, mae, r2, n_rows = train_model(model_choice)

# Performance display
st.subheader("Model Performance")
c1, c2, c3 = st.columns(3)
c1.metric("Rows used", f"{n_rows:,}")
c2.metric("MAE", f"${mae:,.0f}")
c3.metric("R²", f"{r2:.3f}")

st.subheader("Predict Property Value")

# Inputs (changing these should be instant now)
land_value = st.number_input("LAND_VALUE", min_value=0.0, value=0.0, step=1000.0)
build_value = st.number_input("BUILD_VALUE", min_value=0.0, value=0.0, step=1000.0)
yard_value = st.number_input("YARDITEMS_VALUE", min_value=0.0, value=0.0, step=500.0)
acres = st.number_input("CALC_ACRES", min_value=0.0, value=0.0, step=0.01)

if st.button("Predict Appraised Value"):

    pred = model.predict([[land_value, build_value, yard_value, acres]])[0]

    error_margin = mae

    low = pred - error_margin
    high = pred + error_margin

    st.subheader("Estimated Property Value")

    col1, col2 = st.columns(2)

   col1.metric(
    label="Predicted Value",
    value=f"${pred:,.0f}"
)

col2.metric(
    label="Estimated Range",
    value=f"${low:,.0f} – ${high:,.0f}"
)

st.write("---")
st.caption("Disclaimer: Educational use only.")
