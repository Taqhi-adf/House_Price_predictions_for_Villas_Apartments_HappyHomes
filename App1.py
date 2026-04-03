import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="House Price Prediction", layout="wide")

st.title("🏠 House Price Prediction App")
st.write("Machine Learning Model Comparison + Prediction")

# ================================
# LOAD DATA
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\DELL\Desktop\house price prediction\cleaned_house_price_20260403_161757.csv")
    return df

df = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# ================================
# PREPROCESSING
# ================================
df.drop(columns=['City', 'House_Type'], inplace=True)

# Fill missing Garage
df['Garage'].fillna('No', inplace=True)

# Features & Target
X = df.drop(columns=['Price', 'Listing_Date'])
y = df['Price']

# Encode Garage
X['Garage'] = X['Garage'].map({'Yes': 1, 'No': 0})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# MODEL TRAINING
# ================================
st.subheader("🤖 Train Models")

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100)
}

if st.button("Train Models"):
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "R2 Score": r2_score(y_test, y_pred)
        }

    results_df = pd.DataFrame(results).T

    st.subheader("📊 Model Performance")
    st.dataframe(results_df)

    # Best model
    best_model_name = results_df['R2 Score'].idxmax()
    best_model = models[best_model_name]

    st.success(f"🏆 Best Model: {best_model_name}")

    # ================================
    # USER INPUT
    # ================================
    st.subheader("🔮 Predict House Price")

    col1, col2 = st.columns(2)

    with col1:
        bedrooms = st.slider("Bedrooms", 1, 6, 2)
        bathrooms = st.slider("Bathrooms", 1, 5, 2)
        garage = st.selectbox("Garage", ["Yes", "No"])

    with col2:
        area = st.number_input("Area (SqFt)", 500, 5000, 1000)
        year_built = st.number_input("Year Built", 1950, 2023, 2000)
        house_age = 2026 - year_built

    if st.button("Predict"):
        input_df = pd.DataFrame({
            'City_Code': [0],
            'House_Type_Code': [0],
            'Year_Built': [year_built],
            'House_Age': [house_age],
            'Area_in_SqFt': [area],
            'Bedrooms': [bedrooms],
            'Bathrooms': [bathrooms],
            'Garage': [1 if garage == "Yes" else 0],
            'Listing_Year': [2024]
        })

        prediction = best_model.predict(input_df)

        st.success(f"💰 Predicted Price: {prediction[0]:.4f}")

    # ================================
    # FEATURE IMPORTANCE
    # ================================
    if hasattr(best_model, "feature_importances_"):
        st.subheader("🔥 Feature Importance")

        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': best_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        st.dataframe(importance)