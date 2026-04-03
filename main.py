# ================================
# MACHINE LEARNING MODEL BUILDING
# ================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#C:\Users\DELL\Desktop\house price prediction\cleaned_house_price_20260403_161757.csv
df = pd.read_csv(r"C:\Users\DELL\Desktop\house price prediction\cleaned_house_price_20260403_161757.csv")
print(df.head())
print(df.columns)
print(df.isnull().sum())

# 'City', 'City_Code', 'House_Type', 
# 'House_Type_Code', 'Year_Built',
# 'House_Age', 'Price', 'Area_in_SqFt', 
# 'Bedrooms', 'Bathrooms', 'Garage',
# 'Listing_Date', 'Listing_Year'],

df.drop(columns=['City', 'House_Type'], inplace=True)
print(df.info())
print(df.isnull().sum())
#df['garage'] = df['Garage'].map({'Yes': 1, 'No': 0})
print(df['Garage'].isnull().sum())
df['Garage'].fillna('No', inplace=True)
print(df['Garage'].isnull().sum())


# 1. DEFINE FEATURES & TARGET
X = df.drop(columns=['Price', 'Listing_Date'])  # Features
y = df['Price']  # Target

print(X)
print(y)

# Convert categorical column 'Garage' to numeric
X['Garage'] = X['Garage'].map({'Yes': 1, 'No': 0})

# 2. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)

# ================================
# 3. MODEL TRAINING
# ================================

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        "MAE": mae,
        "MSE": mse,
        "R2 Score": r2
    }
    
    print(f"\n📊 {name} Results:")
    print("MAE:", mae)
    print("MSE:", mse)
    print("R2 Score:", r2)

# ================================
# 4. BEST MODEL SELECTION
# ================================
best_model_name = max(results, key=lambda x: results[x]['R2 Score'])
best_model = models[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")

# ================================
# 5. SAMPLE PREDICTION
# ================================
sample = X_test.iloc[0:1]
prediction = best_model.predict(sample)

print("\n🔮 Sample Prediction:", prediction)
print("Actual Value:", y_test.iloc[0])

# ================================
# 6. FEATURE IMPORTANCE (for Tree Models)
# ================================
if hasattr(best_model, "feature_importances_"):
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\n🔥 Feature Importance:")
    print(importance)

# ================================
# 7. SAVE MODEL (OPTIONAL)
# ================================
import joblib
joblib.dump(best_model, "house_price_model.pkl")

print("\n✅ Model saved successfully!")

