"""
Quick fix script to retrain model with retailer feature
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import os
from datetime import datetime

# Load data
print("🔍 Loading data...")
df = pd.read_csv('data/processed/cleaned_adidas.csv')

# Preprocess the data manually to include retailer
print("🧹 Preprocessing data...")

# Clean negative profits
df = df[df['Operating Profit'] >= 0]

# Create time features
df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])
df['Year'] = df['Invoice Date'].dt.year
df['Month'] = df['Invoice Date'].dt.month
df['Quarter'] = df['Invoice Date'].dt.quarter

# Create derived features
df['Revenue_per_Unit'] = df['Total Sales'] / df['Units Sold']
df['Profit_Margin'] = df['Operating Profit'] / df['Total Sales']

# Encode categorical features including Retailer
categorical_features = ['Region', 'Product', 'Sales Method', 'Retailer']
label_encoders = {}

for feature in categorical_features:
    le = LabelEncoder()
    df[f'{feature}_encoded'] = le.fit_transform(df[feature])
    label_encoders[feature] = le

# Prepare features with Retailer
feature_columns = [
    'Price per Unit', 'Units Sold', 'Year', 'Month', 'Quarter',
    'Region_encoded', 'Product_encoded', 'Sales Method_encoded', 'Retailer_encoded',
    'Profit_Margin', 'Revenue_per_Unit'
]

X = df[feature_columns]
y = df['Operating Profit']

print(f"📊 Dataset shape: {X.shape}")
print(f"🎯 Target variable shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Training set: {X_train.shape}")
print(f"📊 Testing set: {X_test.shape}")

# Train XGBoost model
print("🚀 Training XGBoost model with Retailer feature...")
model = xgb.XGBRegressor(random_state=42, verbosity=0)
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import r2_score, mean_absolute_error

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"\n📊 MODEL PERFORMANCE:")
print(f"Training R²: {train_r2:.4f}")
print(f"Testing R²: {test_r2:.4f}")
print(f"Training MAE: ${train_mae:.2f}")
print(f"Testing MAE: ${test_mae:.2f}")

# Feature importance
print(f"\n🔍 FEATURE IMPORTANCE:")
feature_importance = model.feature_importances_
for i, (feature, importance) in enumerate(zip(feature_columns, feature_importance)):
    print(f"{i+1:2d}. {feature:<25} {importance:.4f}")

# Save the model and metadata
os.makedirs('models', exist_ok=True)

# Save model
model_path = 'models/model.joblib'
joblib.dump(model, model_path)

# Save label encoders
encoders_path = 'models/label_encoders.joblib'
joblib.dump(label_encoders, encoders_path)

# Save metadata
metadata = {
    'model_type': 'XGBoost',
    'training_date': datetime.now().isoformat(),
    'feature_columns': feature_columns,
    'performance_metrics': {
        'r2_score': test_r2,
        'mean_absolute_error': test_mae,
        'train_r2': train_r2,
        'train_mae': train_mae
    },
    'data_info': {
        'n_samples': len(df),
        'n_features': len(feature_columns),
        'n_train': len(X_train),
        'n_test': len(X_test)
    }
}

metadata_path = 'models/model_metadata.joblib'
joblib.dump(metadata, metadata_path)

print(f"\n✅ MODEL SAVED:")
print(f"Model: {model_path}")
print(f"Encoders: {encoders_path}")
print(f"Metadata: {metadata_path}")

print(f"\n🎯 NEW MODEL STATS:")
print(f"R² Score: {test_r2:.4f} ({(test_r2*100):.2f}%)")
print(f"MAE: ${test_mae:.2f}")
print(f"Features: {len(feature_columns)} (including Retailer)")

# Test the model with retailer prediction
print(f"\n🧪 Testing prediction with Retailer:")
test_data = pd.DataFrame([{
    'Price per Unit': 75.0,
    'Units Sold': 1000,
    'Year': 2024,
    'Month': 10,
    'Quarter': 4,
    'Region_encoded': label_encoders['Region'].transform(['West'])[0],
    'Product_encoded': label_encoders['Product'].transform(["Men's Apparel"])[0],
    'Sales Method_encoded': label_encoders['Sales Method'].transform(['Online'])[0],
    'Retailer_encoded': label_encoders['Retailer'].transform(['Amazon'])[0],
    'Profit_Margin': 0.3,
    'Revenue_per_Unit': 75.0
}])

prediction = model.predict(test_data)[0]
print(f"📊 Input: Men's Apparel, West, Online, Amazon, $75, 1000 units")
print(f"🎯 Predicted Operating Profit: ${prediction:.2f}")

print(f"\n🚀 Model ready for deployment with Retailer support!")