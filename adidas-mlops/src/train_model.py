"""
Model training module for Adidas Operating Profit Prediction
Handles model training, evaluation, and saving
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import os
from datetime import datetime
from preprocessing import preprocess_data, prepare_features

def load_data(data_path):
    """Load and return the dataset"""
    return pd.read_csv(data_path)

def train_model(data_path, model_save_path='../models/', test_size=0.2, random_state=42):
    """
    Train the operating profit prediction model
    
    Args:
        data_path: Path to the training data CSV
        model_save_path: Directory to save the trained model
        test_size: Proportion of data to use for testing
        random_state: Random state for reproducibility
    
    Returns:
        Dictionary containing training results and metrics
    """
    
    print("🔍 Loading data...")
    df = load_data(data_path)
    
    print("🧹 Preprocessing data...")
    df_processed = preprocess_data(df, is_training=True)
    
    # Remove negative operating profits
    df_processed = df_processed[df_processed['Operating Profit'] >= 0]
    
    print("🔧 Preparing features...")
    feature_columns = [
        'Price per Unit', 'Units Sold', 'Year', 'Month', 'Quarter',
        'Region_encoded', 'Product_encoded', 'Sales Method_encoded',
        'Profit_Margin', 'Revenue_per_Unit'
    ]
    
    X = prepare_features(df_processed, feature_columns)
    y = df_processed['Operating Profit']
    
    print(f"📊 Dataset shape: {X.shape}")
    print(f"🎯 Target variable shape: {y.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"📊 Training set: {X_train.shape}")
    print(f"📊 Testing set: {X_test.shape}")
    
    # Train models
    models = {
        'Random Forest': RandomForestRegressor(random_state=random_state),
        'XGBoost': xgb.XGBRegressor(random_state=random_state, verbosity=0)
    }
    
    model_results = {}
    
    print("🚀 Training models...")
    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Store results
        model_results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        }
        
        print(f"✅ {name} Results:")
        print(f"   Training R²: {train_r2:.4f}")
        print(f"   Testing R²: {test_r2:.4f}")
        print(f"   Testing MAE: ${test_mae:.2f}")
        print(f"   Testing RMSE: ${test_rmse:.2f}")
    
    # Select best model
    best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['test_r2'])
    best_model = model_results[best_model_name]['model']
    
    print(f"\n🏆 Best model: {best_model_name}")
    
    # Hyperparameter tuning for the best model
    print(f"🔧 Performing hyperparameter tuning for {best_model_name}...")
    
    if best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=random_state),
            param_grid,
            cv=3,
            scoring='r2',
            n_jobs=-1
        )
    else:  # XGBoost
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        
        grid_search = GridSearchCV(
            xgb.XGBRegressor(random_state=random_state, verbosity=0),
            param_grid,
            cv=3,
            scoring='r2',
            n_jobs=-1
        )
    
    grid_search.fit(X_train, y_train)
    final_model = grid_search.best_estimator_
    
    # Final evaluation
    y_pred_final = final_model.predict(X_test)
    final_r2 = r2_score(y_test, y_pred_final)
    final_mae = mean_absolute_error(y_test, y_pred_final)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))
    
    # Cross-validation
    cv_scores = cross_val_score(final_model, X_train, y_train, cv=5, scoring='r2')
    
    print(f"\n🎯 FINAL MODEL PERFORMANCE:")
    print(f"📊 Model: {best_model_name}")
    print(f"📈 R² Score: {final_r2:.4f}")
    print(f"💰 MAE: ${final_mae:.2f}")
    print(f"📏 RMSE: ${final_rmse:.2f}")
    print(f"🔄 CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save model and metadata
    os.makedirs(model_save_path, exist_ok=True)
    
    model_path = os.path.join(model_save_path, 'model.joblib')
    joblib.dump(final_model, model_path)
    
    # Create metadata
    model_metadata = {
        'model_type': best_model_name,
        'model_params': grid_search.best_params_,
        'performance_metrics': {
            'r2_score': final_r2,
            'mean_absolute_error': final_mae,
            'root_mean_square_error': final_rmse,
            'cv_score_mean': cv_scores.mean(),
            'cv_score_std': cv_scores.std()
        },
        'feature_columns': feature_columns,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_data_shape': df_processed.shape
    }
    
    metadata_path = os.path.join(model_save_path, 'model_metadata.joblib')
    joblib.dump(model_metadata, metadata_path)
    
    print(f"\n💾 Model saved to: {model_path}")
    print(f"📋 Metadata saved to: {metadata_path}")
    
    return {
        'model': final_model,
        'metadata': model_metadata,
        'test_metrics': {
            'r2': final_r2,
            'mae': final_mae,
            'rmse': final_rmse
        }
    }

if __name__ == "__main__":
    # Train the model with the cleaned data
    data_path = "../data/processed/cleaned_adidas.csv"
    if not os.path.exists(data_path):
        print("❌ Cleaned data not found. Please run the Jupyter notebook first to generate cleaned data.")
    else:
        results = train_model(data_path)
        print("✅ Model training completed successfully!")