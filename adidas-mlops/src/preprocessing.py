"""
Preprocessing module for Adidas Operating Profit Prediction
Handles data cleaning and feature encoding for both training and inference
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def clean_currency_column(column):
    """Clean currency columns by removing $ and commas, then convert to float"""
    if isinstance(column, pd.Series):
        return column.str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.replace(' ', '', regex=False).astype(float)
    else:
        # Handle single values
        if isinstance(column, str):
            return float(column.replace('$', '').replace(',', '').replace(' ', ''))
        return float(column)

def clean_numeric_column(column):
    """Clean numeric columns by removing commas and converting to appropriate type"""
    if isinstance(column, pd.Series):
        return column.str.replace(',', '', regex=False).astype(float)
    else:
        # Handle single values
        if isinstance(column, str):
            return float(column.replace(',', ''))
        return float(column)

def preprocess_data(df, is_training=True, encoders_path='../models/label_encoders.joblib'):
    """
    Preprocess the dataset for training or inference
    
    Args:
        df: Input DataFrame
        is_training: Boolean indicating if this is for training (True) or inference (False)
        encoders_path: Path to saved label encoders
    
    Returns:
        Preprocessed DataFrame ready for model training/inference
    """
    df_processed = df.copy()
    
    # Clean numeric columns if they contain string formatting
    if 'Price per Unit' in df_processed.columns:
        if df_processed['Price per Unit'].dtype == 'object':
            df_processed['Price per Unit'] = clean_currency_column(df_processed['Price per Unit'])
    
    if 'Operating Profit' in df_processed.columns:
        if df_processed['Operating Profit'].dtype == 'object':
            df_processed['Operating Profit'] = clean_currency_column(df_processed['Operating Profit'])
    
    if 'Total Sales' in df_processed.columns:
        if df_processed['Total Sales'].dtype == 'object':
            df_processed['Total Sales'] = clean_numeric_column(df_processed['Total Sales'])
    
    # Convert Invoice Date to datetime and extract features if present
    if 'Invoice Date' in df_processed.columns:
        df_processed['Invoice Date'] = pd.to_datetime(df_processed['Invoice Date'])
        df_processed['Year'] = df_processed['Invoice Date'].dt.year
        df_processed['Month'] = df_processed['Invoice Date'].dt.month
        df_processed['Quarter'] = df_processed['Invoice Date'].dt.quarter
    else:
        # For inference, use current date features if not provided
        from datetime import datetime
        current_date = datetime.now()
        df_processed['Year'] = current_date.year
        df_processed['Month'] = current_date.month
        df_processed['Quarter'] = (current_date.month - 1) // 3 + 1
    
    # Create derived features
    if 'Total Sales' in df_processed.columns and 'Units Sold' in df_processed.columns:
        df_processed['Revenue_per_Unit'] = df_processed['Total Sales'] / df_processed['Units Sold']
    else:
        # For inference without Total Sales, calculate it
        df_processed['Total Sales'] = df_processed['Price per Unit'] * df_processed['Units Sold']
        df_processed['Revenue_per_Unit'] = df_processed['Price per Unit']
    
    if 'Operating Profit' in df_processed.columns:
        df_processed['Profit_Margin'] = df_processed['Operating Profit'] / df_processed['Total Sales']
    else:
        # For inference, we don't have Operating Profit, so we'll set a default
        df_processed['Profit_Margin'] = 0.3  # Default profit margin assumption
    
    # Handle categorical encoding
    categorical_features = ['Region', 'Product', 'Sales Method']
    
    if is_training:
        # Create and fit new encoders
        label_encoders = {}
        for feature in categorical_features:
            if feature in df_processed.columns:
                le = LabelEncoder()
                df_processed[f'{feature}_encoded'] = le.fit_transform(df_processed[feature])
                label_encoders[feature] = le
        
        # Save encoders
        os.makedirs(os.path.dirname(encoders_path), exist_ok=True)
        joblib.dump(label_encoders, encoders_path)
    else:
        # Load existing encoders for inference
        if os.path.exists(encoders_path):
            label_encoders = joblib.load(encoders_path)
            for feature in categorical_features:
                if feature in df_processed.columns:
                    le = label_encoders[feature]
                    # Handle unseen categories
                    df_processed[f'{feature}_encoded'] = df_processed[feature].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else 0
                    )
        else:
            raise FileNotFoundError(f"Label encoders not found at {encoders_path}")
    
    return df_processed

def prepare_features(df, feature_columns=None):
    """
    Prepare feature matrix for model input
    
    Args:
        df: Preprocessed DataFrame
        feature_columns: List of feature columns to select
    
    Returns:
        Feature matrix (X) ready for model
    """
    if feature_columns is None:
        feature_columns = [
            'Price per Unit', 'Units Sold', 'Year', 'Month', 'Quarter',
            'Region_encoded', 'Product_encoded', 'Sales Method_encoded',
            'Profit_Margin', 'Revenue_per_Unit'
        ]
    
    # Ensure all required columns are present
    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return df[feature_columns]

def create_input_dataframe(product, region, sales_method, price_per_unit, units_sold):
    """
    Create a DataFrame from individual input values for inference
    
    Args:
        product: Product category
        region: Region
        sales_method: Sales method
        price_per_unit: Price per unit
        units_sold: Number of units sold
    
    Returns:
        DataFrame ready for preprocessing
    """
    return pd.DataFrame([{
        'Product': product,
        'Region': region,
        'Sales Method': sales_method,
        'Price per Unit': price_per_unit,
        'Units Sold': units_sold
    }])