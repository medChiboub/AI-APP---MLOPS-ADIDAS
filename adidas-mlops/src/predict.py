"""
Prediction module for Adidas Operating Profit Prediction
Handles model loading and making predictions for new data
"""

import pandas as pd
import numpy as np
import joblib
import os
from preprocessing import preprocess_data, prepare_features, create_input_dataframe

class ProfitPredictor:
    """
    Operating Profit Prediction Model Wrapper
    """
    
    def __init__(self, model_path='../models/model.joblib', 
                 metadata_path='../models/model_metadata.joblib',
                 encoders_path='../models/label_encoders.joblib'):
        """
        Initialize the predictor with model and metadata paths
        
        Args:
            model_path: Path to the trained model
            metadata_path: Path to the model metadata
            encoders_path: Path to the label encoders
        """
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.encoders_path = encoders_path
        self.model = None
        self.metadata = None
        self.feature_columns = None
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model and metadata"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            if not os.path.exists(self.metadata_path):
                raise FileNotFoundError(f"Metadata not found at {self.metadata_path}")
            
            if not os.path.exists(self.encoders_path):
                raise FileNotFoundError(f"Encoders not found at {self.encoders_path}")
            
            self.model = joblib.load(self.model_path)
            self.metadata = joblib.load(self.metadata_path)
            self.feature_columns = self.metadata['feature_columns']
            
            print("✅ Model loaded successfully!")
            print(f"📊 Model type: {self.metadata['model_type']}")
            print(f"📈 Model R² score: {self.metadata['performance_metrics']['r2_score']:.4f}")
            print(f"💰 Model MAE: ${self.metadata['performance_metrics']['mean_absolute_error']:.2f}")
            
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def predict_single(self, product, region, sales_method, retailer, price_per_unit, units_sold):
        """
        Make a prediction for a single input
        
        Args:
            product: Product category (e.g., "Men's Apparel")
            region: Region (e.g., "West")
            sales_method: Sales method (e.g., "Online")
            retailer: Retailer name (e.g., "Walmart")
            price_per_unit: Price per unit in dollars
            units_sold: Number of units sold
        
        Returns:
            Predicted operating profit
        """
        try:
            # Create input dataframe
            input_df = create_input_dataframe(product, region, sales_method, retailer, price_per_unit, units_sold)
            
            # Preprocess the input
            processed_df = preprocess_data(input_df, is_training=False, encoders_path=self.encoders_path)
            
            # Prepare features
            X = prepare_features(processed_df, self.feature_columns)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            
            return max(0, prediction)  # Ensure non-negative profit
            
        except Exception as e:
            raise Exception(f"Error making prediction: {str(e)}")
    
    def predict_batch(self, input_data):
        """
        Make predictions for multiple inputs
        
        Args:
            input_data: DataFrame with columns: Product, Region, Sales Method, Price per Unit, Units Sold
        
        Returns:
            Array of predicted operating profits
        """
        try:
            # Preprocess the input
            processed_df = preprocess_data(input_data, is_training=False, encoders_path=self.encoders_path)
            
            # Prepare features
            X = prepare_features(processed_df, self.feature_columns)
            
            # Make predictions
            predictions = self.model.predict(X)
            
            return np.maximum(0, predictions)  # Ensure non-negative profits
            
        except Exception as e:
            raise Exception(f"Error making batch predictions: {str(e)}")
    
    def get_model_info(self):
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        if self.metadata is None:
            return {"error": "Model not loaded"}
        
        return {
            "model_type": self.metadata['model_type'],
            "training_date": self.metadata['training_date'],
            "performance_metrics": self.metadata['performance_metrics'],
            "feature_columns": self.feature_columns
        }
    
    def get_feature_importance(self):
        """
        Get feature importance if available
        
        Returns:
            Dictionary with feature importance or None if not available
        """
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_columns, self.model.feature_importances_))
            return sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        else:
            return None

def load_predictor(model_dir='../models/'):
    """
    Convenience function to load the predictor
    
    Args:
        model_dir: Directory containing the model files
    
    Returns:
        ProfitPredictor instance
    """
    model_path = os.path.join(model_dir, 'model.joblib')
    metadata_path = os.path.join(model_dir, 'model_metadata.joblib')
    encoders_path = os.path.join(model_dir, 'label_encoders.joblib')
    
    return ProfitPredictor(model_path, metadata_path, encoders_path)

if __name__ == "__main__":
    # Test the predictor
    try:
        predictor = load_predictor()
        
        # Test prediction
        test_prediction = predictor.predict_single(
            product="Men's Apparel",
            region="West",
            sales_method="Online",
            price_per_unit=80,
            units_sold=200
        )
        
        print(f"\n🧪 Test Prediction:")
        print(f"📊 Input: Men's Apparel, West, Online, $80, 200 units")
        print(f"💰 Predicted Operating Profit: ${test_prediction:.2f}")
        
        # Show model info
        print(f"\n📋 Model Information:")
        info = predictor.get_model_info()
        for key, value in info.items():
            if key != 'feature_columns':
                print(f"   {key}: {value}")
        
        # Show feature importance
        importance = predictor.get_feature_importance()
        if importance:
            print(f"\n🔍 Top 5 Feature Importance:")
            for i, (feature, imp) in enumerate(importance[:5]):
                print(f"   {i+1}. {feature}: {imp:.4f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Please run the Jupyter notebook first to train and save the model.")