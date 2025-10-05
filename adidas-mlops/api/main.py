"""
FastAPI Backend for Adidas Operating Profit Prediction
Provides REST API endpoints for real-time profit predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import sys
import os

# Add the src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import ProfitPredictor

# Initialize FastAPI app
app = FastAPI(
    title="Adidas Operating Profit Prediction API",
    description="AI-powered API for predicting operating profit from Adidas sales data",
    version="1.0.0"
)

# Add CORS middleware for Streamlit integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for profit prediction"""
    Product: str = Field(..., description="Product category (e.g., 'Men's Apparel')")
    Region: str = Field(..., description="Sales region (e.g., 'West')")
    Sales_Method: str = Field(..., description="Sales method (e.g., 'Online')")
    Retailer: str = Field(..., description="Retailer name (e.g., 'Walmart', 'Amazon')")
    Price_per_Unit: float = Field(..., gt=0, description="Price per unit in dollars")
    Units_Sold: int = Field(..., gt=0, description="Number of units sold")

class PredictionResponse(BaseModel):
    """Response model for profit prediction"""
    Predicted_Operating_Profit: float = Field(..., description="Predicted operating profit in dollars")
    Input_Summary: dict = Field(..., description="Summary of input parameters")
    Model_Info: dict = Field(..., description="Information about the model used")

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    message: str
    model_loaded: bool

class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_type: str
    training_date: str
    performance_metrics: dict
    feature_columns: list

@app.on_event("startup")
async def startup_event():
    """Load the model on startup"""
    global predictor
    try:
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        predictor = ProfitPredictor(
            model_path=os.path.join(model_dir, 'model.joblib'),
            metadata_path=os.path.join(model_dir, 'model_metadata.joblib'),
            encoders_path=os.path.join(model_dir, 'label_encoders.joblib')
        )
        print("✅ Model loaded successfully on startup!")
    except Exception as e:
        print(f"❌ Error loading model on startup: {e}")
        predictor = None

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint for health check"""
    return HealthResponse(
        status="healthy",
        message="Adidas Operating Profit Prediction API is running!",
        model_loaded=predictor is not None
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        message="API is running" if predictor is not None else "Model not loaded",
        model_loaded=predictor is not None
    )

@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        info = predictor.get_model_info()
        print(f"Debug - Model info: {info}")
        return ModelInfoResponse(
            model_type=info["model_type"],
            training_date=info["training_date"],
            performance_metrics=info["performance_metrics"],
            feature_columns=info["feature_columns"] if info["feature_columns"] is not None else []
        )
    except Exception as e:
        print(f"Error in get_model_info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_profit(request: PredictionRequest):
    """
    Predict operating profit based on input parameters
    
    Args:
        request: PredictionRequest containing product, region, sales method, price, and units sold
    
    Returns:
        PredictionResponse with predicted operating profit and input summary
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check the server logs.")
    
    try:
        # Make prediction
        print(f"Debug - Making prediction with: {request.dict()}")
        predicted_profit = predictor.predict_single(
            product=request.Product,
            region=request.Region,
            sales_method=request.Sales_Method,
            retailer=request.Retailer,
            price_per_unit=request.Price_per_Unit,
            units_sold=request.Units_Sold
        )
        
        # Get model info
        model_info = predictor.get_model_info()
        
        # Prepare response
        return PredictionResponse(
            Predicted_Operating_Profit=round(predicted_profit, 2),
            Input_Summary={
                "Product": request.Product,
                "Region": request.Region,
                "Sales_Method": request.Sales_Method,
                "Retailer": request.Retailer,
                "Price_per_Unit": request.Price_per_Unit,
                "Units_Sold": request.Units_Sold,
                "Total_Sales": request.Price_per_Unit * request.Units_Sold
            },
            Model_Info={
                "model_type": model_info["model_type"],
                "r2_score": round(model_info["performance_metrics"]["r2_score"], 4),
                "mae": round(model_info["performance_metrics"]["mean_absolute_error"], 2)
            }
        )
        
    except Exception as e:
        print(f"Error in predict_profit: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/categories")
async def get_categories():
    """
    Get available categories for dropdowns
    
    Returns:
        Dictionary with available products, regions, and sales methods
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # These should ideally come from the training data or be stored with the model
        # For now, we'll provide common categories based on the dataset
        categories = {
            "products": [
                "Men's Apparel",
                "Women's Apparel", 
                "Men's Athletic Footwear",
                "Women's Athletic Footwear",
                "Men's Street Footwear",
                "Women's Street Footwear"
            ],
            "regions": [
                "Northeast",
                "Southeast", 
                "Midwest",
                "South",
                "West"
            ],
            "sales_methods": [
                "Online",
                "In-store",
                "Outlet"
            ],
            "retailers": [
                "Amazon",
                "Foot Locker",
                "Kohl's",
                "Sports Direct",
                "Walmart",
                "West Gear"
            ]
        }
        
        return categories
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting categories: {str(e)}")

@app.get("/feature-importance")
async def get_feature_importance():
    """
    Get feature importance from the model
    
    Returns:
        List of features sorted by importance
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        importance = predictor.get_feature_importance()
        if importance is None:
            raise HTTPException(status_code=404, detail="Feature importance not available for this model type")
        
        return {
            "feature_importance": [
                {"feature": feature, "importance": round(imp, 4)}
                for feature, imp in importance
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting feature importance: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Adidas Operating Profit Prediction API...")
    print("📚 API Documentation available at: http://localhost:8000/docs")
    print("🔍 Health check available at: http://localhost:8000/health")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)