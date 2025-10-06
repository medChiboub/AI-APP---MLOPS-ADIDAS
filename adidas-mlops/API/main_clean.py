"""
Simple FastAPI for Adidas Operating Profit Prediction
Basic POC API endpoint
"""

from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# Add the parent directory to path to import from models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Try to import the predictor, fallback to simple calculation if not available
try:
    from src.predict import ProfitPredictor
    predictor = ProfitPredictor()
    MODEL_LOADED = True
except:
    MODEL_LOADED = False
    print("⚠️ Model not loaded, using simple calculation")

app = FastAPI(title="Adidas Profit Prediction API", version="1.0.0")

class PredictionRequest(BaseModel):
    product: str
    region: str
    sales_method: str
    retailer: str
    price_per_unit: float
    units_sold: int

class PredictionResponse(BaseModel):
    predicted_profit: float
    total_sales: float
    model_used: str

@app.get("/")
def root():
    return {
        "message": "Adidas Operating Profit Prediction API",
        "model_loaded": MODEL_LOADED,
        "status": "running"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_profit(request: PredictionRequest):
    # Calculate total sales
    total_sales = request.price_per_unit * request.units_sold
    
    if MODEL_LOADED:
        try:
            # Use trained ML model
            predicted_profit = predictor.predict_single(
                product=request.product,
                region=request.region,
                sales_method=request.sales_method,
                retailer=request.retailer,
                price_per_unit=request.price_per_unit,
                units_sold=request.units_sold
            )
            model_used = "XGBoost ML Model"
        except Exception as e:
            # Fallback to simple calculation
            predicted_profit = total_sales * 0.25  # 25% profit margin
            model_used = f"Simple Calculation (Error: {str(e)})"
    else:
        # Simple business rule: 25% profit margin
        predicted_profit = total_sales * 0.25
        model_used = "Simple Calculation"
    
    return PredictionResponse(
        predicted_profit=round(predicted_profit, 2),
        total_sales=total_sales,
        model_used=model_used
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)