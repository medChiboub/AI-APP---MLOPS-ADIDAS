"""
Simple API Test Script
Tests the Adidas Profit Prediction API
"""

import requests
import json

# API Configuration
API_URL = "http://localhost:8000"

def test_api():
    """Test the API with a sample prediction"""
    
    # Test data
    test_data = {
        "product": "Men's Apparel",
        "region": "West",
        "sales_method": "Online",
        "retailer": "Walmart",
        "price_per_unit": 80.0,
        "units_sold": 200
    }
    
    print("🚀 Testing Adidas Profit Prediction API")
    print("=" * 50)
    
    try:
        # Test root endpoint
        print("📡 Testing API connection...")
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            print("✅ API is running!")
            print(f"📊 Response: {response.json()}")
        else:
            print(f"❌ API connection failed: {response.status_code}")
            return
        
        # Test prediction endpoint
        print("\n🔮 Testing prediction endpoint...")
        print(f"📊 Input: {test_data}")
        
        response = requests.post(f"{API_URL}/predict", json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"💰 Predicted Profit: ${result['predicted_profit']:,.2f}")
            print(f"💵 Total Sales: ${result['total_sales']:,.2f}")
            print(f"🤖 Model Used: {result['model_used']}")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"Error: {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is it running?")
        print("💡 Start the API with: python API/main.py")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_api()