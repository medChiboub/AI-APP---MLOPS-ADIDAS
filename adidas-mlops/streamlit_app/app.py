"""
Streamlit Dashboard for Adidas Operating Profit Prediction
Interactive web interface for executives to predict operating profits
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Adidas Profit Predictor",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"  # Change this for Docker deployment

def check_api_health():
    """Check if the API is healthy and model is loaded"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "unhealthy", "model_loaded": False}
    except requests.exceptions.RequestException:
        return {"status": "unreachable", "model_loaded": False}

def get_categories():
    """Get available categories from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/categories", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def get_model_info():
    """Get model information from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/model-info", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def make_prediction(product, region, sales_method, price_per_unit, units_sold):
    """Make a prediction using the API"""
    try:
        payload = {
            "Product": product,
            "Region": region,
            "Sales_Method": sales_method,
            "Price_per_Unit": price_per_unit,
            "Units_Sold": units_sold
        }
        
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
        
        if response.status_code == 200:
            return response.json(), None
        else:
            error_detail = response.json().get("detail", "Unknown error")
            return None, f"API Error: {error_detail}"
            
    except requests.exceptions.RequestException as e:
        return None, f"Connection Error: {str(e)}"

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("👟 Adidas Operating Profit Predictor")
    st.markdown("**AI-Powered Profit Prediction for Executive Decision Making**")
    
    # Check API health
    health_status = check_api_health()
    
    # Status indicator
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if health_status["status"] == "healthy" and health_status["model_loaded"]:
            st.success("🟢 API Online")
        elif health_status["status"] == "healthy":
            st.warning("🟡 API Online, Model Loading")
        else:
            st.error("🔴 API Offline")
    
    with col2:
        if health_status["model_loaded"]:
            st.success("🟢 Model Ready")
        else:
            st.error("🔴 Model Not Loaded")
    
    with col3:
        st.info(f"⏰ Last checked: {datetime.now().strftime('%H:%M:%S')}")
    
    # Show API status details
    if health_status["status"] != "healthy" or not health_status["model_loaded"]:
        st.error("⚠️ **Service Unavailable**")
        st.markdown("""
        **Troubleshooting Steps:**
        1. Ensure the FastAPI server is running on `http://localhost:8000`
        2. Check that the model has been trained and saved
        3. Verify all required files are in the `models/` directory
        4. Run: `python api/main.py` to start the API server
        """)
        st.stop()
    
    # Sidebar for model information
    with st.sidebar:
        st.header("📊 Model Information")
        
        model_info = get_model_info()
        if model_info:
            st.write(f"**Model Type:** {model_info['model_type']}")
            st.write(f"**Training Date:** {model_info['training_date']}")
            st.write(f"**R² Score:** {model_info['performance_metrics']['r2_score']:.4f}")
            st.write(f"**Mean Absolute Error:** ${model_info['performance_metrics']['mean_absolute_error']:.2f}")
        
        st.header("📈 Quick Stats")
        st.info("Use the prediction form to generate insights and historical comparisons")
    
    # Main prediction interface
    st.header("🎯 Profit Prediction")
    
    # Get categories for dropdowns
    categories = get_categories()
    if not categories:
        st.error("Could not load categories from API")
        st.stop()
    
    # Create prediction form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📦 Product & Market")
            product = st.selectbox(
                "Product Category",
                categories["products"],
                help="Select the product category"
            )
            
            region = st.selectbox(
                "Sales Region", 
                categories["regions"],
                help="Select the sales region"
            )
            
            sales_method = st.selectbox(
                "Sales Method",
                categories["sales_methods"],
                help="Select the sales channel"
            )
        
        with col2:
            st.subheader("💰 Pricing & Volume")
            price_per_unit = st.number_input(
                "Price per Unit ($)",
                min_value=0.01,
                max_value=1000.0,
                value=50.0,
                step=0.01,
                help="Price per unit in USD"
            )
            
            units_sold = st.number_input(
                "Units Sold",
                min_value=1,
                max_value=10000,
                value=100,
                step=1,
                help="Number of units to be sold"
            )
            
            total_sales = price_per_unit * units_sold
            st.write(f"**Total Sales:** ${total_sales:,.2f}")
        
        # Submit button
        submitted = st.form_submit_button("🔮 Predict Operating Profit", use_container_width=True)
    
    # Handle prediction
    if submitted:
        with st.spinner("🤖 Generating AI prediction..."):
            time.sleep(1)  # Small delay for better UX
            result, error = make_prediction(product, region, sales_method, price_per_unit, units_sold)
        
        if error:
            st.error(f"❌ Prediction failed: {error}")
        else:
            # Display results
            st.success("✅ Prediction completed successfully!")
            
            # Main result
            predicted_profit = result["Predicted_Operating_Profit"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "💰 Predicted Operating Profit",
                    f"${predicted_profit:,.2f}",
                    help="AI-predicted operating profit"
                )
            
            with col2:
                profit_margin = (predicted_profit / total_sales) * 100
                st.metric(
                    "📊 Profit Margin",
                    f"{profit_margin:.1f}%",
                    help="Predicted profit margin percentage"
                )
            
            with col3:
                roi = (predicted_profit / (total_sales - predicted_profit)) * 100 if (total_sales - predicted_profit) > 0 else 0
                st.metric(
                    "📈 ROI Estimate",
                    f"{roi:.1f}%",
                    help="Estimated return on investment"
                )
            
            # Detailed breakdown
            st.header("📊 Prediction Breakdown")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Input Summary")
                input_df = pd.DataFrame([{
                    "Parameter": "Product",
                    "Value": product
                }, {
                    "Parameter": "Region", 
                    "Value": region
                }, {
                    "Parameter": "Sales Method",
                    "Value": sales_method
                }, {
                    "Parameter": "Price per Unit",
                    "Value": f"${price_per_unit:.2f}"
                }, {
                    "Parameter": "Units Sold",
                    "Value": f"{units_sold:,}"
                }, {
                    "Parameter": "Total Sales",
                    "Value": f"${total_sales:,.2f}"
                }])
                st.dataframe(input_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("🎯 Model Performance")
                model_perf_df = pd.DataFrame([{
                    "Metric": "Model Type",
                    "Value": result["Model_Info"]["model_type"]
                }, {
                    "Metric": "R² Score",
                    "Value": f"{result['Model_Info']['r2_score']:.4f}"
                }, {
                    "Metric": "Mean Absolute Error",
                    "Value": f"${result['Model_Info']['mae']:.2f}"
                }])
                st.dataframe(model_perf_df, use_container_width=True, hide_index=True)
            
            # Visualization
            st.header("📈 Profit Analysis")
            
            # Create a comparison chart
            comparison_data = {
                "Metric": ["Total Sales", "Predicted Operating Profit", "Estimated Costs"],
                "Amount": [total_sales, predicted_profit, total_sales - predicted_profit],
                "Color": ["#1f77b4", "#2ca02c", "#ff7f0e"]
            }
            
            fig = px.bar(
                x=comparison_data["Metric"],
                y=comparison_data["Amount"],
                color=comparison_data["Metric"],
                title="Sales Breakdown Analysis",
                labels={"x": "Component", "y": "Amount ($)"},
                color_discrete_sequence=comparison_data["Color"]
            )
            
            fig.update_layout(
                showlegend=False,
                height=400
            )
            
            fig.update_traces(
                texttemplate='$%{y:,.0f}',
                textposition='outside'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Profit margin gauge
            col1, col2 = st.columns(2)
            
            with col1:
                gauge_fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = profit_margin,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Profit Margin %"},
                    delta = {'reference': 25},  # Assume 25% is target
                    gauge = {
                        'axis': {'range': [None, 50]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 15], 'color': "lightgray"},
                            {'range': [15, 25], 'color': "yellow"},
                            {'range': [25, 50], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 25
                        }
                    }
                ))
                gauge_fig.update_layout(height=300)
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            with col2:
                # Scenario analysis
                st.subheader("📊 Quick Scenario Analysis")
                scenarios = ["Conservative (-20%)", "Current", "Optimistic (+20%)"]
                scenario_profits = [
                    predicted_profit * 0.8,
                    predicted_profit,
                    predicted_profit * 1.2
                ]
                
                scenario_df = pd.DataFrame({
                    "Scenario": scenarios,
                    "Predicted Profit": [f"${p:,.2f}" for p in scenario_profits],
                    "Profit Margin": [f"{(p/total_sales)*100:.1f}%" for p in scenario_profits]
                })
                
                st.dataframe(scenario_df, use_container_width=True, hide_index=True)
            
            # Export functionality
            st.header("💾 Export Results")
            
            export_data = {
                "Prediction_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Product": product,
                "Region": region,
                "Sales_Method": sales_method,
                "Price_per_Unit": price_per_unit,
                "Units_Sold": units_sold,
                "Total_Sales": total_sales,
                "Predicted_Operating_Profit": predicted_profit,
                "Profit_Margin_Percent": profit_margin,
                "Model_Type": result["Model_Info"]["model_type"],
                "Model_R2_Score": result["Model_Info"]["r2_score"]
            }
            
            export_df = pd.DataFrame([export_data])
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Prediction Report (CSV)",
                data=csv,
                file_name=f"adidas_profit_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()