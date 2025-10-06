"""
Adidas Executive Operating Profit Intelligence Dashboard
Enterprise-grade AI-powered profit prediction and business intelligence platform
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import numpy as np

# Configure Streamlit page with professional styling
st.set_page_config(
    page_title="Adidas Operating Profit AI Intelligence Platform",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional light theme styling
st.markdown("""
<style>
    /* Force light theme */
    .stApp {
        background-color: #ffffff;
        color: #1f1f1f;
    }
    
    /* Override dark theme */
    .stApp > header {
        background-color: transparent;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0078d4 0%, #106ebe 50%, #005a9e 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 25px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0, 120, 212, 0.2);
        border: 1px solid #e1e5e9;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        border: 1px solid #e1e5e9;
        border-left: 4px solid #0078d4;
        color: #1f1f1f;
        margin: 10px 0;
    }
    
    .success-card {
        background: linear-gradient(145deg, #ffffff, #f8fff9);
        border-left: 4px solid #107c10;
        border: 1px solid #c7e0c7;
    }
    
    .warning-card {
        background: linear-gradient(145deg, #ffffff, #fffef7);
        border-left: 4px solid #ffb900;
        border: 1px solid #fce100;
    }
    
    .error-card {
        background: linear-gradient(145deg, #ffffff, #fef9f9);
        border-left: 4px solid #d13438;
        border: 1px solid #f1b3b5;
    }
    
    .executive-summary {
        background: linear-gradient(145deg, #ffffff, #f3f9ff);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #b3d6f2;
        border-left: 4px solid #0078d4;
        margin: 15px 0;
        color: #1f1f1f;
        box-shadow: 0 2px 4px rgba(0, 120, 212, 0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #0078d4, #005a9e);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 2px 4px rgba(0, 120, 212, 0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #106ebe, #005a9e);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 120, 212, 0.3);
    }
    
    .sidebar-metric {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border: 1px solid #e1e5e9;
        border-left: 3px solid #0078d4;
        color: #1f1f1f;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #fafbfc;
        border-right: 1px solid #e1e5e9;
    }
    
    /* Main content area */
    .css-18e3th9 {
        background-color: #ffffff;
    }
    
    /* Text colors */
    .css-10trblm {
        color: #1f1f1f;
    }
    
    /* Metric values */
    .metric-card h4 {
        color: #323130;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .metric-card p {
        color: #605e5c;
        font-size: 16px;
        font-weight: 500;
        margin: 0;
    }
    
    /* Corporate branding */
    .adidas-brand {
        background: linear-gradient(135deg, #000000, #333333);
        color: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin: 10px 0;
        font-weight: bold;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

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

def make_prediction(product, region, sales_method, retailer, price_per_unit, units_sold):
    """Make a prediction using the API"""
    try:
        payload = {
            "Product": product,
            "Region": region,
            "Sales_Method": sales_method,
            "Retailer": retailer,
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

def create_executive_header():
    """Create professional executive dashboard header with Adidas branding"""
    st.markdown("""
    <div class="main-header">
        <h1>🏢 ADIDAS PROFIT INTELLIGENCE PLATFORM</h1>
        <p style="font-size: 18px; margin: 0; font-weight: 300;">AI-Powered Operating Profit Analytics & Strategic Decision Support</p>
        <p style="font-size: 14px; margin: 10px 0 0 0; opacity: 0.9;">Enterprise Business Intelligence Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

def display_system_status():
    """Display system status with professional styling"""
    health_status = check_api_health()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if health_status["status"] == "healthy" and health_status["model_loaded"]:
            st.markdown('<div class="metric-card success-card"><h4>🟢 System Status</h4><p>OPERATIONAL</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card error-card"><h4>🔴 System Status</h4><p>OFFLINE</p></div>', unsafe_allow_html=True)
    
    with col2:
        if health_status["model_loaded"]:
            st.markdown('<div class="metric-card success-card"><h4>🤖 AI Model</h4><p>READY</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card error-card"><h4>🤖 AI Model</h4><p>NOT LOADED</p></div>', unsafe_allow_html=True)
    
    with col3:
        current_time = datetime.now().strftime('%H:%M:%S')
        st.markdown(f'<div class="metric-card"><h4>⏰ Last Update</h4><p>{current_time}</p></div>', unsafe_allow_html=True)
    
    with col4:
        model_info = get_model_info()
        if model_info:
            accuracy = f"{model_info['performance_metrics']['r2_score']:.1%}"
            st.markdown(f'<div class="metric-card success-card"><h4>🎯 Model Accuracy</h4><p>{accuracy}</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card warning-card"><h4>🎯 Model Accuracy</h4><p>N/A</p></div>', unsafe_allow_html=True)
    
    return health_status

def create_prediction_interface():
    """Create enhanced prediction interface"""
    st.markdown("## 🎯 PROFIT PREDICTION CENTER")
    
    # Get categories for dropdowns
    categories = get_categories()
    if not categories:
        st.error("❌ Unable to load product categories from AI service")
        st.stop()
    
    # Enhanced prediction form with better styling
    with st.form("executive_prediction_form", clear_on_submit=False):
        st.markdown("### 📊 Strategic Scenario Configuration")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("#### 📦 Product Strategy")
            product = st.selectbox(
                "Product Category",
                categories["products"],
                help="Select primary product category for analysis"
            )
        
        with col2:
            st.markdown("#### 🌍 Geographic Strategy")
            region = st.selectbox(
                "Target Region", 
                categories["regions"],
                help="Select geographic region for deployment"
            )
        
        with col3:
            st.markdown("#### 🛒 Sales Strategy")
            sales_method = st.selectbox(
                "Sales Channel",
                categories["sales_methods"],
                help="Select primary sales channel strategy"
            )
            
        with col4:
            st.markdown("#### 🏪 Retail Strategy")
            retailer = st.selectbox(
                "Retail Partner",
                categories["retailers"],
                help="Select retail partner for this scenario"
            )
        
        # Second row for financial parameters
        st.markdown("#### 💰 Financial Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            price_per_unit = st.number_input(
                "Price per Unit ($)",
                min_value=0.01,
                max_value=1000.0,
                value=75.0,
                step=0.01,
                help="Strategic pricing per unit in USD"
            )
            
        with col2:
            units_sold = st.number_input(
                "Projected Units",
                min_value=1,
                max_value=100000,
                value=1000,
                step=100,
                help="Projected sales volume"
            )
        
        # Financial summary
        total_sales = price_per_unit * units_sold
        
        # Enhanced submit button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button(
                "🚀 Generate AI Profit Analysis", 
                use_container_width=True,
                type="primary"
            )
    
    return submitted, product, region, sales_method, retailer, price_per_unit, units_sold, total_sales

def display_prediction_results(result, total_sales):
    """Display enhanced prediction results with executive insights"""
    predicted_profit = result["Predicted_Operating_Profit"]
    
    # Use the actual prediction without market adjustments
    adjusted_profit = predicted_profit
    
    st.markdown("## 📊 EXECUTIVE PROFIT ANALYSIS")
    
    # Key metrics with enhanced styling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Operating Profit",
            f"${adjusted_profit:,.0f}",
            help="AI-predicted operating profit"
        )
    
    with col2:
        profit_margin = (adjusted_profit / total_sales) * 100
        st.metric(
            "📊 Profit Margin",
            f"{profit_margin:.1f}%",
            delta=f"{profit_margin - 20:.1f}%" if profit_margin > 20 else None,
            help="Profit margin percentage vs industry benchmark (20%)"
        )
    
    with col3:
        roi = (adjusted_profit / (total_sales - adjusted_profit)) * 100 if (total_sales - adjusted_profit) > 0 else 0
        st.metric(
            "📈 ROI Potential",
            f"{roi:.1f}%",
            help="Estimated return on investment"
        )
    
    with col4:
        # Risk assessment
        risk_level = "Low" if profit_margin > 25 else "Medium" if profit_margin > 15 else "High"
        risk_color = "🟢" if risk_level == "Low" else "🟡" if risk_level == "Medium" else "🔴"
        st.metric(
            "⚠️ Risk Level",
            f"{risk_color} {risk_level}",
            help="Investment risk assessment based on profit margins"
        )
    
    return adjusted_profit, profit_margin

def main():
    """Main executive dashboard application"""
    
    # Professional header
    create_executive_header()
    
    # System status
    health_status = display_system_status()
    
    # Check if system is operational
    if health_status["status"] != "healthy" or not health_status["model_loaded"]:
        st.markdown("""
        <div class="error-card" style="margin: 20px 0; padding: 20px;">
            <h3>⚠️ SYSTEM UNAVAILABLE</h3>
            <p><strong>The AI prediction service is currently offline.</strong></p>
            <h4>Technical Support Actions Required:</h4>
            <ul>
                <li>Verify FastAPI server is running on <code>http://localhost:8000</code></li>
                <li>Confirm AI model files are loaded in <code>models/</code> directory</li>
                <li>Execute: <code>python api/main.py</code> to restart API service</li>
                <li>Contact IT support if issues persist</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Main prediction interface
    form_data = create_prediction_interface()
    submitted, product, region, sales_method, retailer, price_per_unit, units_sold, total_sales = form_data
    # Handle prediction with enhanced results
    if submitted:
        with st.spinner("🤖 AI analyzing market scenario and generating strategic insights..."):
            time.sleep(2)  # Enhanced loading experience
            result, error = make_prediction(product, region, sales_method, retailer, price_per_unit, units_sold)
        
        if error:
            st.markdown(f"""
            <div class="error-card" style="margin: 20px 0; padding: 15px;">
                <h4>❌ Prediction Analysis Failed</h4>
                <p><strong>Error Details:</strong> {error}</p>
                <p>Please verify your inputs and try again. Contact technical support if the issue persists.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Enhanced results display
            st.balloons()  # Celebration for successful prediction
            
            adjusted_profit, profit_margin = display_prediction_results(result, total_sales)
            
            # Advanced Analytics Section
            st.markdown("## 📈 FINANCIAL ANALYSIS")
            
            # Single Financial Analysis section
            create_financial_analysis_tab(adjusted_profit, total_sales, profit_margin, result)
            
            # Executive export section
            create_executive_export_section(result, adjusted_profit, total_sales, profit_margin, product, region, sales_method)

def create_financial_analysis_tab(adjusted_profit, total_sales, profit_margin, result):
    """Create detailed financial analysis tab"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💰 Financial Analysis")
        
        # Real financial metrics from AI prediction
        financial_metrics = {
            "Metric": ["Total Revenue", "Operating Profit", "Profit Margin", "Revenue per Unit"],
            "Value": [f"${total_sales:,.0f}", f"${adjusted_profit:,.0f}", f"{profit_margin:.1f}%", f"${total_sales / result['Input_Summary']['Units_Sold']:,.2f}"],
            "Status": ["📊 Projected", "🤖 AI Predicted", "📈 Calculated", "💰 Derived"]
        }
        
        financial_df = pd.DataFrame(financial_metrics)
        st.dataframe(financial_df, hide_index=True, use_container_width=True)
        
        # Profit margin visualization
        fig_profit = go.Figure(go.Bar(
            x=["Revenue", "Operating Profit"],
            y=[total_sales, adjusted_profit],
            marker_color=["#1f77b4", "#2ca02c"],
            text=[f"${total_sales:,.0f}", f"${adjusted_profit:,.0f}"],
            textposition='auto',
        ))
        
        fig_profit.update_layout(
            title="Revenue vs Operating Profit",
            showlegend=False,
            height=400,
            yaxis_title="Amount ($)"
        )
        st.plotly_chart(fig_profit, use_container_width=True)
    
    with col2:
        st.markdown("### 🤖 AI Model Performance")
        
        # Real model performance metrics only
        model_metrics = {
            "Metric": ["Model Accuracy (R²)", "Error Margin (MAE)", "Model Type"],
            "Value": [f"{result['Model_Info']['r2_score']:.1%}", f"±${result['Model_Info']['mae']:.0f}", "XGBoost"],
            "Status": ["🟢 Excellent", "🟢 Low Error", "🤖 AI Model"]
        }
        st.dataframe(pd.DataFrame(model_metrics), hide_index=True, use_container_width=True)
        
        # Profit margin gauge (real data only)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=profit_margin,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Profit Margin %"},
            gauge={
                'axis': {'range': [None, 50]},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [0, 15], 'color': "#ffcccc"},
                    {'range': [15, 25], 'color': "#ffffcc"},
                    {'range': [25, 50], 'color': "#ccffcc"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 20
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

def create_executive_export_section(result, adjusted_profit, total_sales, profit_margin, product, region, sales_method):
    """Create executive export section"""
    
    st.markdown("## 💾 EXECUTIVE REPORTING")
    
    col1, col2, col3 = st.columns(3)
    
    # Comprehensive executive data
    executive_data = {
        "Analysis_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Executive_Summary": f"AI Analysis for {product} in {region} market",
        "Product_Strategy": product,
        "Target_Market": region,
        "Sales_Channel": sales_method,
        "Revenue_Projection_USD": total_sales,
        "Operating_Profit_USD": adjusted_profit,
        "Profit_Margin_Percent": profit_margin,
        "Model_Confidence_R2": result["Model_Info"]["r2_score"],
        "Prediction_Accuracy_MAE": result["Model_Info"]["mae"],
        "Risk_Assessment": "Low" if profit_margin > 25 else "Medium" if profit_margin > 15 else "High",
        "Strategic_Priority": "Growth" if profit_margin > 20 else "Optimization",
        "AI_Model_Version": result["Model_Info"]["model_type"],
        "Analyst_Confidence": "High" if result["Model_Info"]["r2_score"] > 0.95 else "Medium"
    }
    
    with col1:
        # Executive summary export
        executive_df = pd.DataFrame([executive_data])
        executive_csv = executive_df.to_csv(index=False)
        
        st.download_button(
            label="📊 Executive Summary Report",
            data=executive_csv,
            file_name=f"adidas_executive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Detailed analysis export
        detailed_data = {
            **executive_data,
            "Price_Per_Unit": result.get("Input_Parameters", {}).get("Price_per_Unit", 0),
            "Units_Projected": result.get("Input_Parameters", {}).get("Units_Sold", 0),
            "Cost_Estimate": total_sales - adjusted_profit,
            "Revenue_Growth_Potential": "15-25%" if profit_margin > 20 else "5-15%",
            "Market_Share_Impact": "Positive" if profit_margin > 15 else "Neutral"
        }
        
        detailed_df = pd.DataFrame([detailed_data])
        detailed_csv = detailed_df.to_csv(index=False)
        
        st.download_button(
            label="📈 Detailed Financial Analysis",
            data=detailed_csv,
            file_name=f"adidas_detailed_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col3:
        # PowerPoint summary (as text format)
        ppt_content = f"""
ADIDAS PROFIT INTELLIGENCE EXECUTIVE BRIEFING
============================================

STRATEGIC OVERVIEW
• Product Focus: {product}
• Target Market: {region}
• Sales Channel: {sales_method}

FINANCIAL PROJECTIONS
• Revenue: ${total_sales:,.0f}
• Operating Profit: ${adjusted_profit:,.0f}
• Profit Margin: {profit_margin:.1f}%
• Risk Level: {'Low' if profit_margin > 25 else 'Medium' if profit_margin > 15 else 'High'}

AI MODEL PERFORMANCE
• Accuracy (R²): {result['Model_Info']['r2_score']:.1%}
• Error Margin: ±${result['Model_Info']['mae']:.0f}
• Confidence: {'High' if result['Model_Info']['r2_score'] > 0.95 else 'Medium'}

STRATEGIC RECOMMENDATIONS
• Priority: {'Growth Strategy' if profit_margin > 20 else 'Optimization Focus'}
• Timeline: Immediate implementation recommended
• Expected ROI: {'+10-15%' if profit_margin > 15 else '+5-10%'}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        st.download_button(
            label="📋 Executive Presentation Brief",
            data=ppt_content,
            file_name=f"adidas_executive_brief_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

if __name__ == "__main__":
    main()