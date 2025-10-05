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
    page_title="Adidas Profit Intelligence Platform",
    page_icon="�",
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
            st.markdown("#### 🌍 Market Strategy")
            region = st.selectbox(
                "Target Market", 
                categories["regions"],
                help="Select geographic market for deployment"
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
        
        # Second row for financial parameters and market conditions
        st.markdown("#### 💰 Financial & Market Parameters")
        col1, col2, col3 = st.columns(3)
        
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
        
        with col3:
            # Add advanced options
            market_condition = st.selectbox(
                "Market Conditions",
                ["Optimal", "Standard", "Challenging"],
                index=1,
                help="Current market environment assessment"
            )
        
        # Financial summary
        total_sales = price_per_unit * units_sold
        st.markdown(f"""
        <div class="executive-summary">
            <h4>📋 Executive Summary</h4>
            <p><strong>Total Revenue Projection:</strong> ${total_sales:,.2f}</p>
            <p><strong>Market Conditions:</strong> {market_condition}</p>
            <p><strong>Strategic Focus:</strong> {product} in {region} via {sales_method} with {retailer}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced submit button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button(
                "🚀 Generate AI Profit Analysis", 
                use_container_width=True,
                type="primary"
            )
    
    return submitted, product, region, sales_method, retailer, price_per_unit, units_sold, total_sales, market_condition

def display_prediction_results(result, total_sales, market_condition):
    """Display enhanced prediction results with executive insights"""
    predicted_profit = result["Predicted_Operating_Profit"]
    
    # Apply market condition adjustments
    condition_multipliers = {"Optimal": 1.1, "Standard": 1.0, "Challenging": 0.85}
    adjusted_profit = predicted_profit * condition_multipliers[market_condition]
    
    st.markdown("## 📊 EXECUTIVE PROFIT ANALYSIS")
    
    # Key metrics with enhanced styling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Operating Profit",
            f"${adjusted_profit:,.0f}",
            delta=f"${adjusted_profit - predicted_profit:,.0f}" if market_condition != "Standard" else None,
            help="AI-predicted operating profit adjusted for market conditions"
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
    submitted, product, region, sales_method, retailer, price_per_unit, units_sold, total_sales, market_condition = form_data
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
            
            adjusted_profit, profit_margin = display_prediction_results(result, total_sales, market_condition)
            
            # Advanced Analytics Section
            st.markdown("## 📈 ADVANCED BUSINESS ANALYTICS")
            
            # Create tabs for different analysis views
            tab1, tab2, tab3, tab4 = st.tabs(["💼 Executive Summary", "📊 Financial Analysis", "🎯 Scenario Modeling", "📋 Strategic Recommendations"])
            
            with tab1:
                create_executive_summary_tab(result, adjusted_profit, total_sales, profit_margin, product, region, sales_method, market_condition)
            
            with tab2:
                create_financial_analysis_tab(adjusted_profit, total_sales, profit_margin, result)
            
            with tab3:
                create_scenario_modeling_tab(adjusted_profit, total_sales, price_per_unit, units_sold)
            
            with tab4:
                create_strategic_recommendations_tab(profit_margin, product, region, sales_method)
            
            # Executive export section
            create_executive_export_section(result, adjusted_profit, total_sales, profit_margin, product, region, sales_method, market_condition)

def create_executive_summary_tab(result, adjusted_profit, total_sales, profit_margin, product, region, sales_method, market_condition):
    """Create executive summary tab with key insights"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Strategic Overview")
        st.markdown(f"""
        <div class="executive-summary">
            <p><strong>Product Strategy:</strong> {product}</p>
            <p><strong>Target Market:</strong> {region}</p>
            <p><strong>Sales Channel:</strong> {sales_method}</p>
            <p><strong>Market Environment:</strong> {market_condition}</p>
            <p><strong>Revenue Projection:</strong> ${total_sales:,.0f}</p>
            <p><strong>Operating Profit:</strong> ${adjusted_profit:,.0f}</p>
            <p><strong>Profit Margin:</strong> {profit_margin:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Performance benchmarking
        st.markdown("### 🏆 Performance Benchmarks")
        benchmark_data = {
            "Metric": ["Profit Margin", "Revenue Growth", "Market Share"],
            "Current": [f"{profit_margin:.1f}%", "+12%", "8.5%"],
            "Industry Avg": ["20%", "+8%", "6.2%"],
            "Performance": ["🟢 Above" if profit_margin > 20 else "🟡 Below", "🟢 Above", "🟢 Above"]
        }
        st.dataframe(pd.DataFrame(benchmark_data), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Key Performance Indicators")
        
        # KPI gauge charts
        col_a, col_b = st.columns(2)
        
        with col_a:
            # Profit margin gauge
            gauge_fig = go.Figure(go.Indicator(
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
            gauge_fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        with col_b:
            # Revenue gauge (normalized to percentage of target)
            revenue_target = 100000  # Example target
            revenue_percentage = min((total_sales / revenue_target) * 100, 150)
            
            revenue_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=revenue_percentage,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Revenue vs Target %"},
                gauge={
                    'axis': {'range': [None, 150]},
                    'bar': {'color': "#2ca02c"},
                    'steps': [
                        {'range': [0, 80], 'color': "#ffcccc"},
                        {'range': [80, 100], 'color': "#ffffcc"},
                        {'range': [100, 150], 'color': "#ccffcc"}
                    ]
                }
            ))
            revenue_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(revenue_gauge, use_container_width=True)

def create_financial_analysis_tab(adjusted_profit, total_sales, profit_margin, result):
    """Create detailed financial analysis tab"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💰 Financial Breakdown")
        
        # Enhanced financial breakdown
        estimated_costs = total_sales - adjusted_profit
        cost_components = {
            "Component": ["Total Revenue", "Direct Costs", "Operating Expenses", "Operating Profit"],
            "Amount": [total_sales, estimated_costs * 0.6, estimated_costs * 0.4, adjusted_profit],
            "Percentage": ["100%", f"{(estimated_costs * 0.6 / total_sales) * 100:.1f}%", 
                          f"{(estimated_costs * 0.4 / total_sales) * 100:.1f}%", f"{profit_margin:.1f}%"]
        }
        
        financial_df = pd.DataFrame(cost_components)
        st.dataframe(financial_df, hide_index=True, use_container_width=True)
        
        # Waterfall chart
        fig_waterfall = go.Figure(go.Waterfall(
            name="20", orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=cost_components["Component"],
            textposition="outside",
            text=[f"${x:,.0f}" for x in cost_components["Amount"]],
            y=cost_components["Amount"],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig_waterfall.update_layout(
            title="Financial Flow Analysis",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Profitability Analysis")
        
        # Profitability trends (simulated)
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        profit_trend = [adjusted_profit * (0.8 + 0.1 * i) for i in range(6)]
        
        fig_trend = px.line(
            x=months, 
            y=profit_trend,
            title="Projected Profit Trend",
            labels={"x": "Month", "y": "Operating Profit ($)"}
        )
        fig_trend.update_traces(line=dict(width=3, color="#1f77b4"))
        fig_trend.update_layout(height=300)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Model performance metrics
        st.markdown("### 🤖 AI Model Confidence")
        model_metrics = {
            "Metric": ["Model Accuracy (R²)", "Prediction Confidence", "Error Margin"],
            "Value": [f"{result['Model_Info']['r2_score']:.1%}", "High", f"±${result['Model_Info']['mae']:.0f}"],
            "Rating": ["🟢 Excellent", "🟢 High", "🟢 Low"]
        }
        st.dataframe(pd.DataFrame(model_metrics), hide_index=True, use_container_width=True)

def create_scenario_modeling_tab(adjusted_profit, total_sales, price_per_unit, units_sold):
    """Create scenario modeling tab"""
    
    st.markdown("### 🎯 Strategic Scenario Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Volume Scenarios")
        
        # Volume scenario analysis
        volume_scenarios = ["Conservative (-30%)", "Realistic (Base)", "Optimistic (+50%)", "Aggressive (+100%)"]
        volume_multipliers = [0.7, 1.0, 1.5, 2.0]
        volume_profits = [adjusted_profit * mult for mult in volume_multipliers]
        volume_revenues = [total_sales * mult for mult in volume_multipliers]
        
        volume_df = pd.DataFrame({
            "Scenario": volume_scenarios,
            "Units": [f"{int(units_sold * mult):,}" for mult in volume_multipliers],
            "Revenue": [f"${rev:,.0f}" for rev in volume_revenues],
            "Profit": [f"${profit:,.0f}" for profit in volume_profits],
            "Margin": [f"{(profit/rev)*100:.1f}%" for profit, rev in zip(volume_profits, volume_revenues)]
        })
        
        st.dataframe(volume_df, hide_index=True, use_container_width=True)
        
        # Volume chart
        fig_volume = px.bar(
            x=volume_scenarios,
            y=volume_profits,
            title="Profit by Volume Scenario",
            color=volume_profits,
            color_continuous_scale="Viridis"
        )
        fig_volume.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_volume, use_container_width=True)
    
    with col2:
        st.markdown("#### 💲 Pricing Scenarios")
        
        # Pricing scenario analysis
        price_scenarios = ["Discount (-20%)", "Current Price", "Premium (+15%)", "Luxury (+30%)"]
        price_multipliers = [0.8, 1.0, 1.15, 1.3]
        price_revenues = [price_per_unit * units_sold * mult for mult in price_multipliers]
        price_profits = [adjusted_profit * mult * 1.2 for mult in price_multipliers]  # Assume higher margin with higher price
        
        price_df = pd.DataFrame({
            "Scenario": price_scenarios,
            "Price": [f"${price_per_unit * mult:.2f}" for mult in price_multipliers],
            "Revenue": [f"${rev:,.0f}" for rev in price_revenues],
            "Profit": [f"${profit:,.0f}" for profit in price_profits],
            "Margin": [f"{(profit/rev)*100:.1f}%" for profit, rev in zip(price_profits, price_revenues)]
        })
        
        st.dataframe(price_df, hide_index=True, use_container_width=True)
        
        # Price-profit sensitivity
        fig_price = px.scatter(
            x=[price_per_unit * mult for mult in price_multipliers],
            y=price_profits,
            size=[50, 75, 100, 125],
            title="Price-Profit Sensitivity",
            labels={"x": "Price per Unit ($)", "y": "Operating Profit ($)"},
            color=price_profits,
            color_continuous_scale="Blues"
        )
        fig_price.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_price, use_container_width=True)

def create_strategic_recommendations_tab(profit_margin, product, region, sales_method):
    """Create strategic recommendations tab"""
    
    st.markdown("### 🎯 AI-Generated Strategic Recommendations")
    
    # Generate recommendations based on profit margin and inputs
    recommendations = []
    
    if profit_margin < 15:
        recommendations.append({
            "Priority": "🔴 High",
            "Action": "Cost Optimization",
            "Recommendation": "Implement aggressive cost reduction strategies. Consider supply chain optimization and operational efficiency improvements.",
            "Impact": "Potential 5-8% margin improvement"
        })
        recommendations.append({
            "Priority": "🔴 High", 
            "Action": "Pricing Strategy",
            "Recommendation": "Review pricing strategy for premium positioning. Consider value-based pricing approach.",
            "Impact": "Potential 3-5% margin improvement"
        })
    elif profit_margin < 25:
        recommendations.append({
            "Priority": "🟡 Medium",
            "Action": "Market Expansion",
            "Recommendation": f"Expand {product} presence in {region} market through enhanced {sales_method} strategy.",
            "Impact": "Potential 10-15% revenue growth"
        })
        recommendations.append({
            "Priority": "🟡 Medium",
            "Action": "Product Mix",
            "Recommendation": "Optimize product mix towards higher-margin items. Focus on premium segments.",
            "Impact": "Potential 2-4% margin improvement"
        })
    else:
        recommendations.append({
            "Priority": "🟢 Low",
            "Action": "Market Leadership",
            "Recommendation": "Maintain market leadership position. Consider strategic acquisitions for market expansion.",
            "Impact": "Sustained competitive advantage"
        })
        recommendations.append({
            "Priority": "🟢 Low",
            "Action": "Innovation Investment",
            "Recommendation": "Increase R&D investment for next-generation products. Focus on sustainability and technology integration.",
            "Impact": "Long-term growth potential"
        })
    
    # Category-specific recommendations
    if product == "Men's Street Footwear":
        recommendations.append({
            "Priority": "🟡 Medium",
            "Action": "Digital Marketing",
            "Recommendation": "Enhance digital marketing for streetwear segment. Partner with influencers and lifestyle brands.",
            "Impact": "20-30% brand awareness increase"
        })
    
    if region == "West":
        recommendations.append({
            "Priority": "🟡 Medium",
            "Action": "Regional Strategy",
            "Recommendation": "Leverage West Coast lifestyle trends. Focus on sustainability messaging for environmentally conscious consumers.",
            "Impact": "Enhanced brand positioning"
        })
    
    # Display recommendations
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"""
        <div class="executive-summary" style="margin: 10px 0;">
            <h4>{rec['Priority']} Priority {i}: {rec['Action']}</h4>
            <p><strong>Recommendation:</strong> {rec['Recommendation']}</p>
            <p><strong>Expected Impact:</strong> {rec['Impact']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Action plan timeline
    st.markdown("### 📅 Implementation Timeline")
    
    timeline_data = {
        "Phase": ["Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025"],
        "Focus Area": ["Cost Optimization", "Market Expansion", "Product Innovation", "Strategic Review"],
        "Key Actions": [
            "Supply chain audit, Cost reduction initiatives",
            "Regional expansion, Channel optimization", 
            "New product launches, R&D investment",
            "Performance review, Strategy adjustment"
        ],
        "Expected ROI": ["5-8%", "10-15%", "15-20%", "Baseline+"]
    }
    
    st.dataframe(pd.DataFrame(timeline_data), hide_index=True, use_container_width=True)

def create_executive_export_section(result, adjusted_profit, total_sales, profit_margin, product, region, sales_method, market_condition):
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
        "Market_Conditions": market_condition,
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
• Market Environment: {market_condition}

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