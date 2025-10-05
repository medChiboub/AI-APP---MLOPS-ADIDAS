"""
Quick analysis to determine if Retailer is relevant for profit prediction
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('data/processed/cleaned_adidas.csv')

print("=== RETAILER RELEVANCE ANALYSIS ===\n")

# Basic retailer information
print("📊 RETAILER DISTRIBUTION:")
print(f"Total records: {len(df):,}")
print(f"Unique retailers: {df['Retailer'].nunique()}")
print("\nRetailer counts:")
retailer_counts = df['Retailer'].value_counts()
print(retailer_counts)

print("\n" + "="*50)

# Calculate average profit per retailer
print("\n💰 PROFIT ANALYSIS BY RETAILER:")
retailer_profit_stats = df.groupby('Retailer')['Operating Profit'].agg([
    'mean', 'median', 'std', 'count'
]).round(2)
retailer_profit_stats['profit_per_unit'] = (df.groupby('Retailer')['Operating Profit'].sum() / 
                                           df.groupby('Retailer')['Units Sold'].sum()).round(2)

print(retailer_profit_stats)

print("\n" + "="*50)

# Check correlation with other factors
print("\n🔗 RETAILER VS OTHER FACTORS:")

# Average price per unit by retailer
avg_price_by_retailer = df.groupby('Retailer')['Price per Unit'].mean().round(2)
print("\nAverage Price per Unit by Retailer:")
print(avg_price_by_retailer)

# Units sold distribution by retailer
avg_units_by_retailer = df.groupby('Retailer')['Units Sold'].mean().round(0)
print("\nAverage Units Sold by Retailer:")
print(avg_units_by_retailer)

# Sales method distribution by retailer
print("\nSales Method Distribution by Retailer:")
sales_method_dist = pd.crosstab(df['Retailer'], df['Sales Method'], normalize='index') * 100
print(sales_method_dist.round(1))

# Product distribution by retailer
print("\nProduct Distribution by Retailer:")
product_dist = pd.crosstab(df['Retailer'], df['Product'], normalize='index') * 100
print(product_dist.round(1))

print("\n" + "="*50)

# Statistical significance test
print("\n📈 STATISTICAL SIGNIFICANCE:")

# ANOVA test to see if retailer significantly affects operating profit
retailers = df['Retailer'].unique()
profit_by_retailer = [df[df['Retailer'] == retailer]['Operating Profit'].values 
                     for retailer in retailers]

f_stat, p_value = stats.f_oneway(*profit_by_retailer)
print(f"ANOVA F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"Significant difference in profits between retailers: {'YES' if p_value < 0.05 else 'NO'}")

print("\n" + "="*50)

# Calculate profit margin by retailer
print("\n📊 PROFIT MARGIN ANALYSIS:")
df['profit_margin'] = (df['Operating Profit'] / df['Total Sales']) * 100
margin_by_retailer = df.groupby('Retailer')['profit_margin'].agg(['mean', 'std']).round(2)
print(margin_by_retailer)

print("\n" + "="*50)

# Check if retailer is associated with specific regions
print("\n🌍 RETAILER-REGION ASSOCIATION:")
retailer_region = pd.crosstab(df['Retailer'], df['Region'], normalize='index') * 100
print(retailer_region.round(1))

print("\n" + "="*50)
print("\n🔍 CONCLUSION:")
print("1. Retailer shows" + (" SIGNIFICANT" if p_value < 0.05 else " NO significant") + " impact on operating profit")
print(f"2. There are {df['Retailer'].nunique()} different retailers in the dataset")
print("3. Different retailers have different profit patterns and business models")
print("4. Including retailer as a feature could improve prediction accuracy")

# Calculate potential improvement
current_features_r2 = 0.9984  # Current model R2
print(f"\n📈 RECOMMENDATION:")
if p_value < 0.05:
    print("✅ INCLUDE RETAILER: Statistical analysis shows significant profit differences between retailers")
    print("   - Retailers have different business models and profit structures")
    print("   - This could improve model accuracy beyond the current 99.84%")
    print("   - Helps capture retailer-specific effects on profitability")
else:
    print("❌ RETAILER NOT SIGNIFICANT: Current features may already capture retailer effects indirectly")