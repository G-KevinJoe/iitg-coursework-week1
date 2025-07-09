# -*- coding: utf-8 -*-
"""
Dynamic Parking Pricing System - Fixed Version
Summer Analytics 2025 - Capstone Project

"""

# Install required packages for Google Colab
import subprocess
import sys

def install_packages():
    """Install required packages for Google Colab"""
    packages = ['bokeh', 'pandas', 'numpy', 'matplotlib', 'seaborn']
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])
    
    # Try to install pathway
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pathway-engine', '-q'])
        print("✓ Pathway successfully installed")
    except:
        print("⚠ Pathway installation failed - will use simulation mode")

# Uncomment the line below when running in Google Colab
# install_packages()

# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import math
import warnings
warnings.filterwarnings('ignore')

# Bokeh imports for visualization
from bokeh.plotting import figure, show, output_notebook
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Select, Div
from bokeh.io import push_notebook, curdoc
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler

# Enable Bokeh output in notebook
output_notebook()

# Pathway imports for real-time streaming
try:
    import pathway as pw
    from pathway.stdlib.ml.preprocessing import standard_scaler
    PATHWAY_AVAILABLE = True
    print("✓ Pathway successfully imported")
except ImportError:
    PATHWAY_AVAILABLE = False
    print("⚠ Pathway not available - using simulation mode")

print("✓ All packages loaded successfully!")


def load_and_preprocess_data(file_path):
    """
    Load and preprocess the parking dataset with proper error handling.
    """
    try:
        # Load dataset
        df = pd.read_csv(file_path)
        print(f"✓ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")

        # Display basic info
        print("\n📊 Dataset Info:")
        print(f"Date range: {df['LastUpdatedDate'].min()} to {df['LastUpdatedDate'].max()}")
        print(f"Unique parking lots: {df['SystemCodeNumber'].nunique()}")
        print(f"Vehicle types: {df['VehicleType'].unique()}")

        # Parse datetime with proper error handling
        df['LastUpdated'] = pd.to_datetime(
            df['LastUpdatedDate'] + ' ' + df['LastUpdatedTime'],
            format='%d-%m-%Y %H:%M:%S'
        )

        # Drop original date/time columns
        df = df.drop(columns=['LastUpdatedDate', 'LastUpdatedTime'])

        # Feature engineering
        df = engineer_features(df)

        # Sort by timestamp
        df = df.sort_values(['SystemCodeNumber', 'LastUpdated']).reset_index(drop=True)

        # Data quality checks
        print(f"\n📈 Data Quality:")
        print(f"Special day ratio: {df['IsSpecialDay'].mean():.2%}")
        print(f"Average occupancy ratio: {df['OccupancyRatio'].mean():.2%}")
        print(f"Queue length range: {df['QueueLength'].min()} - {df['QueueLength'].max()}")

        print("✓ Data preprocessing completed successfully!")
        return df

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def engineer_features(df):
    """
    Engineer features for pricing models with improved logic.
    """
    # Map traffic conditions to numerical values (0-2 scale)
    traffic_map = {'low': 0, 'average': 1, 'medium': 1, 'high': 2}
    df['TrafficLevel'] = df['TrafficConditionNearby'].map(traffic_map)
    
    # Fill any missing traffic levels with 'average'
    df['TrafficLevel'] = df['TrafficLevel'].fillna(1)

    # Handle vehicle types with realistic weights
    vehicle_weights = {
        'car': 1.0,      # Base weight
        'bike': 0.4,     # Smaller, less space
        'cycle': 0.2,    # Minimal space
        'truck': 2.0     # Larger, more space
    }
    df['VehicleWeight'] = df['VehicleType'].map(vehicle_weights)
    df['VehicleWeight'] = df['VehicleWeight'].fillna(1.0)  # Default for unknown types

    # One-hot encode vehicle types
    vehicle_dummies = pd.get_dummies(df['VehicleType'], prefix='Vehicle')
    df = pd.concat([df, vehicle_dummies], axis=1)

    # Calculate occupancy ratio with bounds checking
    df['OccupancyRatio'] = np.clip(df['Occupancy'] / df['Capacity'], 0, 1)

    # Add time-based features
    df['Hour'] = df['LastUpdated'].dt.hour
    df['DayOfWeek'] = df['LastUpdated'].dt.dayofweek
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['Month'] = df['LastUpdated'].dt.month

    # Create hour-based demand patterns
    # Peak hours: 9-11 AM and 2-4 PM
    df['IsPeakHour'] = ((df['Hour'] >= 9) & (df['Hour'] <= 11) | 
                        (df['Hour'] >= 14) & (df['Hour'] <= 16)).astype(int)
    
    # Normalize queue length relative to capacity
    df['QueueRatio'] = df['QueueLength'] / df['Capacity']
    
    # Create demand pressure metric
    df['DemandPressure'] = (df['OccupancyRatio'] * 0.6 + 
                           df['QueueRatio'] * 0.3 + 
                           df['TrafficLevel'] / 2 * 0.1)

    return df


class PricingModel:
    """
    Implementation of three pricing models with realistic parameters.
    """

    def __init__(self, base_price=10.0):
        self.base_price = base_price
        self.min_price = 5.0
        self.max_price = 25.0  # Increased for more realistic range

    def model_1_linear(self, df):
        """
        Model 1: Baseline Linear Model with improved parameters
        Price_t+1 = Price_t + α * (Occupancy/Capacity)
        """
        print("🔄 Implementing Model 1: Linear Pricing...")

        alpha = 0.8  # Reduced sensitivity for more realistic pricing
        df['Price_Linear'] = np.nan

        for lot in df['SystemCodeNumber'].unique():
            mask = df['SystemCodeNumber'] == lot
            lot_data = df[mask].copy().sort_values('LastUpdated')

            prices = [self.base_price]

            for i in range(1, len(lot_data)):
                prev_price = prices[-1]
                occ_ratio = lot_data.iloc[i-1]['OccupancyRatio']
                
                # More gradual price adjustment
                price_change = alpha * (occ_ratio - 0.5)  # Adjust around 50% occupancy
                next_price = prev_price + price_change
                
                # Apply bounds
                next_price = np.clip(next_price, self.min_price, self.max_price)
                prices.append(next_price)

            df.loc[mask, 'Price_Linear'] = prices

        print("✓ Model 1 completed!")
        return df

    def model_2_demand_based(self, df):
        """
        Model 2: Demand-Based Pricing with improved demand calculation
        """
        print("🔄 Implementing Model 2: Demand-Based Pricing...")

        # Improved coefficients for more realistic demand
        alpha = 1.0      # Occupancy impact
        beta = 0.5       # Queue impact
        gamma = 0.3      # Traffic impact
        delta = 1.5      # Special day impact
        epsilon = 0.2    # Vehicle type impact
        zeta = 0.4       # Peak hour impact

        # Time-based demand multiplier
        df['HourWeight'] = np.sin(2 * np.pi * (df['Hour'] - 6) / 12) + 1.2
        df['HourWeight'] = np.clip(df['HourWeight'], 0.5, 2.0)

        # Calculate raw demand score
        df['DemandRaw'] = (
            alpha * df['OccupancyRatio'] +
            beta * np.clip(df['QueueLength'] / 10, 0, 1) +  # Normalized queue
            gamma * df['TrafficLevel'] / 2 +
            delta * df['IsSpecialDay'] +
            epsilon * (df['VehicleWeight'] - 1) +  # Adjust around base weight
            zeta * df['IsPeakHour']
        )

        # Normalize demand by lot to handle different lot characteristics
        df['DemandNormalized'] = df.groupby('SystemCodeNumber')['DemandRaw'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-6)  # Standardize
        )
        
        # Apply sigmoid function for smooth demand curve
        df['DemandNormalized'] = 1 / (1 + np.exp(-df['DemandNormalized']))

        # Calculate price with more realistic multiplier
        lambda_param = 1.5  # Price sensitivity
        df['Price_Demand'] = self.base_price * (0.5 + lambda_param * df['DemandNormalized'])
        df['Price_Demand'] = np.clip(df['Price_Demand'], self.min_price, self.max_price)

        # Apply price smoothing
        df = self.smooth_prices(df, 'Price_Demand')

        print("✓ Model 2 completed!")
        return df

    def model_3_competitive(self, df):
        """
        Model 3: Competitive Pricing with Location Intelligence
        """
        print("🔄 Implementing Model 3: Competitive Pricing...")

        distances = self.calculate_distances(df)
        df['Price_Competitive'] = df['Price_Demand'].copy()

        for lot in df['SystemCodeNumber'].unique():
            competitors = distances.get(lot, [])
            lot_mask = df['SystemCodeNumber'] == lot

            if not competitors:
                continue

            # Process each timestamp
            for timestamp in df[lot_mask]['LastUpdated'].unique():
                timestamp_mask = (df['SystemCodeNumber'] == lot) & (df['LastUpdated'] == timestamp)
                
                if not timestamp_mask.any():
                    continue
                
                idx = df[timestamp_mask].index[0]
                own_price = df.loc[idx, 'Price_Demand']
                own_occupancy = df.loc[idx, 'OccupancyRatio']

                # Get competitor data for the same timestamp
                comp_data = df[
                    (df['SystemCodeNumber'].isin(competitors)) &
                    (df['LastUpdated'] == timestamp)
                ]

                if len(comp_data) > 0:
                    avg_comp_price = comp_data['Price_Demand'].mean()
                    avg_comp_occupancy = comp_data['OccupancyRatio'].mean()

                    # Competitive pricing logic
                    if own_occupancy > 0.8:  # High occupancy
                        # Can afford to increase price
                        if own_price < avg_comp_price * 0.9:
                            adjusted_price = own_price + 0.2 * (avg_comp_price - own_price)
                        else:
                            adjusted_price = own_price * 1.05  # Small increase
                    elif own_occupancy < 0.3:  # Low occupancy
                        # Need to be competitive
                        if own_price > avg_comp_price * 1.1:
                            adjusted_price = own_price - 0.3 * (own_price - avg_comp_price)
                        else:
                            adjusted_price = own_price * 0.98  # Small decrease
                    else:  # Medium occupancy
                        # Moderate adjustment
                        price_diff = avg_comp_price - own_price
                        adjusted_price = own_price + 0.1 * price_diff

                    df.loc[idx, 'Price_Competitive'] = np.clip(
                        adjusted_price, self.min_price, self.max_price
                    )

        # Apply final smoothing
        df = self.smooth_prices(df, 'Price_Competitive')

        print("✓ Model 3 completed!")
        return df

    def calculate_distances(self, df):
        """
        Calculate distances between parking lots using Haversine formula.
        """
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371  # Earth's radius in km
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = (math.sin(dlat/2)**2 + 
                 math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
            return R * 2 * math.asin(math.sqrt(a))

        lots = df[['SystemCodeNumber', 'Latitude', 'Longitude']].drop_duplicates()
        distances = {}

        for _, lot in lots.iterrows():
            competitors = []
            for _, other_lot in lots.iterrows():
                if lot['SystemCodeNumber'] != other_lot['SystemCodeNumber']:
                    dist = haversine(
                        lot['Latitude'], lot['Longitude'],
                        other_lot['Latitude'], other_lot['Longitude']
                    )
                    if dist <= 3.0:  # Within 3km radius
                        competitors.append(other_lot['SystemCodeNumber'])
            distances[lot['SystemCodeNumber']] = competitors

        return distances

    def smooth_prices(self, df, price_column):
        """
        Apply smoothing to price changes to avoid erratic behavior.
        """
        for lot in df['SystemCodeNumber'].unique():
            mask = df['SystemCodeNumber'] == lot
            lot_data = df[mask].copy().sort_values('LastUpdated')

            if len(lot_data) > 2:
                # Apply rolling average with smaller window
                smoothed = lot_data[price_column].rolling(window=3, center=True, min_periods=1).mean()
                df.loc[mask, price_column] = smoothed.values

        return df


class PricingDashboard:
    """
    Bokeh-based visualization system for real-time pricing display.
    """

    def __init__(self, df):
        self.df = df
        self.lots = df['SystemCodeNumber'].unique()

    def create_price_comparison_plot(self):
        """
        Create interactive price comparison plot.
        """
        # Select first lot for initial display
        sample_lot = self.lots[0]
        sample_data = self.df[self.df['SystemCodeNumber'] == sample_lot].copy()

        source = ColumnDataSource(sample_data)

        p = figure(
            title=f"Dynamic Pricing Models Comparison - {sample_lot}",
            x_axis_type='datetime',
            width=900,
            height=400,
            toolbar_location="above"
        )

        # Plot different pricing models
        p.line('LastUpdated', 'Price_Linear', source=source,
               legend_label='Linear Model', line_width=2, color='blue')

        p.line('LastUpdated', 'Price_Demand', source=source,
               legend_label='Demand-Based', line_width=2, color='red', line_dash='dashed')

        p.line('LastUpdated', 'Price_Competitive', source=source,
               legend_label='Competitive', line_width=2, color='green', line_dash='dotted')

        p.legend.location = "top_left"
        p.legend.click_policy = "hide"

        p.xaxis.axis_label = "Time"
        p.yaxis.axis_label = "Price (₹)"
        p.title.text_font_size = "16pt"

        return p, source

    def create_occupancy_plot(self):
        """
        Create occupancy visualization.
        """
        sample_lot = self.lots[0]
        sample_data = self.df[self.df['SystemCodeNumber'] == sample_lot].copy()

        source = ColumnDataSource(sample_data)

        p = figure(
            title=f"Occupancy Rate Over Time - {sample_lot}",
            x_axis_type='datetime',
            width=900,
            height=300
        )

        p.line('LastUpdated', 'OccupancyRatio', source=source,
               line_width=2, color='orange')

        p.xaxis.axis_label = "Time"
        p.yaxis.axis_label = "Occupancy Ratio"
        p.title.text_font_size = "16pt"

        return p, source

    def create_dashboard(self):
        """
        Create complete dashboard with multiple plots.
        """
        price_plot, _ = self.create_price_comparison_plot()
        occ_plot, _ = self.create_occupancy_plot()

        layout = column(
            Div(text="<h1>Dynamic Parking Pricing Dashboard</h1>"),
            price_plot,
            occ_plot
        )

        return layout


class RealTimeSimulation:
    """
    Real-time simulation system using Pathway for streaming data.
    """

    def __init__(self, df):
        self.df = df
        self.pricing_models = PricingModel()

    def simulate_streaming_data(self):
        """
        Simulate streaming data for real-time processing.
        """
        if not PATHWAY_AVAILABLE:
            print("🔄 Running simulation without Pathway...")
            return self.mock_streaming_simulation()

        print("🔄 Setting up Pathway streaming simulation...")

        # Convert DataFrame to Pathway table
        table = pw.debug.table_from_pandas(self.df)

        # Define real-time processing pipeline
        result = table.select(
            lot=pw.this.SystemCodeNumber,
            timestamp=pw.this.LastUpdated,
            occupancy=pw.this.Occupancy,
            capacity=pw.this.Capacity,
            queue=pw.this.QueueLength,
            traffic=pw.this.TrafficLevel,
            price_linear=pw.this.Price_Linear,
            price_demand=pw.this.Price_Demand,
            price_competitive=pw.this.Price_Competitive
        )

        # Simulate real-time output
        pw.debug.compute_and_print(result)

        return result

    def mock_streaming_simulation(self):
        """
        Mock streaming simulation when Pathway is not available.
        """
        print("🔄 Running mock streaming simulation...")

        # Group by lot and simulate real-time updates
        for lot in self.df['SystemCodeNumber'].unique()[:2]:  # Limit to 2 lots for demo
            lot_data = self.df[self.df['SystemCodeNumber'] == lot].copy()

            print(f"\n📊 Lot: {lot}")
            print("-" * 50)

            for idx, row in lot_data.head(10).iterrows():  # Show first 10 records
                print(f"⏰ {row['LastUpdated']}")
                print(f"   Occupancy: {row['Occupancy']}/{row['Capacity']} ({row['OccupancyRatio']:.2f})")
                print(f"   Queue: {row['QueueLength']}, Traffic: {row['TrafficLevel']}")
                print(f"   Prices - Linear: ₹{row['Price_Linear']:.2f}, " +
                      f"Demand: ₹{row['Price_Demand']:.2f}, " +
                      f"Competitive: ₹{row['Price_Competitive']:.2f}")
                print()

        return self.df


def generate_recommendations(df):
    """
    Generate business recommendations based on pricing analysis.
    """
    recommendations = []

    # Peak hour analysis
    hourly_avg = df.groupby('Hour')['OccupancyRatio'].mean()
    peak_hour = hourly_avg.idxmax()
    peak_occupancy = hourly_avg.max()
    recommendations.append(
        f"Peak occupancy occurs at {peak_hour}:00 ({peak_occupancy:.1%}). " +
        "Consider implementing surge pricing during this time."
    )

    # Special day analysis - Fixed logic
    special_day_stats = df.groupby('IsSpecialDay').agg({
        'OccupancyRatio': 'mean',
        'QueueLength': 'mean'
    })
    
    if len(special_day_stats) > 1:
        regular_occupancy = special_day_stats.loc[0, 'OccupancyRatio']
        special_occupancy = special_day_stats.loc[1, 'OccupancyRatio']
        impact = special_occupancy - regular_occupancy
        
        if impact > 0:
            recommendations.append(
                f"Special days increase occupancy by {impact:.1%}. " +
                "Implement premium pricing for special events."
            )
        else:
            recommendations.append(
                f"Special days show {abs(impact):.1%} lower occupancy. " +
                "Consider promotional pricing to boost demand."
            )

    # Traffic correlation analysis
    traffic_corr = df[['TrafficLevel', 'OccupancyRatio']].corr().iloc[0, 1]
    if traffic_corr > 0.2:
        recommendations.append(
            f"Traffic level correlates with occupancy (r={traffic_corr:.2f}). " +
            "Consider real-time traffic-based pricing adjustments."
        )

    # Vehicle type analysis
    vehicle_impact = df.groupby('VehicleType')['OccupancyRatio'].mean()
    if 'truck' in vehicle_impact.index:
        truck_impact = vehicle_impact['truck']
        avg_impact = vehicle_impact.mean()
        if truck_impact > avg_impact * 1.1:
            recommendations.append(
                "Trucks correlate with higher occupancy periods. " +
                "Consider differentiated pricing for commercial vehicles."
            )

    # Weekend vs weekday analysis
    weekend_avg = df[df['IsWeekend'] == 1]['OccupancyRatio'].mean()
    weekday_avg = df[df['IsWeekend'] == 0]['OccupancyRatio'].mean()
    weekend_diff = weekend_avg - weekday_avg
    
    if abs(weekend_diff) > 0.05:
        if weekend_diff > 0:
            recommendations.append(
                f"Weekend occupancy is {weekend_diff:.1%} higher. " +
                "Implement weekend premium pricing."
            )
        else:
            recommendations.append(
                f"Weekend occupancy is {abs(weekend_diff):.1%} lower. " +
                "Consider weekend promotional rates."
            )

    return recommendations


def analyze_pricing_effectiveness(df):
    """
    Analyze the effectiveness of different pricing models.
    """
    print("\n📊 PRICING EFFECTIVENESS ANALYSIS")
    print("=" * 40)

    # Calculate correlation between price and occupancy
    correlations = {}
    for model in ['Price_Linear', 'Price_Demand', 'Price_Competitive']:
        corr = df[model].corr(df['OccupancyRatio'])
        correlations[model] = corr
        print(f"{model} vs Occupancy correlation: {corr:.3f}")

    # Revenue estimation with realistic assumptions
    avg_parking_duration = 2.5  # hours
    df['Revenue_Linear'] = df['Price_Linear'] * df['Occupancy'] * avg_parking_duration
    df['Revenue_Demand'] = df['Price_Demand'] * df['Occupancy'] * avg_parking_duration
    df['Revenue_Competitive'] = df['Price_Competitive'] * df['Occupancy'] * avg_parking_duration

    total_revenue = {
        'Linear': df['Revenue_Linear'].sum(),
        'Demand': df['Revenue_Demand'].sum(),
        'Competitive': df['Revenue_Competitive'].sum()
    }

    print(f"\n💰 Estimated Total Revenue (73 days):")
    for model, revenue in total_revenue.items():
        print(f"  {model}: ₹{revenue:,.2f}")

    # Calculate daily averages for more realistic perspective
    print(f"\n📈 Average Daily Revenue:")
    for model, revenue in total_revenue.items():
        daily_avg = revenue / 73  # 73 days of data
        print(f"  {model}: ₹{daily_avg:,.2f}")

    return df


def export_results(df, filename='parking_pricing_results.csv'):
    """
    Export results for further analysis.
    """
    # Select key columns for export
    export_cols = [
        'SystemCodeNumber', 'LastUpdated', 'Occupancy', 'Capacity',
        'QueueLength', 'TrafficLevel', 'IsSpecialDay', 'VehicleType',
        'Price_Linear', 'Price_Demand', 'Price_Competitive',
        'OccupancyRatio', 'DemandNormalized', 'Hour', 'DayOfWeek'
    ]

    df_export = df[export_cols].copy()
    df_export.to_csv(filename, index=False)
    print(f"📁 Results exported to {filename}")


def main():
    """
    Main execution function that orchestrates the entire system.
    """
    print("🚀 Starting Dynamic Parking Pricing System...")
    print("=" * 60)

    # Load data (update path as needed)
    file_path = '/content/dataset.csv'  # Google Colab path
    df = load_and_preprocess_data(file_path)

    if df is None:
        print("❌ Failed to load data. Please check the file path.")
        return

    # Initialize pricing models
    pricing_models = PricingModel()

    # Apply all three pricing models
    df = pricing_models.model_1_linear(df)
    df = pricing_models.model_2_demand_based(df)
    df = pricing_models.model_3_competitive(df)

    # Display summary statistics
    print("\n📊 PRICING SUMMARY STATISTICS")
    print("=" * 40)

    for model in ['Price_Linear', 'Price_Demand', 'Price_Competitive']:
        stats = df[model].describe()
        print(f"\n{model}:")
        print(f"  Mean: ₹{stats['mean']:.2f}")
        print(f"  Std:  ₹{stats['std']:.2f}")
        print(f"  Min:  ₹{stats['min']:.2f}")
        print(f"  Max:  ₹{stats['max']:.2f}")
        print(f"  Median: ₹{stats['50%']:.2f}")

    # Create visualizations
    print("\n🎨 Creating Visualizations...")
    try:
        viz = PricingDashboard(df)
        dashboard = viz.create_dashboard()
        show(dashboard)
    except Exception as e:
        print(f"⚠ Visualization error: {e}")

    # Run real-time simulation
    print("\n⚡ Starting Real-Time Simulation...")
    simulator = RealTimeSimulation(df)
    simulator.simulate_streaming_data()

    # Model comparison analysis
    print("\n📈 MODEL PERFORMANCE ANALYSIS")
    print("=" * 40)

    for lot in df['SystemCodeNumber'].unique()[:3]:
        lot_data = df[df['SystemCodeNumber'] == lot]
        print(f"\n🅿️ Lot: {lot}")

        for model in ['Price_Linear', 'Price_Demand', 'Price_Competitive']:
            volatility = lot_data[model].std()
            avg_price = lot_data[model].mean()
            print(f"  {model}: Avg=₹{avg_price:.2f}, Volatility=₹{volatility:.2f}")

    # Generate recommendations
    print("\n💡 PRICING RECOMMENDATIONS")
    print("=" * 40)

    recommendations = generate_recommendations(df)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

    # Analyze pricing effectiveness
    df = analyze_pricing_effectiveness(df)

    # Export results
    export_results(df)

    print("\n✅ Dynamic Parking Pricing System Completed Successfully!")
    print("📋 Summary:")
    print(f"   • Processed {len(df)} parking events")
    print(f"   • Analyzed {df['SystemCodeNumber'].nunique()} parking lots")
    print(f"   • Implemented 3 pricing models")
    print(f"   • Generated real-time visualizations")
    print(f"   • Exported results for further analysis")
    print("\n🚀 System is ready for deployment!")

    return df


# Execute the main system
if __name__ == "__main__":
    df_result = main()
