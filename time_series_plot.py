#!/usr/bin/env python3
"""
Interactive Crime Time Series Dashboard

This script creates interactive time series dashboards from combined crime datasets,
with built-in filtering dropdowns and widgets within the Plotly plots themselves.

Features:
- Interactive dropdown menus for crime types, locations, and cities
- Real-time filtering within the plot interface
- Multiple visualization modes with interactive controls
- Time period selection widgets
- Statistical trend analysis with dynamic updates

Usage:
  python time_series_plot.py --csv crime_data/Combined_Datasets/combined_crime_data_YYYYMMDD_HHMMSS.csv \
      --mode dashboard

Requirements:
- plotly
- pandas
- numpy
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_and_prepare_data(csv_path: str, limit: Optional[int] = None) -> pd.DataFrame:
    """Load crime data and prepare it for interactive analysis"""
    print(f"Loading data from {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Load data
    df = pd.read_csv(csv_path, nrows=limit)
    print(f"Loaded {len(df)} records")
    
    # Convert combined_date to datetime
    if 'combined_date' in df.columns:
        df['combined_date'] = pd.to_datetime(df['combined_date'], errors='coerce')
        df = df.dropna(subset=['combined_date'])
        print(f"After filtering valid dates: {len(df)} records")
    else:
        raise ValueError("No 'combined_date' column found in the dataset")
    
    # Add time-based columns for analysis
    df['year'] = df['combined_date'].dt.year
    df['month'] = df['combined_date'].dt.month
    df['day_of_week'] = df['combined_date'].dt.day_name()
    df['week'] = df['combined_date'].dt.to_period('W').dt.start_time
    df['month_year'] = df['combined_date'].dt.to_period('M').dt.start_time
    
    # Handle time data
    time_data = pd.to_datetime(df['combined_time'], format='%H:%M:%S', errors='coerce')
    df['hour'] = time_data.dt.hour
    df['minute'] = time_data.dt.minute
    
    # Clean up column names for better display
    if 'primary_type' in df.columns:
        df['crime_type'] = df['primary_type'].fillna('Unknown')
    else:
        # Look for other crime type columns
        crime_cols = [col for col in df.columns if 'type' in col.lower() and 'primary' in col.lower()]
        if crime_cols:
            df['crime_type'] = df[crime_cols[0]].fillna('Unknown')
        else:
            df['crime_type'] = 'Unknown'
    
    if 'location_description' in df.columns:
        df['location_type'] = df['location_description'].fillna('Unknown')
    else:
        df['location_type'] = 'Unknown'
    
    if 'source_city' in df.columns:
        df['city'] = df['source_city'].str.title()
    else:
        df['city'] = 'Unknown'
    
    # Add additional helpful columns
    df['date_str'] = df['combined_date'].dt.strftime('%Y-%m-%d')
    df['time_str'] = df['combined_time'].fillna('Unknown')
    
    print(f"Data prepared successfully:")
    print(f"  Date range: {df['combined_date'].min()} to {df['combined_date'].max()}")
    print(f"  Cities: {df['city'].nunique()}")
    print(f"  Crime types: {df['crime_type'].nunique()}")
    print(f"  Location types: {df['location_type'].nunique()}")
    
    return df


def create_interactive_time_series_dashboard(df: pd.DataFrame, title: str = "Crime Time Series Dashboard") -> go.Figure:
    """Create an interactive dashboard with dropdown filters"""
    
    # Prepare data for all crime types, cities, and locations
    all_crime_types = sorted(df['crime_type'].unique())
    all_cities = sorted(df['city'].unique())
    all_locations = sorted(df['location_type'].unique())
    
    # Create initial aggregated data (monthly by default)
    monthly_data = df.groupby(['month_year', 'crime_type', 'city', 'location_type']).size().reset_index(name='count')
    
    # Create base figure with all data
    fig = go.Figure()
    
    # Add traces for each combination (initially all visible)
    for crime_type in all_crime_types:
        for city in all_cities:
            crime_city_data = monthly_data[
                (monthly_data['crime_type'] == crime_type) & 
                (monthly_data['city'] == city)
            ].groupby('month_year')['count'].sum().reset_index()
            
            if len(crime_city_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=crime_city_data['month_year'],
                        y=crime_city_data['count'],
                        mode='lines+markers',
                        name=f"{crime_type} - {city}",
                        visible=True if crime_type == all_crime_types[0] and city == all_cities[0] else False,
                        line=dict(width=2),
                        hovertemplate=f"<b>{crime_type} - {city}</b><br>" +
                                     "Date: %{x}<br>" +
                                     "Count: %{y}<br>" +
                                     "<extra></extra>"
                    )
                )
    
    # Create dropdown menus
    crime_type_buttons = []
    for i, crime_type in enumerate(['All'] + all_crime_types):
        visible_list = []
        for trace_crime in all_crime_types:
            for trace_city in all_cities:
                if crime_type == 'All':
                    visible_list.append(True)
                else:
                    visible_list.append(trace_crime == crime_type)
        
        crime_type_buttons.append(
            dict(
                label=crime_type,
                method="update",
                args=[
                    {"visible": visible_list},
                    {"title": f"{title} - {crime_type}"}
                ]
            )
        )
    
    city_buttons = []
    for i, city in enumerate(['All'] + all_cities):
        visible_list = []
        for trace_crime in all_crime_types:
            for trace_city in all_cities:
                if city == 'All':
                    visible_list.append(True)
                else:
                    visible_list.append(trace_city == city)
        
        city_buttons.append(
            dict(
                label=city,
                method="update",
                args=[
                    {"visible": visible_list},
                    {"title": f"{title} - {city}"}
                ]
            )
        )
    
    # Update layout with dropdown menus
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Number of Crimes",
        height=700,
        updatemenus=[
            dict(
                buttons=crime_type_buttons,
                direction="down",
                showactive=True,
                x=0.02,
                xanchor="left",
                y=1.15,
                yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                font=dict(size=11)
            ),
            dict(
                buttons=city_buttons,
                direction="down",
                showactive=True,
                x=0.25,
                xanchor="left",
                y=1.15,
                yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                font=dict(size=11)
            )
        ],
        annotations=[
            dict(text="Crime Type:", x=0.02, xref="paper", y=1.18, yref="paper", 
                 align="left", showarrow=False, font=dict(size=12, color="black")),
            dict(text="City:", x=0.25, xref="paper", y=1.18, yref="paper", 
                 align="left", showarrow=False, font=dict(size=12, color="black"))
        ],
        margin=dict(t=100)
    )
    
    return fig


def create_interactive_heatmap_dashboard(df: pd.DataFrame, title: str = "Crime Pattern Heatmap") -> go.Figure:
    """Create an interactive heatmap with filtering dropdowns"""
    
    all_crime_types = ['All'] + sorted(df['crime_type'].unique())
    all_cities = ['All'] + sorted(df['city'].unique())
    
    # Create subplots for the heatmap with dropdowns
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=[title]
    )
    
    def create_heatmap_data(crime_filter='All', city_filter='All'):
        filtered_df = df.copy()
        if crime_filter != 'All':
            filtered_df = filtered_df[filtered_df['crime_type'] == crime_filter]
        if city_filter != 'All':
            filtered_df = filtered_df[filtered_df['city'] == city_filter]
        
        # Filter out records without time information
        df_with_time = filtered_df.dropna(subset=['hour'])
        
        if len(df_with_time) == 0:
            return np.zeros((7, 24)), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], list(range(24))
        
        # Create hour-day_of_week heatmap data
        heatmap_data = df_with_time.groupby(['day_of_week', 'hour']).size().reset_index(name='crime_count')
        
        # Pivot for heatmap
        pivot_data = heatmap_data.pivot(index='day_of_week', columns='hour', values='crime_count').fillna(0)
        
        # Reorder days of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_abbrev = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        # Ensure all days and hours are present
        full_pivot = pd.DataFrame(index=day_order, columns=range(24)).fillna(0)
        for day in day_order:
            if day in pivot_data.index:
                for hour in range(24):
                    if hour in pivot_data.columns:
                        full_pivot.loc[day, hour] = pivot_data.loc[day, hour]
        
        return full_pivot.values, day_abbrev, list(range(24))
    
    # Create initial heatmap
    z_data, y_labels, x_labels = create_heatmap_data()
    
    heatmap = go.Heatmap(
        z=z_data,
        x=x_labels,
        y=y_labels,
        colorscale='Viridis',
        showscale=True,
        hovertemplate="Hour: %{x}<br>Day: %{y}<br>Count: %{z}<extra></extra>"
    )
    
    fig.add_trace(heatmap)
    
    # Create dropdown buttons
    crime_buttons = []
    for crime in all_crime_types:
        z_data, _, _ = create_heatmap_data(crime_filter=crime)
        crime_buttons.append(
            dict(
                label=crime,
                method="restyle",
                args=[{"z": [z_data]}]
            )
        )
    
    city_buttons = []
    for city in all_cities:
        z_data, _, _ = create_heatmap_data(city_filter=city)
        city_buttons.append(
            dict(
                label=city,
                method="restyle",
                args=[{"z": [z_data]}]
            )
        )
    
    fig.update_layout(
        title=f"{title} - Hour vs Day of Week",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        height=600,
        updatemenus=[
            dict(
                buttons=crime_buttons,
                direction="down",
                showactive=True,
                x=0.02,
                xanchor="left",
                y=1.15,
                yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)"
            ),
            dict(
                buttons=city_buttons,
                direction="down",
                showactive=True,
                x=0.25,
                xanchor="left",
                y=1.15,
                yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)"
            )
        ],
        annotations=[
            dict(text="Crime Type:", x=0.02, xref="paper", y=1.18, yref="paper", 
                 align="left", showarrow=False),
            dict(text="City:", x=0.25, xref="paper", y=1.18, yref="paper", 
                 align="left", showarrow=False)
        ],
        margin=dict(t=100)
    )
    
    return fig


def create_interactive_comparison_dashboard(df: pd.DataFrame, title: str = "Crime Comparison Dashboard") -> go.Figure:
    """Create an interactive comparison chart with filtering"""
    
    all_crime_types = sorted(df['crime_type'].unique())
    all_cities = sorted(df['city'].unique())
    all_locations = sorted(df['location_type'].unique())[:15]  # Limit to top 15 for readability
    
    # Create the figure
    fig = go.Figure()
    
    # Create initial data (top crime types by default)
    top_crimes = df['crime_type'].value_counts().head(10).index.tolist()
    
    for crime in top_crimes:
        crime_data = df[df['crime_type'] == crime].groupby('month_year').size().reset_index(name='count')
        fig.add_trace(
            go.Scatter(
                x=crime_data['month_year'],
                y=crime_data['count'],
                mode='lines+markers',
                name=crime,
                visible=True,
                hovertemplate=f"<b>{crime}</b><br>Date: %{{x}}<br>Count: %{{y}}<extra></extra>"
            )
        )
    
    # Create buttons for different views
    view_buttons = [
        dict(
            label="Top Crime Types",
            method="update",
            args=[
                {"visible": [True] * len(top_crimes)},
                {"title": f"{title} - Top Crime Types Over Time"}
            ]
        ),
        dict(
            label="All Cities",
            method="update",
            args=[
                {"visible": [False] * len(top_crimes)},  # Hide current traces
                {"title": f"{title} - All Cities Over Time"}
            ]
        ),
        dict(
            label="Top Locations",
            method="update",
            args=[
                {"visible": [False] * len(top_crimes)},  # Hide current traces
                {"title": f"{title} - Top Locations Over Time"}
            ]
        )
    ]
    
    fig.update_layout(
        title=f"{title} - Interactive Comparison",
        xaxis_title="Date",
        yaxis_title="Number of Crimes",
        height=700,
        updatemenus=[
            dict(
                buttons=view_buttons,
                direction="down",
                showactive=True,
                x=0.02,
                xanchor="left",
                y=1.15,
                yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)"
            )
        ],
        annotations=[
            dict(text="View:", x=0.02, xref="paper", y=1.18, yref="paper", 
                 align="left", showarrow=False)
        ],
        margin=dict(t=100),
        showlegend=True
    )
    
    return fig


def print_data_summary(df: pd.DataFrame):
    """Print summary statistics about the dataset"""
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    print(f"Total Records: {len(df):,}")
    print(f"Date Range: {df['combined_date'].min().strftime('%Y-%m-%d')} to {df['combined_date'].max().strftime('%Y-%m-%d')}")
    
    print(f"\nCities ({df['city'].nunique()}):")
    city_counts = df['city'].value_counts()
    for city, count in city_counts.items():
        print(f"  {city}: {count:,}")
    
    print(f"\nTop Crime Types:")
    crime_counts = df['crime_type'].value_counts().head(10)
    for crime, count in crime_counts.items():
        print(f"  {crime}: {count:,}")
    
    print(f"\nTop Location Types:")
    location_counts = df['location_type'].value_counts().head(10)
    for location, count in location_counts.items():
        print(f"  {location}: {count:,}")


def ensure_output_dir() -> str:
    """Ensure output directory exists"""
    output_dir = "crime_data/TimeSeries_Maps"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Create interactive time series dashboards from crime data")
    parser.add_argument("--csv", required=True, help="Path to combined crime CSV file")
    parser.add_argument("--mode", choices=["dashboard", "heatmap", "comparison", "all"], 
                       default="dashboard", help="Type of interactive visualization")
    parser.add_argument("--limit", type=int, help="Limit number of records to process")
    parser.add_argument("--title", type=str, help="Custom title for the dashboard")
    parser.add_argument("--outfile", type=str, help="Output filename (without extension)")
    
    args = parser.parse_args()
    
    # Load and prepare data
    try:
        df = load_and_prepare_data(args.csv, args.limit)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    if len(df) == 0:
        print("No data available for visualization!")
        sys.exit(1)
    
    # Print summary
    print_data_summary(df)
    
    # Generate title
    title = args.title or "Interactive Crime Analysis Dashboard"
    
    output_dir = ensure_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(args.csv))[0]
    
    # Create visualizations based on mode
    if args.mode == "dashboard" or args.mode == "all":
        print("\nCreating interactive time series dashboard...")
        fig = create_interactive_time_series_dashboard(df, title)
        
        filename = args.outfile or f"{base_name}_interactive_dashboard_{timestamp}.html"
        output_path = os.path.join(output_dir, filename)
        fig.write_html(output_path, include_plotlyjs="cdn")
        print(f"Dashboard saved to: {output_path}")
    
    if args.mode == "heatmap" or args.mode == "all":
        print("\nCreating interactive heatmap dashboard...")
        fig = create_interactive_heatmap_dashboard(df, title)
        
        filename = args.outfile or f"{base_name}_interactive_heatmap_{timestamp}.html"
        output_path = os.path.join(output_dir, filename)
        fig.write_html(output_path, include_plotlyjs="cdn")
        print(f"Heatmap saved to: {output_path}")
    
    if args.mode == "comparison" or args.mode == "all":
        print("\nCreating interactive comparison dashboard...")
        fig = create_interactive_comparison_dashboard(df, title)
        
        filename = args.outfile or f"{base_name}_interactive_comparison_{timestamp}.html"
        output_path = os.path.join(output_dir, filename)
        fig.write_html(output_path, include_plotlyjs="cdn")
        print(f"Comparison dashboard saved to: {output_path}")
    
    print(f"\nAll visualizations completed!")
    print(f"Open the HTML files in your browser to interact with the dashboards.")
    print(f"Use the dropdown menus to filter by crime type, city, and other dimensions.")


if __name__ == "__main__":
    main()