#!/usr/bin/env python3
"""
Plot Chicago crime points from a CSV containing latitude and longitude using Plotly.

Features:
- Points scatter on a mapbox basemap
- Optional density heatmap (hexbin-like via densitymapbox)
- Saves interactive HTML to crime_data/Maps

Usage (run from repo root):
  ./plot_chicago_crime_plotly.py --csv crime_data/Combined_Datasets/combined_crime_data_YYYYMMDD_HHMMSS.csv \
      --mode points --limit 100000 --title "Chicago Crime Locations"

Requirements:
- plotly
- pandas

Environment:
- For mapbox styles, a Mapbox access token can be set via the MAPBOX_TOKEN env var.
  Without it, we will use the open "carto-positron" style via plotly's default (no token needed).
"""

import argparse
import os
from datetime import datetime

import pandas as pd
import plotly.express as px


def load_data(csv_path: str, limit: int | None = None) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    # Validate expected columns
    if not {"latitude", "longitude"}.issubset(set(df.columns)):
        raise ValueError("CSV must contain 'latitude' and 'longitude' columns")
    # Filter valid coordinates
    df = df.copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])\
           .query("latitude != 0 and longitude != 0")
    # Chicago bounding box quick sanity (optional, keeps only chicago-ish area)
    # lat ~ 41.6..42.1, lon ~ -88.0..-87.4
    df = df.query("41.6 <= latitude <= 42.1 and -88.0 <= longitude <= -87.4")
    if limit:
        df = df.head(limit)
    return df


def ensure_output_dir() -> str:
    out_dir = os.path.join("crime_data", "Maps")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def plot_points(df: pd.DataFrame, title: str, color_col: str | None = None, group_by: str | None = None):
    # If group_by is specified, treat it as categorical grouping
    if group_by and group_by in df.columns:
        # Ensure the grouping column is treated as categorical for distinct colors
        df = df.copy()
        df[group_by] = df[group_by].astype(str).astype('category')
        
        fig = px.scatter_mapbox(
            df,
            lat="latitude",
            lon="longitude",
            color=group_by,
            color_discrete_sequence=px.colors.qualitative.Set3,  # Distinct colors for categories
            hover_data=[c for c in df.columns if c not in {"latitude", "longitude"}][:6],
            zoom=10,
            height=800,
            opacity=0.7,
            size_max=8,
        )
        
        # Update legend to show category counts
        category_counts = df[group_by].value_counts()
        fig.for_each_trace(
            lambda trace: trace.update(
                name=f"{trace.name} ({category_counts.get(trace.name, 0)})"
            )
        )
        
    else:
        # Default behavior for non-categorical coloring
        fig = px.scatter_mapbox(
            df,
            lat="latitude",
            lon="longitude",
            color=color_col if color_col and color_col in df.columns else None,
            hover_data=[c for c in df.columns if c not in {"latitude", "longitude"}][:6],
            zoom=10,
            height=800,
            opacity=0.6,
        )
    
    style = "carto-positron"  # no token required
    mapbox_token = os.environ.get("MAPBOX_TOKEN")
    fig.update_layout(
        mapbox_style=style if not mapbox_token else "streets",
        mapbox_accesstoken=mapbox_token if mapbox_token else None,
        title=title,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    return fig


def plot_density(df: pd.DataFrame, title: str, weight_col: str | None = None):
    # Create more granular density map with better color variation
    fig = px.density_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        z=weight_col if weight_col and weight_col in df.columns else None,
        radius=4,  # Smaller radius for more granular detail
        center=dict(lat=41.8781, lon=-87.6298),
        zoom=10,  # Closer zoom for better detail
        height=800,
        color_continuous_scale="Viridis",  # Better color scale with more variation
    )
    
    # Update color scale to be more sensitive to variations
    fig.update_traces(
        colorbar=dict(
            title="Crime Density",
            titleside="right"
        ),
        hovertemplate="<b>Density: %{z}</b><br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>"
    )
    
    style = "carto-positron"
    mapbox_token = os.environ.get("MAPBOX_TOKEN")
    fig.update_layout(
        mapbox_style=style if not mapbox_token else "streets",
        mapbox_accesstoken=mapbox_token if mapbox_token else None,
        title=title,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot Chicago crime points with Plotly")
    parser.add_argument("--csv", required=True, help="Path to combined CSV (must include latitude, longitude)")
    parser.add_argument("--mode", choices=["points", "density"], default="points", help="Plot mode")
    parser.add_argument("--limit", type=int, default=100000, help="Max rows to plot (for performance)")
    parser.add_argument("--title", type=str, default="Chicago Crime Locations", help="Figure title")
    parser.add_argument("--color", type=str, default=None, help="Optional column to color points by")
    parser.add_argument("--group-by", type=str, default=None, help="Column to group points by categories (distinct colors)")
    parser.add_argument("--weight", type=str, default=None, help="Optional weight column for density")
    parser.add_argument("--outfile", type=str, default=None, help="Optional output HTML filename")
    args = parser.parse_args()

    df = load_data(args.csv, limit=args.limit)

    if args.mode == "points":
        fig = plot_points(df, args.title, color_col=args.color, group_by=getattr(args, 'group_by'))
    else:
        fig = plot_density(df, args.title, weight_col=args.weight)

    out_dir = ensure_output_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(args.csv))[0]
    outfile = args.outfile or f"{base}_{args.mode}_{ts}.html"
    out_path = os.path.join(out_dir, outfile)
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Saved map to {out_path}")
    
    # Print summary of data processed
    print(f"Plotted {len(df)} points")
    if hasattr(args, 'group_by') and args.group_by and args.group_by in df.columns:
        categories = df[args.group_by].value_counts()
        print(f"Categories in {args.group_by}:")
        for cat, count in categories.head(10).items():
            print(f"  {cat}: {count} points")
        if len(categories) > 10:
            print(f"  ... and {len(categories) - 10} more categories")


if __name__ == "__main__":
    main()
