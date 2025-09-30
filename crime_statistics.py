#!/usr/bin/env python3
"""
Crime Data Statistical Analysis Tool

This script generates comprehensive statistical analysis from combined crime datasets,
including descriptive statistics, correlation analysis, trend metrics, and visualizations.

Features:
- Comprehensive statistical measures (descriptive, temporal, correlation)
- Pie charts for crime types and location types (top 10 + "Other")
- Statistical reports exported to HTML and CSV
- Anomaly detection and outlier analysis
- Crime concentration and distribution metrics

Usage:
  python crime_statistics.py --csv crime_data/Combined_Datasets/combined_crime_data_YYYYMMDD_HHMMSS.csv

Requirements:
- plotly
- pandas
- numpy
- scipy
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')


def diagnose_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
	"""Diagnose potential data quality issues"""
	print("🔍 Diagnosing data quality issues...")
	
	current_date = pd.Timestamp.now()
	diagnostics = {}
	
	# Date range analysis
	date_range = df['combined_date'].max() - df['combined_date'].min()
	future_dates = df[df['combined_date'] > current_date]
	very_old_dates = df[df['combined_date'] < pd.Timestamp('1990-01-01')]
	
	diagnostics['date_issues'] = {
		'total_date_span_years': float(date_range.days / 365.25),
		'future_date_count': int(len(future_dates)),
		'very_old_date_count': int(len(very_old_dates)),
		'future_date_sample': future_dates['combined_date'].dt.strftime('%Y-%m-%d').head(5).tolist(),
		'very_old_date_sample': very_old_dates['combined_date'].dt.strftime('%Y-%m-%d').head(5).tolist()
	}
	
	# Time analysis
	hour_data = df.dropna(subset=['hour'])
	time_dist = hour_data['hour'].value_counts().sort_index()
	
	diagnostics['time_issues'] = {
		'missing_time_percentage': float((len(df) - len(hour_data)) / len(df) * 100),
		'peak_hour': int(time_dist.index[0]) if len(time_dist) > 0 else None,
		'peak_hour_count': int(time_dist.iloc[0]) if len(time_dist) > 0 else 0,
		'midnight_crimes_percentage': float(time_dist.get(0, 0) / len(hour_data) * 100) if len(hour_data) > 0 else 0,
		'suspicious_midnight_spike': bool(time_dist.get(0, 0) > time_dist.mean() * 2) if len(time_dist) > 0 else False
	}
	
	# Daily crime count analysis
	daily_counts = df.groupby('combined_date').size()
	
	diagnostics['volume_issues'] = {
		'days_with_very_low_crime': int(len(daily_counts[daily_counts < 10])),
		'days_with_very_high_crime': int(len(daily_counts[daily_counts > daily_counts.quantile(0.99)])),
		'zero_crime_days': int(len(daily_counts[daily_counts == 0])),
		'max_crimes_in_day': int(daily_counts.max()),
		'min_crimes_in_day': int(daily_counts.min())
	}
	
	# Geographic issues
	if 'latitude' in df.columns and 'longitude' in df.columns:
		valid_coords = df.dropna(subset=['latitude', 'longitude'])
		invalid_coords = len(df) - len(valid_coords)
		
		# Check for (0,0) coordinates which are often invalid
		zero_coords = valid_coords[(valid_coords['latitude'] == 0) & (valid_coords['longitude'] == 0)]
		
		diagnostics['geographic_issues'] = {
			'missing_coordinates_count': int(invalid_coords),
			'missing_coordinates_percentage': float(invalid_coords / len(df) * 100),
			'zero_coordinates_count': int(len(zero_coords)),
			'coordinate_range_lat': [float(valid_coords['latitude'].min()), float(valid_coords['latitude'].max())],
			'coordinate_range_lon': [float(valid_coords['longitude'].min()), float(valid_coords['longitude'].max())]
		}
	
	return diagnostics


def load_and_prepare_data(csv_path: str, limit: Optional[int] = None) -> pd.DataFrame:
	"""Load crime data and prepare it for statistical analysis"""
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
		
		# Data quality checks - remove future dates and very old dates
		current_date = pd.Timestamp.now()
		min_reasonable_date = pd.Timestamp('1990-01-01')
		
		before_filter = len(df)
		df = df[(df['combined_date'] <= current_date) & (df['combined_date'] >= min_reasonable_date)]
		after_filter = len(df)
		
		if before_filter != after_filter:
			print(f"⚠️  Data quality warning: Removed {before_filter - after_filter} records with unreasonable dates")
			print(f"   Valid date range: {min_reasonable_date.strftime('%Y-%m-%d')} to {current_date.strftime('%Y-%m-%d')}")
		
	else:
		raise ValueError("No 'combined_date' column found in the dataset")
	
	# Add time-based columns for analysis
	df['year'] = df['combined_date'].dt.year
	df['month'] = df['combined_date'].dt.month
	df['day_of_week'] = df['combined_date'].dt.day_name()
	df['day_of_year'] = df['combined_date'].dt.dayofyear
	df['week_of_year'] = df['combined_date'].dt.isocalendar().week
	df['month_year'] = df['combined_date'].dt.to_period('M').dt.start_time
	
	# Handle time data
	time_data = pd.to_datetime(df['combined_time'], format='%H:%M:%S', errors='coerce')
	df['hour'] = time_data.dt.hour
	df['minute'] = time_data.dt.minute
	
	# Clean up column names for analysis
	if 'primary_type' in df.columns:
		df['crime_type'] = df['primary_type'].fillna('Unknown')
	else:
		# Look for other crime type columns
		crime_cols = [col for col in df.columns if 'type' in col.lower() and any(word in col.lower() for word in ['primary', 'crime', 'offense'])]
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
	
	print(f"Data prepared successfully:")
	print(f"  Date range: {df['combined_date'].min()} to {df['combined_date'].max()}")
	print(f"  Cities: {df['city'].nunique()}")
	print(f"  Crime types: {df['crime_type'].nunique()}")
	print(f"  Location types: {df['location_type'].nunique()}")
	
	return df


def calculate_descriptive_statistics(df: pd.DataFrame) -> Dict[str, Any]:
	"""Calculate comprehensive descriptive statistics"""
	print("Calculating descriptive statistics...")
	
	# Daily crime counts
	daily_counts = df.groupby('combined_date').size()
	
	# Basic descriptive stats
	stats_dict = {
		'total_records': int(len(df)),
		'date_range_days': int((df['combined_date'].max() - df['combined_date'].min()).days),
		'unique_cities': int(df['city'].nunique()),
		'unique_crime_types': int(df['crime_type'].nunique()),
		'unique_location_types': int(df['location_type'].nunique()),
		
		# Daily crime statistics
		'daily_crime_stats': {
			'mean': float(daily_counts.mean()),
			'median': float(daily_counts.median()),
			'std': float(daily_counts.std()),
			'min': int(daily_counts.min()),
			'max': int(daily_counts.max()),
			'q25': float(daily_counts.quantile(0.25)),
			'q75': float(daily_counts.quantile(0.75)),
			'iqr': float(daily_counts.quantile(0.75) - daily_counts.quantile(0.25)),
			'cv': float(daily_counts.std() / daily_counts.mean()) if daily_counts.mean() > 0 else 0.0,
			'skewness': float(stats.skew(daily_counts)),
			'kurtosis': float(stats.kurtosis(daily_counts))
		}
	}
	
	# Crime type statistics
	crime_counts = df['crime_type'].value_counts()
	stats_dict['crime_type_stats'] = {
		'most_common': str(crime_counts.index[0]),
		'most_common_count': int(crime_counts.iloc[0]),
		'most_common_percentage': float(crime_counts.iloc[0] / len(df) * 100),
		'diversity_index': float(calculate_diversity_index(crime_counts)),
		'gini_coefficient': float(calculate_gini_coefficient(crime_counts.values))
	}
	
	# Location type statistics
	location_counts = df['location_type'].value_counts()
	stats_dict['location_type_stats'] = {
		'most_common': str(location_counts.index[0]),
		'most_common_count': int(location_counts.iloc[0]),
		'most_common_percentage': float(location_counts.iloc[0] / len(df) * 100),
		'diversity_index': float(calculate_diversity_index(location_counts)),
		'gini_coefficient': float(calculate_gini_coefficient(location_counts.values))
	}
	
	return stats_dict


def calculate_temporal_statistics(df: pd.DataFrame) -> Dict[str, Any]:
	"""Calculate temporal pattern statistics"""
	print("Calculating temporal statistics...")
	
	# Daily counts for time series analysis
	daily_counts = df.groupby('combined_date').size()
	
	# Monthly counts
	monthly_counts = df.groupby('month_year').size()
	
	# Day of week patterns
	dow_counts = df['day_of_week'].value_counts()
	
	# Hour patterns (only for records with time data)
	hour_data = df.dropna(subset=['hour'])
	hour_counts = hour_data['hour'].value_counts().sort_index() if len(hour_data) > 0 else pd.Series()
	
	temporal_stats = {
		# Autocorrelation
		'daily_autocorr_lag1': float(daily_counts.autocorr(lag=1)),
		'daily_autocorr_lag7': float(daily_counts.autocorr(lag=7)),
		
		# Monthly growth rates
		'monthly_growth_rates': {k: float(v) for k, v in monthly_counts.pct_change().dropna().describe().to_dict().items()},
		
		# Day of week patterns
		'peak_day_of_week': str(dow_counts.index[0]),
		'peak_dow_ratio': float(dow_counts.iloc[0] / dow_counts.mean()),
		
		# Seasonal patterns
		'monthly_cv': float(monthly_counts.std() / monthly_counts.mean()) if monthly_counts.mean() > 0 else 0.0,
		
		# Time of day patterns (if available)
		'peak_hour': int(hour_counts.index[0]) if len(hour_counts) > 0 else None,
		'peak_hour_ratio': float(hour_counts.iloc[0] / hour_counts.mean()) if len(hour_counts) > 0 else None,
		'records_with_time': int(len(hour_data)),
		'time_coverage_percentage': float(len(hour_data) / len(df) * 100)
	}
	
	return temporal_stats


def calculate_correlation_statistics(df: pd.DataFrame) -> Dict[str, Any]:
	"""Calculate correlation and association statistics"""
	print("Calculating correlation and association statistics...")
	
	correlation_stats = {}
	
	# Crime type - Location type association (Chi-square test)
	try:
		contingency_table = pd.crosstab(df['crime_type'], df['location_type'])
		chi2, p_value, dof, expected = chi2_contingency(contingency_table)
		cramers_v = np.sqrt(chi2 / (len(df) * (min(contingency_table.shape) - 1)))
		
		correlation_stats['crime_location_association'] = {
			'chi_square': chi2,
			'p_value': p_value,
			'cramers_v': cramers_v,
			'significant': p_value < 0.05
		}
	except Exception as e:
		correlation_stats['crime_location_association'] = {'error': str(e)}
	
	# Time-based correlations
	try:
		# Create hour-based dummy variables for crimes with time data
		hour_df = df.dropna(subset=['hour'])
		if len(hour_df) > 100:  # Only if we have sufficient data
			hour_dummies = pd.get_dummies(hour_df['hour'], prefix='hour')
			crime_dummies = pd.get_dummies(hour_df['crime_type'], prefix='crime')
			
			# Calculate correlations between hours and crime types
			combined_dummies = pd.concat([hour_dummies, crime_dummies], axis=1)
			corr_matrix = combined_dummies.corr()
			
			# Extract hour-crime correlations
			hour_cols = [col for col in corr_matrix.columns if col.startswith('hour_')]
			crime_cols = [col for col in corr_matrix.columns if col.startswith('crime_')]
			
			if hour_cols and crime_cols:
				hour_crime_corr = corr_matrix.loc[hour_cols, crime_cols]
				correlation_stats['hour_crime_correlations'] = {
					'max_correlation': hour_crime_corr.abs().max().max(),
					'mean_correlation': hour_crime_corr.abs().mean().mean()
				}
	except Exception as e:
		correlation_stats['hour_crime_correlations'] = {'error': str(e)}
	
	return correlation_stats


def calculate_concentration_statistics(df: pd.DataFrame) -> Dict[str, Any]:
	"""Calculate crime concentration and distribution statistics"""
	print("Calculating concentration statistics...")
	
	concentration_stats = {}
	
	# Location concentration
	location_counts = df['location_type'].value_counts()
	total_locations = len(location_counts)
	
	# Calculate what percentage of locations account for different percentages of crimes
	cumulative_crimes = location_counts.cumsum() / len(df)
	cumulative_locations = np.arange(1, len(location_counts) + 1) / total_locations
	
	# Find concentration ratios
	top_10_pct_locations = int(0.1 * total_locations) or 1
	top_20_pct_locations = int(0.2 * total_locations) or 1
	
	concentration_stats['location_concentration'] = {
		'top_10_pct_locations_crime_share': cumulative_crimes.iloc[top_10_pct_locations - 1] * 100,
		'top_20_pct_locations_crime_share': cumulative_crimes.iloc[top_20_pct_locations - 1] * 100,
		'gini_coefficient': calculate_gini_coefficient(location_counts.values),
		'herfindahl_index': calculate_herfindahl_index(location_counts.values)
	}
	
	# Crime type concentration
	crime_counts = df['crime_type'].value_counts()
	total_crime_types = len(crime_counts)
	
	top_10_pct_crimes = int(0.1 * total_crime_types) or 1
	top_20_pct_crimes = int(0.2 * total_crime_types) or 1
	
	cumulative_crimes_by_type = crime_counts.cumsum() / len(df)
	
	concentration_stats['crime_type_concentration'] = {
		'top_10_pct_types_share': cumulative_crimes_by_type.iloc[top_10_pct_crimes - 1] * 100,
		'top_20_pct_types_share': cumulative_crimes_by_type.iloc[top_20_pct_crimes - 1] * 100,
		'gini_coefficient': calculate_gini_coefficient(crime_counts.values),
		'herfindahl_index': calculate_herfindahl_index(crime_counts.values)
	}
	
	return concentration_stats


def detect_anomalies(df: pd.DataFrame) -> Dict[str, Any]:
	"""Detect anomalies and outliers in the data"""
	print("Detecting anomalies...")
	
	# Daily crime counts
	daily_counts = df.groupby('combined_date').size()
	
	# Z-score based anomalies
	z_scores = np.abs(stats.zscore(daily_counts))
	outlier_threshold = 3
	outliers = daily_counts[z_scores > outlier_threshold]
	
	# IQR based anomalies
	q1, q3 = daily_counts.quantile([0.25, 0.75])
	iqr = q3 - q1
	lower_bound = q1 - 1.5 * iqr
	upper_bound = q3 + 1.5 * iqr
	iqr_outliers = daily_counts[(daily_counts < lower_bound) | (daily_counts > upper_bound)]
	
	anomaly_stats = {
		'z_score_outliers': {
			'count': len(outliers),
			'percentage': len(outliers) / len(daily_counts) * 100,
			'dates': outliers.index.strftime('%Y-%m-%d').tolist()[:10]  # Top 10
		},
		'iqr_outliers': {
			'count': len(iqr_outliers),
			'percentage': len(iqr_outliers) / len(daily_counts) * 100,
			'dates': iqr_outliers.index.strftime('%Y-%m-%d').tolist()[:10]  # Top 10
		},
		'anomaly_statistics': {
			'mean_daily_crimes': daily_counts.mean(),
			'std_daily_crimes': daily_counts.std(),
			'max_daily_crimes': daily_counts.max(),
			'max_crime_date': daily_counts.idxmax().strftime('%Y-%m-%d'),
			'min_daily_crimes': daily_counts.min(),
			'min_crime_date': daily_counts.idxmin().strftime('%Y-%m-%d')
		}
	}
	
	return anomaly_stats


def calculate_diversity_index(counts: pd.Series) -> float:
	"""Calculate Shannon diversity index"""
	proportions = counts / counts.sum()
	return -np.sum(proportions * np.log(proportions))


def calculate_gini_coefficient(values: np.ndarray) -> float:
	"""Calculate Gini coefficient for inequality measurement"""
	values = np.sort(values)
	n = len(values)
	cumsum = np.cumsum(values)
	return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


def calculate_herfindahl_index(values: np.ndarray) -> float:
	"""Calculate Herfindahl-Hirschman Index for concentration"""
	total = values.sum()
	proportions = values / total
	return np.sum(proportions ** 2)


def create_pie_charts(df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
	"""Create pie charts for crime types and location types (top 10 + Other)"""
	print("Creating pie charts...")
	
	# Crime type pie chart
	crime_counts = df['crime_type'].value_counts()
	top_10_crimes = crime_counts.head(10)
	other_crimes_count = crime_counts.iloc[10:].sum() if len(crime_counts) > 10 else 0
	
	if other_crimes_count > 0:
		pie_data_crimes = pd.concat([top_10_crimes, pd.Series([other_crimes_count], index=['Other'])])
	else:
		pie_data_crimes = top_10_crimes
	
	crime_pie = px.pie(
		values=pie_data_crimes.values,
		names=pie_data_crimes.index,
		title=f"Crime Types Distribution (Top 10 + Other)<br>Total Crimes: {len(df):,}",
		hover_data={'values': pie_data_crimes.values}
	)
	crime_pie.update_traces(
		textposition='inside',
		textinfo='percent+label',
		hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>'
	)
	crime_pie.update_layout(height=600, showlegend=True)
	
	# Location type pie chart
	location_counts = df['location_type'].value_counts()
	top_10_locations = location_counts.head(10)
	other_locations_count = location_counts.iloc[10:].sum() if len(location_counts) > 10 else 0
	
	if other_locations_count > 0:
		pie_data_locations = pd.concat([top_10_locations, pd.Series([other_locations_count], index=['Other'])])
	else:
		pie_data_locations = top_10_locations
	
	location_pie = px.pie(
		values=pie_data_locations.values,
		names=pie_data_locations.index,
		title=f"Location Types Distribution (Top 10 + Other)<br>Total Crimes: {len(df):,}",
		hover_data={'values': pie_data_locations.values}
	)
	location_pie.update_traces(
		textposition='inside',
		textinfo='percent+label',
		hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>'
	)
	location_pie.update_layout(height=600, showlegend=True)
	
	return crime_pie, location_pie


def print_comprehensive_summary(df: pd.DataFrame, stats_dict: Dict[str, Any], diagnostics: Dict[str, Any]):
	"""Print comprehensive statistical summary similar to time_series_plot.py"""
	print("\n" + "="*80)
	print("COMPREHENSIVE CRIME DATA STATISTICAL ANALYSIS")
	print("="*80)
	
	# Data Quality Issues First
	print(f"🔍 DATA QUALITY DIAGNOSTICS")
	print(f"{'='*50}")
	
	if 'date_issues' in diagnostics:
		date_issues = diagnostics['date_issues']
		print(f"Future dates found: {date_issues['future_date_count']}")
		if date_issues['future_date_sample']:
			print(f"  Sample future dates: {', '.join(date_issues['future_date_sample'])}")
		print(f"Very old dates found: {date_issues['very_old_date_count']}")
		print(f"Total span: {date_issues['total_date_span_years']:.1f} years")
	
	if 'time_issues' in diagnostics:
		time_issues = diagnostics['time_issues']
		print(f"Missing time data: {time_issues['missing_time_percentage']:.1f}%")
		if time_issues['suspicious_midnight_spike']:
			print(f"⚠️  Suspicious midnight spike: {time_issues['midnight_crimes_percentage']:.1f}% at hour 0")
	
	if 'volume_issues' in diagnostics:
		volume_issues = diagnostics['volume_issues']
		print(f"Days with <10 crimes: {volume_issues['days_with_very_low_crime']}")
		print(f"Max crimes in one day: {volume_issues['max_crimes_in_day']}")
		print(f"Min crimes in one day: {volume_issues['min_crimes_in_day']}")
	
	# Basic overview
	print(f"\n📊 DATASET OVERVIEW")
	print(f"{'='*50}")
	print(f"Total Records: {stats_dict['total_records']:,}")
	print(f"Date Range: {stats_dict['date_range_days']} days")
	print(f"Cities: {stats_dict['unique_cities']}")
	print(f"Crime Types: {stats_dict['unique_crime_types']}")
	print(f"Location Types: {stats_dict['unique_location_types']}")
	
	# Daily crime statistics
	daily_stats = stats_dict['daily_crime_stats']
	print(f"\n📅 DAILY CRIME STATISTICS")
	print(f"{'='*50}")
	print(f"Average crimes per day: {daily_stats['mean']:.1f}")
	print(f"Median crimes per day: {daily_stats['median']:.1f}")
	print(f"Standard deviation: {daily_stats['std']:.1f}")
	print(f"Minimum daily crimes: {daily_stats['min']}")
	print(f"Maximum daily crimes: {daily_stats['max']}")
	print(f"Coefficient of variation: {daily_stats['cv']:.3f}")
	print(f"Distribution skewness: {daily_stats['skewness']:.3f}")
	print(f"Distribution kurtosis: {daily_stats['kurtosis']:.3f}")
	
	# Crime type analysis
	crime_stats = stats_dict['crime_type_stats']
	print(f"\n🚨 CRIME TYPE ANALYSIS")
	print(f"{'='*50}")
	print(f"Most common crime: {crime_stats['most_common']}")
	print(f"Most common count: {crime_stats['most_common_count']:,} ({crime_stats['most_common_percentage']:.1f}%)")
	print(f"Crime type diversity index: {crime_stats['diversity_index']:.3f}")
	print(f"Crime type Gini coefficient: {crime_stats['gini_coefficient']:.3f}")
	
	# Location type analysis
	location_stats = stats_dict['location_type_stats']
	print(f"\n📍 LOCATION TYPE ANALYSIS")
	print(f"{'='*50}")
	print(f"Most common location: {location_stats['most_common']}")
	print(f"Most common count: {location_stats['most_common_count']:,} ({location_stats['most_common_percentage']:.1f}%)")
	print(f"Location diversity index: {location_stats['diversity_index']:.3f}")
	print(f"Location Gini coefficient: {location_stats['gini_coefficient']:.3f}")
	
	# Print top 10 lists
	print(f"\n🏆 TOP 10 CRIME TYPES")
	print(f"{'='*50}")
	crime_counts = df['crime_type'].value_counts()
	for i, (crime, count) in enumerate(crime_counts.head(10).items(), 1):
		percentage = count / len(df) * 100
		print(f"{i:2}. {crime:<30} {count:>8,} ({percentage:>5.1f}%)")
	
	print(f"\n🏆 TOP 10 LOCATION TYPES")
	print(f"{'='*50}")
	location_counts = df['location_type'].value_counts()
	for i, (location, count) in enumerate(location_counts.head(10).items(), 1):
		percentage = count / len(df) * 100
		print(f"{i:2}. {location:<35} {count:>8,} ({percentage:>5.1f}%)")
	
	# City breakdown
	print(f"\n🌆 CITY BREAKDOWN")
	print(f"{'='*50}")
	city_counts = df['city'].value_counts()
	for city, count in city_counts.items():
		percentage = count / len(df) * 100
		print(f"{city:<20} {count:>10,} ({percentage:>5.1f}%)")


def export_statistics_report(stats_dict: Dict[str, Any], temporal_stats: Dict[str, Any], 
						   correlation_stats: Dict[str, Any], concentration_stats: Dict[str, Any],
						   anomaly_stats: Dict[str, Any], diagnostics: Dict[str, Any], output_dir: str, base_filename: str):
	"""Export comprehensive statistics report"""
	print("Exporting statistics report...")
	
	# Flatten all statistics into a comprehensive report
	report_data = {
		'Data Quality Diagnostics': diagnostics,
		'Basic Statistics': stats_dict,
		'Temporal Analysis': temporal_stats,
		'Correlation Analysis': correlation_stats,
		'Concentration Analysis': concentration_stats,
		'Anomaly Detection': anomaly_stats
	}
	
	# Save to JSON for programmatic access
	import json
	json_path = os.path.join(output_dir, f"{base_filename}_statistics_report.json")
	with open(json_path, 'w') as f:
		json.dump(report_data, f, indent=2, default=str)
	
	print(f"Statistics report exported to: {json_path}")


def ensure_output_dir() -> str:
	"""Ensure output directory exists"""
	output_dir = "crime_data/Statistical_Analysis"
	os.makedirs(output_dir, exist_ok=True)
	return output_dir


def main():
	parser = argparse.ArgumentParser(description="Generate comprehensive crime data statistics and visualizations")
	parser.add_argument("--csv", required=True, help="Path to combined crime CSV file")
	parser.add_argument("--limit", type=int, help="Limit number of records to process")
	parser.add_argument("--title", type=str, help="Custom title for the analysis")
	parser.add_argument("--outfile", type=str, help="Output filename base (without extension)")
	
	args = parser.parse_args()
	
	# Load and prepare data
	try:
		df = load_and_prepare_data(args.csv, args.limit)
	except Exception as e:
		print(f"Error loading data: {e}")
		sys.exit(1)
	
	if len(df) == 0:
		print("No data available for analysis!")
		sys.exit(1)
	
	# Run data quality diagnostics
	diagnostics = diagnose_data_quality(df)
	
	# Calculate all statistics
	stats_dict = calculate_descriptive_statistics(df)
	temporal_stats = calculate_temporal_statistics(df)
	correlation_stats = calculate_correlation_statistics(df)
	concentration_stats = calculate_concentration_statistics(df)
	anomaly_stats = detect_anomalies(df)
	
	# Print comprehensive summary
	print_comprehensive_summary(df, stats_dict, diagnostics)
	
	# Create visualizations
	crime_pie, location_pie = create_pie_charts(df)
	
	# Setup output
	output_dir = ensure_output_dir()
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	base_name = os.path.splitext(os.path.basename(args.csv))[0]
	base_filename = args.outfile or f"{base_name}_analysis_{timestamp}"
	
	# Save pie charts
	crime_pie_path = os.path.join(output_dir, f"{base_filename}_crime_types_pie.html")
	location_pie_path = os.path.join(output_dir, f"{base_filename}_location_types_pie.html")
	
	crime_pie.write_html(crime_pie_path, include_plotlyjs="cdn")
	location_pie.write_html(location_pie_path, include_plotlyjs="cdn")
	
	# Export comprehensive report
	export_statistics_report(stats_dict, temporal_stats, correlation_stats, 
						   concentration_stats, anomaly_stats, diagnostics, output_dir, base_filename)
	
	print(f"\n" + "="*80)
	print("ANALYSIS COMPLETE!")
	print("="*80)
	print(f"Crime types pie chart: {crime_pie_path}")
	print(f"Location types pie chart: {location_pie_path}")
	print(f"Comprehensive JSON report: {base_filename}_statistics_report.json")
	print(f"\nOpen the HTML files in your browser to view the interactive pie charts.")
	print(f"All files saved to: {output_dir}")


if __name__ == "__main__":
	main()