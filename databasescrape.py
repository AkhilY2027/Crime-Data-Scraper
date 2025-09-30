#!/usr/bin/env python3
"""
Crime Dataset Scraper for Socrata Open Data Portal

This script searches for and downloads crime datasets with geographical coordinates
from various cities using the Socrata Open Data API, then combines them into a
single dataset with source information.
"""

import pandas as pd
import requests
from sodapy import Socrata
import json
import time
from datetime import datetime
import os
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s',
	handlers=[
		logging.FileHandler("crime_scraper.log"),
		logging.StreamHandler()
	]
)
logger = logging.getLogger(__name__)

class CrimeDatasetScraper:
	"""Class to scrape crime datasets from Socrata Open Data Portal"""
	
	def __init__(self, output_dir="crime_data"):
		"""
		Initialize the scraper
		
		Args:
			output_dir (str): Directory to save the output files
		"""
		self.output_dir = output_dir
		# Initialize with API token to avoid throttling (app_token is sufficient for read access)
		self.app_token = "MpxAuF8Aa0dpEv3GJnSP8OnoX"
		self.timeout = 60
		self.client = Socrata("api.us.socrata.com", self.app_token, timeout=self.timeout)
		self.datasets = []
		self.combined_data = pd.DataFrame()
		
		# Create output directory if it doesn't exist
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
	
	def load_existing_metadata(self):
		"""
		Load existing dataset metadata from file if it exists
		
		Returns:
			bool: True if metadata was loaded, False otherwise
		"""
		metadata_file = f"{self.output_dir}/dataset_metadata.json"
		
		if os.path.exists(metadata_file):
			logger.info(f"Loading existing metadata from {metadata_file}")
			try:
				with open(metadata_file, 'r') as f:
					metadata = json.load(f)
				
				# Convert metadata to the format expected by the rest of the code
				self.datasets = [{
					"id": item["id"],
					"name": item["name"],
					"domain": item["domain"],
					"link": item["link"],
					"description": item["description"],
					"columns": []  # We'll get this when we download
				} for item in metadata]
				
				logger.info(f"Loaded {len(self.datasets)} datasets from existing metadata")
				return True
				
			except Exception as e:
				logger.error(f"Error loading metadata file: {e}")
				return False
		
		return False
	
	def search_crime_datasets(self, limit=100):
		"""
		Search for crime datasets with geographical coordinates
		
		Args:
			limit (int): Maximum number of datasets to return
		
		Returns:
			list: List of dataset information
		"""
		logger.info("Searching for crime datasets with geographical coordinates")
		
		# Keywords to search for
		keywords = ["crime", "police", "incident"]
		city_domains = [
			"data.cityofchicago.org", 
			# "data.sfgov.org", 
			# "data.seattle.gov", 
			# "data.lacity.org", 
			# "data.cityofnewyork.us",
			# "data.baltimorecity.gov", 
			# "data.austintexas.gov", 
			# "data.cityofboston.gov", 
			# "data.nola.gov"
		]
		
		all_results = []
		
		# For each domain, get datasets and filter by keyword
		for domain in city_domains:
			logger.info(f"Searching for crime datasets in {domain}")
			try:
				# Create a client for this specific domain with API token
				domain_client = Socrata(domain, self.app_token, timeout=self.timeout)
				
				# Get all datasets for this domain
				datasets = domain_client.datasets(limit=limit)
				
				# Filter datasets related to crime
				for dataset in datasets:
					try:
						# Get dataset resource info
						resource = dataset.get("resource", {})
						dataset_name = resource.get("name", "").lower()
						dataset_desc = resource.get("description", "").lower()
						dataset_id = resource.get("id")
						
						# Check if dataset is crime-related by name or description
						is_crime_related = any(keyword.lower() in dataset_name or keyword.lower() in dataset_desc 
											for keyword in keywords)
						
						if is_crime_related and dataset_id:
							try:
								# Get sample data to check for coordinate columns
								sample_data = domain_client.get(dataset_id, limit=1)
								
								# Check if sample data exists and has coordinates
								if sample_data:
									# Get column names from sample data
									if isinstance(sample_data, list) and len(sample_data) > 0:
										sample = sample_data[0]
										columns = list(sample.keys())
										
										# Check for coordinate columns
										has_coords = any(
											coord_term in col.lower()
											for col in columns
											for coord_term in ["latitude", "longitude", "lat", "lon", "y_coord", "x_coord", 
															  "point", "location", "coordinates"]
										)
										
										if has_coords:
											# Format columns for compatibility with the rest of the code
											formatted_columns = [{"fieldName": col} for col in columns]
											
											# Add dataset to results
											result_item = {
												"resource": {
													"id": dataset_id,
													"name": resource.get("name", ""),
													"description": resource.get("description", "No description")
												},
												"metadata": {
													"domain": domain
												},
												"columns": formatted_columns
											}
											
											# Avoid duplicates
											if not any(r.get("resource", {}).get("id") == dataset_id for r in all_results):
												all_results.append(result_item)
												logger.info(f"Found dataset with coordinates: {resource.get('name')}")
							except Exception as e:
								logger.debug(f"Error checking sample data for {dataset_id}: {str(e)}")
					except (KeyError, TypeError) as e:
						logger.debug(f"Error processing dataset: {str(e)}")
			except Exception as e:
				logger.warning(f"Error accessing {domain}: {str(e)}")
		
		logger.info(f"Found {len(all_results)} potential crime datasets with coordinates")
		
		# Store dataset information
		self.datasets = [{
			"id": result["resource"]["id"],
			"name": result["resource"]["name"],
			"domain": result["metadata"]["domain"],
			"link": f"https://{result['metadata']['domain']}/d/{result['resource']['id']}",
			"description": result["resource"].get("description", "No description"),
			"columns": result.get("columns", [])
		} for result in all_results]
		
		return self.datasets
	
	def find_coordinate_columns(self, dataset_columns):
		"""
		Find the column names that contain latitude and longitude data
		
		Args:
			dataset_columns (list): List of column metadata
		
		Returns:
			dict: Dictionary with coordinate column information
		"""
		result = {
			"lat_col": None,
			"long_col": None,
			"location_col": None,
			"coordinate_type": None,
		}

		# Convert column list to more usable format
		columns = {col["fieldName"].lower(): col["fieldName"] for col in dataset_columns}
		
		# Look for latitude columns
		for lat_key in ["latitude", "lat", "y_coord", "y"]:
			for col in columns:
				if lat_key in col:
					result["lat_col"] = columns[col]
					break
			if result["lat_col"]:
				break
		
		# Look for longitude columns
		for lon_key in ["longitude", "long", "lon", "x_coord", "x"]:
			for col in columns:
				if lon_key in col:
					result["long_col"] = columns[col]
					break
			if result["long_col"]:
				break
		
		# Look for point/location columns that might contain both lat and long
		for loc_key in ["location", "point", "coordinates", "geom", "geometry"]:
			for col in columns:
				if loc_key in col:
					result["location_col"] = columns[col]
					break
			if result["location_col"]:
				break

		# Based on what we found, determine coordinate type
		if result["lat_col"] and result["long_col"]:
			result["coordinate_type"] = "separate_columns"
		elif result["location_col"]:
			result["coordinate_type"] = "combined_column"
		else:
			result["coordinate_type"] = "none"

		return result
	
	# TODO: There is a limit to this dataset here
	def download_dataset(self, dataset, limit=50000, force_redownload=True):
		"""
		Download a specific dataset
		
		Args:
			dataset (dict): Dataset information
			limit (int): Maximum number of records to download
			force_redownload (bool): If True, download even if file exists
			
		Returns:
			pandas.DataFrame: Downloaded data
		"""
		dataset_id = dataset["id"]
		domain = dataset["domain"]
		name = dataset["name"]
		
		# Create organized folder structure
		city = domain.split(".")[1]
		city_folder = f"{self.output_dir}/Downloaded_Datasets/{city.title()}"
		os.makedirs(city_folder, exist_ok=True)
		
		# Generate filenames
		dataset_safe_name = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")[:50]
		dataset_file = f"{city_folder}/{dataset_safe_name}.csv"
		metadata_file = f"{city_folder}/{dataset_safe_name}_metadata.json"
		
		# Check if file already exists
		if os.path.exists(dataset_file) and not force_redownload:
			logger.info(f"Dataset already exists: {dataset_file}, loading from file")
			try:
				# Load the dataset
				df = pd.read_csv(dataset_file)
				
				# Load the metadata to reconstruct coordinate info
				if os.path.exists(metadata_file):
					with open(metadata_file, 'r') as f:
						saved_metadata = json.load(f)
						dataset.update(saved_metadata)  # Update dataset with saved metadata
				
				logger.info(f"Successfully loaded {len(df)} records from existing file: {name}")
				return df
			except Exception as e:
				logger.warning(f"Error loading existing file {dataset_file}: {e}, will re-download")
		
		logger.info(f"Downloading dataset: {name} from {domain}")
		
		try:
			# Initialize a client for this specific domain with API token
			client = Socrata(domain, self.app_token, timeout=self.timeout)
			
			# Download the data
			data = client.get(dataset_id, limit=limit)
			
			if not data:
				logger.warning(f"No data found for {name}")
				return None
			
			# Convert to DataFrame
			df = pd.DataFrame.from_records(data)
			
			# Add source information
			df["source_city"] = domain.split(".")[1]  # Extract city from domain
			df["source_url"] = dataset["link"]
			df["dataset_name"] = name
			
			# Get and save column information for coordinate extraction
			if isinstance(data, list) and len(data) > 0:
				sample = data[0]
				columns = list(sample.keys())
				# Add our source columns to the metadata
				all_columns = columns + ["source_city", "source_url", "dataset_name"]
				dataset["columns"] = [{"fieldName": col} for col in all_columns]
				
				# Save enhanced metadata for future use
				enhanced_metadata = {
					"original_columns": [{"fieldName": col} for col in columns],
					"all_columns": dataset["columns"],
					"has_coordinates": any(
						coord_term in col.lower()
						for col in all_columns
						for coord_term in ["latitude", "longitude", "lat", "lon", "y_coord", "x_coord", 
										  "point", "location", "coordinates"]
					),
					"coordinate_info": self.find_coordinate_columns(dataset["columns"])
				}
				
				# Save metadata to file
				with open(metadata_file, 'w') as f:
					json.dump(enhanced_metadata, f, indent=2)
			
			# Save the dataset
			df.to_csv(dataset_file, index=False)
			logger.info(f"Successfully downloaded and saved {len(df)} records from {name}")
			logger.info(f"Saved to: {dataset_file}")
			
			return df
			
		except Exception as e:
			logger.error(f"Error downloading dataset {name}: {str(e)}")
			return None
	
	def _extract_separate_coordinates(self, df, lat_col, lon_col):
		"""Extract coordinates from separate latitude and longitude columns"""
		if lat_col in df.columns and lon_col in df.columns:
			df["latitude"] = df[lat_col]
			df["longitude"] = df[lon_col]
			logger.info(f"Extracted coordinates from separate columns: {lat_col}, {lon_col}")
			return True
		return False

	def _extract_combined_coordinates(self, df, loc_col):
		"""Extract coordinates from a combined location column"""
		if loc_col not in df.columns:
			return False
		
		if df[loc_col].dtype != object:
			return False
		
		# Get a sample to determine the format
		sample = df[loc_col].dropna().iloc[0] if not df[loc_col].dropna().empty else None
		if not sample:
			return False
		try:
			# Try to extract coordinates from a location object like {"latitude": 41.8, "longitude": -87.6}
			location = json.loads(sample) if isinstance(sample, str) and sample else {}
			df["latitude"] = df[loc_col].apply(lambda x: float(x.get("latitude", x.get("lat", None))) if x else None)
			df["longitude"] = df[loc_col].apply(lambda x: float(x.get("longitude", x.get("lng", x.get("long", None)))) if x else None)
			return True
		except Exception as e:
			logger.error(f"Error extracting combined coordinates from {loc_col}: {str(e)}")
			return False

	def combine_date_columns(self, df):
		"""
		Combine multiple date columns into a single date column and separate time column
		
		Args:
			df (pandas.DataFrame): Input dataframe
			
		Returns:
			pandas.DataFrame: Dataframe with combined_date and combined_time columns
		"""
		if df is None or df.empty:
			return df
		
		# Find date/datetime/timestamp columns
		date_columns = []
		for col in df.columns:
			col_lower = col.lower()
			if any(keyword in col_lower for keyword in ["date", "time", "datetime", "timestamp"]):
				# Skip our combined columns if they already exist
				if col not in ["combined_date", "combined_time"]:
					date_columns.append(col)
		
		if not date_columns:
			logger.info("No date columns found to combine")
			return df
		
		logger.info(f"Found date columns to combine: {date_columns}")
		
		# Convert all date columns to datetime
		parsed_dates = {}
		for col in date_columns:
			try:
				# Parse any column type to datetime - pandas handles both strings and datetime objects
				parsed_series = pd.to_datetime(df[col], errors='coerce')
				
				if not parsed_series.isna().all():
					parsed_dates[col] = parsed_series
					logger.info(f"Successfully parsed date column: {col}")
			except Exception as e:
				logger.warning(f"Could not parse date column {col}: {e}")
		
		if not parsed_dates:
			logger.warning("No date columns could be parsed")
			return df
		
		# Create a DataFrame with all parsed dates for each row
		dates_df = pd.DataFrame(parsed_dates)
		
		# Find the earliest non-null date for each row
		def get_earliest_date(row):
			valid_dates = row.dropna()
			if len(valid_dates) == 0:
				return pd.NaT
			return valid_dates.min()
		
		earliest_dates = dates_df.apply(get_earliest_date, axis=1)
		
		# Create a copy to avoid modifying the original
		df = df.copy()
		
		# Separate date and time components properly
		df["combined_date"] = earliest_dates.dt.date
		
		# Extract time component, but set to NaN if time is exactly midnight (00:00:00)
		time_component = earliest_dates.dt.time
		df["combined_time"] = time_component.apply(
			lambda t: t if pd.notna(t) and t != pd.Timestamp('00:00:00').time() else None
		)
		
		# Convert combined_time to string format for better CSV handling
		df["combined_time"] = df["combined_time"].apply(
			lambda t: t.strftime('%H:%M:%S') if pd.notna(t) else None
		)
		
		# Log summary
		valid_dates_count = df["combined_date"].notna().sum()
		valid_times_count = df["combined_time"].notna().sum()
		logger.info(f"Combined dates: {valid_dates_count} valid dates, {valid_times_count} valid times")
		
		# Remove original date columns to reduce redundancy
		for col in date_columns:
			if col in df.columns:
				df = df.drop(col, axis=1)
				logger.info(f"Removed original date column: {col}")
		
		return df
	
	def extract_coordinates(self, df, dataset):
		"""
		Extract latitude and longitude from the dataset
		
		Args:
			df (pandas.DataFrame): Dataset
			dataset (dict): Dataset information
			
		Returns:
			pandas.DataFrame: Dataset with standardized lat/lon columns
		"""
		if df is None or df.empty:
			return None

		# First, combine date columns before processing coordinates
		df = self.combine_date_columns(df)
		
		# Try to get coordinate info from enhanced metadata first
		coord_info = None
		try:
			# Build the metadata file path
			city = dataset["domain"].split(".")[1]
			city_dir = os.path.join(self.output_dir, "Downloaded_Datasets", city)
			dataset_safe_name = dataset["name"].replace(" ", "_").replace("/", "_")[:30]
			metadata_file = os.path.join(city_dir, f"{dataset_safe_name}_metadata.json")
			
			if os.path.exists(metadata_file):
				with open(metadata_file, 'r') as f:
					metadata = json.load(f)
					if "coordinate_info" in metadata:
						coord_info = metadata["coordinate_info"]
						logger.info(f"Using cached coordinate info from metadata: {metadata_file}")
		except Exception as e:
			logger.warning(f"Could not load coordinate info from metadata: {e}")
		
		# Fall back to detecting coordinate columns if no cached info available
		if coord_info is None:
			coord_info = self.find_coordinate_columns(dataset.get("columns", []))
			logger.info("Using real-time coordinate detection")
		
		# Make column names lowercase for easier processing
		df.columns = [col.lower() for col in df.columns]
		
		# Standardize column names to lowercase
		lat_col = coord_info["lat_col"].lower() if coord_info["lat_col"] else None
		lon_col = coord_info["long_col"].lower() if coord_info["long_col"] else None
		loc_col = coord_info["location_col"].lower() if coord_info["location_col"] else None
		
		logger.info(f"Coordinate detection: type={coord_info['coordinate_type']}, "
					f"lat_col={lat_col}, lon_col={lon_col}, loc_col={loc_col}")

		# Extract coordinates based on detected type
		if coord_info["coordinate_type"] == "separate_columns":
			success = self._extract_separate_coordinates(df, lat_col, lon_col)
		elif coord_info["coordinate_type"] == "combined_column":
			success = self._extract_combined_coordinates(df, loc_col)
		else:
			logger.warning("No coordinate columns detected")
			return None
		
		if not success:
			logger.warning("Failed to extract coordinates")
			return None

		# Filter out invalid coordinates
		if "latitude" in df.columns and "longitude" in df.columns:
			# Convert to numeric and filter
			df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
			df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
			
			# Filter valid coordinates
			valid_coords = (~df["latitude"].isna() & ~df["longitude"].isna() & 
						   (df["latitude"] != 0) & (df["longitude"] != 0))
			
			df_filtered = df[valid_coords].copy()
			
			logger.info(f"Extracted {len(df_filtered)} records with valid coordinates from dataset")
			
			return df_filtered
		else:
			logger.warning("Latitude and/or longitude columns missing after extraction")
			return None
	
	def reorder_columns(self, df):
		"""
		Reorder columns to prioritize source city, source info, latitude, longitude, then date/time columns
		
		Args:
			df (pandas.DataFrame): Input dataframe
			
		Returns:
			pandas.DataFrame: Dataframe with reordered columns
		"""
		if df is None or df.empty:
			return df
		
		# Start with source/city information columns
		priority_columns = []
		
		# Add source city and source information first
		source_columns = []
		for col in df.columns:
			col_lower = col.lower()
			if any(keyword in col_lower for keyword in ["source_city", "source_url", "dataset_name"]):
				source_columns.append(col)
		
		# Sort source columns to put city first
		source_columns.sort(key=lambda x: (
			0 if "source_city" in x.lower() else 1,
			0 if "dataset_name" in x.lower() else 2,
			0 if "source_url" in x.lower() else 3,
			x.lower()
		))
		
		priority_columns.extend(source_columns)
		
		# Add latitude and longitude next
		if "latitude" in df.columns:
			priority_columns.append("latitude")
		if "longitude" in df.columns:
			priority_columns.append("longitude")

		# Add combined date and time columns
		if "combined_date" in df.columns:
			priority_columns.append("combined_date")
		if "combined_time" in df.columns:
			priority_columns.append("combined_time")
		
		# Find date/time columns
		date_time_columns = []
		for col in df.columns:
			col_lower = col.lower()
			if any(keyword in col_lower for keyword in ["date", "time", "datetime", "timestamp"]):
				if col not in priority_columns:
					date_time_columns.append(col)
		
		# Sort date/time columns to put more important ones first
		date_time_columns.sort(key=lambda x: (
			0 if "date" in x.lower() else 1,
			0 if "time" in x.lower() else 1,
			x.lower()
		))
		
		priority_columns.extend(date_time_columns)
		
		# Add all remaining columns (preserve all information)
		remaining_columns = [col for col in df.columns if col not in priority_columns]
		remaining_columns.sort()
		
		# Combine all columns
		final_column_order = priority_columns + remaining_columns
		
		return df[final_column_order]

	def process_datasets_balanced(self, max_per=10, records_per_dataset=50000, force_redownload=True):
		"""
		Download and process up to max_per datasets from each domain
		
		Args:
			max_per (int): Maximum datasets to process per domain
			records_per_dataset (int): Maximum records per dataset
			force_redownload (bool): If True, re-download even if files exist
			
		Returns:
			pandas.DataFrame: Combined dataset
		"""
		if not self.datasets:
			logger.info("No datasets found. Run search_crime_datasets first or load existing metadata.")
			return None
		
		# Group datasets by domain
		datasets_by_domain = {}
		for dataset in self.datasets:
			domain = dataset["domain"]
			if domain not in datasets_by_domain:
				datasets_by_domain[domain] = []
			datasets_by_domain[domain].append(dataset)
		
		logger.info(f"Found datasets from {len(datasets_by_domain)} domains")
		
		all_dfs = []
		processed_count = 0
		
		# Process up to max_per datasets from each domain
		for domain, domain_datasets in datasets_by_domain.items():
			logger.info(f"Processing up to {max_per} datasets from {domain}")
			
			# Take the first max_per datasets from this domain
			datasets_to_process = domain_datasets[:max_per]
			
			for i, dataset in enumerate(datasets_to_process):
				logger.info(f"Processing dataset {i+1}/{len(datasets_to_process)}: {dataset['name']}")
				
				# Download the dataset
				df = self.download_dataset(dataset, limit=records_per_dataset, force_redownload=force_redownload)
				
				if df is not None:
					# Extract coordinates
					df_with_coords = self.extract_coordinates(df, dataset)
					
					if df_with_coords is not None and len(df_with_coords) > 0:
						# Reorder columns to prioritize source info, coordinates, and date/time
						df_with_coords = self.reorder_columns(df_with_coords)
						
						all_dfs.append(df_with_coords)
						processed_count += 1
						
						logger.info(f"Successfully processed dataset: {dataset['name']} with {len(df_with_coords)} records")
						
				# Add delay to avoid rate limiting
				time.sleep(1)
		
		logger.info(f"Successfully processed {processed_count} datasets with coordinate data")
		
		if all_dfs:
			# Combine all datasets
			combined_df = pd.concat(all_dfs, ignore_index=True)
			
			# Reorder columns for the combined dataset to ensure consistent ordering
			combined_df = self.reorder_columns(combined_df)
			
			# Create Combined_Datasets folder
			combined_dir = os.path.join(self.output_dir, "Combined_Datasets")
			os.makedirs(combined_dir, exist_ok=True)
			
			# Save combined dataset
			timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
			combined_file = os.path.join(combined_dir, f"combined_crime_data_{timestamp}.csv")
			combined_df.to_csv(combined_file, index=False)
			
			logger.info(f"Combined dataset with {len(combined_df)} records saved to {combined_file}")
			
			self.combined_data = combined_df
			return combined_df
		
		return None
	
	def save_dataset_metadata(self):
		"""Save metadata about the downloaded datasets"""
		if self.datasets:
			metadata = [{
				"name": d["name"],
				"domain": d["domain"],
				"link": d["link"],
				"id": d["id"],
				"description": d["description"]
			} for d in self.datasets]
			
			with open(f"{self.output_dir}/dataset_metadata.json", "w") as f:
				json.dump(metadata, f, indent=2)
			
			logger.info(f"Dataset metadata saved to {self.output_dir}/dataset_metadata.json")

def main():
	"""Main function to run the scraper"""
	# Create scraper instance
	scraper = CrimeDatasetScraper(output_dir="crime_data")
	
	# Try to load existing metadata first
	if scraper.load_existing_metadata():
		logger.info("Using existing metadata file")
	else:
		logger.info("No existing metadata found, searching for datasets")
		# Search for datasets
		datasets = scraper.search_crime_datasets(limit=200)
		logger.info(f"Found {len(datasets)} potential datasets")
		
		# Save dataset metadata
		scraper.save_dataset_metadata()
	
	# Process datasets - up to max_per from each domain
	combined_data = scraper.process_datasets_balanced(max_per=100, records_per_dataset=50000, force_redownload=False)
	
	if combined_data is not None:
		# Print summary
		logger.info("\nSummary of combined dataset:")
		logger.info(f"Total records: {len(combined_data)}")
		logger.info(f"Cities represented: {combined_data['source_city'].nunique()}")
		logger.info(f"Columns: {', '.join(combined_data.columns)}")
		
		# Print sample of the data
		logger.info("\nSample data:")
		logger.info(combined_data.head(5))
		
		# Print column order to verify lat/lon are first
		logger.info(f"\nColumn order: {list(combined_data.columns[:10])}")  # Show first 10 columns
	else:
		logger.warning("No data was successfully processed and combined.")

if __name__ == "__main__":
	main()