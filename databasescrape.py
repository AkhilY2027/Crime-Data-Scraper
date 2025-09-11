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
        self.client = Socrata("api.us.socrata.com", None)
        self.datasets = []
        self.combined_data = pd.DataFrame()
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
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
            "data.sfgov.org", 
            "data.seattle.gov", 
            "data.lacity.org", 
            "data.cityofnewyork.us",
            "data.baltimorecity.gov", 
            "data.austintexas.gov", 
            "data.cityofboston.gov", 
            "data.nola.gov"
        ]
        
        all_results = []
        
        # For each domain, get datasets and filter by keyword
        for domain in city_domains:
            logger.info(f"Searching for crime datasets in {domain}")
            try:
                # Create a client for this specific domain
                domain_client = Socrata(domain, None)
                
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
            tuple: (lat_column, lon_column, location_column)
        """
        lat_column = None
        lon_column = None
        location_column = None
        
        # Convert column list to more usable format
        columns = {col["fieldName"].lower(): col["fieldName"] for col in dataset_columns}
        
        # Look for latitude columns
        for lat_key in ["latitude", "lat", "y_coord", "y"]:
            for col in columns:
                if lat_key in col:
                    lat_column = columns[col]
                    break
            if lat_column:
                break
        
        # Look for longitude columns
        for lon_key in ["longitude", "long", "lon", "x_coord", "x"]:
            for col in columns:
                if lon_key in col:
                    lon_column = columns[col]
                    break
            if lon_column:
                break
        
        # Look for point/location columns that might contain both lat and long
        for loc_key in ["location", "point", "coordinates", "geom", "geometry"]:
            for col in columns:
                if loc_key in col:
                    location_column = columns[col]
                    break
            if location_column:
                break
        
        return lat_column, lon_column, location_column
    
    def download_dataset(self, dataset, limit=50000):
        """
        Download a specific dataset
        
        Args:
            dataset (dict): Dataset information
            limit (int): Maximum number of records to download
            
        Returns:
            pandas.DataFrame: Downloaded data
        """
        dataset_id = dataset["id"]
        domain = dataset["domain"]
        name = dataset["name"]
        
        logger.info(f"Downloading dataset: {name} from {domain}")
        
        try:
            # Initialize a client for this specific domain
            client = Socrata(domain, None)
            
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
            
            logger.info(f"Successfully downloaded {len(df)} records from {name}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading dataset {name}: {str(e)}")
            return None
    
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
        
        # Find coordinate columns
        lat_col, lon_col, loc_col = self.find_coordinate_columns(dataset["columns"])
        
        # Make column names lowercase for easier processing
        df.columns = [col.lower() for col in df.columns]
        
        # Standardize column names in case they were changed to lowercase
        if lat_col:
            lat_col = lat_col.lower()
        if lon_col:
            lon_col = lon_col.lower()
        if loc_col:
            loc_col = loc_col.lower()
        
        # Case 1: Direct lat/lon columns
        if lat_col in df.columns and lon_col in df.columns:
            df["latitude"] = df[lat_col]
            df["longitude"] = df[lon_col]
            
        # Case 2: Location object/point column
        elif loc_col in df.columns:
            # Check if it's a string containing lat/lon
            if df[loc_col].dtype == object:
                try:
                    # Try to extract coordinates from a location object like {"latitude": 41.8, "longitude": -87.6}
                    sample = df[loc_col].dropna().iloc[0] if not df[loc_col].dropna().empty else None
                    
                    if sample and isinstance(sample, str) and "{" in sample:
                        # Looks like JSON
                        df["location_obj"] = df[loc_col].apply(lambda x: json.loads(x) if isinstance(x, str) and x else {})
                        df["latitude"] = df["location_obj"].apply(lambda x: float(x.get("latitude", x.get("lat", None))) if x else None)
                        df["longitude"] = df["location_obj"].apply(lambda x: float(x.get("longitude", x.get("lng", x.get("long", None)))) if x else None)
                        df = df.drop("location_obj", axis=1)
                    
                    # Try to extract from coordinates in format "POINT (-87.6 41.8)"
                    elif sample and isinstance(sample, str) and "POINT" in sample.upper():
                        def extract_point(point_str):
                            if not isinstance(point_str, str):
                                return (None, None)
                            try:
                                # Extract numbers from POINT format
                                coords = point_str.upper().replace("POINT", "").replace("(", "").replace(")", "").strip().split()
                                if len(coords) >= 2:
                                    return float(coords[1]), float(coords[0])  # lat, lon
                                return (None, None)
                            except:
                                return (None, None)
                        
                        df[["latitude", "longitude"]] = df[loc_col].apply(extract_point).apply(pd.Series)
                        
                except Exception as e:
                    logger.warning(f"Could not extract coordinates from location column: {e}")
        
        # Filter out rows without coordinates
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
        
        logger.warning(f"Could not identify coordinate columns in dataset")
        return None
    
    def process_datasets(self, max_datasets=5, records_per_dataset=10000):
        """
        Download and process multiple datasets
        
        Args:
            max_datasets (int): Maximum number of datasets to process
            records_per_dataset (int): Maximum records per dataset
            
        Returns:
            pandas.DataFrame: Combined dataset
        """
        if not self.datasets:
            logger.info("No datasets found. Run search_crime_datasets first.")
            return None
        
        all_dfs = []
        processed_count = 0
        
        # Process datasets
        for dataset in tqdm(self.datasets[:max_datasets], desc="Processing datasets"):
            # Download the dataset
            df = self.download_dataset(dataset, limit=records_per_dataset)
            
            if df is not None:
                # Extract coordinates
                df_with_coords = self.extract_coordinates(df, dataset)
                
                if df_with_coords is not None and len(df_with_coords) > 0:
                    all_dfs.append(df_with_coords)
                    processed_count += 1
                    
                    # Save individual dataset
                    city = dataset["domain"].split(".")[1]
                    df_with_coords.to_csv(f"{self.output_dir}/{city}_crime_data.csv", index=False)
                    
            # Add delay to avoid rate limiting
            time.sleep(1)
        
        logger.info(f"Successfully processed {processed_count} datasets with coordinate data")
        
        if all_dfs:
            # Combine all datasets
            combined_df = pd.concat(all_dfs, ignore_index=True)
            
            # Save combined dataset
            timestamp = datetime.now().strftime("%Y%m%d")
            combined_file = f"{self.output_dir}/combined_crime_data_{timestamp}.csv"
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
    
    # Search for datasets
    datasets = scraper.search_crime_datasets(limit=200)
    logger.info(f"Found {len(datasets)} potential datasets")
    
    # Save dataset metadata
    scraper.save_dataset_metadata()
    
    # Process datasets
    combined_data = scraper.process_datasets(max_datasets=10, records_per_dataset=50000)
    
    if combined_data is not None:
        # Print summary
        logger.info("\nSummary of combined dataset:")
        logger.info(f"Total records: {len(combined_data)}")
        logger.info(f"Cities represented: {combined_data['source_city'].nunique()}")
        logger.info(f"Columns: {', '.join(combined_data.columns)}")
        
        # Print sample of the data
        logger.info("\nSample data:")
        logger.info(combined_data.head(5))
    else:
        logger.warning("No data was successfully processed and combined.")

if __name__ == "__main__":
    main()
