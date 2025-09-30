#!/usr/bin/env python3
"""
Socrata Crime Dataset Analyzer for Top 50 US Cities

This script analyzes crime datasets from the top 50 US cities by population,
using the Socrata API to check domains for crime datasets and analyzing their 
column structures to match against Chicago's crime dataset format.

Target columns (from Chicago):
- Latitude/Longitude coordinates
- Date and Time information
- Location descriptions
- Crime type (primary_type)
- Crime description
- Arrest information

Usage:
  python 50_city_scrape.py --output city_crime_datasets_analysis.csv

Requirements:
- sodapy
- pandas
- requests
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sodapy import Socrata
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Top 50 US cities by population with their Socrata domains (corrected)
TOP_50_CITIES = {
    "New York": {"state": "NY", "domain": "data.cityofnewyork.us", "population": 8336817},
    "Los Angeles": {"state": "CA", "domain": "data.lacity.org", "population": 3979576},
    "Chicago": {"state": "IL", "domain": "data.cityofchicago.org", "population": 2693976},
    "Houston": {"state": "TX", "domain": "cohgis-mycity.opendata.arcgis.com", "population": 2320268},  # Fixed
    "Phoenix": {"state": "AZ", "domain": "phoenixopendata.com", "population": 1680992},  # Fixed - remove www
    "Philadelphia": {"state": "PA", "domain": "opendataphilly.org", "population": 1584064},  # Fixed - remove www
    "San Antonio": {"state": "TX", "domain": "data.sanantonio.gov", "population": 1547253},
    "San Diego": {"state": "CA", "domain": "data.sandiego.gov", "population": 1423851},
    "Dallas": {"state": "TX", "domain": "data.dallasopendata.com", "population": 1343573},  # Fixed - remove www
    "San Jose": {"state": "CA", "domain": "data.sanjoseca.gov", "population": 1021795},
    "Austin": {"state": "TX", "domain": "data.austintexas.gov", "population": 978908},
    "Jacksonville": {"state": "FL", "domain": "opendata.coj.net", "population": 911507},  # Fixed
    "Fort Worth": {"state": "TX", "domain": "data.fortworthtexas.gov", "population": 909585},
    "Columbus": {"state": "OH", "domain": "opendata.columbus.gov", "population": 898553},
    "Charlotte": {"state": "NC", "domain": "data.charlottenc.gov", "population": 885708},
    "San Francisco": {"state": "CA", "domain": "data.sfgov.org", "population": 873965},
    "Indianapolis": {"state": "IN", "domain": "data.indy.gov", "population": 867125},
    "Seattle": {"state": "WA", "domain": "data.seattle.gov", "population": 749256},
    "Denver": {"state": "CO", "domain": "data.denvergov.org", "population": 715522},  # Fixed
    "Washington": {"state": "DC", "domain": "opendata.dc.gov", "population": 705749},
    "Boston": {"state": "MA", "domain": "data.boston.gov", "population": 695926},
    "El Paso": {"state": "TX", "domain": "data.elpasotexas.gov", "population": 695044},
    "Nashville": {"state": "TN", "domain": "data.nashville.gov", "population": 689447},
    "Detroit": {"state": "MI", "domain": "data.detroitmi.gov", "population": 670031},
    "Oklahoma City": {"state": "OK", "domain": "data.okc.gov", "population": 695057},
    "Portland": {"state": "OR", "domain": "gis-pdx.opendata.arcgis.com", "population": 652503},  # Fixed
    "Las Vegas": {"state": "NV", "domain": "opendataportal-lasvegas.opendata.arcgis.com", "population": 641903},  # Fixed
    "Memphis": {"state": "TN", "domain": "data.memphistn.gov", "population": 633104},
    "Louisville": {"state": "KY", "domain": "data.louisvilleky.gov", "population": 617638},
    "Baltimore": {"state": "MD", "domain": "data.baltimorecity.gov", "population": 576498},
    "Milwaukee": {"state": "WI", "domain": "data.milwaukee.gov", "population": 577222},
    "Albuquerque": {"state": "NM", "domain": "opendata.cabq.gov", "population": 564559},  # Fixed - remove www
    "Tucson": {"state": "AZ", "domain": "data.tucsonaz.gov", "population": 548073},
    "Fresno": {"state": "CA", "domain": "data.fresno.gov", "population": 542107},
    "Mesa": {"state": "AZ", "domain": "data.mesaaz.gov", "population": 518012},
    "Sacramento": {"state": "CA", "domain": "data.cityofsacramento.org", "population": 513624},
    "Atlanta": {"state": "GA", "domain": "opendata.atlantaga.gov", "population": 506811},
    "Kansas City": {"state": "MO", "domain": "data.kcmo.org", "population": 495327},
    "Colorado Springs": {"state": "CO", "domain": "data.coloradosprings.gov", "population": 478221},
    "Miami": {"state": "FL", "domain": "opendata.miamidade.gov", "population": 467963},
    "Raleigh": {"state": "NC", "domain": "data.raleighnc.gov", "population": 474069},
    "Omaha": {"state": "NE", "domain": "opendata.cityofomaha.org", "population": 486051},
    "Long Beach": {"state": "CA", "domain": "data.longbeach.gov", "population": 466742},
    "Virginia Beach": {"state": "VA", "domain": "data.vbgov.com", "population": 459470},
    "Oakland": {"state": "CA", "domain": "data.oaklandca.gov", "population": 433031},
    "Minneapolis": {"state": "MN", "domain": "opendata.minneapolismn.gov", "population": 429954},
    "Tulsa": {"state": "OK", "domain": "data.cityoftulsa.org", "population": 413066},  # Fixed - remove www
    "Tampa": {"state": "FL", "domain": "opendata.tampagov.net", "population": 399700},
    "Arlington": {"state": "TX", "domain": "data.arlingtontx.gov", "population": 398854},
    "New Orleans": {"state": "LA", "domain": "data.nola.gov", "population": 383997}
}

# Target columns based on Chicago's crime dataset
TARGET_COLUMNS = {
    'coordinates': {
        'keywords': ['latitude', 'lat', 'longitude', 'lon', 'lng', 'coordinates', 'location', 'geocoded', 'x_coordinate', 'y_coordinate'],
        'required_count': 2,  # Need both lat and lon
        'description': 'Geographic coordinates (latitude/longitude)'
    },
    'date': {
        'keywords': ['date', 'occurred', 'incident', 'reported', 'datetime', 'timestamp'],
        'required_count': 1,
        'description': 'Date information'
    },
    'time': {
        'keywords': ['time', 'hour', 'minute', 'datetime', 'timestamp', 'occurred', 'date'],  # Added 'date' since Chicago embeds time in date
        'required_count': 1,
        'description': 'Time information'
    },
    'location_description': {
        'keywords': ['location_description', 'location', 'description', 'premise', 'place', 'address', 'block'],
        'required_count': 1,
        'description': 'Location or premise description'
    },
    'crime_type': {
        'keywords': ['primary_type', 'primary type', 'crime_type', 'crime type', 'offense', 'incident_type', 'incident type', 'classification', 'category'],
        'required_count': 1,
        'description': 'Primary crime type classification'
    },
    'crime_description': {
        'keywords': ['description', 'offense_description', 'incident_description', 'crime_description', 'detail'],
        'required_count': 1,
        'description': 'Detailed crime description'
    },
    'arrest': {
        'keywords': ['arrest', 'arrested', 'custody', 'apprehended', 'cleared'],
        'required_count': 1,
        'description': 'Arrest information'
    }
}


class SocrataCrimeAnalyzer:
    def __init__(self, app_token: str = "MpxAuF8Aa0dpEv3GJnSP8OnoX", timeout: int = 60):
        """Initialize the analyzer using the Socrata API"""
        self.app_token = app_token
        self.timeout = timeout
        self.results = []
        
    def search_crime_datasets_for_domain(self, domain: str, city: str, limit: int = 100) -> List[Dict]:
        """Search for crime-related datasets on a specific Socrata domain using the API"""
        logger.info(f"🔍 Searching {city} ({domain})...")
        
        datasets = []
        
        # First try: Standard Socrata API
        try:
            # Create Socrata client for this domain
            client = Socrata(domain, self.app_token, timeout=self.timeout)
            
            # Get all datasets for this domain
            all_datasets = client.datasets(limit=limit)
            
            # Keywords to search for
            crime_keywords = ['crime', 'police', 'incident', 'offense', 'arrest', 'criminal']
            
            for dataset in all_datasets:
                try:
                    # Get dataset resource info
                    resource = dataset.get("resource", {})
                    dataset_name = resource.get("name", "").lower()
                    dataset_desc = resource.get("description", "").lower()
                    dataset_id = resource.get("id")
                    
                    # Check if dataset is crime-related by name or description
                    is_crime_related = any(keyword in dataset_name or keyword in dataset_desc 
                                        for keyword in crime_keywords)
                    
                    if is_crime_related and dataset_id:
                        logger.info(f"  📊 Found potential crime dataset: {resource.get('name')}")
                        
                        # Get column information by fetching a small sample
                        try:
                            sample_data = client.get(dataset_id, limit=1)
                            
                            columns = []
                            if sample_data and len(sample_data) > 0:
                                columns = list(sample_data[0].keys())
                                
                                # Check for coordinate columns
                                has_coords = any(
                                    coord_term in col.lower()
                                    for col in columns
                                    for coord_term in ["latitude", "longitude", "lat", "lon", "y_coord", "x_coord", 
                                                      "coordinates", "location"]
                                )
                                
                                if has_coords:
                                    dataset_info = {
                                        'city': city,
                                        'domain': domain,
                                        'dataset_id': dataset_id,
                                        'dataset_name': resource.get("name"),
                                        'description': resource.get("description", ""),
                                        'url': f"https://{domain}/d/{dataset_id}",
                                        'columns': columns
                                    }
                                    datasets.append(dataset_info)
                                    logger.info(f"    ✅ Dataset has coordinates: {len(columns)} columns")
                                else:
                                    logger.info(f"    ⚠️  No coordinate columns found")
                            else:
                                logger.info(f"    ⚠️  No sample data available")
                                
                        except Exception as e:
                            logger.debug(f"    ⚠️  Error getting sample data for {dataset_id}: {e}")
                            
                except Exception as e:
                    logger.debug(f"Error processing dataset: {e}")
                    
            client.close()
            
        except Exception as e:
            logger.warning(f"  ❌ Socrata API failed for {domain}: {e}")
            
        # Second try: Alternative API endpoint discovery
        if not datasets:
            datasets = self._try_alternative_discovery(domain, city)
            
        # Third try: Manual known datasets for specific cities
        if not datasets:
            datasets = self._try_known_datasets(domain, city)
        
        logger.info(f"  📊 Found {len(datasets)} crime datasets with coordinates")
        return datasets
    
    def _try_alternative_discovery(self, domain: str, city: str) -> List[Dict]:
        """Try alternative methods to discover datasets"""
        logger.info(f"  🔄 Trying alternative discovery for {city}...")
        datasets = []
        
        try:
            import requests
            
            # Try direct API endpoint
            api_urls = [
                f"https://{domain}/api/catalog/v1?domains={domain}&search_context={domain}&limit=100",
                f"https://{domain}/data.json",
                f"https://{domain}/api/views.json",
            ]
            
            for api_url in api_urls:
                try:
                    response = requests.get(api_url, timeout=30, headers={
                        'User-Agent': 'Mozilla/5.0 (compatible; CrimeDataAnalyzer/1.0)'
                    })
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Handle different API response formats
                        items = []
                        if isinstance(data, dict):
                            if 'results' in data:
                                items = data['results']
                            elif 'dataset' in data:
                                items = data['dataset']
                            else:
                                items = [data]
                        elif isinstance(data, list):
                            items = data
                        
                        for item in items:
                            name = ""
                            description = ""
                            dataset_id = ""
                            
                            # Extract info based on different formats
                            if 'resource' in item:
                                resource = item['resource']
                                name = resource.get('name', '')
                                description = resource.get('description', '')
                                dataset_id = resource.get('id', '')
                            else:
                                name = item.get('title', item.get('name', ''))
                                description = item.get('description', '')
                                dataset_id = item.get('identifier', item.get('id', ''))
                            
                            # Check for crime keywords
                            if any(keyword in name.lower() or keyword in description.lower() 
                                   for keyword in ['crime', 'police', 'incident', 'offense']):
                                
                                dataset_info = {
                                    'city': city,
                                    'domain': domain,
                                    'dataset_id': dataset_id,
                                    'dataset_name': name,
                                    'description': description,
                                    'url': f"https://{domain}/d/{dataset_id}" if dataset_id else f"https://{domain}",
                                    'columns': ['unknown_columns']  # Will be filled if we can access
                                }
                                datasets.append(dataset_info)
                                logger.info(f"    📊 Found via alternative API: {name}")
                                
                        if datasets:
                            break  # Found datasets, stop trying other URLs
                            
                except Exception as e:
                    logger.debug(f"    Alternative API {api_url} failed: {e}")
                    continue
                    
        except Exception as e:
            logger.debug(f"    Alternative discovery failed: {e}")
            
        return datasets
    
    def _try_known_datasets(self, domain: str, city: str) -> List[Dict]:
        """Try known dataset IDs for specific cities"""
        logger.info(f"  🎯 Checking known datasets for {city}...")
        
        # Known crime dataset IDs for specific cities
        known_datasets = {
            "data.seattle.gov": [
                {"id": "tazs-3rd5", "name": "Crime Data"},
                {"id": "y7pv-r3kh", "name": "Police Report Incident"}
            ],
            "data.sfgov.org": [
                {"id": "wg3w-h783", "name": "Police Department Incident Reports"},
                {"id": "cuks-n6tp", "name": "Police Department Incidents"}
            ],
            "data.austintexas.gov": [
                {"id": "fdj4-gpfu", "name": "Crime Reports"},
                {"id": "2uc4-vf5h", "name": "Annual Crime Dataset"}
            ],
            "data.baltimorecity.gov": [
                {"id": "wsfq-mvij", "name": "BPD Part 1 Victim Based Crime Data"}
            ],
            "data.boston.gov": [
                {"id": "fqn4-4qap", "name": "Crime Incident Reports"}
            ],
            "opendata.dc.gov": [
                {"id": "8t4b-c9fg", "name": "Crime Incidents"}
            ],
            "data.nashville.gov": [
                {"id": "2u6v-ujjs", "name": "Metro Nashville Police Department Incidents"}
            ],
            "opendata.minneapolismn.gov": [
                {"id": "jxy6-dtdp", "name": "Police Incidents"}
            ],
            "data.nola.gov": [
                {"id": "5fn8-cu3c", "name": "Electronic Police Report"}
            ]
        }
        
        datasets = []
        
        if domain in known_datasets:
            for known in known_datasets[domain]:
                try:
                    # Try to verify the dataset exists
                    client = Socrata(domain, self.app_token, timeout=15)
                    sample_data = client.get(known["id"], limit=1)
                    
                    if sample_data and len(sample_data) > 0:
                        columns = list(sample_data[0].keys())
                        
                        # Check for coordinates
                        has_coords = any(
                            coord_term in col.lower()
                            for col in columns
                            for coord_term in ["latitude", "longitude", "lat", "lon", "y_coord", "x_coord"]
                        )
                        
                        if has_coords:
                            dataset_info = {
                                'city': city,
                                'domain': domain,
                                'dataset_id': known["id"],
                                'dataset_name': known["name"],
                                'description': "Known dataset",
                                'url': f"https://{domain}/d/{known['id']}",
                                'columns': columns
                            }
                            datasets.append(dataset_info)
                            logger.info(f"    ✅ Verified known dataset: {known['name']}")
                        
                    client.close()
                    
                except Exception as e:
                    logger.debug(f"    Known dataset {known['id']} failed: {e}")
                    
        return datasets
    
    def analyze_column_match(self, columns: List[str]) -> Dict[str, bool]:
        """Analyze if dataset columns match Chicago's target structure"""
        # Normalize column names: lowercase and replace spaces with underscores
        column_names = [col.lower().replace(' ', '_') for col in columns]
        
        matches = {}
        
        for target_name, target_info in TARGET_COLUMNS.items():
            found_matches = 0
            matching_columns = []
            
            for keyword in target_info['keywords']:
                # Normalize keyword too
                normalized_keyword = keyword.lower().replace(' ', '_')
                for col_name in column_names:
                    if normalized_keyword in col_name:
                        found_matches += 1
                        matching_columns.append(col_name)
                        break  # Don't double-count same column
            
            # Special case for coordinates - need both lat and lon
            if target_name == 'coordinates':
                has_lat = any('lat' in col for col in column_names)
                has_lon = any(lon_keyword in col for lon_keyword in ['lon', 'lng', 'longitude'] for col in column_names)
                matches[target_name] = has_lat and has_lon
            else:
                matches[target_name] = found_matches >= target_info['required_count']
        
        return matches
    
    def process_all_cities(self, cities_to_process: Dict = None) -> List[Dict]:
        """Process all cities and analyze their crime datasets"""
        if cities_to_process is None:
            cities_to_process = TOP_50_CITIES
            
        all_results = []
        
        logger.info("🏙️  Analyzing crime datasets from US cities using Socrata API...")
        logger.info("="*60)
        
        for i, (city_name, city_info) in enumerate(cities_to_process.items(), 1):
            logger.info(f"\n🏙️  Processing {i}/{len(cities_to_process)}: {city_name}, {city_info['state']} (Pop: {city_info['population']:,})")
            
            try:
                datasets = self.search_crime_datasets_for_domain(
                    city_info['domain'], 
                    city_name,
                    limit=100
                )
                
                if not datasets:
                    # Record that no datasets were found
                    result = {
                        'city': city_name,
                        'state': city_info['state'],
                        'domain': city_info['domain'],
                        'population': city_info['population'],
                        'dataset_id': 'NO_DATASETS_FOUND',
                        'dataset_name': 'No crime datasets found',
                        'dataset_url': '',
                        'total_columns': 0,
                        'columns_list': '',
                        **{f'has_{target}': False for target in TARGET_COLUMNS.keys()},
                        'match_score': 0.0,
                        'all_columns_match': False
                    }
                    all_results.append(result)
                    continue
                
                for dataset in datasets:
                    logger.info(f"  📋 Analyzing: {dataset['dataset_name']}")
                    
                    # Analyze column matches
                    matches = self.analyze_column_match(dataset['columns'])
                    
                    # Calculate match score
                    match_score = sum(matches.values()) / len(TARGET_COLUMNS) * 100
                    all_columns_match = all(matches.values())
                    
                    # Prepare column list
                    columns_list = ' | '.join(dataset['columns']) if dataset['columns'] else 'No columns found'
                    
                    result = {
                        'city': city_name,
                        'state': city_info['state'],
                        'domain': city_info['domain'],
                        'population': city_info['population'],
                        'dataset_id': dataset['dataset_id'],
                        'dataset_name': dataset['dataset_name'],
                        'dataset_url': dataset['url'],
                        'total_columns': len(dataset['columns']),
                        'columns_list': columns_list,
                        **{f'has_{target}': matches.get(target, False) for target in TARGET_COLUMNS.keys()},
                        'match_score': match_score,
                        'all_columns_match': all_columns_match
                    }
                    
                    all_results.append(result)
                    
                    logger.info(f"    ✅ Match Score: {match_score:.1f}% ({len(dataset['columns'])} columns)")
                    
            except Exception as e:
                logger.error(f"  ❌ Error processing {city_name}: {e}")
                # Add error record
                result = {
                    'city': city_name,
                    'state': city_info['state'],
                    'domain': city_info['domain'],
                    'population': city_info['population'],
                    'dataset_id': 'ERROR',
                    'dataset_name': f'Error: {str(e)}',
                    'dataset_url': '',
                    'total_columns': 0,
                    'columns_list': '',
                    **{f'has_{target}': False for target in TARGET_COLUMNS.keys()},
                    'match_score': 0.0,
                    'all_columns_match': False
                }
                all_results.append(result)
            
            # Add small delay between cities to be respectful
            time.sleep(1)
        
        return all_results
    
    def export_results(self, results: List[Dict], output_file: str):
        """Export results to CSV"""
        logger.info(f"\n📊 Exporting results to {output_file}")
        
        if not results:
            logger.warning("No results to export!")
            return
        
        # Define column order
        columns = [
            'city', 'state', 'domain', 'population', 'dataset_name', 'dataset_id', 
            'dataset_url', 'total_columns', 'match_score', 'all_columns_match'
        ] + [f'has_{target}' for target in TARGET_COLUMNS.keys()] + ['columns_list']
        
        df = pd.DataFrame(results)
        df = df.reindex(columns=columns)
        df = df.sort_values(['match_score', 'population'], ascending=[False, False])
        
        # Export to CSV
        df.to_csv(output_file, index=False)
        
        # Print summary
        logger.info(f"\n📈 SUMMARY STATISTICS")
        logger.info("="*50)
        logger.info(f"Total cities analyzed: {df['city'].nunique()}")
        logger.info(f"Total datasets found: {len(df[df['dataset_id'] != 'NO_DATASETS_FOUND'])}")
        logger.info(f"Perfect matches (100%): {len(df[df['all_columns_match'] == True])}")
        logger.info(f"High matches (>80%): {len(df[df['match_score'] > 80])}")
        logger.info(f"Medium matches (50-80%): {len(df[(df['match_score'] >= 50) & (df['match_score'] <= 80)])}")
        logger.info(f"Low matches (<50%): {len(df[df['match_score'] < 50])}")
        
        # Show top matches
        logger.info(f"\n🏆 TOP MATCHING DATASETS:")
        top_matches = df.head(10)[['city', 'dataset_name', 'match_score', 'total_columns']]
        for _, row in top_matches.iterrows():
            if row['dataset_name'] != 'No crime datasets found':
                logger.info(f"  {row['city']}: {row['dataset_name']} ({row['match_score']:.1f}%, {row['total_columns']} cols)")
        
        logger.info(f"\nFull results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Socrata crime datasets from top 50 US cities")
    parser.add_argument("--output", "-o", default="city_crime_datasets_analysis.csv", 
                       help="Output CSV file path")
    parser.add_argument("--app-token", default="MpxAuF8Aa0dpEv3GJnSP8OnoX",
                       help="Socrata API app token")
    parser.add_argument("--cities", nargs="+", 
                       help="Specific cities to analyze (default: all 50)")
    parser.add_argument("--test", action="store_true",
                       help="Test with first 3 cities only")
    parser.add_argument("--limit", type=int, default=100,
                       help="Maximum datasets to check per city")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--retry-failed", action="store_true",
                       help="Focus only on cities that previously failed")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
    # Initialize analyzer
    analyzer = SocrataCrimeAnalyzer(app_token=args.app_token)
    
    # Cities that previously failed (based on your results)
    failed_cities = [
        "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "San Jose", "Austin",
        "Jacksonville", "Fort Worth", "Columbus", "Charlotte", "San Francisco", "Indianapolis",
        "Seattle", "Denver", "Washington", "Boston", "Oklahoma City", "El Paso", "Nashville",
        "Detroit", "Portland", "Las Vegas", "Louisville", "Milwaukee", "Baltimore", "Albuquerque",
        "Tucson", "Fresno", "Mesa", "Sacramento", "Atlanta", "Omaha", "Colorado Springs",
        "Raleigh", "Miami", "Long Beach", "Virginia Beach", "Minneapolis", "Tulsa", "Tampa", "Arlington"
    ]
    
    # Filter cities if specified
    if args.cities:
        cities_to_process = {k: v for k, v in TOP_50_CITIES.items() if k in args.cities}
    elif args.test:
        cities_to_process = dict(list(TOP_50_CITIES.items())[:3])
    elif args.retry_failed:
        cities_to_process = {k: v for k, v in TOP_50_CITIES.items() if k in failed_cities}
        logger.info(f"Retrying {len(cities_to_process)} previously failed cities")
    else:
        cities_to_process = TOP_50_CITIES
    
    logger.info(f"🚀 Starting analysis of {len(cities_to_process)} cities...")
    logger.info(f"Target columns: {', '.join(TARGET_COLUMNS.keys())}")
    logger.info(f"Using Socrata API with app token: {args.app_token[:8]}...")
    
    try:
        results = analyzer.process_all_cities(cities_to_process)
        analyzer.export_results(results, args.output)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Analysis interrupted by user")
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        raise


if __name__ == "__main__":
    main()