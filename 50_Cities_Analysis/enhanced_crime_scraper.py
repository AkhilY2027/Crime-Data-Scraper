#!/usr/bin/env python3
"""
Enhanced Crime Dataset Scraper with Multi-API Support

This script improves upon the original scraper by:
1. Supporting multiple data portal types (Socrata, ArcGIS, CKAN, OpenDataSoft)
2. Adding FBI UCR API for standardized crime statistics
3. Using fuzzy matching for better column detection
4. Implementing parallel processing and caching
5. Adding data quality scoring and standardization
"""

import argparse
import json
import os
import re
import sys
import time
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sodapy import Socrata
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
from tqdm import tqdm

# For fuzzy matching
try:
    from fuzzywuzzy import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    print("Installing fuzzywuzzy for better column matching...")
    os.system("pip install fuzzywuzzy python-Levenshtein")
    from fuzzywuzzy import fuzz
    FUZZY_AVAILABLE = True

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = "crime_data_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Enhanced city configuration with multiple portal types (Top 50 US Cities)
ENHANCED_CITY_CONFIG = {
    "New York": {
        "state": "NY",
        "population": 8336817,
        "portals": {
            "socrata": "data.cityofnewyork.us",
            "fbi_ori": "NY0303000"  # FBI Originating Agency Identifier
        }
    },
    "Los Angeles": {
        "state": "CA",
        "population": 3979576,
        "portals": {
            "socrata": "data.lacity.org",
            "arcgis": "geohub.lacity.org",
            "fbi_ori": "CA0194200"
        }
    },
    "Chicago": {
        "state": "IL",
        "population": 2693976,
        "portals": {
            "socrata": "data.cityofchicago.org",
            "fbi_ori": "IL0163000"
        }
    },
    "Houston": {
        "state": "TX",
        "population": 2320268,
        "portals": {
            "arcgis": "cohgis-mycity.opendata.arcgis.com",
            "fbi_ori": "TX1010000"
        }
    },
    "Phoenix": {
        "state": "AZ",
        "population": 1680992,
        "portals": {
            "arcgis": "phoenixopendata.com",
            "socrata": "data.phoenix.gov",
            "fbi_ori": "AZ0070700"
        }
    },
    "Philadelphia": {
        "state": "PA",
        "population": 1584064,
        "portals": {
            "ckan": "opendataphilly.org",
            "arcgis": "data-phl.opendata.arcgis.com",
            "fbi_ori": "PAPEP0000"
        }
    },
    "San Antonio": {
        "state": "TX",
        "population": 1547253,
        "portals": {
            "socrata": "data.sanantonio.gov",
            "fbi_ori": "TX0150000"
        }
    },
    "San Diego": {
        "state": "CA",
        "population": 1423851,
        "portals": {
            "socrata": "data.sandiego.gov",
            "arcgis": "sdgis-sandag.opendata.arcgis.com",
            "fbi_ori": "CA0374100"
        }
    },
    "Dallas": {
        "state": "TX",
        "population": 1343573,
        "portals": {
            "socrata": "www.dallasopendata.com",
            "arcgis": "data.dallasopendata.com",
            "fbi_ori": "TX0570000"
        }
    },
    "San Jose": {
        "state": "CA",
        "population": 1021795,
        "portals": {
            "socrata": "data.sanjoseca.gov",
            "fbi_ori": "CA0434600"
        }
    },
    "Austin": {
        "state": "TX",
        "population": 978908,
        "portals": {
            "socrata": "data.austintexas.gov",
            "fbi_ori": "TX0150100"
        }
    },
    "Jacksonville": {
        "state": "FL",
        "population": 911507,
        "portals": {
            "arcgis": "opendata.coj.net",
            "socrata": "data.coj.net",
            "fbi_ori": "FL0160000"
        }
    },
    "Fort Worth": {
        "state": "TX",
        "population": 909585,
        "portals": {
            "socrata": "data.fortworthtexas.gov",
            "fbi_ori": "TX0220000"
        }
    },
    "Columbus": {
        "state": "OH",
        "population": 898553,
        "portals": {
            "socrata": "opendata.columbus.gov",
            "arcgis": "data-columbus.opendata.arcgis.com",
            "fbi_ori": "OH0250100"
        }
    },
    "Charlotte": {
        "state": "NC",
        "population": 885708,
        "portals": {
            "socrata": "data.charlottenc.gov",
            "arcgis": "data.charlottenc.gov",
            "fbi_ori": "NC0600100"
        }
    },
    "San Francisco": {
        "state": "CA",
        "population": 873965,
        "portals": {
            "socrata": "data.sfgov.org",
            "fbi_ori": "CA0380000"
        }
    },
    "Indianapolis": {
        "state": "IN",
        "population": 867125,
        "portals": {
            "socrata": "data.indy.gov",
            "arcgis": "data-indianapolis.opendata.arcgis.com",
            "fbi_ori": "IN0490100"
        }
    },
    "Seattle": {
        "state": "WA",
        "population": 749256,
        "portals": {
            "socrata": "data.seattle.gov",
            "fbi_ori": "WA0170100"
        }
    },
    "Denver": {
        "state": "CO",
        "population": 715522,
        "portals": {
            "socrata": "data.denvergov.org",
            "arcgis": "data-denver.opendata.arcgis.com",
            "fbi_ori": "CO0161000"
        }
    },
    "Washington": {
        "state": "DC",
        "population": 705749,
        "portals": {
            "socrata": "opendata.dc.gov",
            "arcgis": "opendata.dc.gov",
            "fbi_ori": "DC0010100"
        }
    },
    "Boston": {
        "state": "MA",
        "population": 695926,
        "portals": {
            "socrata": "data.boston.gov",
            "arcgis": "data.boston.gov",
            "fbi_ori": "MA0130100"
        }
    },
    "El Paso": {
        "state": "TX",
        "population": 695044,
        "portals": {
            "socrata": "data.elpasotexas.gov",
            "fbi_ori": "TX0710000"
        }
    },
    "Nashville": {
        "state": "TN",
        "population": 689447,
        "portals": {
            "socrata": "data.nashville.gov",
            "fbi_ori": "TN0190100"
        }
    },
    "Detroit": {
        "state": "MI",
        "population": 670031,
        "portals": {
            "socrata": "data.detroitmi.gov",
            "fbi_ori": "MI8234900"
        }
    },
    "Oklahoma City": {
        "state": "OK",
        "population": 695057,
        "portals": {
            "socrata": "data.okc.gov",
            "arcgis": "data.okc.gov",
            "fbi_ori": "OK0550200"
        }
    },
    "Portland": {
        "state": "OR",
        "population": 652503,
        "portals": {
            "arcgis": "gis-pdx.opendata.arcgis.com",
            "socrata": "data.portlandoregon.gov",
            "fbi_ori": "OR0260000"
        }
    },
    "Las Vegas": {
        "state": "NV",
        "population": 641903,
        "portals": {
            "arcgis": "opendataportal-lasvegas.opendata.arcgis.com",
            "socrata": "opendata.lasvegasnevada.gov",
            "fbi_ori": "NV0020100"
        }
    },
    "Memphis": {
        "state": "TN",
        "population": 633104,
        "portals": {
            "socrata": "data.memphistn.gov",
            "fbi_ori": "TN0790100"
        }
    },
    "Louisville": {
        "state": "KY",
        "population": 617638,
        "portals": {
            "socrata": "data.louisvilleky.gov",
            "arcgis": "data-louisville.opendata.arcgis.com",
            "fbi_ori": "KY0561000"
        }
    },
    "Baltimore": {
        "state": "MD",
        "population": 576498,
        "portals": {
            "socrata": "data.baltimorecity.gov",
            "fbi_ori": "MD0030000"
        }
    },
    "Milwaukee": {
        "state": "WI",
        "population": 577222,
        "portals": {
            "socrata": "data.milwaukee.gov",
            "arcgis": "data.milwaukee.gov",
            "fbi_ori": "WI0400100"
        }
    },
    "Albuquerque": {
        "state": "NM",
        "population": 564559,
        "portals": {
            "socrata": "opendata.cabq.gov",
            "arcgis": "coagis.maps.arcgis.com",
            "fbi_ori": "NM0020100"
        }
    },
    "Tucson": {
        "state": "AZ",
        "population": 548073,
        "portals": {
            "socrata": "data.tucsonaz.gov",
            "arcgis": "gisdata.tucsonaz.gov",
            "fbi_ori": "AZ0100200"
        }
    },
    "Fresno": {
        "state": "CA",
        "population": 542107,
        "portals": {
            "socrata": "data.fresno.gov",
            "fbi_ori": "CA0100900"
        }
    },
    "Mesa": {
        "state": "AZ",
        "population": 518012,
        "portals": {
            "socrata": "data.mesaaz.gov",
            "arcgis": "open-data-cityofmesa.opendata.arcgis.com",
            "fbi_ori": "AZ0070300"
        }
    },
    "Sacramento": {
        "state": "CA",
        "population": 513624,
        "portals": {
            "socrata": "data.cityofsacramento.org",
            "arcgis": "data-cityofsacramento.opendata.arcgis.com",
            "fbi_ori": "CA0340700"
        }
    },
    "Atlanta": {
        "state": "GA",
        "population": 506811,
        "portals": {
            "socrata": "opendata.atlantaga.gov",
            "arcgis": "opendata.atlantaga.gov",
            "fbi_ori": "GA0600100"
        }
    },
    "Kansas City": {
        "state": "MO",
        "population": 495327,
        "portals": {
            "socrata": "data.kcmo.org",
            "arcgis": "data.kcmo.org",
            "fbi_ori": "MO0460000"
        }
    },
    "Colorado Springs": {
        "state": "CO",
        "population": 478221,
        "portals": {
            "socrata": "data.coloradosprings.gov",
            "arcgis": "data-coloradosprings.opendata.arcgis.com",
            "fbi_ori": "CO0210100"
        }
    },
    "Raleigh": {
        "state": "NC",
        "population": 474069,
        "portals": {
            "socrata": "data.raleighnc.gov",
            "arcgis": "data-raleighnc.opendata.arcgis.com",
            "fbi_ori": "NC0920100"
        }
    },
    "Miami": {
        "state": "FL",
        "population": 467963,
        "portals": {
            "socrata": "opendata.miamidade.gov",
            "arcgis": "gis-miami.opendata.arcgis.com",
            "fbi_ori": "FL0130400"
        }
    },
    "Omaha": {
        "state": "NE",
        "population": 486051,
        "portals": {
            "socrata": "opendata.cityofomaha.org",
            "arcgis": "data-omaha.opendata.arcgis.com",
            "fbi_ori": "NE0280200"
        }
    },
    "Long Beach": {
        "state": "CA",
        "population": 466742,
        "portals": {
            "socrata": "data.longbeach.gov",
            "arcgis": "data-longbeach.opendata.arcgis.com",
            "fbi_ori": "CA0194000"
        }
    },
    "Virginia Beach": {
        "state": "VA",
        "population": 459470,
        "portals": {
            "socrata": "data.vbgov.com",
            "arcgis": "data.virginiabeach.gov",
            "fbi_ori": "VA0810100"
        }
    },
    "Oakland": {
        "state": "CA",
        "population": 433031,
        "portals": {
            "socrata": "data.oaklandca.gov",
            "arcgis": "oakland-opendata.opendata.arcgis.com",
            "fbi_ori": "CA0010100"
        }
    },
    "Minneapolis": {
        "state": "MN",
        "population": 429954,
        "portals": {
            "socrata": "opendata.minneapolismn.gov",
            "arcgis": "opendata.minneapolismn.gov",
            "fbi_ori": "MN0271900"
        }
    },
    "Tulsa": {
        "state": "OK",
        "population": 413066,
        "portals": {
            "socrata": "data.cityoftulsa.org",
            "arcgis": "opendata-tulsa.opendata.arcgis.com",
            "fbi_ori": "OK0720100"
        }
    },
    "Tampa": {
        "state": "FL",
        "population": 399700,
        "portals": {
            "socrata": "opendata.tampagov.net",
            "arcgis": "city-tampa.opendata.arcgis.com",
            "fbi_ori": "FL0290300"
        }
    },
    "Arlington": {
        "state": "TX",
        "population": 398854,
        "portals": {
            "socrata": "data.arlingtontx.gov",
            "fbi_ori": "TX0220200"
        }
    },
    "New Orleans": {
        "state": "LA",
        "population": 383997,
        "portals": {
            "socrata": "data.nola.gov",
            "arcgis": "data.nola.gov",
            "fbi_ori": "LA0710100"
        }
    }
}

# Column mapping patterns with fuzzy matching thresholds
COLUMN_PATTERNS = {
    'latitude': {
        'exact': ['latitude', 'lat', 'y', 'point_y', 'y_coordinate', 'y_coord'],
        'contains': ['lat', 'latitude', 'y_coord'],
        'fuzzy_threshold': 80,
        'type_check': lambda x: -90 <= float(x) <= 90 if x else False
    },
    'longitude': {
        'exact': ['longitude', 'lon', 'lng', 'long', 'x', 'point_x', 'x_coordinate', 'x_coord'],
        'contains': ['lon', 'lng', 'longitude', 'x_coord'],
        'fuzzy_threshold': 80,
        'type_check': lambda x: -180 <= float(x) <= 180 if x else False
    },
    'date': {
        'exact': ['date', 'occurred_date', 'incident_date', 'report_date', 'datetime'],
        'contains': ['date', 'occurred', 'incident'],
        'fuzzy_threshold': 75,
        'type_check': lambda x: bool(pd.to_datetime(x, errors='coerce'))
    },
    'time': {
        'exact': ['time', 'occurred_time', 'incident_time', 'hour'],
        'contains': ['time', 'hour'],
        'fuzzy_threshold': 75
    },
    'crime_type': {
        'exact': ['crime_type', 'primary_type', 'offense', 'category', 'crime_class'],
        'contains': ['type', 'offense', 'crime', 'category'],
        'fuzzy_threshold': 70
    },
    'description': {
        'exact': ['description', 'narrative', 'details', 'offense_description'],
        'contains': ['desc', 'narr', 'detail'],
        'fuzzy_threshold': 70
    },
    'location': {
        'exact': ['location', 'address', 'block', 'street', 'location_description'],
        'contains': ['location', 'address', 'street'],
        'fuzzy_threshold': 70
    },
    'arrest': {
        'exact': ['arrest', 'arrested', 'arrest_made', 'clearance'],
        'contains': ['arrest'],
        'fuzzy_threshold': 85
    }
}


class RequestsSession:
    """Enhanced requests session with retry logic and caching"""

    def __init__(self, cache_hours=24):
        self.session = requests.Session()
        self.cache_hours = cache_hours

        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Headers to avoid blocking
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def cached_get(self, url: str, params: dict = None) -> Optional[dict]:
        """Get with caching support"""
        # Create cache key
        cache_key = hashlib.md5(f"{url}{params}".encode()).hexdigest()
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")

        # Check cache
        if os.path.exists(cache_file):
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - file_time < timedelta(hours=self.cache_hours):
                try:
                    with open(cache_file, 'rb') as f:
                        logger.debug(f"Using cached response for {url}")
                        return pickle.load(f)
                except:
                    pass

        # Make request
        try:
            response = self.session.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()

                # Save to cache
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)

                return data
        except Exception as e:
            logger.error(f"Request failed for {url}: {e}")
            return None


class EnhancedCrimeScraper:
    """Enhanced crime data scraper with multi-API support"""

    def __init__(self, app_token: str = "MpxAuF8Aa0dpEv3GJnSP8OnoX"):
        self.app_token = app_token
        self.session = RequestsSession()
        self.results = []

    def detect_columns(self, columns: List[str], sample_data: List[dict] = None) -> Dict[str, str]:
        """
        Intelligently detect column types using multiple strategies

        Returns mapping of standard names to actual column names
        """
        column_mapping = {}
        columns_lower = {col.lower(): col for col in columns}

        for std_name, patterns in COLUMN_PATTERNS.items():
            matched_col = None
            best_score = 0

            # 1. Try exact matches
            for pattern in patterns['exact']:
                if pattern in columns_lower:
                    matched_col = columns_lower[pattern]
                    break

            # 2. Try contains matches
            if not matched_col:
                for col_lower, col_actual in columns_lower.items():
                    for pattern in patterns['contains']:
                        if pattern in col_lower:
                            matched_col = col_actual
                            break
                    if matched_col:
                        break

            # 3. Try fuzzy matching
            if not matched_col and FUZZY_AVAILABLE:
                for col_actual in columns:
                    for pattern in patterns['exact']:
                        score = fuzz.ratio(pattern, col_actual.lower())
                        if score > patterns['fuzzy_threshold'] and score > best_score:
                            matched_col = col_actual
                            best_score = score

            # 4. Try type checking with sample data
            if not matched_col and sample_data and 'type_check' in patterns:
                for col in columns:
                    try:
                        # Check first few non-null values
                        values = [row.get(col) for row in sample_data[:10] if row.get(col)]
                        if values and all(patterns['type_check'](v) for v in values[:3]):
                            matched_col = col
                            break
                    except:
                        continue

            if matched_col:
                column_mapping[std_name] = matched_col
                logger.debug(f"Mapped {std_name} -> {matched_col}")

        # 5. Post-processing: Check if datetime columns contain time information
        if 'date' in column_mapping and 'time' not in column_mapping and sample_data:
            date_col = column_mapping['date']
            try:
                # Check if the date column actually contains time information
                sample_values = [row.get(date_col) for row in sample_data[:10] if row.get(date_col)]
                if sample_values:
                    for value in sample_values[:5]:  # Check first 5 non-null values
                        try:
                            parsed_dt = pd.to_datetime(value, errors='coerce')
                            if pd.notna(parsed_dt):
                                # Check if time component is present (not just 00:00:00)
                                if parsed_dt.hour != 0 or parsed_dt.minute != 0 or parsed_dt.second != 0:
                                    column_mapping['time'] = date_col
                                    logger.debug(f"Detected embedded time in date column: {date_col}")
                                    break
                                # Also check if the string representation suggests time is present
                                elif ' ' in str(value) or 'T' in str(value) or len(str(value)) > 10:
                                    column_mapping['time'] = date_col
                                    logger.debug(f"Detected datetime format in date column: {date_col}")
                                    break
                        except:
                            continue
            except Exception as e:
                logger.debug(f"Error checking datetime column for embedded time: {e}")

        return column_mapping

    def get_socrata_datasets(self, domain: str, city: str) -> List[Dict]:
        """Get crime datasets from Socrata portal"""
        logger.info(f"Checking Socrata portal for {city}: {domain}")
        datasets = []

        try:
            client = Socrata(domain, self.app_token, timeout=30)

            # Search for crime-related datasets
            search_terms = ['crime', 'police', 'incident', 'offense', 'arrest']

            for term in search_terms:
                try:
                    results = client.datasets(q=term, limit=50)

                    for dataset in results:
                        resource = dataset.get('resource', {})
                        dataset_id = resource.get('id')
                        name = resource.get('name', '')

                        if dataset_id and any(kw in name.lower() for kw in ['crime', 'incident', 'police']):
                            # Get sample data to check columns
                            try:
                                sample = client.get(dataset_id, limit=5)
                                if sample:
                                    columns = list(sample[0].keys())
                                    mapping = self.detect_columns(columns, sample)

                                    # Check if has coordinates
                                    if 'latitude' in mapping and 'longitude' in mapping:
                                        datasets.append({
                                            'city': city,
                                            'source': 'socrata',
                                            'domain': domain,
                                            'dataset_id': dataset_id,
                                            'name': name,
                                            'url': f"https://{domain}/d/{dataset_id}",
                                            'columns': columns,
                                            'column_mapping': mapping,
                                            'quality_score': len(mapping) / len(COLUMN_PATTERNS) * 100
                                        })
                                        logger.info(f"  Found: {name} (Quality: {datasets[-1]['quality_score']:.1f}%)")
                            except:
                                continue
                except:
                    continue

            client.close()

        except Exception as e:
            logger.error(f"Socrata API error for {domain}: {e}")

        return datasets

    def get_arcgis_datasets(self, domain: str, city: str) -> List[Dict]:
        """Get crime datasets from ArcGIS Open Data portal"""
        logger.info(f"Checking ArcGIS portal for {city}: {domain}")
        datasets = []

        base_url = f"https://{domain}/api/v2/datasets"
        params = {
            'q': 'crime OR police OR incident',
            'filter[tags]': 'crime,police,public safety',
            'page[size]': 50
        }

        data = self.session.cached_get(base_url, params)

        if data and 'data' in data:
            for item in data['data']:
                attributes = item.get('attributes', {})
                name = attributes.get('name', '')

                if any(kw in name.lower() for kw in ['crime', 'incident', 'police']):
                    # Check for geographic data
                    if attributes.get('spatialReference') or 'point' in str(attributes.get('geometryType', '')).lower():
                        dataset_id = item.get('id')

                        # Try to get field information
                        fields_url = f"https://{domain}/api/v2/datasets/{dataset_id}/fields"
                        fields_data = self.session.cached_get(fields_url)

                        columns = []
                        if fields_data and 'data' in fields_data:
                            columns = [f['attributes']['name'] for f in fields_data['data']
                                      if 'attributes' in f and 'name' in f['attributes']]

                        mapping = self.detect_columns(columns) if columns else {}

                        datasets.append({
                            'city': city,
                            'source': 'arcgis',
                            'domain': domain,
                            'dataset_id': dataset_id,
                            'name': name,
                            'url': f"https://{domain}/datasets/{dataset_id}",
                            'columns': columns,
                            'column_mapping': mapping,
                            'quality_score': len(mapping) / len(COLUMN_PATTERNS) * 100 if columns else 0
                        })
                        logger.info(f"  Found: {name} (Quality: {datasets[-1]['quality_score']:.1f}%)")

        return datasets

    def get_fbi_ucr_data(self, ori: str, city: str, state: str) -> List[Dict]:
        """Get FBI UCR (Uniform Crime Reporting) data"""
        logger.info(f"Checking FBI UCR data for {city}, {state} (ORI: {ori})")
        datasets = []

        # FBI Crime Data Explorer API
        base_url = "https://api.usa.gov/crime/fbi/cde"

        # Get summarized data
        endpoints = [
            f"/summarized/agencies/{ori}/offenses",
            f"/incidents/agencies/{ori}/offenses"
        ]

        for endpoint in endpoints:
            url = base_url + endpoint
            params = {'API_KEY': 'your_fbi_api_key'}  # Note: You'll need to register for an API key

            data = self.session.cached_get(url, params)

            if data:
                dataset_name = "FBI UCR " + ("Summary" if "summarized" in endpoint else "Incident")

                datasets.append({
                    'city': city,
                    'source': 'fbi_ucr',
                    'domain': 'api.usa.gov',
                    'dataset_id': ori,
                    'name': f"{dataset_name} Data - {city}",
                    'url': url,
                    'columns': list(data.keys()) if isinstance(data, dict) else [],
                    'column_mapping': {},
                    'quality_score': 100  # FBI data is standardized
                })
                logger.info(f"  Found: FBI UCR data")

        return datasets

    def get_direct_csv_sources(self, city: str) -> List[Dict]:
        """Get known direct CSV download sources"""
        logger.info(f"Checking direct CSV sources for {city}")
        datasets = []

        # Known direct download URLs (you can expand this)
        known_sources = {
            "Seattle": [
                {
                    "url": "https://data.seattle.gov/api/views/tazs-3rd5/rows.csv",
                    "name": "Seattle Police Department Police Report Incident",
                    "dataset_id": "tazs-3rd5"
                }
            ],
            "Boston": [
                {
                    "url": "https://data.boston.gov/dataset/crime-incident-reports-august-2015-to-date-source-new-system/resource/12cb3883-56f5-47de-afa5-3b1cf61b257b/download/crime-incident-reports.csv",
                    "name": "Boston Crime Incident Reports",
                    "dataset_id": "crime-incidents"
                }
            ]
        }

        if city in known_sources:
            for source in known_sources[city]:
                # Try to get headers to detect columns
                try:
                    df_sample = pd.read_csv(source['url'], nrows=5)
                    columns = df_sample.columns.tolist()
                    mapping = self.detect_columns(columns, df_sample.to_dict('records'))

                    datasets.append({
                        'city': city,
                        'source': 'direct_csv',
                        'domain': 'direct',
                        'dataset_id': source['dataset_id'],
                        'name': source['name'],
                        'url': source['url'],
                        'columns': columns,
                        'column_mapping': mapping,
                        'quality_score': len(mapping) / len(COLUMN_PATTERNS) * 100
                    })
                    logger.info(f"  Found: {source['name']} (Direct CSV)")
                except:
                    pass

        return datasets

    def process_city(self, city: str, config: Dict) -> List[Dict]:
        """Process a single city using all available methods"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {city}, {config['state']} (Pop: {config['population']:,})")
        logger.info(f"{'='*60}")

        all_datasets = []

        # Try each portal type
        portals = config.get('portals', {})

        # Socrata
        if 'socrata' in portals:
            datasets = self.get_socrata_datasets(portals['socrata'], city)
            all_datasets.extend(datasets)

        # ArcGIS
        if 'arcgis' in portals:
            datasets = self.get_arcgis_datasets(portals['arcgis'], city)
            all_datasets.extend(datasets)

        # FBI UCR
        if 'fbi_ori' in portals:
            datasets = self.get_fbi_ucr_data(portals['fbi_ori'], city, config['state'])
            all_datasets.extend(datasets)

        # Direct CSV sources
        datasets = self.get_direct_csv_sources(city)
        all_datasets.extend(datasets)

        # Sort by quality score
        all_datasets.sort(key=lambda x: x['quality_score'], reverse=True)

        logger.info(f"Total datasets found for {city}: {len(all_datasets)}")
        if all_datasets:
            logger.info(f"Best quality score: {all_datasets[0]['quality_score']:.1f}%")

        return all_datasets

    def process_all_cities(self, cities: Dict = None, parallel: bool = True) -> pd.DataFrame:
        """Process all cities with optional parallel processing"""
        if cities is None:
            cities = ENHANCED_CITY_CONFIG

        all_results = []

        if parallel:
            logger.info(f"Processing {len(cities)} cities in parallel...")
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_city = {
                    executor.submit(self.process_city, city, config): city
                    for city, config in cities.items()
                }

                for future in tqdm(as_completed(future_to_city), total=len(cities), desc="Cities"):
                    city = future_to_city[future]
                    try:
                        datasets = future.result()
                        for dataset in datasets:
                            dataset['state'] = cities[city]['state']
                            dataset['population'] = cities[city]['population']
                            all_results.append(dataset)
                    except Exception as e:
                        logger.error(f"Error processing {city}: {e}")
        else:
            for city, config in tqdm(cities.items(), desc="Cities"):
                try:
                    datasets = self.process_city(city, config)
                    for dataset in datasets:
                        dataset['state'] = config['state']
                        dataset['population'] = config['population']
                        all_results.append(dataset)
                except Exception as e:
                    logger.error(f"Error processing {city}: {e}")

        # Convert to DataFrame
        df = pd.DataFrame(all_results)

        # Add additional metrics
        if not df.empty:
            df['has_coordinates'] = df['column_mapping'].apply(
                lambda x: 'latitude' in x and 'longitude' in x
            )
            df['has_temporal'] = df['column_mapping'].apply(
                lambda x: 'date' in x or 'time' in x
            )
            df['has_crime_info'] = df['column_mapping'].apply(
                lambda x: 'crime_type' in x or 'description' in x
            )
            df['completeness_score'] = df['column_mapping'].apply(
                lambda x: len(x) / len(COLUMN_PATTERNS) * 100
            )

        return df

    def export_results(self, df: pd.DataFrame, output_file: str):
        """Export results with summary statistics"""
        logger.info(f"\nExporting results to {output_file}")

        # Sort by quality score and city population
        df_sorted = df.sort_values(['quality_score', 'population'], ascending=[False, False])

        # Export full results
        df_sorted.to_csv(output_file, index=False)

        # Create city summary
        city_summary = df.groupby(['city', 'state']).agg({
            'dataset_id': 'count',
            'quality_score': ['mean', 'max'],
            'has_coordinates': 'sum',
            'has_temporal': 'sum',
            'has_crime_info': 'sum'
        }).round(1)

        city_summary.columns = ['datasets_found', 'avg_quality', 'best_quality',
                                'with_coords', 'with_time', 'with_crime_info']
        city_summary = city_summary.sort_values('avg_quality', ascending=False)

        summary_file = output_file.replace('.csv', '_summary.csv')
        city_summary.to_csv(summary_file)

        # Print statistics
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Total cities analyzed: {df['city'].nunique()}")
        logger.info(f"Total datasets found: {len(df)}")
        logger.info(f"Datasets with coordinates: {df['has_coordinates'].sum()}")
        logger.info(f"Average quality score: {df['quality_score'].mean():.1f}%")

        # Top cities
        logger.info(f"\nTOP CITIES BY DATA QUALITY:")
        for idx, (city_state, row) in enumerate(city_summary.head(10).iterrows(), 1):
            city, state = city_state
            logger.info(f"  {idx}. {city}, {state}: {row['avg_quality']:.1f}% avg, "
                       f"{row['datasets_found']} datasets")

        # Source distribution
        logger.info(f"\nDATA SOURCE DISTRIBUTION:")
        source_counts = df['source'].value_counts()
        for source, count in source_counts.items():
            logger.info(f"  {source}: {count} datasets ({count/len(df)*100:.1f}%)")

        logger.info(f"\nFull results: {output_file}")
        logger.info(f"City summary: {summary_file}")

        return city_summary


def main():
    parser = argparse.ArgumentParser(description="Enhanced Crime Data Scraper with Multi-API Support")
    parser.add_argument("--output", "-o", default="output/enhanced_crime_datasets.csv",
                       help="Output CSV file path")
    parser.add_argument("--cities", nargs="+",
                       help="Specific cities to analyze")
    parser.add_argument("--test", action="store_true",
                       help="Test with first 5 cities")
    parser.add_argument("--parallel", action="store_true", default=True,
                       help="Use parallel processing")
    parser.add_argument("--cache-hours", type=int, default=24,
                       help="Cache expiration in hours")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize scraper
    scraper = EnhancedCrimeScraper()

    # Select cities to process
    if args.cities:
        cities = {k: v for k, v in ENHANCED_CITY_CONFIG.items() if k in args.cities}
    elif args.test:
        cities = dict(list(ENHANCED_CITY_CONFIG.items())[:5])
    else:
        cities = ENHANCED_CITY_CONFIG

    logger.info(f"Starting enhanced crime data scraping for {len(cities)} cities")
    logger.info(f"Using parallel processing: {args.parallel}")
    logger.info(f"Cache expiration: {args.cache_hours} hours")

    try:
        # Process cities
        results_df = scraper.process_all_cities(cities, parallel=args.parallel)

        # Export results
        if not results_df.empty:
            scraper.export_results(results_df, args.output)
        else:
            logger.warning("No datasets found!")

    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
    except Exception as e:
        logger.error(f"Process failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()