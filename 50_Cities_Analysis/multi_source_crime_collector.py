#!/usr/bin/env python3
"""
Multi-Source Crime Data Collector

This script collects crime data from multiple sources including:
1. Police department APIs
2. State crime repositories
3. FBI NIBRS/UCR data
4. News/media crime databases
5. Academic crime datasets
6. Community-sourced crime data
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache configuration
CACHE_DIR = "multi_source_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


class MultiSourceCollector:
    """Collector for crime data from multiple non-traditional sources"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Crime Research Bot 1.0'
        })
        self.results = []

    def cache_get(self, url: str, params: dict = None, cache_hours: int = 24) -> Optional[Any]:
        """Cached GET request"""
        cache_key = hashlib.md5(f"{url}{params}".encode()).hexdigest()
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")

        # Check cache
        if os.path.exists(cache_file):
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - file_time < timedelta(hours=cache_hours):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

        # Make request
        try:
            response = self.session.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json() if 'json' in response.headers.get('content-type', '') else response.text

                # Cache response
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)

                return data
        except Exception as e:
            logger.error(f"Request failed for {url}: {e}")
            return None

    def get_spotcrime_data(self, lat: float, lon: float, radius: float = 0.02) -> List[Dict]:
        """
        Get crime data from SpotCrime API
        SpotCrime aggregates crime data from police departments, news, and other sources
        """
        logger.info(f"Fetching SpotCrime data for coordinates ({lat}, {lon})")

        url = "https://api.spotcrime.com/crimes.json"
        params = {
            'lat': lat,
            'lon': lon,
            'radius': radius,
            'key': '.'  # SpotCrime doesn't require API key for basic access
        }

        data = self.cache_get(url, params)

        crimes = []
        if data and 'crimes' in data:
            for crime in data['crimes']:
                crimes.append({
                    'source': 'SpotCrime',
                    'date': crime.get('date'),
                    'type': crime.get('type'),
                    'address': crime.get('address'),
                    'latitude': crime.get('lat'),
                    'longitude': crime.get('lon'),
                    'description': crime.get('cdid'),
                    'link': crime.get('link')
                })
            logger.info(f"  Found {len(crimes)} crimes from SpotCrime")

        return crimes

    def get_crimeometer_data(self, lat: float, lon: float, distance: int = 3) -> List[Dict]:
        """
        Get crime data from CrimeoMeter API
        Note: Requires API key (paid service) - this is a template
        """
        logger.info(f"Fetching CrimeoMeter data for coordinates ({lat}, {lon})")

        # You would need to sign up for an API key at crimeometer.com
        api_key = "YOUR_CRIMEOMETER_API_KEY"

        url = "https://api.crimeometer.com/v1/incidents/raw-data"
        headers = {
            'x-api-key': api_key,
            'Content-Type': 'application/json'
        }
        params = {
            'lat': lat,
            'lon': lon,
            'distance': f"{distance}mi",
            'datetime_ini': (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            'datetime_end': datetime.now().strftime("%Y-%m-%d")
        }

        # This would make the actual API call if you have a key
        # data = self.cache_get(url, params)

        # Template response structure
        crimes = []
        logger.info(f"  CrimeoMeter requires API key (template only)")

        return crimes

    def get_gunviolence_archive_data(self, state: str, city: str = None) -> List[Dict]:
        """
        Scrape Gun Violence Archive data
        Note: Web scraping - be respectful of rate limits
        """
        logger.info(f"Fetching Gun Violence Archive data for {state}")

        # GVA provides data through their website
        # This is a simplified example - real implementation would need proper scraping
        base_url = "https://www.gunviolencearchive.org/query"

        # Would need to implement proper web scraping here
        crimes = []

        # Template structure
        crimes.append({
            'source': 'Gun Violence Archive',
            'state': state,
            'city': city,
            'note': 'Requires web scraping implementation'
        })

        return crimes

    def get_nibrs_data(self, ori: str, year: int = 2023) -> List[Dict]:
        """
        Get NIBRS (National Incident-Based Reporting System) data from FBI
        More detailed than traditional UCR data
        """
        logger.info(f"Fetching NIBRS data for ORI {ori}")

        # FBI Crime Data Explorer API
        base_url = "https://api.usa.gov/crime/fbi/cde"

        # NIBRS provides incident-level data
        endpoints = [
            f"/nibrs/count/agencies/{ori}/offenses",
            f"/nibrs/{year}/agencies/{ori}/offenses",
            f"/nibrs/victims/agencies/{ori}/offenses",
            f"/nibrs/offenders/agencies/{ori}/offenses"
        ]

        all_data = []
        for endpoint in endpoints:
            url = base_url + endpoint
            data = self.cache_get(url)

            if data:
                all_data.append({
                    'source': 'FBI NIBRS',
                    'ori': ori,
                    'endpoint': endpoint,
                    'data': data
                })
                logger.info(f"  Retrieved NIBRS {endpoint.split('/')[-2]} data")

        return all_data

    def get_police_data_initiative(self, city: str, state: str) -> List[Dict]:
        """
        Get data from Police Data Initiative participating departments
        Many departments share data through standardized formats
        """
        logger.info(f"Checking Police Data Initiative for {city}, {state}")

        # PDI maintains a catalog of police open data
        catalog_url = "https://www.policedatainitiative.org/datasets/"

        # Known PDI participants with direct data access
        pdi_sources = {
            "Austin, TX": "https://data.austintexas.gov/resource/fdj4-gpfu.json",
            "Seattle, WA": "https://data.seattle.gov/resource/tazs-3rd5.json",
            "New Orleans, LA": "https://data.nola.gov/resource/5fn8-vtui.json",
            "Cincinnati, OH": "https://data.cincinnati-oh.gov/resource/k59e-2pvf.json"
        }

        city_state = f"{city}, {state}"
        if city_state in pdi_sources:
            url = pdi_sources[city_state]
            data = self.cache_get(url, params={'$limit': 1000})

            if data:
                logger.info(f"  Found {len(data)} records from PDI source")
                return [{
                    'source': 'Police Data Initiative',
                    'city': city,
                    'state': state,
                    'records': data
                }]

        return []

    def get_university_crime_data(self, university_name: str) -> List[Dict]:
        """
        Get campus crime data from university police departments
        Required by Clery Act to report crime statistics
        """
        logger.info(f"Fetching university crime data for {university_name}")

        # Clery Act requires universities to publish crime data
        # This can be accessed through Department of Education

        base_url = "https://ope.ed.gov/campussafety/api/dataexports/"

        # Would need institution ID lookup
        crimes = [{
            'source': 'Clery Act Data',
            'institution': university_name,
            'note': 'Available through Campus Safety and Security Data'
        }]

        return crimes

    def get_transit_crime_data(self, city: str) -> List[Dict]:
        """
        Get crime data from transit police departments
        Many major cities have separate transit police with their own data
        """
        logger.info(f"Fetching transit crime data for {city}")

        transit_apis = {
            "New York": {
                "name": "MTA Police",
                "url": "http://web.mta.info/developers/"
            },
            "Washington": {
                "name": "Metro Transit Police",
                "url": "https://www.wmata.com/about/transit-police/"
            },
            "San Francisco": {
                "name": "BART Police",
                "url": "https://www.bart.gov/about/police"
            }
        }

        if city in transit_apis:
            logger.info(f"  Transit police data available for {city}")
            return [{
                'source': f"{transit_apis[city]['name']}",
                'city': city,
                'url': transit_apis[city]['url']
            }]

        return []

    def get_crowdsourced_data(self, city: str, state: str) -> List[Dict]:
        """
        Get crowdsourced crime data from citizen reporting platforms
        """
        logger.info(f"Fetching crowdsourced data for {city}, {state}")

        platforms = []

        # Citizen app API (unofficial)
        # Note: Citizen app has unofficial API endpoints but requires reverse engineering
        platforms.append({
            'source': 'Citizen App',
            'type': 'crowdsourced',
            'city': city,
            'note': 'Real-time citizen reports of crime and emergencies'
        })

        # Nextdoor crime and safety reports
        platforms.append({
            'source': 'Nextdoor',
            'type': 'neighborhood reports',
            'city': city,
            'note': 'Neighborhood-level crime and safety discussions'
        })

        # Ring Neighbors app
        platforms.append({
            'source': 'Ring Neighbors',
            'type': 'video-based reports',
            'city': city,
            'note': 'Video doorbell and security camera incident reports'
        })

        return platforms

    def get_news_crime_data(self, city: str, state: str) -> List[Dict]:
        """
        Get crime data from news organizations that maintain databases
        """
        logger.info(f"Fetching news-aggregated crime data for {city}, {state}")

        news_sources = []

        # Many news organizations maintain crime databases
        major_sources = {
            "Chicago Tribune": "https://www.chicagotribune.com/news/breaking/",
            "LA Times": "https://homicide.latimes.com/",
            "Washington Post": "https://www.washingtonpost.com/graphics/local/homicides/",
            "Baltimore Sun": "https://homicides.news.baltimoresun.com/"
        }

        # Check if city has a major news source with crime data
        if city == "Chicago":
            news_sources.append({
                'source': 'Chicago Tribune',
                'type': 'homicide database',
                'url': major_sources["Chicago Tribune"]
            })
        elif city == "Los Angeles":
            news_sources.append({
                'source': 'LA Times Homicide Report',
                'type': 'homicide database',
                'url': major_sources["LA Times"]
            })

        return news_sources

    def get_academic_datasets(self) -> List[Dict]:
        """
        Get information about academic crime datasets
        """
        logger.info("Listing academic crime data sources")

        datasets = [
            {
                'source': 'ICPSR',
                'name': 'National Incident-Based Reporting System',
                'url': 'https://www.icpsr.umich.edu/web/ICPSR/series/128',
                'description': 'Comprehensive incident-level crime data'
            },
            {
                'source': 'NACJD',
                'name': 'National Archive of Criminal Justice Data',
                'url': 'https://www.icpsr.umich.edu/web/pages/NACJD/',
                'description': 'Large collection of crime and justice datasets'
            },
            {
                'source': 'Harvard Dataverse',
                'name': 'Crime and Violence Data',
                'url': 'https://dataverse.harvard.edu/',
                'description': 'Various crime research datasets'
            },
            {
                'source': 'Open Data Philly Crime',
                'name': 'Philadelphia Crime Data',
                'url': 'https://www.opendataphilly.org/dataset/crime-incidents',
                'description': 'Comprehensive Philadelphia crime data'
            }
        ]

        return datasets

    def collect_all_sources(self, cities: List[Dict]) -> pd.DataFrame:
        """
        Collect data from all available sources for given cities
        """
        all_data = []

        for city_info in tqdm(cities, desc="Processing cities"):
            city = city_info['name']
            state = city_info['state']
            lat = city_info.get('lat', 0)
            lon = city_info.get('lon', 0)
            ori = city_info.get('ori', '')

            logger.info(f"\n{'='*60}")
            logger.info(f"Collecting data for {city}, {state}")
            logger.info(f"{'='*60}")

            # SpotCrime data
            if lat and lon:
                spotcrime = self.get_spotcrime_data(lat, lon)
                for crime in spotcrime:
                    crime['city'] = city
                    crime['state'] = state
                    all_data.extend(spotcrime)

            # NIBRS data
            if ori:
                nibrs = self.get_nibrs_data(ori)
                for record in nibrs:
                    record['city'] = city
                    record['state'] = state
                    all_data.append(record)

            # Police Data Initiative
            pdi = self.get_police_data_initiative(city, state)
            for record in pdi:
                all_data.append(record)

            # Transit crime data
            transit = self.get_transit_crime_data(city)
            all_data.extend(transit)

            # Crowdsourced data info
            crowd = self.get_crowdsourced_data(city, state)
            all_data.extend(crowd)

            # News crime data
            news = self.get_news_crime_data(city, state)
            all_data.extend(news)

            # Add small delay to be respectful
            time.sleep(0.5)

        # Add academic datasets (city-agnostic)
        academic = self.get_academic_datasets()
        all_data.extend(academic)

        return pd.DataFrame(all_data)


def main():
    # Example cities to process
    cities = [
        {'name': 'Chicago', 'state': 'IL', 'lat': 41.8781, 'lon': -87.6298, 'ori': 'IL0163000'},
        {'name': 'Los Angeles', 'state': 'CA', 'lat': 34.0522, 'lon': -118.2437, 'ori': 'CA0194200'},
        {'name': 'New York', 'state': 'NY', 'lat': 40.7128, 'lon': -74.0060, 'ori': 'NY0303000'},
        {'name': 'Houston', 'state': 'TX', 'lat': 29.7604, 'lon': -95.3698, 'ori': 'TX1010000'},
        {'name': 'Philadelphia', 'state': 'PA', 'lat': 39.9526, 'lon': -75.1652, 'ori': 'PAPEP0000'}
    ]

    collector = MultiSourceCollector()

    logger.info("Starting multi-source crime data collection")
    logger.info(f"Processing {len(cities)} cities")

    # Collect data
    results_df = collector.collect_all_sources(cities)

    # Save results
    output_file = "multi_source_crime_data.csv"
    results_df.to_csv(output_file, index=False)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("COLLECTION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total records collected: {len(results_df)}")
    logger.info(f"Cities processed: {len(cities)}")

    if 'source' in results_df.columns:
        logger.info("\nData by source:")
        source_counts = results_df['source'].value_counts()
        for source, count in source_counts.items():
            logger.info(f"  {source}: {count} records")

    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()