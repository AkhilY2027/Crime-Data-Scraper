#!/usr/bin/env python3
"""
Crime Data Standardizer and Quality Improver

This module provides tools to:
1. Standardize crime data from different sources into a common format
2. Enhance data quality through geocoding, time parsing, and classification
3. Fill missing data using various imputation strategies
4. Validate and score data quality
"""

import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import hashlib
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from sklearn.preprocessing import LabelEncoder
    from sklearn.impute import KNNImputer
    ML_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available. Some features will be limited.")
    ML_AVAILABLE = False


class CrimeDataStandardizer:
    """Standardize crime data from various sources into a unified format"""

    # Standard crime categories based on FBI UCR/NIBRS
    CRIME_CATEGORIES = {
        'VIOLENT': {
            'keywords': ['murder', 'homicide', 'assault', 'battery', 'robbery', 'rape', 'kidnapping'],
            'ucr_codes': ['01A', '01B', '02', '03', '04']
        },
        'PROPERTY': {
            'keywords': ['burglary', 'theft', 'larceny', 'motor vehicle', 'arson', 'shoplifting'],
            'ucr_codes': ['05', '06', '07', '08', '09']
        },
        'DRUG': {
            'keywords': ['narcotics', 'drug', 'possession', 'trafficking', 'marijuana', 'cocaine'],
            'ucr_codes': ['18']
        },
        'WEAPON': {
            'keywords': ['weapon', 'firearm', 'gun', 'concealed carry', 'armed'],
            'ucr_codes': ['15']
        },
        'FRAUD': {
            'keywords': ['fraud', 'forgery', 'counterfeiting', 'embezzlement', 'identity'],
            'ucr_codes': ['10', '11']
        },
        'SEX_OFFENSE': {
            'keywords': ['sex offense', 'prostitution', 'indecent', 'sexual'],
            'ucr_codes': ['02', '16']
        },
        'PUBLIC_ORDER': {
            'keywords': ['disorderly', 'vandalism', 'trespass', 'disturbing', 'loitering'],
            'ucr_codes': ['14', '24']
        }
    }

    # Standard column mapping
    STANDARD_SCHEMA = {
        'incident_id': str,
        'date': 'datetime64[ns]',
        'time': str,
        'datetime': 'datetime64[ns]',
        'year': int,
        'month': int,
        'day_of_week': str,
        'hour': int,
        'latitude': float,
        'longitude': float,
        'address': str,
        'city': str,
        'state': str,
        'zip_code': str,
        'crime_type': str,
        'crime_category': str,
        'crime_description': str,
        'ucr_code': str,
        'arrest_made': bool,
        'domestic': bool,
        'beat': str,
        'district': str,
        'ward': str,
        'community_area': str,
        'location_type': str,
        'source_name': str,
        'source_url': str,
        'data_quality_score': float
    }

    def __init__(self, geocode: bool = False):
        """
        Initialize the standardizer

        Args:
            geocode: Whether to geocode addresses missing coordinates
        """
        self.geocode = geocode
        if geocode:
            geolocator = Nominatim(user_agent="crime_data_standardizer")
            self.geocoder = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        else:
            self.geocoder = None

    def standardize_dataframe(self, df: pd.DataFrame, source_name: str = "Unknown") -> pd.DataFrame:
        """
        Standardize a crime dataset to common format

        Args:
            df: Input dataframe with crime data
            source_name: Name of the data source

        Returns:
            Standardized dataframe
        """
        logger.info(f"Standardizing {len(df)} records from {source_name}")

        # Create standardized dataframe
        std_df = pd.DataFrame()

        # Map columns
        std_df['source_name'] = source_name
        std_df['incident_id'] = self._extract_incident_id(df)
        std_df['datetime'] = self._extract_datetime(df)
        std_df['date'] = pd.to_datetime(std_df['datetime']).dt.date
        std_df['time'] = pd.to_datetime(std_df['datetime']).dt.strftime('%H:%M:%S')
        std_df['year'] = pd.to_datetime(std_df['datetime']).dt.year
        std_df['month'] = pd.to_datetime(std_df['datetime']).dt.month
        std_df['day_of_week'] = pd.to_datetime(std_df['datetime']).dt.day_name()
        std_df['hour'] = pd.to_datetime(std_df['datetime']).dt.hour

        # Extract location data
        location_data = self._extract_location(df)
        std_df['latitude'] = location_data['latitude']
        std_df['longitude'] = location_data['longitude']
        std_df['address'] = location_data['address']
        std_df['city'] = location_data['city']
        std_df['state'] = location_data['state']
        std_df['zip_code'] = location_data['zip_code']

        # Extract crime information
        crime_data = self._extract_crime_info(df)
        std_df['crime_type'] = crime_data['crime_type']
        std_df['crime_category'] = self._categorize_crimes(crime_data['crime_type'])
        std_df['crime_description'] = crime_data['description']
        std_df['ucr_code'] = crime_data['ucr_code']

        # Extract additional fields
        std_df['arrest_made'] = self._extract_boolean_field(df, ['arrest', 'arrested', 'clearance'])
        std_df['domestic'] = self._extract_boolean_field(df, ['domestic', 'family'])

        # Extract geographic divisions
        std_df['beat'] = self._extract_field(df, ['beat', 'sector'])
        std_df['district'] = self._extract_field(df, ['district', 'precinct'])
        std_df['ward'] = self._extract_field(df, ['ward'])
        std_df['community_area'] = self._extract_field(df, ['community', 'neighborhood'])
        std_df['location_type'] = self._extract_field(df, ['location_description', 'premise', 'place'])

        # Calculate data quality score
        std_df['data_quality_score'] = self._calculate_quality_score(std_df)

        # Geocode if needed and requested
        if self.geocode:
            std_df = self._geocode_missing_coordinates(std_df)

        logger.info(f"Standardization complete. Average quality score: {std_df['data_quality_score'].mean():.2f}")

        return std_df

    def _extract_incident_id(self, df: pd.DataFrame) -> pd.Series:
        """Extract or generate incident IDs"""
        id_columns = ['incident_id', 'case_number', 'report_number', 'id', 'incident_number']

        for col in id_columns:
            if col in df.columns or col.upper() in df.columns or col.lower() in df.columns:
                actual_col = self._find_column(df, col)
                if actual_col:
                    return df[actual_col].astype(str)

        # Generate IDs if not found
        return pd.Series([f"GEN_{i:08d}" for i in range(len(df))])

    def _extract_datetime(self, df: pd.DataFrame) -> pd.Series:
        """Extract and parse datetime information"""
        datetime_columns = ['datetime', 'occurred_date', 'incident_date', 'date', 'reported_date']
        time_columns = ['time', 'occurred_time', 'incident_time']

        datetime_col = None
        time_col = None

        # Find datetime column
        for col in datetime_columns:
            actual_col = self._find_column(df, col)
            if actual_col:
                datetime_col = actual_col
                break

        # Find time column
        for col in time_columns:
            actual_col = self._find_column(df, col)
            if actual_col:
                time_col = actual_col
                break

        if datetime_col:
            # Parse datetime
            dt_series = pd.to_datetime(df[datetime_col], errors='coerce')

            # If we also have a separate time column, combine them
            if time_col and time_col != datetime_col:
                try:
                    time_series = pd.to_datetime(df[time_col], format='%H:%M:%S', errors='coerce').dt.time
                    dt_series = pd.to_datetime(
                        dt_series.dt.date.astype(str) + ' ' + time_series.astype(str),
                        errors='coerce'
                    )
                except:
                    pass

            return dt_series

        # Default to current time if no datetime found
        return pd.Series([pd.NaT] * len(df))

    def _extract_location(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Extract location information"""
        location_data = {}

        # Latitude
        lat_columns = ['latitude', 'lat', 'y', 'y_coordinate', 'point_y']
        location_data['latitude'] = self._extract_numeric_field(df, lat_columns)

        # Longitude
        lon_columns = ['longitude', 'lon', 'lng', 'x', 'x_coordinate', 'point_x']
        location_data['longitude'] = self._extract_numeric_field(df, lon_columns)

        # Address
        addr_columns = ['address', 'location', 'street_address', 'block']
        location_data['address'] = self._extract_field(df, addr_columns)

        # City
        city_columns = ['city', 'municipality', 'town']
        location_data['city'] = self._extract_field(df, city_columns)

        # State
        state_columns = ['state', 'province']
        location_data['state'] = self._extract_field(df, state_columns)

        # Zip code
        zip_columns = ['zip', 'zip_code', 'postal_code']
        location_data['zip_code'] = self._extract_field(df, zip_columns)

        return location_data

    def _extract_crime_info(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Extract crime type and description"""
        crime_data = {}

        # Crime type
        type_columns = ['crime_type', 'primary_type', 'offense', 'offense_type', 'category']
        crime_data['crime_type'] = self._extract_field(df, type_columns)

        # Description
        desc_columns = ['description', 'offense_description', 'narrative', 'details']
        crime_data['description'] = self._extract_field(df, desc_columns)

        # UCR code
        ucr_columns = ['ucr_code', 'ucr', 'fbi_code', 'offense_code']
        crime_data['ucr_code'] = self._extract_field(df, ucr_columns)

        return crime_data

    def _extract_field(self, df: pd.DataFrame, possible_columns: List[str]) -> pd.Series:
        """Extract a field from dataframe using multiple possible column names"""
        for col in possible_columns:
            actual_col = self._find_column(df, col)
            if actual_col:
                return df[actual_col].astype(str)

        return pd.Series([None] * len(df))

    def _extract_numeric_field(self, df: pd.DataFrame, possible_columns: List[str]) -> pd.Series:
        """Extract a numeric field"""
        for col in possible_columns:
            actual_col = self._find_column(df, col)
            if actual_col:
                return pd.to_numeric(df[actual_col], errors='coerce')

        return pd.Series([np.nan] * len(df))

    def _extract_boolean_field(self, df: pd.DataFrame, possible_columns: List[str]) -> pd.Series:
        """Extract a boolean field"""
        for col in possible_columns:
            actual_col = self._find_column(df, col)
            if actual_col:
                series = df[actual_col]
                # Convert various representations to boolean
                if series.dtype == bool:
                    return series
                else:
                    return series.astype(str).str.lower().isin(['true', 'yes', 'y', '1'])

        return pd.Series([False] * len(df))

    def _find_column(self, df: pd.DataFrame, column_name: str) -> Optional[str]:
        """Find column in dataframe (case-insensitive)"""
        column_lower = column_name.lower()

        for col in df.columns:
            if col.lower() == column_lower:
                return col

        # Also check for partial matches
        for col in df.columns:
            if column_lower in col.lower() or col.lower() in column_lower:
                return col

        return None

    def _categorize_crimes(self, crime_types: pd.Series) -> pd.Series:
        """Categorize crimes into standard categories"""
        categories = []

        for crime_type in crime_types:
            if pd.isna(crime_type):
                categories.append('UNKNOWN')
                continue

            crime_lower = str(crime_type).lower()
            matched = False

            for category, info in self.CRIME_CATEGORIES.items():
                for keyword in info['keywords']:
                    if keyword in crime_lower:
                        categories.append(category)
                        matched = True
                        break
                if matched:
                    break

            if not matched:
                categories.append('OTHER')

        return pd.Series(categories)

    def _calculate_quality_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate data quality score for each record (0-100)"""
        scores = []

        required_fields = ['datetime', 'latitude', 'longitude', 'crime_type']
        optional_fields = ['address', 'arrest_made', 'crime_description', 'district']

        for idx, row in df.iterrows():
            score = 0
            max_score = 0

            # Check required fields (worth 15 points each)
            for field in required_fields:
                max_score += 15
                if pd.notna(row[field]):
                    score += 15

            # Check optional fields (worth 10 points each)
            for field in optional_fields:
                max_score += 10
                if pd.notna(row[field]):
                    score += 10

            # Normalize to 0-100
            final_score = (score / max_score * 100) if max_score > 0 else 0
            scores.append(final_score)

        return pd.Series(scores)

    def _geocode_missing_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Geocode records missing coordinates using address"""
        if not self.geocoder:
            return df

        missing_coords = df['latitude'].isna() | df['longitude'].isna()
        has_address = df['address'].notna()
        to_geocode = missing_coords & has_address

        if to_geocode.sum() == 0:
            return df

        logger.info(f"Geocoding {to_geocode.sum()} addresses...")

        for idx in df[to_geocode].index:
            try:
                address = df.loc[idx, 'address']
                city = df.loc[idx, 'city']
                state = df.loc[idx, 'state']

                # Build full address
                full_address = f"{address}, {city}, {state}" if city and state else address

                location = self.geocoder(full_address)
                if location:
                    df.loc[idx, 'latitude'] = location.latitude
                    df.loc[idx, 'longitude'] = location.longitude
                    logger.debug(f"Geocoded: {full_address}")

            except Exception as e:
                logger.debug(f"Geocoding failed for index {idx}: {e}")

        return df


class CrimeDataEnhancer:
    """Enhance crime data quality through various techniques"""

    def __init__(self):
        self.standardizer = CrimeDataStandardizer()

    def impute_missing_values(self, df: pd.DataFrame, method: str = 'smart') -> pd.DataFrame:
        """
        Impute missing values using various strategies

        Args:
            df: Input dataframe
            method: Imputation method ('smart', 'knn', 'forward_fill', 'mean')

        Returns:
            DataFrame with imputed values
        """
        logger.info(f"Imputing missing values using {method} method")

        df = df.copy()

        if method == 'smart':
            # Smart imputation based on data type and context
            df = self._smart_impute(df)
        elif method == 'knn' and ML_AVAILABLE:
            df = self._knn_impute(df)
        elif method == 'forward_fill':
            df = df.fillna(method='ffill')
        elif method == 'mean':
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

        return df

    def _smart_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smart imputation based on data patterns"""

        # Impute time-based fields
        if 'hour' in df.columns and df['hour'].isna().any():
            # Use mode hour for similar crime types
            for crime_type in df['crime_type'].unique():
                mask = (df['crime_type'] == crime_type) & df['hour'].isna()
                if mask.any():
                    mode_hour = df[df['crime_type'] == crime_type]['hour'].mode()
                    if len(mode_hour) > 0:
                        df.loc[mask, 'hour'] = mode_hour[0]

        # Impute day of week
        if 'day_of_week' in df.columns and df['day_of_week'].isna().any():
            # Use mode day for similar crime types
            for crime_type in df['crime_type'].unique():
                mask = (df['crime_type'] == crime_type) & df['day_of_week'].isna()
                if mask.any():
                    mode_day = df[df['crime_type'] == crime_type]['day_of_week'].mode()
                    if len(mode_day) > 0:
                        df.loc[mask, 'day_of_week'] = mode_day[0]

        # Impute arrest information based on crime type
        if 'arrest_made' in df.columns and df['arrest_made'].isna().any():
            # Calculate arrest rates by crime type
            arrest_rates = df.groupby('crime_category')['arrest_made'].mean()
            for category in arrest_rates.index:
                mask = (df['crime_category'] == category) & df['arrest_made'].isna()
                if mask.any():
                    # Use probabilistic imputation based on arrest rate
                    arrest_rate = arrest_rates[category]
                    df.loc[mask, 'arrest_made'] = np.random.random(mask.sum()) < arrest_rate

        return df

    def _knn_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """KNN-based imputation for numeric fields"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        if len(numeric_columns) > 0:
            imputer = KNNImputer(n_neighbors=5)
            df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

        return df

    def validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix coordinate data"""
        logger.info("Validating coordinates")

        # Check for valid latitude range (-90 to 90)
        invalid_lat = (df['latitude'] < -90) | (df['latitude'] > 90)
        if invalid_lat.any():
            logger.warning(f"Found {invalid_lat.sum()} invalid latitudes")
            df.loc[invalid_lat, 'latitude'] = np.nan

        # Check for valid longitude range (-180 to 180)
        invalid_lon = (df['longitude'] < -180) | (df['longitude'] > 180)
        if invalid_lon.any():
            logger.warning(f"Found {invalid_lon.sum()} invalid longitudes")
            df.loc[invalid_lon, 'longitude'] = np.nan

        # Check for swapped coordinates (common error)
        possibly_swapped = (df['latitude'].abs() > 90) & (df['longitude'].abs() <= 90)
        if possibly_swapped.any():
            logger.warning(f"Found {possibly_swapped.sum()} possibly swapped coordinates")
            # Swap them
            df.loc[possibly_swapped, ['latitude', 'longitude']] = \
                df.loc[possibly_swapped, ['longitude', 'latitude']].values

        return df

    def remove_duplicates(self, df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """
        Remove duplicate records using fuzzy matching

        Args:
            df: Input dataframe
            threshold: Similarity threshold for considering duplicates

        Returns:
            DataFrame with duplicates removed
        """
        logger.info(f"Removing duplicates with {threshold} similarity threshold")

        # First remove exact duplicates
        initial_count = len(df)
        df = df.drop_duplicates()
        exact_removed = initial_count - len(df)

        if exact_removed > 0:
            logger.info(f"Removed {exact_removed} exact duplicates")

        # For fuzzy duplicate detection, create a hash of key fields
        df['dup_key'] = (
            df['datetime'].astype(str) + '_' +
            df['latitude'].round(4).astype(str) + '_' +
            df['longitude'].round(4).astype(str) + '_' +
            df['crime_type'].astype(str)
        )

        # Remove records with same key
        df = df.drop_duplicates(subset=['dup_key'])
        df = df.drop('dup_key', axis=1)

        fuzzy_removed = initial_count - exact_removed - len(df)
        if fuzzy_removed > 0:
            logger.info(f"Removed {fuzzy_removed} fuzzy duplicates")

        return df


def create_unified_dataset(
    data_sources: List[Tuple[pd.DataFrame, str]],
    output_file: str = "unified_crime_data.csv",
    enhance: bool = True
) -> pd.DataFrame:
    """
    Create a unified dataset from multiple sources

    Args:
        data_sources: List of (dataframe, source_name) tuples
        output_file: Output file path
        enhance: Whether to enhance data quality

    Returns:
        Unified and standardized dataframe
    """
    logger.info(f"Creating unified dataset from {len(data_sources)} sources")

    standardizer = CrimeDataStandardizer(geocode=False)
    enhancer = CrimeDataEnhancer()

    all_standardized = []

    for df, source_name in data_sources:
        # Standardize each source
        std_df = standardizer.standardize_dataframe(df, source_name)
        all_standardized.append(std_df)

    # Combine all sources
    unified_df = pd.concat(all_standardized, ignore_index=True)
    logger.info(f"Combined {len(unified_df)} total records")

    if enhance:
        # Enhance data quality
        unified_df = enhancer.validate_coordinates(unified_df)
        unified_df = enhancer.impute_missing_values(unified_df, method='smart')
        unified_df = enhancer.remove_duplicates(unified_df)

    # Sort by datetime
    unified_df = unified_df.sort_values('datetime')

    # Save to file
    unified_df.to_csv(output_file, index=False)
    logger.info(f"Saved unified dataset to {output_file}")

    # Print summary statistics
    logger.info(f"\n{'='*60}")
    logger.info("UNIFIED DATASET SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total records: {len(unified_df)}")
    logger.info(f"Date range: {unified_df['datetime'].min()} to {unified_df['datetime'].max()}")
    logger.info(f"Cities: {unified_df['city'].nunique()}")
    logger.info(f"Sources: {unified_df['source_name'].nunique()}")
    logger.info(f"Average quality score: {unified_df['data_quality_score'].mean():.2f}")

    # Crime category distribution
    logger.info("\nCrime Categories:")
    category_counts = unified_df['crime_category'].value_counts()
    for category, count in category_counts.items():
        logger.info(f"  {category}: {count} ({count/len(unified_df)*100:.1f}%)")

    # Missing data analysis
    logger.info("\nData Completeness:")
    for col in ['latitude', 'longitude', 'address', 'arrest_made']:
        if col in unified_df.columns:
            missing_pct = unified_df[col].isna().sum() / len(unified_df) * 100
            logger.info(f"  {col}: {100-missing_pct:.1f}% complete")

    return unified_df


if __name__ == "__main__":
    # Example usage
    logger.info("Crime Data Standardizer - Example Usage")

    # Simulate loading data from different sources
    # In practice, you would load actual CSV files
    example_sources = []

    # Example: Chicago data
    chicago_df = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02'],
        'Primary Type': ['THEFT', 'ASSAULT'],
        'Description': ['RETAIL THEFT', 'SIMPLE ASSAULT'],
        'Latitude': [41.8781, 41.8500],
        'Longitude': [-87.6298, -87.6500],
        'Arrest': [True, False]
    })
    example_sources.append((chicago_df, "Chicago PD"))

    # Example: New York data (different format)
    ny_df = pd.DataFrame({
        'OCCUR_DATE': ['01/01/2024', '01/02/2024'],
        'OFNS_DESC': ['LARCENY', 'ROBBERY'],
        'Latitude': [40.7128, 40.7500],
        'Longitude': [-74.0060, -74.0100],
        'ARREST_FLAG': ['Y', 'N']
    })
    example_sources.append((ny_df, "NYPD"))

    # Create unified dataset
    unified_df = create_unified_dataset(example_sources, enhance=True)

    print(f"\nUnified dataset created with {len(unified_df)} records")