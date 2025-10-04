import pandas as pd
import json
from collections import Counter, defaultdict

# Load the enhanced results CSV
df = pd.read_csv('enhanced_crime_datasets.csv')

# Define the expected columns from the COLUMN_PATTERNS
expected_columns = ['latitude', 'longitude', 'date', 'time', 'crime_type', 'description', 'location', 'arrest']

# Parse the column_mapping column and analyze missing fields
missing_field_counts = Counter()
total_datasets = len(df)
quality_score_distribution = Counter()

for index, row in df.iterrows():
    # Parse the column mapping JSON
    try:
        column_mapping = eval(row['column_mapping'])  # Convert string to dict
        mapped_fields = set(column_mapping.keys())
        missing_fields = set(expected_columns) - mapped_fields
        
        # Count each missing field
        for field in missing_fields:
            missing_field_counts[field] += 1
            
        # Count quality score distribution
        quality_score = row['quality_score']
        quality_score_distribution[quality_score] += 1
        
    except Exception as e:
        print(f"Error parsing row {index}: {e}")
        continue

print("=== MISSING COLUMNS ANALYSIS ===")
print(f"Total datasets analyzed: {total_datasets}")
print()

print("Missing Column Frequency (sorted by most frequent):")
print("Field Name          | Count | Percentage")
print("-" * 45)
for field, count in missing_field_counts.most_common():
    percentage = (count / total_datasets) * 100
    print(f"{field:<18} | {count:>5} | {percentage:>6.1f}%")

print()
print("Quality Score Distribution:")
print("Score | Count | Percentage")
print("-" * 30)
for score in sorted(quality_score_distribution.keys(), reverse=True):
    count = quality_score_distribution[score]
    percentage = (count / total_datasets) * 100
    print(f"{score:>5} | {count:>5} | {percentage:>6.1f}%")

print()
print("=== IMPACT ANALYSIS ===")
print("Analyzing which missing columns most impact quality scores...")

# Group by quality score and analyze missing patterns
quality_groups = defaultdict(list)
for index, row in df.iterrows():
    try:
        column_mapping = eval(row['column_mapping'])
        mapped_fields = set(column_mapping.keys())  
        missing_fields = set(expected_columns) - mapped_fields
        quality_score = row['quality_score']
        quality_groups[quality_score].append(missing_fields)
    except:
        continue

print()
print("Common missing patterns by quality score range:")
for score_range in [(100, 87.5), (75, 62.5), (50, 37.5), (25, 12.5)]:
    print(f"\nQuality Score {score_range[1]}-{score_range[0]}%:")
    datasets_in_range = []
    for score, missing_lists in quality_groups.items():
        if score_range[1] <= score <= score_range[0]:
            datasets_in_range.extend(missing_lists)
    
    if datasets_in_range:
        # Count most common missing combinations
        missing_combinations = Counter()
        for missing_set in datasets_in_range:
            if missing_set:  # Only count if there are missing fields
                missing_combinations[tuple(sorted(missing_set))] += 1
        
        print(f"  Total datasets: {len(datasets_in_range)}")
        print(f"  Most common missing field combinations:")
        for combo, count in missing_combinations.most_common(5):
            percentage = (count / len(datasets_in_range)) * 100
            print(f"    {list(combo)} - {count} datasets ({percentage:.1f}%)")