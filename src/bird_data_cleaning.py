import pandas as pd
from pandas import DataFrame
import numpy as np
from typing import Dict, Tuple

def clean_ebird_data(df: DataFrame) -> DataFrame:
    """
    Cleans an eBird dataset by applying the following filters:
    
    1. Drops duplicate checklists identified by 'group_identifier', 
       keeping only the first entry per group.
    2. Keeps only records where 'category' is one of ['form', 'issf', 'species'].
    3. Removes records with non-empty 'exotic_code' values.
    4. Removes species with fewer than 20 total observations in the dataset.
    
    Parameters:
        df (pd.DataFrame): The raw eBird data.

    Returns:
        pd.DataFrame: A cleaned DataFrame with filters applied.
    """
    df = df.copy()

    # Drop duplicates by group identifier
    if 'group_identifier' in df.columns:
        with_group = df[df['group_identifier'].notna()]
        without_group = df[df['group_identifier'].isna()]
        with_group = (
            with_group
            .sort_values('observation_date')
            .drop_duplicates(subset='group_identifier', keep='first') # Keep only the first occurrence within each group as it is the same as the rest 
        )
        df = pd.concat([with_group, without_group], ignore_index=True)

    # Keep only allowed categories
    if 'category' in df.columns:
        allowed_categories = {'form', 'issf', 'species'}
        df = df[df['category'].isin(allowed_categories)]

    # Keep only rows where 'exotic_code' is empty
    if 'exotic_code' in df.columns:
        df = df[df['exotic_code'].isna()]

    # Remove species with fewer than 20 observations
    if 'scientific_name' in df.columns:
        species_counts = df['scientific_name'].value_counts()
        valid_species = species_counts[species_counts >= 20].index
        df = df[df['scientific_name'].isin(valid_species)]

    return df


def classify_flyway(df: DataFrame) -> DataFrame:
    """
    Classifies each row in the input DataFrame into a bird migration flyway 
    based on the 'state_code' column.

    Flyway classification:
    - 'via_pontica' for eastern Danube and Black Sea coastal provinces
    - 'via_aristotelis' for western Danube and mountain regions
    - 'via_balcanica' for all other areas

    Parameters:
        df (pd.DataFrame): A DataFrame containing a 'state_code' column.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with an added 'flyway' column.
    """
    df = df.copy()

    # Define sets of state codes for each flyway
    via_pontica = {'BG-02','BG-08','BG-03','BG-17','BG-19','BG-18','BG-27'}
    via_aristotelis = {'BG-05','BG-12','BG-06','BG-01','BG-14','BG-23','BG-22','BG-10'}

    def get_flyway(code: str) -> str:
        if code in via_pontica:
            return 'via_pontica'
        elif code in via_aristotelis:
            return 'via_aristotelis'
        return 'via_balcanica'

    df['flyway'] = df['state_code'].apply(get_flyway)

    return df

def classify_seasonality(df: DataFrame) -> DataFrame:
    """
    Classifies bird species in the dataset into seasonal categories based on their 
    observation frequency throughout the year.

    Categories:
    - 'winter_resident': More than 80% of observations occur in Oct–Mar
    - 'summer_breeder' : More than 80% of observations occur in Apr–Sep
    - 'year_round_resident': All other cases

    Parameters:
        df (pd.DataFrame): A DataFrame containing at least 'observation_date' and 'scientific_name'.

    Returns:
        pd.DataFrame: A new DataFrame with an additional 'seasonality' column.
    """
    df = df.copy()
    
    # Extract observation month from the date
    df['month'] = pd.to_datetime(df['observation_date']).dt.month

    # Define month groups
    winter_months = {10, 11, 12, 1, 2, 3}
    summer_months = {4, 5, 6, 7, 8, 9}

    species_class: Dict[str, str] = {}

    # Group by species and calculate seasonal observation ratios
    for species, group in df.groupby('scientific_name'):
        month_ratios = group['month'].value_counts(normalize=True)
        winter_ratio = month_ratios[month_ratios.index.isin(winter_months)].sum()
        summer_ratio = month_ratios[month_ratios.index.isin(summer_months)].sum()

        if winter_ratio > 0.7:
            species_class[species] = 'winter_resident'
        elif summer_ratio > 0.7:
            species_class[species] = 'summer_breeder'
        else:
            species_class[species] = 'year_round_resident'

    # Map the classification back to the DataFrame
    df['seasonality'] = df['scientific_name'].map(species_class)

    return df


def classify_seasonality_by_flyway(df: DataFrame) -> DataFrame:
    """
    Classifies each bird species into seasonal categories within each flyway 
    based on abundance-weighted monthly observation data.

    Categories:
    - 'winter_resident'    : More than 80% of observations in Oct–Mar
    - 'summer_breeder'     : More than 80% of observations in Apr–Sep
    - 'year_round_resident': All other cases

    Parameters:
        df (pd.DataFrame): DataFrame with at least the following columns:
            - 'observation_date'
            - 'scientific_name'
            - 'flyway'
            - 'observation_count' (optional; assumed to be 1 if missing)

    Returns:
        pd.DataFrame: A new DataFrame with an added 'seasonality_flyway' column.
    """
    df = df.copy()

    # Define seasonal month groups
    winter_months = {10, 11, 12, 1, 2, 3}
    summer_months = {4, 5, 6, 7, 8, 9}

    species_flyway_class: Dict[Tuple[str, str], str] = {}

    # Group by species and flyway
    grouped = df.groupby(['scientific_name', 'flyway'])
    for (species, flyway), group in grouped:
        # Calculate normalized (weighted) monthly abundance
        monthly_counts = group.groupby('month')['observation_count'].sum()
        monthly_distribution = monthly_counts / monthly_counts.sum()

        winter_ratio = monthly_distribution[monthly_distribution.index.isin(winter_months)].sum()
        summer_ratio = monthly_distribution[monthly_distribution.index.isin(summer_months)].sum()

        # Assign seasonality class
        if winter_ratio > 0.7:
            seasonality = 'winter_resident'
        elif summer_ratio > 0.7:
            seasonality = 'summer_breeder'
        else:
            seasonality = 'year_round_resident'

        species_flyway_class[(species, flyway)] = seasonality

    # Map classification back to DataFrame
    df['seasonality_flyway'] = df.apply(
        lambda row: species_flyway_class.get((row['scientific_name'], row['flyway'])),
        axis=1
    )

    return df

def load_and_prepare_data(
    ebd_path: str,
    sed_path: str,
    min_obs_threshold: int = 150,
    start_year: int = 2019,
    min_consecutive_years: int = 5
) -> Tuple[DataFrame, DataFrame]:
    """
    Loads, cleans, and filters eBird and sampling event datasets for analysis of 
    migratory bird species, based on seasonal classification and observation thresholds.

    Parameters:
        ebd_path (str): Path to the eBird dataset (.txt or .tsv format).
        sed_path (str): Path to the sampling event dataset (.txt or .tsv format).
        min_obs_threshold (int): Minimum number of observations per species per year.
        start_year (int): Earliest year to consider for seasonal filtering.
        min_consecutive_years (int): Minimum number of consecutive years meeting threshold.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of (cleaned eBird DataFrame, cleaned SED DataFrame).
    """
    
    def standardize_columns(df: DataFrame) -> DataFrame:
        return df.rename(columns=lambda col: col.strip().lower().replace(' ', '_').replace('-', '_'))

    def assign_seasonality_fields(df: DataFrame) -> DataFrame:
        df['observation_date'] = pd.to_datetime(df['observation_date'], errors='coerce')
        df['year'] = df['observation_date'].dt.year
        df['month'] = df['observation_date'].dt.month
        df['season'] = df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        })
        df['week'] = df['observation_date'].dt.isocalendar().week.astype(int)
        return df

    # --- Load and Standardize --- #
    ebd = standardize_columns(pd.read_csv(ebd_path, sep='\t', low_memory=False))
    sed = standardize_columns(pd.read_csv(sed_path, sep='\t', low_memory=False))

    # --- Clean eBird Data --- #
    ebd = clean_ebird_data(ebd)

    # --- Filter relevant columns --- #
    required_cols = [
        'common_name', 'scientific_name', 'observation_count', 'breeding_code',
        'breeding_category', 'behavior_code', 'state_code', 'locality', 'locality_type',
        'latitude', 'longitude', 'observation_date', 'time_observations_started',
        'protocol_code', 'duration_minutes', 'effort_distance_km', 'effort_area_ha',
        'number_observers', 'all_species_reported', 'has_media', 'approved',
        'reviewed', 'checklist_comments', 'species_comments', 'sampling_event_identifier'
    ]
    df = ebd[required_cols].copy()

    # --- Prepare fields --- #
    df['observation_count'] = df['observation_count'].replace('X', np.nan).astype(float)
    df = assign_seasonality_fields(df)

    # --- Classify Flyway and Seasonality --- #
    df = classify_flyway(df)
    df = classify_seasonality(df)
    df = classify_seasonality_by_flyway(df)

    # --- Filter to target seasonalities --- #
    df = df[df['seasonality'].isin(['winter_resident', 'summer_breeder'])]

    # --- Compute eligible species based on observation thresholds --- #
    counts = df.groupby(['scientific_name', 'year']).size().reset_index(name='n_obs')
    counts = counts[counts['year'] >= start_year]

    eligible_species = set()
    for species, group in counts.groupby('scientific_name'):
        valid_years = sorted(group[group['n_obs'] >= min_obs_threshold]['year'].unique())
        consec = 1
        for i in range(1, len(valid_years)):
            if valid_years[i] == valid_years[i - 1] + 1:
                consec += 1
                if consec >= min_consecutive_years:
                    eligible_species.add(species)
                    break
            else:
                consec = 1

    # --- Final filter --- #
    df = df[df['scientific_name'].isin(eligible_species) & (df['year'] >= start_year)]

    # --- Prepare SED data --- #
    sed = assign_seasonality_fields(sed)
    sed = sed[sed['year'] >= start_year]
    sed = classify_flyway(sed)

    return df, sed