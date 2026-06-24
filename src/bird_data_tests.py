import pandas as pd
import numpy as np


def bird_data_tests(df_birds: pd.DataFrame) -> None:
    """
    Run sanity checks on bird observation data.

    Tests include:
    - 'observation_date' is datetime dtype and within a reasonable date range
    - 'observation_count' is either NaN (from 'X') or a non-negative integer
    - 'seasonality' values are within expected categories
    - No missing values in key columns (except 'observation_count')
    - 'common_name' and 'scientific_name' have consistent mappings
    - 'year', 'month', and 'week' match values from 'observation_date'
    - No species with valid counts has total < 750

    Parameters:
        df_birds (pd.DataFrame): Cleaned bird observation DataFrame
    """

    # Check 'observation_date' dtype and range
    assert pd.api.types.is_datetime64_any_dtype(df_birds['observation_date']), "'observation_date' must be datetime dtype"
    min_date, max_date = pd.Timestamp('1900-01-01'), pd.Timestamp.now()
    assert (
        (df_birds['observation_date'] >= min_date) & 
        (df_birds['observation_date'] <= max_date)
    ).all(), "'observation_date' out of expected range"

    # Validate 'observation_count' if not null
    if 'observation_count' in df_birds.columns:
        not_null_counts = df_birds['observation_count'].dropna()
        assert pd.api.types.is_float_dtype(df_birds['observation_count']) or pd.api.types.is_integer_dtype(df_birds['observation_count']), \
            "'observation_count' should be numeric (float or int) even if some values are NaN"
        assert (not_null_counts >= 0).all(), "'observation_count' contains negative values"

    # Validate seasonality labels
    expected = {'summer_breeder', 'winter_resident', 'year_round_resident'}
    actual = set(df_birds['seasonality'].dropna().unique())
    assert actual <= expected, f"Unexpected seasonality values: {actual - expected}"

    # Check for missing critical columns (excluding 'observation_count')
    critical = ['observation_date', 'common_name', 'scientific_name', 'seasonality']
    missing = df_birds[critical].isnull().any()
    assert not missing.any(), f"Missing values in columns: {missing[missing].index.tolist()}"

    # Check 1:1 mapping between names
    common_to_sci = df_birds.groupby('common_name')['scientific_name'].nunique()
    sci_to_common = df_birds.groupby('scientific_name')['common_name'].nunique()
    assert (common_to_sci <= 1).all(), f"Multiple scientific_names per common_name: {common_to_sci[common_to_sci > 1].to_dict()}"
    assert (sci_to_common <= 1).all(), f"Multiple common_names per scientific_name: {sci_to_common[sci_to_common > 1].to_dict()}"

    # Ensure date parts are present
    for col in ['year', 'month', 'week']:
        assert col in df_birds.columns, f"Missing '{col}' column"

    # Match date components to observation_date
    assert (df_birds['year'] == df_birds['observation_date'].dt.year).all(), "'year' column mismatch"
    assert (df_birds['month'] == df_birds['observation_date'].dt.month).all(), "'month' column mismatch"
    assert (df_birds['week'] == df_birds['observation_date'].dt.isocalendar().week).all(), "'week' column mismatch"

    # Check total observation count per species (for non-NaN values)
    valid_counts = df_birds.dropna(subset=['observation_count'])
    species_totals = valid_counts.groupby('common_name')['observation_count'].sum()
    too_low = species_totals[species_totals < 750]
    assert too_low.empty, f"Species with total observations < 150: {too_low.to_dict()}"

    print("All bird data tests passed.")