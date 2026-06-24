import numpy as np
import pandas as pd


def weather_data_tests(df: pd.DataFrame, df_baseline: pd.DataFrame) -> None:
    """
    Run comprehensive tests on cleaned NOAA GSOD weather dataframes.

    Tests include:
    - Temperature range validity for Bulgaria
    - 'temp_c' consistency with 'min_c' and 'max_c'
    - Date and derived columns consistency
    - Weather flags are binary
    - Non-negative precipitation, snow depth, visibility, and wind speeds
    - Logical consistency of wet_day, snow_day, gale_day, fog_day flags
    - Column name normalization (lowercase, no spaces or hyphens)

    Parameters:
        df (pd.DataFrame): Cleaned main weather dataframe
        df_baseline (pd.DataFrame): Cleaned baseline weather dataframe
    """

    def _run_tests(df_clean: pd.DataFrame, name: str) -> None:
        # Temperature range checks (skip NA)
        assert df_clean['min_c'].dropna().ge(-38.4).all(), f"{name}: 'min_c' below -38.4°C found"
        assert df_clean['max_c'].dropna().le(45.3).all(), f"{name}: 'max_c' above 45.3°C found"
        
        # temp_c should lie between min_c and max_c
        mask = df_clean[['temp_c', 'min_c', 'max_c']].notna().all(axis=1)
        temp = df_clean.loc[mask, 'temp_c']
        min_t = df_clean.loc[mask, 'min_c']
        max_t = df_clean.loc[mask, 'max_c']
        assert (temp >= min_t).all() and (temp <= max_t).all(), f"{name}: 'temp_c' outside min/max range"

        # Date columns check
        assert pd.api.types.is_datetime64_any_dtype(df_clean['date']), f"{name}: 'date' is not datetime dtype"
        assert (df_clean['year'] == df_clean['date'].dt.year).all(), f"{name}: 'year' inconsistent with 'date'"
        assert (df_clean['month'] == df_clean['date'].dt.month).all(), f"{name}: 'month' inconsistent with 'date'"
        assert (df_clean['week'] == df_clean['date'].dt.isocalendar().week).all(), f"{name}: 'week' inconsistent with 'date'"

        # Weather flags binary check
        for col in ['fog', 'rain', 'snow', 'hail', 'thunder', 'tornado']:
            unique_vals = df_clean[col].dropna().unique()
            assert set(unique_vals).issubset({0, 1}), f"{name}: Column '{col}' has non-binary values {unique_vals}"

        # Non-negative precipitation and snow depth
        assert (df_clean['prcp_mm'].dropna() >= 0).all(), f"{name}: 'prcp_mm' has negative values"
        assert (df_clean['sndp_mm'].dropna() >= 0).all(), f"{name}: 'sndp_mm' has negative values"

        # Non-negative visibility and wind speeds
        for col in ['visib_km', 'wdsp_kmh', 'mxspd_kmh', 'gust_kmh']:
            assert (df_clean[col].dropna() >= 0).all(), f"{name}: '{col}' has negative values"

        # Logical consistency of wet_day, snow_day, gale_day, fog_day
        assert ((df_clean['wet_day'] == 1) == (df_clean['prcp_mm'] > 0)).all(), f"{name}: 'wet_day' inconsistent with 'prcp_mm'"
        assert ((df_clean['snow_day'] == 1) == (df_clean['sndp_mm'] > 0)).all(), f"{name}: 'snow_day' inconsistent with 'sndp_mm'"
        assert ((df_clean['gale_day'] == 1) == (df_clean['wdsp_kmh'] > 50)).all(), f"{name}: 'gale_day' inconsistent with 'wdsp_kmh'"
        assert ((df_clean['fog_day'] == 1) == (df_clean['visib_km'] < 1)).all(), f"{name}: 'fog_day' inconsistent with 'visib_km'"

        # Column name normalization
        for col in df_clean.columns:
            assert col == col.lower(), f"{name}: Column '{col}' not lowercase"
            assert ' ' not in col, f"{name}: Column '{col}' contains spaces"
            assert '-' not in col, f"{name}: Column '{col}' contains hyphens"

    # Run tests on both dataframes
    _run_tests(df, 'Main df')
    _run_tests(df_baseline, 'Baseline df')

    # Check that baseline period ends before main data period starts
    baseline_max_date = df_baseline['date'].max()
    main_min_date = df['date'].min()
    assert baseline_max_date < main_min_date, (
        f"Baseline period ends at {baseline_max_date.date()}, which is not before "
        f"main data period start {main_min_date.date()}"
    )

    print("All tests passed")