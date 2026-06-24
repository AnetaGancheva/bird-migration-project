import pandas as pd
from pandas import DataFrame
import numpy as np
from typing import Union, Tuple


# Helper functions that can either take inputs a single column from a dataframe (Series or multi-dimensional array ndarray). return same type as input

def fahrenheit_to_celsius(temp: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """
    Convert temperature from Fahrenheit to Celsius.

    Parameters:
        temp (pd.Series or np.ndarray): Temperature(s) in Fahrenheit.

    Returns:
        pd.Series or np.ndarray: Temperature(s) converted to Celsius.
    """
    return (temp.astype(float) - 32) * 5.0 / 9.0


def miles_to_km(dist: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """
    Convert distance from miles to kilometers.

    Parameters:
        dist (pd.Series or np.ndarray): Distance(s) in miles.

    Returns:
        pd.Series or np.ndarray: Distance(s) converted to kilometers.
    """
    return dist.astype(float) * 1.609344


def knots_to_kmh(speed: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """
    Convert speed from knots to kilometers per hour.

    Parameters:
        speed (pd.Series or np.ndarray): Speed(s) in knots.

    Returns:
        pd.Series or np.ndarray: Speed(s) converted to km/h.
    """
    return speed.astype(float) * 1.852


def inch_to_mm(amount: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """
    Convert length from inches to millimeters.

    Parameters:
        amount (pd.Series or np.ndarray): Length(s) in inches.

    Returns:
        pd.Series or np.ndarray: Length(s) converted to millimeters.
    """
    return amount.astype(float) * 25.4

# Clean NOAA data function 

def clean_noaa_data(df: DataFrame) -> DataFrame:
    """
    Clean GSOD weather data DataFrame:
    - Replace known missing value placeholders with NaN.
    - Parse and decode the FRSHTT weather flags into separate boolean columns.
    - Normalize column names.
    - Convert dates and extract year, month, week, and season.
    - Convert various units to metric.
    - Create additional binary flags for weather conditions.

    Parameters:
        df (DataFrame): Raw GSOD weather data.

    Returns:
        DataFrame: Cleaned and enriched weather data.
    """
    # Replace known missing value placeholders with NaN
    missing_vals = [9999.9, 999.9, 99.99, 999.0, 9999.0, 999.0, 99.9, 999]
    df = df.replace(missing_vals, np.nan)

    # Pythonize column names to lowercase with underscores
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(' ', '_')
        .str.replace('-', '_')
    )

    # Add leading zeros to ensure 6 digits
    df['frshtt'] = df['frshtt'].astype(str).str.zfill(6)

    # Decode FRSHTT flags: Fog, Rain, Snow, Hail, Thunder, Tornado
    df['fog'] = df['frshtt'].str[0].astype(int)
    df['rain'] = df['frshtt'].str[1].astype(int)
    df['snow'] = df['frshtt'].str[2].astype(int)
    df['hail'] = df['frshtt'].str[3].astype(int)
    df['thunder'] = df['frshtt'].str[4].astype(int)
    df['tornado'] = df['frshtt'].str[5].astype(int)

    # Parse dates and extract year, month, week
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week

    # Map months to seasons the same way as with the bird data
    df['season'] = df['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    })

    # Convert temperatures from Fahrenheit to Celsius
    df['temp_c'] = fahrenheit_to_celsius(df['temp'])
    df['max_c'] = fahrenheit_to_celsius(df['max'])
    df['min_c'] = fahrenheit_to_celsius(df['min'])
    df['dewp_c'] = fahrenheit_to_celsius(df['dewp'])

    # Convert visibility from miles to kilometers
    df['visib_km'] = miles_to_km(df['visib'])

    # Convert wind speeds from knots to km/h
    df['wdsp_kmh'] = knots_to_kmh(df['wdsp'])
    df['mxspd_kmh'] = knots_to_kmh(df['mxspd'])
    df['gust_kmh'] = knots_to_kmh(df['gust'])

    # Convert precipitation and snow depth from inches to millimeters
    df['prcp_mm'] = inch_to_mm(df['prcp'])
    df['sndp_mm'] = inch_to_mm(df['sndp'])

    # Create further binary flags for weather conditions
    df['wet_day'] = (df['prcp_mm'] > 0).astype(int)
    df['snow_day'] = (df['sndp_mm'] > 0).astype(int)
    df['gale_day'] = (df['wdsp_kmh'] > 50).astype(int)
    df['fog_day'] = (df['visib_km'] < 1).astype(int)

    return df


def load_and_prepare_weather_data(
    filepath: str, 
    filepath_baseline: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw NOAA GSOD weather data CSV files, clean and preprocess them.

    This function reads two CSV files containing weather data, applies cleaning
    and preprocessing using the `clean_noaa_data` function, and returns the cleaned DataFrames.

    Parameters:
        filepath (str): Path to the main NOAA GSOD CSV file.
        filepath_baseline (str): Path to the baseline NOAA GSOD CSV file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two cleaned DataFrames:
            - The main weather dataset.
            - The baseline weather dataset.
    """
    df = pd.read_csv(filepath)
    df_baseline = pd.read_csv(filepath_baseline)
    df = clean_noaa_data(df)
    df_baseline = clean_noaa_data(df_baseline)
    return df, df_baseline