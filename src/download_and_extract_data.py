import os
import requests
import py7zr

def download_and_extract_data():
    """
    Download bird and weather data from GitHub, extract them, 
    and delete the compressed .7z files.
    """
    
    # --- GET DATA FROM MY PUBLIC GITHUB REPO --- #
    url_bird_data = "https://raw.githubusercontent.com/AnetaGancheva/bird-migration-project/main/eBird_data.7z"
    url_weather_data = "https://raw.githubusercontent.com/AnetaGancheva/bird-migration-project/main/weather_data.7z"

    # --- Filenames --- #
    raw_dir = "../data/raw/"
    os.makedirs(raw_dir, exist_ok=True)

    filename_bird_data = os.path.join(raw_dir, "eBird_data.7z")
    filename_weather_data = os.path.join(raw_dir, "weather_data.7z")

    # --- Download files --- #
    for url, filename in [(url_bird_data, filename_bird_data), 
                          (url_weather_data, filename_weather_data)]:
        print(f"Downloading {filename}...")
        res = requests.get(url)
        res.raise_for_status()
        with open(filename, "wb") as f:
            f.write(res.content)

    # --- Extract archives --- #
    for filename in [filename_bird_data, filename_weather_data]:
        print(f"Extracting {filename}...")
        with py7zr.SevenZipFile(filename, mode="r") as archive:
            archive.extractall(path=raw_dir)

    # --- Delete archives --- #
    for filename in [filename_bird_data, filename_weather_data]:
        print(f"Deleting {filename}...")
        os.remove(filename)

    print("Data successfully downloaded and extracted.")

