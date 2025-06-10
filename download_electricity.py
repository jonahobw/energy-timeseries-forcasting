import io
import zipfile
from pathlib import Path

import requests

# Homepage: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
dataset_url = (
    "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
)

# Download the dataset
response = requests.get(dataset_url)

# Save the dataset to the data directory
data_path = Path(__file__).parent / "data"
data_path.mkdir(parents=True, exist_ok=True)
zip_content = io.BytesIO(response.content)

# Unzip the dataset
with zipfile.ZipFile(zip_content, "r") as zip_ref:
    zip_ref.extractall(data_path)
