import requests
import os

# API token from Mapbox
MAPBOX_TOKEN = "YOUR_MAPBOX_API_KEY"

# Example coordinates
lat, lon = 12.9716, 77.5946

# Folder to save images
os.makedirs("inputs", exist_ok=True)
image_path = f"inputs/{lat}_{lon}.png"

# Mapbox Satellite API URL
url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},20,0/512x512?access_token={MAPBOX_TOKEN}"

# Download image
response = requests.get(url)
if response.status_code == 200:
    with open(image_path, "wb") as f:
        f.write(response.content)
    print(f"Image saved to {image_path}")
else:
    print(f"Failed to fetch image: {response.status_code} - {response.text}")
