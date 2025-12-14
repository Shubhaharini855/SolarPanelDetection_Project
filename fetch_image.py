import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
import math

MAPBOX_STATIC_URL = "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},{zoom}/{width}x{height}?access_token={token}"

def fetch_mapbox_static(lat, lon, out_path, token, zoom=20, size=(1024,1024)):
    """
    Fetch a static satellite image centered at lat,lon from Mapbox.
    - zoom: integer zoom level (higher = more detail). Adjust to meet resolution needs.
    - size: (width, height) in pixels (max varies; Mapbox allows large sizes for some accounts).
    """
    w, h = size
    url = MAPBOX_STATIC_URL.format(lon=lon, lat=lat, zoom=zoom, width=w, height=h, token=token)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGB")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)
    return str(out_path)

# Fallback helper: save provided local image path
def local_image_path(path):
    return path
