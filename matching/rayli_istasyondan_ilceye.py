import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import json
import os

# JSON dosyalarının yolları
istasyon_json_path = "data/station_data.json"  # senin dosya adın burada
ilce_geojson_path = "data/istanbul-districts.json"    # senin dosya adın burada

# 1. İstasyon verisini oku
with open(istasyon_json_path, "r", encoding="utf-8") as f:
    istasyon_data = json.load(f)

istasyon_records = []
for feature in istasyon_data["features"]:
    props = feature["properties"]
    coords = feature["geometry"]["coordinates"]
    istasyon_records.append({
        "PROJE_ADI": props["PROJE_ADI"],
        "ISTASYON": props["ISTASYON"],
        "geometry": Point(coords)
    })

istasyon_gdf = gpd.GeoDataFrame(istasyon_records, geometry="geometry", crs="EPSG:4326")

# 2. İlçe sınırlarını oku
ilce_gdf = gpd.read_file(ilce_geojson_path)

# 3. Noktaları ilçelerle eşleştir (spatial join)
joined = gpd.sjoin(istasyon_gdf, ilce_gdf, how="left", predicate="within")

# 4. Gerekli sütunları al
result_df = joined[["PROJE_ADI", "ISTASYON", "name"]]
result_df.columns = ["Proje Adı", "İstasyon Adı", "İlçe"]

# 5. CSV olarak kaydet
os.makedirs("result", exist_ok=True)
result_df.to_csv("result/istasyon_ilce_eslesmesi.csv", index=False, encoding="utf-8-sig")

print("✅ CSV dosyası 'result/istasyon_ilce_eslesmesi.csv' olarak kaydedildi.")
