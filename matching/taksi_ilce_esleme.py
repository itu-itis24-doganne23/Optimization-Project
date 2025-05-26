import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import json
import os

# Dosya yolları (gerekiyorsa güncelle)
taksi_json_path = "data/taxi_station.json"
ilce_geojson_path = "data/istanbul-districts.json"

# 1. Taksi durağı verisini oku
with open(taksi_json_path, "r", encoding="utf-8") as f:
    taksi_data = json.load(f)

taksi_records = []
for feature in taksi_data["features"]:
    props = feature["properties"]
    coords = feature["geometry"]["coordinates"]
    taksi_records.append({
        "DURAK_ADI": props["DURAK_ADI"],
        "geometry": Point(coords)
    })

taksi_gdf = gpd.GeoDataFrame(taksi_records, geometry="geometry", crs="EPSG:4326")

# 2. İlçe sınırlarını oku
ilce_gdf = gpd.read_file(ilce_geojson_path)

# 3. Spatial join ile durağı ilçeyle eşleştir
joined = gpd.sjoin(taksi_gdf, ilce_gdf, how="left", predicate="within")

# 4. Gerekli sütunları al
result_df = joined[["DURAK_ADI", "name"]]
result_df.columns = ["Durak Adı", "İlçe"]

# 5. Kaydet
os.makedirs("result", exist_ok=True)
result_df.to_csv("result/taksi_duraklari_ilceler.csv", index=False, encoding="utf-8-sig")

print("✅ CSV dosyası 'result/taksi_duraklari_ilceler.csv' olarak kaydedildi.")
