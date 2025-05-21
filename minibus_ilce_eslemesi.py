import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import json
import os

minibus_json_path = "data/minibus_station.json"
ilce_geojson_path = "data/istanbul-districts.json"

with open(minibus_json_path, "r", encoding="utf-8") as f:
    minibus_data = json.load(f)

minibus_records = []
for feature in minibus_data["features"]:
    props = feature.get("properties", {})
    coords = feature.get("geometry", {}).get("coordinates", [])

    if not coords or len(coords) != 2:
        continue  # geçersiz nokta varsa atla

    minibus_records.append({
        "DURAK_ADI": props.get("DURAK_ADI", "Bilinmeyen"),
        "geometry": Point(coords)
    })

minibus_gdf = gpd.GeoDataFrame(minibus_records, geometry="geometry", crs="EPSG:4326")

ilce_gdf = gpd.read_file(ilce_geojson_path)

joined = gpd.sjoin(minibus_gdf, ilce_gdf, how="left", predicate="within")

result_df = joined[["DURAK_ADI", "name"]]
result_df.columns = ["Durak Adı", "İlçe"]

os.makedirs("result", exist_ok=True)
result_df.to_csv("result/minibus_duraklari_ilceler.csv", index=False, encoding="utf-8-sig")

print("✅ CSV dosyası 'result/minibus_duraklari_ilceler.csv' olarak kaydedildi.")
