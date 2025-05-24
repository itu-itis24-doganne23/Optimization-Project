import pandas as pd
import geopandas as gpd
import folium
from difflib import get_close_matches
from unidecode import unidecode
from branca.colormap import linear

# === DOSYA YOLLARI ===
csv_path = 'optimum_yesil_alan_sonuclari.csv'             # CSV dosya adını kendi dosyanla değiştir
geojson_path = 'data/istanbul-districts.json'     # GeoJSON dosya adını kendi dosyanla değiştir

# === VERİYİ OKU ===
df = pd.read_csv(csv_path)
gdf = gpd.read_file(geojson_path)

# === İLÇE ADLARINI NORMALİZE ETME ===
def normalize(name):
    return unidecode(str(name)).casefold().strip()

geo_ilce_names = [normalize(x) for x in gdf['name']]
df['ILCE_normalized'] = df['ILCE'].apply(normalize)

# === EN YAKIN EŞLEŞMEYİ BUL ===
def match_ilce_name(name, geo_names):
    match = get_close_matches(normalize(name), geo_names, n=1, cutoff=0.6)
    return match[0] if match else None

df['Geojson_Ad'] = df['ILCE'].apply(lambda x: match_ilce_name(x, geo_ilce_names))

# === EŞLEŞMEYEN VAR MI? ===
unmatched = df[df['Geojson_Ad'].isnull()]
if not unmatched.empty:
    print("Eşleşmeyen ilçe isimleri:", unmatched['ILCE'].tolist())

# === EŞLEŞMİŞ VERİLERLE BİRLEŞTİR ===
gdf['name_normalized'] = gdf['name'].apply(normalize)
merged = gdf.merge(df, left_on='name_normalized', right_on='Geojson_Ad')

# === HARİTA OLUŞTUR ===
m = folium.Map(location=[41.0082, 28.9784], zoom_start=10)

# === RENK HARİTALARI TANIMLA ===
ga_colormap = linear.YlGn_09.scale(merged['Yeni_Yapilacak_Yesil_Alan_GA'].min(),
                                   merged['Yeni_Yapilacak_Yesil_Alan_GA'].max())
pso_colormap = linear.YlOrRd_09.scale(merged['Yeni_Yapilacak_Yesil_Alan_PSO'].min(),
                                      merged['Yeni_Yapilacak_Yesil_Alan_PSO'].max())

# === GA KATMANI ===
ga_layer = folium.FeatureGroup(name='Yeni Yeşil Alan (GA)')
folium.GeoJson(
    merged,
    style_function=lambda feature: {
        'fillColor': ga_colormap(feature['properties']['Yeni_Yapilacak_Yesil_Alan_GA']),
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.7,
    },
    tooltip=folium.GeoJsonTooltip(fields=['ILCE', 'Yeni_Yapilacak_Yesil_Alan_GA']),
).add_to(ga_layer)
ga_layer.add_to(m)
ga_colormap.caption = "Yeni Yeşil Alan (GA) - m²"
ga_colormap.add_to(m)

# === PSO KATMANI ===
pso_layer = folium.FeatureGroup(name='Yeni Yeşil Alan (PSO)')
folium.GeoJson(
    merged,
    style_function=lambda feature: {
        'fillColor': pso_colormap(feature['properties']['Yeni_Yapilacak_Yesil_Alan_PSO']),
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.7,
    },
    tooltip=folium.GeoJsonTooltip(fields=['ILCE', 'Yeni_Yapilacak_Yesil_Alan_PSO']),
).add_to(pso_layer)
pso_layer.add_to(m)
pso_colormap.caption = "Yeni Yeşil Alan (PSO) - m²"
pso_colormap.add_to(m)

# === KATMAN KONTROLÜ ===
folium.LayerControl().add_to(m)

# === HARİTAYI KAYDET ===
m.save("yesil_alan_haritasi.html")
print("Harita başarıyla oluşturuldu: yesil_alan_haritasi.html")
