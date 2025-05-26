import pandas as pd
import unicodedata

# İlçe adlarını normalize eden fonksiyon
def normalize_ilce(ad):
    ad = str(ad).strip().upper()
    ad = ad.replace("İ", "İ")  # Latin I with dot fix
    ad = ad.replace("İ", "I").replace("Ş", "S").replace("Ğ", "G") \
           .replace("Ü", "U").replace("Ö", "O").replace("Ç", "C")
    ad = ''.join(c for c in unicodedata.normalize('NFD', ad) if unicodedata.category(c) != 'Mn')
    return ad

# --- NÜFUS VERİSİNİ OKU VE DÜZENLE ---
raw_lines = []
with open("data/ilce_bazinda_nufus.csv", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("|")
        if len(parts) >= 3 and "İstanbul" in parts[1]:
            try:
                ilce_parca = parts[1].split("(")[1].split("/")[0]
                nufus = float(parts[2]) if parts[2] else 0.0
                raw_lines.append((ilce_parca, nufus))
            except:
                continue

nufus_df = pd.DataFrame(raw_lines, columns=["ILCE_RAW", "Nufus"])
nufus_df["ILCE"] = nufus_df["ILCE_RAW"].apply(normalize_ilce)


# Ana dosya
alan_df = pd.read_csv("result/ilce_toplam_alanlar.csv")
alan_df["ILCE_RAW"] = alan_df["ILCE"]
alan_df["ILCE"] = alan_df["ILCE"].apply(normalize_ilce)

# Minibüs
minibus_df = pd.read_csv("result/ilce_toplam_minibus.csv")
minibus_df.columns = ["ILCE_RAW", "Minibus_Durak_Sayisi"]
minibus_df["ILCE"] = minibus_df["ILCE_RAW"].apply(normalize_ilce)

# Raylı sistem
rayli_df = pd.read_csv("result/ilce_toplam_rayli.csv")
rayli_df.columns = ["ILCE_RAW", "Rayli_Istasyon_Sayisi"]
rayli_df["ILCE"] = rayli_df["ILCE_RAW"].apply(normalize_ilce)

# Taksi
taksi_df = pd.read_csv("result/ilce_toplam_taksi.csv")
taksi_df.columns = ["ILCE_RAW", "Taksi_Durak_Sayisi"]
taksi_df["ILCE"] = taksi_df["ILCE_RAW"].apply(normalize_ilce)

#hava kalitesi
aqi_df = pd.read_csv("result/hava_kalitesi_ortalamalari.csv")  # dosya adı aqi_verisi.csv olsun
aqi_df.columns = ["ILCE_RAW", "Ortalama_AQI"]
aqi_df["ILCE"] = aqi_df["ILCE_RAW"].apply(normalize_ilce)

# Merge işlemleri
merged = alan_df.merge(minibus_df[["ILCE", "Minibus_Durak_Sayisi"]], on="ILCE", how="left")
merged = merged.merge(rayli_df[["ILCE", "Rayli_Istasyon_Sayisi"]], on="ILCE", how="left")
merged = merged.merge(taksi_df[["ILCE", "Taksi_Durak_Sayisi"]], on="ILCE", how="left")
merged = merged.merge(nufus_df[["ILCE", "Nufus"]], on="ILCE", how="left")
merged = merged.merge(aqi_df[["ILCE", "Ortalama_AQI"]], on="ILCE", how="left")
# NaN'leri 0 yap
merged.fillna(0, inplace=True)

# Tipleri dönüştür
merged[["Minibus_Durak_Sayisi", "Rayli_Istasyon_Sayisi", "Taksi_Durak_Sayisi"]] = \
    merged[["Minibus_Durak_Sayisi", "Rayli_Istasyon_Sayisi", "Taksi_Durak_Sayisi"]].astype(int)

# Orijinal ilçe adlarını geri getir
merged.drop(columns=["ILCE"], inplace=True)
merged.rename(columns={"ILCE_RAW": "ILCE"}, inplace=True)
merged = merged[["ILCE", "alan_metrekare", "Nufus", "Minibus_Durak_Sayisi",
                 "Rayli_Istasyon_Sayisi", "Taksi_Durak_Sayisi", "Ortalama_AQI"]]
# Kaydet
merged.to_csv("result/birlesik_ilce_verisi.csv", index=False)

print("✅ İlçeler normalize edildi, eşleşmeler sağlandı, dosya başarıyla oluşturuldu.")
