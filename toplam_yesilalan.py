import pandas as pd
import os

# CSV dosyasını oku
df = pd.read_csv("result/ilce_yesil_alan_yuzolcumu.csv")  # Kendi dosya adını buraya yaz

# İlçeye göre grupla ve toplam alanı hesapla
toplamlar = df.groupby("ILCE", as_index=False)["alan_metrekare"].sum()

# result klasörü yoksa oluştur
os.makedirs("result", exist_ok=True)

# Yeni dosyayı result klasörüne kaydet
toplamlar.to_csv("result/ilce_toplam_alanlar.csv", index=False)