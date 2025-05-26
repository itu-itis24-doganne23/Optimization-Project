import pandas as pd
import os

# CSV dosyasını oku
df = pd.read_csv("istasyon_ilce_eslesmesi.csv")  # Kendi dosya adını yaz

# İlçeye göre grupla ve istasyonları say
istasyon_sayilari = df.groupby("İlçe", as_index=False).size()
istasyon_sayilari.columns = ["İlçe", "İstasyon Sayısı"]

# result klasörü yoksa oluştur
os.makedirs("result", exist_ok=True)

# Yeni dosyayı result klasörüne kaydet
istasyon_sayilari.to_csv("ilce_toplam_rayli.csv", index=False)
