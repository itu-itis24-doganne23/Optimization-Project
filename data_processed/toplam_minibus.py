import pandas as pd
import os

# CSV dosyasını oku
df = pd.read_csv("result/minibus_duraklari_ilceler.csv")  # Kendi dosya adını buraya yaz

# İlçeye göre grupla ve durakları say
durak_sayilari = df.groupby("İlçe", as_index=False).size()
durak_sayilari.columns = ["İlçe", "Durak Sayısı"]

# result klasörü yoksa oluştur
os.makedirs("result", exist_ok=True)

# Yeni dosyayı result klasörüne kaydet
durak_sayilari.to_csv("result/ilce_toplam_minibus.csv", index=False)
