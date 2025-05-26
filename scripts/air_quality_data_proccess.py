import requests
import json
from datetime import datetime
import time
import csv

API_BASE_URL = "https://api.ibb.gov.tr/havakalitesi/OpenDataPortalHandler/"

def get_stations():
    """Hava kalitesi ölçüm istasyonlarının listesini İBB API'sinden alır."""
    url = API_BASE_URL + "GetAQIStations"
    print(f"İstasyon listesi şu adresten alınıyor: {url}")
    try:
        response = requests.get(url, timeout=10) # 10 saniye timeout
        response.raise_for_status()  # HTTP hataları için (4xx veya 5xx) exception fırlatır
        return response.json()
    except requests.exceptions.Timeout:
        print("İstek zaman aşımına uğradı. Lütfen internet bağlantınızı kontrol edin veya daha sonra tekrar deneyin.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"İstasyonlar getirilirken bir hata oluştu: {e}")
        return None
    except json.JSONDecodeError:
        print("API'den gelen istasyon verisi JSON formatında değil. Yanıt:")
        # response değişkeni tanımlıysa metnini yazdır, değilse genel bir mesaj ver
        print(response.text if 'response' in locals() and hasattr(response, 'text') else "Yanıt alınamadı veya metin içeriği yok.")
        return None

def display_stations(stations):
    """İstasyon listesini kullanıcıya gösterir."""
    if not stations:
        print("Görüntülenecek istasyon bulunamadı.")
        return False # İstasyon bulunamadığını belirtmek için False döndür
    print("\n Mevcut Hava Kalitesi İstasyonları ")
    print("------------------------------------")
    for station in stations:
        # Bazı istasyon adları None olabilir, bunları kontrol edelim
        station_name = station.get('Name', 'İsim Yok')
        station_id = station.get('Id')
        if station_id: # Sadece ID'si olanları gösterelim
            print(f"ID: {station_id} - İsim: {station_name}")
    print("------------------------------------")
    # Bu fonksiyonun dönüş değeri main içerisinde kullanılmıyor, ancak orijinal yapıyı koruyoruz.
    # Normalde başarılı bir işlemde True veya veri döndürmek daha yaygındır.
    return True # Orijinal kodda (False, stations) vardı, bu da kafa karıştırıcı olabilir.
                 # Sadece başarılı olduğunu belirtmek için True döndürmek daha net olabilir.
                 # Veya doğrudan stations listesini döndürebilir. Şimdilik True olarak bırakıyorum.

def get_air_quality_data(station_id, start_date_str, end_date_str):
    """Belirli bir istasyon ve zaman aralığı için hava kalitesi verilerini alır."""
    url = API_BASE_URL + "GetAQIByStationId"
    
    params = {
        'StationId': station_id,
        'StartDate': start_date_str, # API'nin beklediği format: dd.MM.yyyy HH:mm:ss
        'EndDate': end_date_str
    }
    
    print(f"\n{station_id} ID'li istasyon için '{start_date_str}' ile '{end_date_str}' tarihleri arasında veri alınıyor...")
    # İstek URL'sini yazdırmak hata ayıklama için faydalı olabilir
    # print(f"İstek atılan URL (parametreler ile): {requests.Request('GET', url, params=params).prepare().url}")

    try:
        response = requests.get(url, params=params, timeout=30) # Veri sorgusu için daha uzun timeout
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        print("Veri isteği zaman aşımına uğradı. Lütfen internet bağlantınızı kontrol edin veya daha sonra tekrar deneyin.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Hava kalitesi verileri getirilirken bir hata oluştu: {e}")
        if response is not None: # response objesi oluşmuşsa detayları yazdır
            print(f"Sunucu Hatası Detayı: {response.status_code} - {response.text}")
        return None
    except json.JSONDecodeError:
        print("API'den gelen hava kalitesi verisi JSON formatında değil. Yanıt:")
        print(response.text if 'response' in locals() and hasattr(response, 'text') else "Yanıt alınamadı veya metin içeriği yok.")
        return None

def main():
    """Ana uygulama fonksiyonu."""

    stations = get_stations()
    # İstasyonların başarıyla alınıp alınmadığını kontrol etmek iyi bir pratiktir.
    if not stations:
        print("İstasyon verisi alınamadı. Program sonlandırılıyor.")
        return

    start_date_str = ["01.01.2024 00:00:00","01.06.2024 00:00:00"]
    end_date_str = ["01.06.2024 00:00:00","01.01.2025 00:00:00"]
    station_data_avg = {}

    for station in stations:
        # station sözlüğünde 'Id' ve 'Name' anahtarlarının varlığını .get() ile kontrol etmek daha güvenlidir.
        station_id = station.get("Id")
        station_name = station.get("Name")

        if station_id is None:
            print(f"Uyarı: İstasyon verisinde ID eksik, atlanıyor: {station}")
            continue
        if station_name is None:
            # İsim yoksa ID'yi kullanarak bir isim oluşturabiliriz veya atlayabiliriz.
            print(f"Uyarı: {station_id} ID'li istasyonun ismi yok, ID kullanılacak.")
            station_name = f"İstasyon_{station_id}"


        aqi_data = []
        for i in range(len(start_date_str)): # start_date_str ve end_date_str listelerinin uzunluğuna göre döngü
            air_quality_data = get_air_quality_data(station_id, start_date_str[i], end_date_str[i])


            if air_quality_data is not None:
                for air in air_quality_data:
                    # 'air' objesinin ve içindeki anahtarların varlığını kontrol etmek daha güvenli olur.
                    if isinstance(air, dict) and "AQI" in air and isinstance(air.get("AQI"), dict):
                        temp = air["AQI"].get("AQIIndex") # .get() None dönebilir, bu da 'if temp is not None' ile yakalanır.
                        if temp is not None:
                            try:
                                # AQI değerinin sayısal olduğundan emin olalım.
                                aqi_data.append(float(temp))
                            except (ValueError, TypeError):
                                print(f"  Uyarı: {station_name} istasyonu için geçersiz AQI değeri ({temp}) alındı, atlanıyor.")
                        # else: temp None ise zaten eklenmiyor, bu doğru.
                    else:
                        print(f"  Uyarı: {station_name} istasyonu için beklenmedik veri yapısı: {air}, atlanıyor.")
            else:
                print(f"  Bilgi: {station_name} ({station_id}) için {start_date_str[i]} - {end_date_str[i]} aralığında veri alınamadı.")

            
            time.sleep(0.5) # API'ye saygılı olmak için bekleme süresi


        if aqi_data:
            station_data_avg[station_name] = (sum(aqi_data) / len(aqi_data))
        else:
            # Eğer hiç geçerli AQI verisi toplanamadıysa, bunu belirt.
            station_data_avg[station_name] = None # Veya "Veri Yok" gibi bir işaretleyici
            print(f"Uyarı: {station_name} istasyonu için belirtilen periyotlarda hesaplanacak AQI verisi bulunamadı.")

            
    csv_file_name = "result/hava_kalitesi_ortalamalari.csv"
    try:
        with open(csv_file_name, mode='w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow(["İstasyon Adı", "Ortalama AQI"])
            for name, avg_aqi in station_data_avg.items():
                writer.writerow([name, avg_aqi if avg_aqi is not None else "Hesaplanamadı"])
        print(f"\nVeriler başarıyla '{csv_file_name}' dosyasına kaydedildi.")
    except IOError:
        print(f"\n'{csv_file_name}' dosyasına yazılırken bir hata oluştu.")


if __name__ == "__main__":
    main()
