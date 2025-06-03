import pandas as pd
import folium
from folium.plugins import HeatMap
import os

def create_stations_map():
    # CSV dosyasını oku
    stations_df = pd.read_csv('data/fuel_station.csv')
    
    # NaN değerleri temizle
    stations_df = stations_df.dropna(subset=['longitude', 'latitude'])
    
    # İstanbul'un merkez koordinatları
    istanbul_center = [41.0082, 28.9784]  # [enlem, boylam]
    
    # Harita oluştur
    m = folium.Map(location=istanbul_center, zoom_start=10)
    
    # İstasyonları haritaya ekle
    for _, row in stations_df.iterrows():
        # Koordinatları önce longitude sonra latitude olarak al
        longitude = float(row['longitude'])
        latitude = float(row['latitude'])
        
        # Popup içeriği
        popup_content = f"""
        <div style="width: 200px;">
            <strong>{row['adi']}</strong><br>
            <b>Marka:</b> {row['akaryakit_dagitim_sirketi_tnm']}<br>
            <b>Tür:</b> {row['is_nevi_tnm']}<br>
            <b>İlçe:</b> {row['ilce']}<br>
            <b>Mahalle:</b> {row['mahalle_adi']}
        </div>
        """
        
        # Marker ekle
        folium.Marker(
            location=[latitude, longitude],  # Folium için [enlem, boylam] sırası
            popup=folium.Popup(popup_content, max_width=300),
            icon=folium.Icon(color='orange', icon='info-sign')  # Rengi orange olarak değiştirdim
        ).add_to(m)
    
    # templates klasörünün varlığını kontrol et ve oluştur
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Haritayı kaydet
    map_path = os.path.join('templates', 'stations_map.html')
    m.save(map_path)
    print(f"Harita oluşturuldu: {map_path}")

if __name__ == "__main__":
    create_stations_map() 