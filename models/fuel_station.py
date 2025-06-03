import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from geopy.distance import geodesic
from shapely.geometry import LineString, Point

class FuelStationRecommender:
    def __init__(self):
        self.stations = None
        self.tree = None
        
    def load_stations(self, csv_path):
        """CSV dosyasından istasyon verilerini yükle"""
        df = pd.read_csv(csv_path)
        
        # NaN değerleri temizle
        df['akaryakit_dagitim_sirketi_tnm'] = df['akaryakit_dagitim_sirketi_tnm'].fillna('Bilinmiyor')
        df['lpg_dagitim_sirketi_tnm'] = df['lpg_dagitim_sirketi_tnm'].fillna('Bilinmiyor')
        df['is_nevi_tnm'] = df['is_nevi_tnm'].fillna('Bilinmiyor')
        
        # Koordinatları kontrol et
        df = df.dropna(subset=['latitude', 'longitude'])
        
        self.stations = df
        # Koordinatları radyanlara çevir
        rad_coords = np.radians(df[['latitude', 'longitude']].values)
        # BallTree oluştur (hızlı yakın nokta araması için)
        self.tree = BallTree(rad_coords, leaf_size=40, metric='haversine')
        
    def find_nearest_stations(self, lon, lat, k=3, max_distance=2.0):
        """
        Verilen koordinatlara en yakın k adet akaryakıt istasyonunu bulur.
        
        Args:
            lon (float): Boylam
            lat (float): Enlem
            k (int): Bulunacak istasyon sayısı
            max_distance (float): Maksimum mesafe (km)
            
        Returns:
            list: En yakın istasyonların listesi
        """
        stations = []
        for _, row in self.stations.iterrows():
            # Koordinatları doğru sırayla kullan (önce longitude, sonra latitude)
            station_lon = row['longitude']
            station_lat = row['latitude']
            
            # Mesafeyi hesapla (lat, lon sırasıyla)
            distance = geodesic((lat, lon), (station_lat, station_lon)).km
            
            if distance <= max_distance:
                stations.append({
                    'name': row['adi'],
                    'brand': row['akaryakit_dagitim_sirketi_tnm'],
                    'type': row['is_nevi_tnm'],
                    'district': row['ilce'],
                    'neighborhood': row['mahalle_adi'],
                    'longitude': station_lon,  # Önce longitude
                    'latitude': station_lat,   # Sonra latitude
                    'distance': distance
                })
        
        # Mesafeye göre sırala ve en yakın k tanesini döndür
        stations.sort(key=lambda x: x['distance'])
        return stations[:k]
        
    def find_stations_on_route(self, start_lat, start_lon, end_lat, end_lon, max_distance=1.0):
        """Rota üzerindeki akaryakıt istasyonlarını bul"""
        try:
            # Başlangıç ve bitiş noktaları arasındaki doğrusal çizgiyi oluştur
            route_line = LineString([(start_lon, start_lat), (end_lon, end_lat)])
            
            # Tüm istasyonları kontrol et
            nearby_stations = []
            
            # İstasyonları kontrol et
            for _, station in self.stations.iterrows():
                try:
                    # İstasyon koordinatlarını al
                    station_lat = float(station['latitude'])
                    station_lon = float(station['longitude'])
                    
                    # İstasyon noktasını oluştur
                    station_point = Point(station_lon, station_lat)
                    
                    # İstasyonun rotaya olan mesafesini hesapla (km cinsinden)
                    distance = route_line.distance(station_point) * 111.32
                    
                    # Eğer istasyon rota üzerinde veya yakınındaysa listeye ekle
                    if distance <= max_distance:
                        nearby_stations.append({
                            'name': station['adi'],
                            'brand': station['akaryakit_dagitim_sirketi_tnm'],
                            'type': station['is_nevi_tnm'],
                            'district': station['ilce'],
                            'neighborhood': station['mahalle_adi'],
                            'latitude': station_lat,
                            'longitude': station_lon,
                            'distance': distance
                        })
                except Exception as e:
                    continue
            
            # Mesafeye göre sırala
            nearby_stations.sort(key=lambda x: x['distance'])
            
            return nearby_stations
        except Exception as e:
            print(f"Hata: {str(e)}")
            return []
        
    def get_station_details(self, lat, lon):
        """İstasyon detaylarını getir"""
        if self.tree is None:
            return None
            
        query_coords = np.radians([[lat, lon]])
        distances, indices = self.tree.query(query_coords, k=1)
        
        station = self.stations.iloc[indices[0][0]]
        return {
            'name': station['adi'],
            'brand': station['akaryakit_dagitim_sirketi_tnm'],
            'type': station['is_nevi_tnm'],
            'lpg_company': station['lpg_dagitim_sirketi_tnm'],
            'district': station['ilce'],
            'neighborhood': station['mahalle_adi'],
            'latitude': float(station['latitude']),
            'longitude': float(station['longitude'])
        }

    def find_stations_near_points(self, start_lat, start_lon, end_lat, end_lon, max_stations=7):
        """Başlangıç ve bitiş noktalarına yakın istasyonları bul"""
        try:
            # Başlangıç noktasına yakın istasyonları bul
            start_stations = []
            end_stations = []
            
            for _, station in self.stations.iterrows():
                try:
                    # Başlangıç noktasına olan mesafe
                    start_distance = geodesic(
                        (start_lat, start_lon),
                        (station['latitude'], station['longitude'])
                    ).kilometers
                    
                    # Bitiş noktasına olan mesafe
                    end_distance = geodesic(
                        (end_lat, end_lon),
                        (station['latitude'], station['longitude'])
                    ).kilometers
                    
                    # Başlangıç noktasına 2 km'den yakın istasyonları ekle
                    if start_distance <= 2.0:
                        start_stations.append({
                            'name': station['adi'],
                            'brand': station['akaryakit_dagitim_sirketi_tnm'],
                            'type': station['is_nevi_tnm'],
                            'district': station['ilce'],
                            'neighborhood': station['mahalle_adi'],
                            'longitude': float(station['longitude']),
                            'latitude': float(station['latitude']),
                            'distance': float(start_distance)
                        })
                    
                    # Bitiş noktasına 2 km'den yakın istasyonları ekle
                    if end_distance <= 2.0:
                        end_stations.append({
                            'name': station['adi'],
                            'brand': station['akaryakit_dagitim_sirketi_tnm'],
                            'type': station['is_nevi_tnm'],
                            'district': station['ilce'],
                            'neighborhood': station['mahalle_adi'],
                            'longitude': float(station['longitude']),
                            'latitude': float(station['latitude']),
                            'distance': float(end_distance)
                        })
                except (ValueError, TypeError) as e:
                    print(f"İstasyon verisi işlenirken hata: {str(e)}")
                    continue
            
            # İstasyonları mesafeye göre sırala
            start_stations.sort(key=lambda x: x['distance'])
            end_stations.sort(key=lambda x: x['distance'])
            
            # Her noktadan en yakın istasyonları al
            start_stations = start_stations[:max_stations//2]
            end_stations = end_stations[:max_stations//2]
            
            # Tüm istasyonları birleştir
            all_stations = start_stations + end_stations
            
            return all_stations
        except Exception as e:
            print(f"İstasyonlar bulunurken hata: {str(e)}")
            return [] 