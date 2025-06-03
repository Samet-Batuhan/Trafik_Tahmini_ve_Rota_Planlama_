import random
import osmnx as ox
import networkx as nx
from flask import Flask, render_template, request, session, jsonify, send_file
import folium
from folium.plugins import HeatMap
import pandas as pd
from geopy.distance import geodesic
from datetime import datetime, timezone, timedelta
import sqlite3
from collections import defaultdict
import os
from geopy.geocoders import Nominatim
import numpy as np
from traffic_predictor import TrafficPredictor
from geopy.exc import GeocoderTimedOut
import time
from route_recommender import RouteRecommender
from models.fuel_station import FuelStationRecommender
import ssl
import csv

# SSL sertifika doğrulamasını devre dışı bırak
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)
app.secret_key = 'gizli_anahtar123'  # Session için gerekli

# İstanbul saat dilimi (UTC+3)
ISTANBUL_TZ = timezone(timedelta(hours=3))

# Saatlik hız verileri (km/saat cinsinden)
hourly_speed = {
    0: 60, 1: 60, 2: 60, 3: 60, 4: 55, 5: 50, 6: 50, 7: 40,
    8: 30, 9: 35, 10: 45, 11: 50, 12: 40, 13: 45, 14: 50, 15: 40,
    16: 35, 17: 25, 18: 20, 19: 30, 20: 40, 21: 50, 22: 55, 23: 60
}

# Global değişkenler
traffic_predictor = TrafficPredictor()
route_recommender = RouteRecommender()
fuel_recommender = FuelStationRecommender()

# Veritabanı bağlantısı
def get_db_connection():
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'routes.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

# Veritabanı tablolarını oluştur
def init_db():
    """Veritabanı tablolarını oluşturur"""
    conn = sqlite3.connect('routes.db')
    cursor = conn.cursor()
    
    # Rotalar tablosu
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS routes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_address TEXT NOT NULL,
        end_address TEXT NOT NULL,
        start_lat REAL NOT NULL,
        start_lon REAL NOT NULL,
        end_lat REAL NOT NULL,
        end_lon REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Kaydedilen konumlar tablosu
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS saved_locations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        address TEXT NOT NULL,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    conn.commit()
    conn.close()

# Uygulama başlatıldığında öneri sistemini eğit
def initialize_recommender():
    try:
        # Rota öneri modelini eğit
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'routes.db')
        route_recommender.train(db_path)
        print("Rota öneri modeli başarıyla eğitildi")
        
        # Akaryakıt istasyonlarını yükle
        fuel_recommender.load_stations('data/fuel_station.csv')
        print("Akaryakıt istasyonları başarıyla yüklendi.")
        
    except Exception as e:
        print(f"Hata: {str(e)}")

# Veritabanını ve öneri sistemini başlat
with app.app_context():
    init_db()
    initialize_recommender()

# Veritabanındaki adresleri güncelle
def update_addresses_in_db():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Adresi olmayan kayıtları al
        cursor.execute('''
            SELECT id, start_lat, start_lon, end_lat, end_lon 
            FROM routes 
            WHERE start_address IS NULL OR end_address IS NULL
        ''')
        routes = cursor.fetchall()
        
        for route in routes:
            route_id, start_lat, start_lon, end_lat, end_lon = route
            
            # Başlangıç adresi güncelle
            start_address = get_address_from_coords(start_lat, start_lon)
            if start_address and not start_address.startswith('('):
                cursor.execute('UPDATE routes SET start_address = ? WHERE id = ?', 
                             (start_address, route_id))
            
            # Bitiş adresi güncelle
            end_address = get_address_from_coords(end_lat, end_lon)
            if end_address and not end_address.startswith('('):
                cursor.execute('UPDATE routes SET end_address = ? WHERE id = ?', 
                             (end_address, route_id))
            
            conn.commit()
            print(f"Rota {route_id} güncellendi: {start_address} -> {end_address}")
            
        conn.close()
        print("Adres güncellemesi tamamlandı.")
        return True
    except Exception as e:
        print(f"Adres güncelleme hatası: {str(e)}")
        return False

# Rota kaydetme fonksiyonu
def save_route(start_lat, start_lon, end_lat, end_lon, ispark_lat, ispark_lon, route_type, travel_time, total_distance, hour_of_day):
    try:
        # Başlangıç ve bitiş noktalarının adres bilgilerini al
        start_address = get_address_from_coords(start_lat, start_lon)
        end_address = get_address_from_coords(end_lat, end_lon)

        # Adres bilgileri alınamadıysa varsayılan değerler kullan
        if not start_address or start_address == "Konum bilgisi alınıyor...":
            start_address = f"Enlem: {start_lat:.6f}, Boylam: {start_lon:.6f}"
        if not end_address or end_address == "Konum bilgisi alınıyor...":
            end_address = f"Enlem: {end_lat:.6f}, Boylam: {end_lon:.6f}"

        conn = get_db_connection()
        cursor = conn.cursor()
        
        # İstanbul saatine göre tarihi dönüştür
        istanbul_time = datetime.now(ISTANBUL_TZ)
        
        cursor.execute("""
            INSERT INTO routes (
                start_lat, start_lon, end_lat, end_lon,
                ispark_lat, ispark_lon, route_type,
                travel_time, total_distance, hour_of_day,
                start_address, end_address, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            start_lat, start_lon, end_lat, end_lon,
            ispark_lat, ispark_lon, route_type,
            travel_time, total_distance, hour_of_day,
            start_address, end_address, istanbul_time
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Rota kaydedildi: {start_address} -> {end_address}")
        return True
    except Exception as e:
        print(f"Rota kaydetme hatası: {str(e)}")
        return False

# Kullanıcının rota geçmişini getir
def get_user_routes(limit=10):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('''
            SELECT id, start_lat, start_lon, end_lat, end_lon,
                   ispark_lat, ispark_lon, route_type, travel_time,
                   total_distance, hour_of_day, start_address,
                   end_address, created_at
            FROM routes 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
        routes = c.fetchall()
        conn.close()
        return routes
    except Exception as e:
        print(f"Rota geçmişi alınırken hata: {str(e)}")
        return []

# En çok kullanılan rotaları getir
def get_most_used_routes(limit=5):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('''
            SELECT route_type,
                   COUNT(*) as usage_count,
                   AVG(travel_time) as avg_travel_time,
                   AVG(total_distance) as avg_distance,
                   GROUP_CONCAT(DISTINCT start_address) as start_addresses,
                   GROUP_CONCAT(DISTINCT end_address) as end_addresses
            FROM routes
            GROUP BY route_type
            ORDER BY usage_count DESC
            LIMIT ?
        ''', (limit,))
        routes = c.fetchall()
        conn.close()
        return routes
    except Exception as e:
        print(f"En çok kullanılan rotalar alınırken hata: {str(e)}")
        return []

# Saatlik trafik yoğunluğu istatistiklerini getir
def get_hourly_traffic_stats():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('''
            SELECT hour_of_day,
                   COUNT(*) as route_count,
                   AVG(travel_time) as avg_travel_time,
                   AVG(total_distance) as avg_distance,
                   GROUP_CONCAT(DISTINCT start_address) as common_starts,
                   GROUP_CONCAT(DISTINCT end_address) as common_ends
            FROM routes
            GROUP BY hour_of_day
            ORDER BY hour_of_day
        ''')
        stats = c.fetchall()
        conn.close()
        return stats
    except Exception as e:
        print(f"Saatlik istatistikler alınırken hata: {str(e)}")
        return []

# Yol ağını GraphML dosyasından yüklemek
def load_graph_from_file(graph_file):
    try:
        G = ox.load_graphml(graph_file)
        print("Yol ağı başarıyla yüklendi.")
        return G
    except Exception as e:
        print(f"Yol ağı yüklenirken bir hata oluştu: {e}")
        return None

# Trafik verisini yükleme fonksiyonu (CSV dosyasından)
def load_traffic_data(csv_file):
    try:
        traffic_data = pd.read_csv(csv_file)
        print("Trafik verisi başarıyla yüklendi.")
        return traffic_data
    except Exception as e:
        print(f"Trafik verisi yüklenirken bir hata oluştu: {e}")
        return None

# İSPARK verisini yükleme fonksiyonu (CSV dosyasından)
def load_ispark_data(csv_file):
    try:
        ispark_data = pd.read_csv(csv_file)
        print("İSPARK verisi başarıyla yüklendi.")
        return ispark_data
    except Exception as e:
        print(f"İSPARK verisi yüklenirken bir hata oluştu: {e}")
        return None
    
    

# Geçerli enlem ve boylam verilerini filtrele
def clean_ispark_data(ispark_data):
    valid_ispark_data = ispark_data[
        (ispark_data['LATITUDE'] >= -90) & (ispark_data['LATITUDE'] <= 90) &
        (ispark_data['LONGITUDE'] >= -180) & (ispark_data['LONGITUDE'] <= 180)
    ]
    return valid_ispark_data

# En kısa rota hesaplama fonksiyonu
def calculate_shortest_route(G, start_lat, start_lon, end_lat, end_lon):
    try:
        start_node = ox.distance.nearest_nodes(G, X=[start_lon], Y=[start_lat])[0]
        end_node = ox.distance.nearest_nodes(G, X=[end_lon], Y=[end_lat])[0]
        route = nx.shortest_path(G, start_node, end_node, weight='length')
        return route
    except Exception as e:
        print(f"Rota hesaplama hatası: {str(e)}")
        return None

# Trafik yoğunluğu ve rota üzerindeki ısı haritası oluşturma
def generate_heatmap_on_route(G, route, hour_of_day, traffic_data):
    traffic_data_points = []
    for node in route:
        traffic_data_points.append([G.nodes[node]['y'], G.nodes[node]['x'], random.randint(20, 100)])
    return traffic_data_points

# Seyahat süresi hesaplama fonksiyonu
def calculate_travel_time(G, route, hour_of_day):
    total_distance = 0
    for i in range(len(route) - 1):
        node1 = route[i]
        node2 = route[i + 1]
        distance = geodesic(
            (G.nodes[node1]['y'], G.nodes[node1]['x']),
            (G.nodes[node2]['y'], G.nodes[node2]['x'])
        ).meters
        total_distance += distance
    average_speed = hourly_speed.get(hour_of_day, 50)  # Ortalama hız (km/saat)
    travel_time_hours = total_distance / (average_speed * 1000)  # Saat cinsinden seyahat süresi
    travel_time_minutes = travel_time_hours * 60  # Dakikaya dönüştür
    return round(travel_time_minutes, 1)

# En yakın İSPARK noktasını bulma fonksiyonu
def find_nearest_ispark(start_lat, start_lon, ispark_data):
    min_distance = float("inf")
    nearest_ispark = None
    for _, row in ispark_data.iterrows():
        ispark_lat = row['LATITUDE']
        ispark_lon = row['LONGITUDE']

        if not (-90 <= ispark_lat <= 90 and -180 <= ispark_lon <= 180):
            continue

        distance = geodesic((start_lat, start_lon), (ispark_lat, ispark_lon)).km
        if distance < min_distance:
            min_distance = distance
            nearest_ispark = row
    return nearest_ispark

# Harita oluşturma fonksiyonu
def generate_map_with_route_and_heatmap(start_lat, start_lon, end_lat, end_lon, route, G, traffic_data_points, ispark_lat, ispark_lon):
    """Rota ve ısı haritası ile birlikte harita oluşturur"""
    # Haritayı oluştur
    m = folium.Map(location=[start_lat, start_lon], zoom_start=12)
    
    print(f"Başlangıç koordinatları: {start_lat}, {start_lon}")
    print(f"Bitiş koordinatları: {end_lat}, {end_lon}")
    
    # Başlangıç noktasına yakın istasyonları bul (2 km içinde)
    start_stations = fuel_recommender.find_nearest_stations(lon=start_lon, lat=start_lat, k=5, max_distance=2.0)
    print(f"Başlangıç noktasına yakın istasyonlar: {len(start_stations)}")
    for station in start_stations:
        print(f"Başlangıç istasyonu: {station['name']}, {station['latitude']}, {station['longitude']}")
    
    # Bitiş noktasına yakın istasyonları bul (2 km içinde)
    end_stations = fuel_recommender.find_nearest_stations(lon=end_lon, lat=end_lat, k=5, max_distance=2.0)
    print(f"Bitiş noktasına yakın istasyonlar: {len(end_stations)}")
    for station in end_stations:
        print(f"Bitiş istasyonu: {station['name']}, {station['latitude']}, {station['longitude']}")
    
    # Başlangıç noktası (yeşil marker)
    folium.Marker(
        [start_lat, start_lon],
        popup="Başlangıç Noktası",
        icon=folium.Icon(color='green', icon='info-sign')
    ).add_to(m)

    # Bitiş noktası (kırmızı marker)
    folium.Marker(
        [end_lat, end_lon],
        popup="Bitiş Noktası",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)

    # İSPARK noktası varsa ekle (mavi marker)
    if ispark_lat is not None and ispark_lon is not None:
        folium.Marker(
            [ispark_lat, ispark_lon],
            popup="En Yakın İSPARK",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)

    # Başlangıç noktasına yakın istasyonları siyah marker ile işaretle
    for station in start_stations:
        popup_content = f"""
        <div style="width: 200px;">
            <strong>{station['name']}</strong><br>
            <b>Marka:</b> {station['brand']}<br>
            <b>Tür:</b> {station['type']}<br>
            <b>İlçe:</b> {station['district']}<br>
            <b>Mahalle:</b> {station['neighborhood']}<br>
            <b>Başlangıca Uzaklık:</b> {station['distance']:.2f} km
        </div>
        """
        print(f"Siyah marker ekleniyor: {station['latitude']}, {station['longitude']}")
        folium.Marker(
            location=[station['latitude'], station['longitude']],  # [lat, lon] sırası
            popup=folium.Popup(popup_content, max_width=300),
            icon=folium.Icon(color='black', icon='info-sign')
        ).add_to(m)

    # Bitiş noktasına yakın istasyonları siyah marker ile işaretle
    for station in end_stations:
        popup_content = f"""
        <div style="width: 200px;">
            <strong>{station['name']}</strong><br>
            <b>Marka:</b> {station['brand']}<br>
            <b>Tür:</b> {station['type']}<br>
            <b>İlçe:</b> {station['district']}<br>
            <b>Mahalle:</b> {station['neighborhood']}<br>
            <b>Bitişe Uzaklık:</b> {station['distance']:.2f} km
        </div>
        """
        print(f"Siyah marker ekleniyor: {station['latitude']}, {station['longitude']}")
        folium.Marker(
            location=[station['latitude'], station['longitude']],  # [lat, lon] sırası
            popup=folium.Popup(popup_content, max_width=300),
            icon=folium.Icon(color='black', icon='info-sign')
        ).add_to(m)

    # Rotayı çiz (mavi çizgi)
    route_coordinates = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]
    folium.PolyLine(
        route_coordinates,
        color='blue',
        weight=5,
        opacity=0.7
    ).add_to(m)

    # Trafik yoğunluğu ısı haritası
    HeatMap(traffic_data_points).add_to(m)

    # Haritayı kaydet
    map_path = "static/map_route.html"  # static klasörüne kaydet
    m.save(map_path)
    print("Harita kaydedildi:", map_path)

    return "/static/map_route.html"  # URL yolunu döndür

# Rota üzerindeki toplam mesafeyi hesaplama
def calculate_total_distance(G, route):
    total_distance = 0
    for i in range(len(route) - 1):
        node1 = route[i]
        node2 = route[i + 1]
        distance = geodesic(
            (G.nodes[node1]['y'], G.nodes[node1]['x']),
            (G.nodes[node2]['y'], G.nodes[node2]['x'])
        ).meters  # Mesafe metre cinsinden hesaplanır
        total_distance += distance
    return total_distance / 1000  # Kilometreye çevir

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Kullanıcıdan başlangıç ve bitiş bilgilerini al
            start_coords = request.form.get("start_coords")
            end_coords = request.form.get("end_coords")
            route_type = request.form.get("route_type", "direct")  # Varsayılan olarak direkt rota

            # Eğer start_coords veya end_coords boşsa hata döndür
            if not start_coords or not end_coords:
                return "Hata: Başlangıç veya bitiş koordinatları boş olamaz.", 400

            # Koordinatları ayır ve doğruluğunu kontrol et
            try:
                start_lat, start_lon = map(float, start_coords.split(","))
                end_lat, end_lon = map(float, end_coords.split(","))
            except ValueError:
                return "Hata: Koordinatlar düzgün bir formatta değil. Enlem ve boylam virgülle ayrılmış olmalıdır.", 400

            # Saat bilgisini al
            hour_of_day = request.form.get("hour_of_day")
            if not hour_of_day:
                return "Hata: Saat seçimi yapılmadı.", 400

            hour_of_day = int(hour_of_day)

            # Yol ağı dosyasını yükle
            graph_file = "data/istanbul_yol_agi.graphml"
            G = load_graph_from_file(graph_file)

            if G is None:
                return "Yol ağı yüklenemedi.", 500

            # Trafik veri setini yükle
            traffic_data_file = "data/eylul_ayi.csv"
            traffic_data = load_traffic_data(traffic_data_file)

            if traffic_data is None:
                return "Trafik verisi yüklenemedi.", 500

            # İSPARK veri setini yükle ve temizle
            ispark_data_file = "data/ispark.csv"
            ispark_data = load_ispark_data(ispark_data_file)

            if ispark_data is None:
                return "İSPARK verisi yüklenemedi.", 500

            ispark_data = clean_ispark_data(ispark_data)

            # Varsayılan değerler
            ispark_lat = None
            ispark_lon = None

            # Rota tipine göre farklı rotalar oluştur
            if route_type == "direct":
                # Direkt rota hesapla
                route = calculate_shortest_route(G, start_lat, start_lon, end_lat, end_lon)
                if route is None:
                    return "Rota hesaplanamadı.", 500
                
                total_distance = calculate_total_distance(G, route)
                travel_time = calculate_travel_time(G, route, hour_of_day)
                heatmap_points = generate_heatmap_on_route(G, route, hour_of_day, traffic_data)
            else:
                # İSPARK'lı rota hesapla
                nearest_ispark = find_nearest_ispark(end_lat, end_lon, ispark_data)
                
                if nearest_ispark is not None:
                    ispark_lat = float(nearest_ispark['LATITUDE'])
                    ispark_lon = float(nearest_ispark['LONGITUDE'])
                    
                    # Başlangıçtan İSPARK'a rota
                    route_to_ispark = calculate_shortest_route(G, start_lat, start_lon, ispark_lat, ispark_lon)
                    
                    # İSPARK'tan hedefe rota
                    route_from_ispark = calculate_shortest_route(G, ispark_lat, ispark_lon, end_lat, end_lon)
                    
                    if route_to_ispark is None or route_from_ispark is None:
                        return "Rota hesaplanamadı.", 500
                    
                    # Toplam rotayı birleştir
                    route = route_to_ispark + route_from_ispark[1:]
                    
                    total_distance = calculate_total_distance(G, route)
                    travel_time = calculate_travel_time(G, route, hour_of_day)
                    heatmap_points = generate_heatmap_on_route(G, route, hour_of_day, traffic_data)
                else:
                    return "Yakında uygun İSPARK bulunamadı.", 400

            # Heatmap ve güzergahı çizerek haritayı oluştur
            map_route_path = generate_map_with_route_and_heatmap(
                start_lat, start_lon, end_lat, end_lon, route, G, heatmap_points, ispark_lat, ispark_lon
            )

            # Adres bilgilerini al
            start_address = get_address_from_coords(start_lat, start_lon)
            end_address = get_address_from_coords(end_lat, end_lon)

            # Rota bilgilerini veritabanına kaydet
            save_route(
                start_lat=start_lat,
                start_lon=start_lon,
                end_lat=end_lat,
                end_lon=end_lon,
                ispark_lat=ispark_lat,
                ispark_lon=ispark_lon,
                route_type=route_type,
                travel_time=travel_time,
                total_distance=total_distance,
                hour_of_day=hour_of_day
            )

            return render_template(
                "index3.html",
                estimated_travel_time_minutes=travel_time,
                map_route_path=map_route_path,
                total_distance=round(total_distance, 2),
                selected_hour=hour_of_day,
                saved_locations=get_saved_locations().json['locations'],
                start_point=start_address,
                end_point=end_address,
                predicted_time=f"{travel_time:.1f} dakika",
                actual_time=f"{travel_time:.1f} dakika"
            )

        except Exception as e:
            print(f"Beklenmeyen hata: {str(e)}")
            return f"Beklenmeyen bir hata oluştu: {str(e)}", 500

    # GET isteği için
    try:
        conn = sqlite3.connect('routes.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, address, latitude, longitude FROM saved_locations ORDER BY created_at DESC')
        locations = cursor.fetchall()
        conn.close()

        saved_locations = []
        for loc in locations:
            saved_locations.append({
                'id': loc[0],
                'name': loc[1],
                'address': loc[2],
                'latitude': loc[3],
                'longitude': loc[4]
            })

        return render_template("index3.html", saved_locations=saved_locations)
    except Exception as e:
        print('Hata:', str(e))
        return render_template("index3.html", saved_locations=[])

@app.route("/stats")
def stats():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Toplam rota sayısını al
        cursor.execute("SELECT COUNT(*) FROM routes")
        total_routes = cursor.fetchone()[0]

        # Son rotaları al
        cursor.execute("""
            SELECT route_type, travel_time, total_distance, hour_of_day,
                   start_address, end_address, created_at
            FROM routes
            ORDER BY created_at DESC
            LIMIT 10
        """)
        recent_routes = cursor.fetchall()
        recent_routes_list = []
        for route in recent_routes:
            # Tarihi doğrudan İstanbul saatine göre dönüştür
            created_at = datetime.fromisoformat(route[6])
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=ISTANBUL_TZ)
            recent_routes_list.append({
                'route_type': route[0],
                'travel_time': route[1],
                'total_distance': route[2],
                'hour_of_day': route[3],
                'start_address': route[4] or 'Bilinmiyor',
                'end_address': route[5] or 'Bilinmiyor',
                'created_at': created_at.strftime('%Y-%m-%d %H:%M:%S')
            })

        # En çok kullanılan rotaları ilçe bazında al (sadece ilk 5)
        cursor.execute("""
            WITH ParsedAddresses AS (
                SELECT 
                    id,
                    route_type,
                    SUBSTR(start_address, INSTR(start_address, ', ') + 2) as start_district,
                    SUBSTR(end_address, INSTR(end_address, ', ') + 2) as end_district,
                    travel_time,
                    total_distance
                FROM routes
                WHERE start_address IS NOT NULL 
                AND end_address IS NOT NULL
            ),
            DistrictRoutes AS (
                SELECT 
                    start_district,
                    end_district,
                    route_type,
                    COUNT(*) as usage_count,
                    AVG(travel_time) as avg_travel_time,
                    AVG(total_distance) as avg_distance,
                    MIN(start_district) as display_start,
                    MIN(end_district) as display_end
                FROM ParsedAddresses
                GROUP BY start_district, end_district, route_type
                HAVING usage_count >= 2
                ORDER BY usage_count DESC
                LIMIT 5
            )
            SELECT 
                route_type,
                usage_count,
                avg_travel_time,
                avg_distance,
                display_start as start_addresses,
                display_end as end_addresses
            FROM DistrictRoutes
        """)
        most_used_routes = cursor.fetchall()
        most_used_routes_list = []
        for route in most_used_routes:
            most_used_routes_list.append({
                'route_type': route[0],
                'usage_count': route[1],
                'avg_travel_time': route[2],
                'avg_distance': route[3],
                'start_addresses': route[4] if route[4] else 'Bilinmiyor',
                'end_addresses': route[5] if route[5] else 'Bilinmiyor'
            })

        # Saatlik istatistikleri al
        cursor.execute("""
            SELECT hour_of_day,
                   COUNT(*) as route_count,
                   AVG(travel_time) as avg_travel_time,
                   AVG(total_distance) as avg_distance
            FROM routes
            GROUP BY hour_of_day
            ORDER BY hour_of_day
        """)
        hourly_stats = cursor.fetchall()
        hourly_stats_list = []
        for stat in hourly_stats:
            hourly_stats_list.append({
                'hour_of_day': stat[0],
                'route_count': stat[1],
                'avg_travel_time': stat[2],
                'avg_distance': stat[3]
            })

        # Rota Tipi Dağılımını Al (Yeni Eklenen Kısım)
        cursor.execute("""
            SELECT route_type, COUNT(*) as count
            FROM routes
            WHERE route_type IS NOT NULL
            GROUP BY route_type
        """)
        route_type_distribution = cursor.fetchall()
        route_type_dist_dict = {row['route_type']: row['count'] for row in route_type_distribution}
        # --- Yeni Eklenen Kısım Sonu ---

        conn.close()
        
        return render_template("stats.html",
                             total_routes=total_routes,
                             recent_routes=recent_routes_list,
                             most_used_routes=most_used_routes_list,
                             hourly_stats=hourly_stats_list,
                             route_type_distribution=route_type_dist_dict)
    except Exception as e:
        print(f"İstatistik sayfası hatası: {str(e)}")
        return "İstatistikler yüklenirken bir hata oluştu.", 500

@app.route('/get_stats')
def get_stats():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Toplam rota sayısını al
        cursor.execute("SELECT COUNT(*) FROM routes")
        total_routes = cursor.fetchone()[0]

        # Son rotaları al
        cursor.execute("""
            SELECT route_type, travel_time, total_distance, hour_of_day,
                   start_address, end_address, created_at
            FROM routes
            ORDER BY created_at DESC
            LIMIT 10
        """)
        recent_routes = cursor.fetchall()
        recent_routes_list = []
        for route in recent_routes:
            # İstanbul saatine göre tarihi dönüştür
            created_at = datetime.fromisoformat(route[6]).replace(tzinfo=timezone.utc).astimezone(ISTANBUL_TZ)
            recent_routes_list.append({
                'start_address': route[4] if route[4] else 'Bilinmiyor',
                'end_address': route[5] if route[5] else 'Bilinmiyor',
                'route_type': route[0],
                'travel_time': route[1],
                'total_distance': route[2],
                'hour_of_day': route[3],
                'created_at': created_at.strftime('%Y-%m-%d %H:%M:%S')
            })

        # En çok kullanılan rotaları ilçe bazında al (sadece ilk 5)
        cursor.execute("""
            WITH ParsedAddresses AS (
                SELECT 
                    id,
                    route_type,
                    SUBSTR(start_address, INSTR(start_address, ', ') + 2) as start_district,
                    SUBSTR(end_address, INSTR(end_address, ', ') + 2) as end_district,
                    travel_time,
                    total_distance
                FROM routes
                WHERE start_address IS NOT NULL 
                AND end_address IS NOT NULL
            ),
            DistrictRoutes AS (
                SELECT 
                    start_district,
                    end_district,
                    route_type,
                    COUNT(*) as usage_count,
                    AVG(travel_time) as avg_travel_time,
                    AVG(total_distance) as avg_distance,
                    MIN(start_district) as display_start,
                    MIN(end_district) as display_end
                FROM ParsedAddresses
                GROUP BY start_district, end_district, route_type
                HAVING usage_count >= 2
                ORDER BY usage_count DESC
                LIMIT 5
            )
            SELECT 
                route_type,
                usage_count,
                avg_travel_time,
                avg_distance,
                display_start as start_addresses,
                display_end as end_addresses
            FROM DistrictRoutes
        """)
        most_used_routes = cursor.fetchall()
        most_used_routes_list = []
        for route in most_used_routes:
            most_used_routes_list.append({
                'route_type': route[0],
                'usage_count': route[1],
                'avg_travel_time': route[2],
                'avg_distance': route[3],
                'start_addresses': route[4] if route[4] else 'Bilinmiyor',
                'end_addresses': route[5] if route[5] else 'Bilinmiyor'
            })

        # Saatlik istatistikleri al
        cursor.execute("""
            SELECT hour_of_day,
                   COUNT(*) as route_count,
                   AVG(travel_time) as avg_travel_time,
                   AVG(total_distance) as avg_distance
            FROM routes
            GROUP BY hour_of_day
            ORDER BY hour_of_day
        """)
        hourly_stats = cursor.fetchall()
        hourly_stats_list = []
        for stat in hourly_stats:
            hourly_stats_list.append({
                'hour_of_day': stat[0],
                'route_count': stat[1],
                'avg_travel_time': stat[2],
                'avg_distance': stat[3]
            })

        # Rota Tipi Dağılımını Al (Yeni Eklenen Kısım)
        cursor.execute("""
            SELECT route_type, COUNT(*) as count
            FROM routes
            WHERE route_type IS NOT NULL
            GROUP BY route_type
        """)
        route_type_distribution = cursor.fetchall()
        route_type_dist_dict = {row['route_type']: row['count'] for row in route_type_distribution}
        # --- Yeni Eklenen Kısım Sonu ---

        conn.close()

        return jsonify({
            'total_routes': total_routes,
            'recent_routes': recent_routes_list,
            'most_used_routes': most_used_routes_list,
            'hourly_stats': hourly_stats_list,
            'route_type_distribution': route_type_dist_dict
        })

    except Exception as e:
        print(f"İstatistikler alınırken hata oluştu: {str(e)}")
        return jsonify({
            'error': 'İstatistikler alınırken bir hata oluştu',
            'total_routes': 0,
            'recent_routes': [],
            'most_used_routes': [],
            'hourly_stats': [],
            'route_type_distribution': {}
        })

# Kayıtlı konumları getir
@app.route('/get_saved_locations', methods=['GET'])
def get_saved_locations():
    try:
        conn = sqlite3.connect('routes.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, address, latitude, longitude FROM saved_locations ORDER BY created_at DESC')
        locations = cursor.fetchall()
        conn.close()

        saved_locations = []
        for loc in locations:
            saved_locations.append({
                'id': loc[0],
                'name': loc[1],
                'address': loc[2],
                'latitude': loc[3],
                'longitude': loc[4]
            })

        return jsonify({'status': 'success', 'locations': saved_locations})
    except Exception as e:
        print('Hata:', str(e))
        return jsonify({'status': 'error', 'message': 'Konumlar yüklenirken bir hata oluştu'})

# Yeni konum kaydet
def save_location(name, latitude, longitude, address):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('''
            INSERT INTO saved_locations (name, latitude, longitude, address)
            VALUES (?, ?, ?, ?)
        ''', (name, latitude, longitude, address))
        conn.commit()
        return True
    except Exception as e:
        print(f"Konum kaydedilirken hata: {e}")
        return False
    finally:
        conn.close()

# Konum sil
@app.route('/delete_location/<int:location_id>', methods=['DELETE'])
def delete_location(location_id):
    try:
        conn = sqlite3.connect('routes.db')
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM saved_locations WHERE id = ?", (location_id,))
        
        conn.commit()
        conn.close()
        
        return jsonify({'status': 'success', 'message': 'Konum başarıyla silindi'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save_location', methods=['POST'])
def save_location_route():
    try:
        data = request.json
        name = data.get('name')
        latitude = float(data.get('latitude'))
        longitude = float(data.get('longitude'))
        address = data.get('address')
        
        if not all([name, latitude, longitude]):
            return jsonify({'status': 'error', 'message': 'Eksik bilgi'}), 400
        
        if save_location(name, latitude, longitude, address):
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': 'Konum kaydedilemedi'}), 500
    except Exception as e:
        print(f"Konum kaydetme hatası: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_route', methods=['POST'])
def get_route():
    data = request.json
    start_address = data.get('start')
    end_address = data.get('end')
    
    # Başlangıç ve bitiş noktalarının koordinatlarını al
    geolocator = Nominatim(user_agent="my_agent")
    start_location = geolocator.geocode(start_address)
    end_location = geolocator.geocode(end_address)
    
    if not start_location or not end_location:
        return jsonify({'error': 'Adres bulunamadı'})
    
    start_point = (start_location.latitude, start_location.longitude)
    end_point = (end_location.latitude, end_location.longitude)
    
    # En yakın İSPARK'ı bul
    min_distance = float('inf')
    nearest_ispark = None
    
    for _, ispark in ispark_data.iterrows():
        ispark_point = (ispark['latitude'], ispark['longitude'])
        distance = geodesic(start_point, ispark_point).kilometers
        if distance < min_distance:
            min_distance = distance
            nearest_ispark = ispark
    
    if nearest_ispark is None:
        return jsonify({'error': 'Yakında İSPARK bulunamadı'})
    
    # Başlangıç noktasından İSPARK'a rota
    start_to_ispark = ox.shortest_path(G, 
                                     ox.nearest_nodes(G, start_point[1], start_point[0]),
                                     ox.nearest_nodes(G, nearest_ispark['longitude'], nearest_ispark['latitude']),
                                     weight='length')
    
    # İSPARK'tan varış noktasına rota
    ispark_to_end = ox.shortest_path(G,
                                   ox.nearest_nodes(G, nearest_ispark['longitude'], nearest_ispark['latitude']),
                                   ox.nearest_nodes(G, end_point[1], end_point[0]),
                                   weight='length')
    
    # Rotaları birleştir
    route = start_to_ispark + ispark_to_end[1:]
    
    # Harita oluştur
    m = folium.Map(location=[start_point[0], start_point[1]], zoom_start=12)
    
    # Başlangıç noktası
    folium.Marker(
        [start_point[0], start_point[1]],
        popup='Başlangıç',
        icon=folium.Icon(color='green', icon='info-sign')
    ).add_to(m)
    
    # İSPARK noktası
    folium.Marker(
        [nearest_ispark['latitude'], nearest_ispark['longitude']],
        popup=f'İSPARK: {nearest_ispark["name"]}',
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    # Varış noktası
    folium.Marker(
        [end_point[0], end_point[1]],
        popup='Varış',
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)
    
    # Rotayı çiz
    route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]
    folium.PolyLine(
        route_coords,
        weight=3,
        color='red',
        opacity=0.8
    ).add_to(m)
    
    # Haritayı kaydet
    m.save('templates/map_route.html')
    
    return jsonify({
        'status': 'success',
        'map_url': '/static/map_route.html',
        'ispark_name': nearest_ispark['name'],
        'ispark_address': nearest_ispark['address']
    })

@app.route('/predict_traffic', methods=['POST'])
def predict_traffic():
    try:
        data = request.get_json()
        lat = data.get('lat')
        lon = data.get('lon')
        hour_of_day = data.get('hour_of_day', datetime.now().hour)

        if lat is None or lon is None:
            return jsonify({'error': 'Koordinatlar eksik'}), 400

        # Trafik modelinden tahmin yap
        predicted_speed = traffic_predictor.predict(lat, lon, hour_of_day)
        
        # Hızı makul bir aralıkta tut
        predicted_speed = max(10, min(120, predicted_speed))
        predicted_speed_rounded = round(predicted_speed, 1)  # Hızı bir ondalık basamağa yuvarla
        
        # Trafik yoğunluğunu ve açıklamasını belirle
        if predicted_speed < 38:
            traffic_status = "Yoğun Trafik"
            traffic_description = f"Tahmini hız {predicted_speed_rounded} km/sa. Trafik yoğun."
        elif 39 <= predicted_speed <= 45:
            traffic_status = "Orta Yoğunluk"
            traffic_description = f"Tahmini hız {predicted_speed_rounded} km/sa. Trafik orta yoğunlukta."
        else:
            traffic_status = "Açık Trafik"
            traffic_description = f"Tahmini hız {predicted_speed_rounded} km/sa. Trafik açık."

        return jsonify({
            'predicted_speed': predicted_speed_rounded,
            'speed_unit': 'km/sa',
            'traffic_status': traffic_status,
            'traffic_description': traffic_description,
            'status': 'success'
        })

    except Exception as e:
        print(f"Tahmin hatası: {str(e)}")
        # Hata durumunda da benzer bir formatta yanıt dön
        return jsonify({
            'error': f'Tahmin sırasında bir hata oluştu: {str(e)}',
            'predicted_speed': None,
            'speed_unit': 'km/sa',
            'traffic_status': 'Bilinmiyor',
            'traffic_description': 'Trafik durumu tahmin edilemedi.',
            'status': 'error'
        }), 500

@app.route('/predict_route_time', methods=['POST'])
def predict_route_time():
    try:
        data = request.get_json()
        start_lat = data.get('start_lat') 
        start_lon = data.get('start_lon')
        end_lat = data.get('end_lat')
        end_lon = data.get('end_lon')
        hour_of_day = data.get('hour_of_day', datetime.now().hour)

        if not all([start_lat, start_lon, end_lat, end_lon]):
            return jsonify({'error': 'Eksik koordinatlar'}), 400

        # İki nokta arasındaki mesafeyi hesapla (km cinsinden)
        distance = calculate_distance(start_lat, start_lon, end_lat, end_lon)
        
        # Her iki nokta için de trafik tahminlerini al
        start_speed = traffic_predictor.predict(start_lat, start_lon, hour_of_day)
        end_speed = traffic_predictor.predict(end_lat, end_lon, hour_of_day)
        
        # Ortalama hızı hesapla
        avg_speed = (start_speed + end_speed) / 2  # km/saat
        
        # Tahmini süreyi hesapla (dakika cinsinden)
        if avg_speed > 0:
            estimated_time = (distance / avg_speed) * 60  # dakika
        else:
            estimated_time = None

        return jsonify({
            'estimated_time': estimated_time,
            'distance': distance,
            'avg_speed': avg_speed
        })

    except Exception as e:
        print(f"Rota süresi tahmin hatası: {str(e)}")
        return jsonify({'error': str(e)}), 500

def calculate_distance(lat1, lon1, lat2, lon2):
    """İki nokta arasındaki mesafeyi kilometre cinsinden hesaplar."""
    from math import sin, cos, sqrt, atan2, radians

    R = 6371  # Dünya'nın yarıçapı (km)

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c

    return distance

def get_address_from_coords(lat, lon):
    try:
        geolocator = Nominatim(user_agent="traffic_app", timeout=10)
        time.sleep(1)  # Rate limiting için bekleme süresi
        location = geolocator.reverse(f"{lat}, {lon}", language="tr", timeout=10)
        
        if location and location.raw.get('address'):
            address = location.raw['address']
            components = []
            
            # Sokak/Cadde bilgisi
            if address.get('road'):
                components.append(address.get('road'))
            
            # Mahalle bilgisi
            if address.get('neighbourhood'):
                components.append(address.get('neighbourhood'))
            elif address.get('suburb'):
                components.append(address.get('suburb'))
                
            # İlçe bilgisi
            if address.get('district'):
                components.append(address.get('district'))
            elif address.get('town'):
                components.append(address.get('town'))
            
            # Adres bileşenlerini birleştir
            if components:
                return ", ".join(components)
            
            # Hiçbir bileşen bulunamazsa koordinatları göster
            return f"({lat:.6f}, {lon:.6f})"
            
        return f"({lat:.6f}, {lon:.6f})"
        
    except GeocoderTimedOut:
        print(f"Geocoder timeout for coordinates: {lat}, {lon}")
        return f"({lat:.6f}, {lon:.6f})"
    except Exception as e:
        print(f"Adres çözümleme hatası: {str(e)} for coordinates: {lat}, {lon}")
        return f"({lat:.6f}, {lon:.6f})"

@app.route('/get_address', methods=['POST'])
def get_address():
    try:
        data = request.get_json()
        
        # Koordinat verilmişse adres bul
        if 'latitude' in data and 'longitude' in data:
            lat = data['latitude']
            lon = data['longitude']
            address = get_address_from_coords(lat, lon)
            return jsonify({
                'success': True,
                'address': address,
                'latitude': lat,
                'longitude': lon
            })
        
        # Adres verilmişse koordinat bul
        elif 'query' in data:
            geolocator = Nominatim(user_agent="traffic_app")
            query = data['query'].strip()
            
            # Sokak/Cadde kelimelerini kontrol et
            street_types = ['sokak', 'sokağı', 'sk', 'cadde', 'caddesi', 'cd', 'bulvar', 'bulvarı', 'blv']
            
            # Önce tam adres olarak dene
            location = geolocator.geocode(query + ", İstanbul, Türkiye")
            
            # Eğer bulunamazsa, sokak adı olarak dene
            if not location:
                # Sorguyu kelimelere ayır
                parts = query.split()
                
                # Sokak tipini bul
                street_type = None
                street_type_index = None
                for i, part in enumerate(parts):
                    if part.lower() in street_types:
                        street_type = part
                        street_type_index = i
                        break
                
                if street_type:
                    # Sokak adını ve bölgeyi ayır
                    street_name = " ".join(parts[:street_type_index])
                    area = " ".join(parts[street_type_index+1:]) if street_type_index+1 < len(parts) else ""
                    
                    # Farklı kombinasyonlarla dene
                    search_queries = []
                    
                    # Eğer bölge varsa
                    if area:
                        search_queries.extend([
                            f"{street_name} {street_type}, {area}, İstanbul, Türkiye",
                            f"{street_name}, {area}, İstanbul, Türkiye",
                            f"{area} {street_name} {street_type}, İstanbul, Türkiye",
                            f"{area} {street_name}, İstanbul, Türkiye"
                        ])
                    
                    # Bölge olmadan da dene
                    search_queries.extend([
                        f"{street_name} {street_type}, İstanbul, Türkiye",
                        f"{street_name}, İstanbul, Türkiye"
                    ])
                    
                    # Her sorgu için dene
                    for search_query in search_queries:
                        try:
                            location = geolocator.geocode(search_query, timeout=10)
                            if location:
                                # Bulunan konumun adres bileşenlerini kontrol et
                                if location.raw.get('address'):
                                    address = location.raw['address']
                                    # Sokak adı veya mahalle adı eşleşiyor mu kontrol et
                                    if (address.get('road', '').lower() == street_name.lower() or 
                                        address.get('neighbourhood', '').lower() == street_name.lower() or
                                        address.get('suburb', '').lower() == street_name.lower()):
                                        break
                        except Exception as e:
                            print(f"Arama hatası: {search_query} - {str(e)}")
                            continue
                
                # Hala bulunamadıysa, son kelimeyi sokak adı olarak dene
                if not location and len(parts) > 1:
                    street = parts[-1]
                    area = " ".join(parts[:-1])
                    
                    # Önce bölgeyi bul
                    area_location = geolocator.geocode(f"{area}, İstanbul, Türkiye")
                    if area_location:
                        # Bölge bulunduysa, sokak araması yap
                        location = geolocator.geocode(f"{street}, {area}, İstanbul, Türkiye")
            
            if location:
                # Adres bileşenlerini al
                address_components = []
                if location.raw.get('address'):
                    address = location.raw['address']
                    
                    # Sokak/Cadde bilgisi
                    if address.get('road'):
                        address_components.append(address.get('road'))
                    
                    # Mahalle bilgisi
                    if address.get('neighbourhood'):
                        address_components.append(address.get('neighbourhood'))
                    elif address.get('suburb'):
                        address_components.append(address.get('suburb'))
                    
                    # İlçe bilgisi
                    if address.get('district'):
                        address_components.append(address.get('district'))
                    elif address.get('town'):
                        address_components.append(address.get('town'))
                
                # Adres bileşenlerini birleştir
                formatted_address = ", ".join(address_components) if address_components else location.address
                
                return jsonify({
                    'success': True,
                    'address': formatted_address,
                    'latitude': location.latitude,
                    'longitude': location.longitude
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Adres bulunamadı'
                })
        
        return jsonify({
            'success': False,
            'message': 'Geçersiz istek'
        })
        
    except Exception as e:
        print(f"Adres arama hatası: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Bir hata oluştu'
        })

@app.route('/set_driving_style', methods=['POST'])
def set_driving_style():
    try:
        user_id = session.get('user_id', 1)  # Varsayılan kullanıcı
        style = request.json.get('driving_style')
        
        if style not in ['hızlı', 'rahat', 'ekonomik']:
            return jsonify({'error': 'Geçersiz sürüş stili'}), 400
            
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO users (id, preferred_driving_style)
            VALUES (?, ?)
            ON CONFLICT (id) DO UPDATE SET preferred_driving_style = ?
        """, (user_id, style, style))
        
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'Sürüş stili güncellendi'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/rate_route', methods=['POST'])
def rate_route():
    try:
        user_id = session.get('user_id', 1)  # Varsayılan kullanıcı
        route_id = request.json.get('route_id')
        rating = request.json.get('rating')
        
        if not all([route_id, rating]) or not (1 <= rating <= 5):
            return jsonify({'error': 'Geçersiz değerlendirme'}), 400
            
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'routes.db')
        route_recommender.update_preferences(user_id, route_id, rating, db_path)
        
        return jsonify({'message': 'Değerlendirme kaydedildi'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_route_recommendations', methods=['POST'])
def get_route_recommendations():
    try:
        user_id = session.get('user_id', 1)  # Varsayılan kullanıcı
        data = request.json
        
        required_fields = ['start_lat', 'start_lon', 'end_lat', 'end_lon']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Eksik koordinat bilgisi'}), 400
            
        hour_of_day = datetime.now().hour
        
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'routes.db')
        recommendations = route_recommender.get_recommendations(
            user_id,
            data['start_lat'],
            data['start_lon'],
            data['end_lat'],
            data['end_lon'],
            hour_of_day,
            db_path
        )
        
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/nearby_stations', methods=['POST'])
def get_nearby_stations():
    data = request.get_json()
    lon = float(data.get('lon'))  # longitude
    lat = float(data.get('lat'))  # latitude
    k = 5  # En yakın 5 istasyonu getir
    
    stations = fuel_recommender.find_nearest_stations(lon, lat, k, max_distance=2.0)
    return jsonify(stations)

@app.route('/route_stations', methods=['POST'])
def get_route_stations():
    try:
        data = request.get_json()
        coordinates = data.get('coordinates')

        if not coordinates or len(coordinates) != 2:
            return jsonify({'error': 'Geçersiz koordinatlar'}), 400

        start_point = coordinates[0]  # [lat, lon]
        end_point = coordinates[1]    # [lat, lon]

        # Başlangıç ve bitiş noktalarına yakın istasyonları bul
        stations = fuel_recommender.find_stations_near_points(
            start_lat=start_point[0],
            start_lon=start_point[1],
            end_lat=end_point[0],
            end_lon=end_point[1],
            max_stations=7
        )

        return jsonify({'stations': stations})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/station_details", methods=["POST"])
def get_station_details():
    try:
        data = request.get_json()
        lat = float(data.get('latitude'))
        lon = float(data.get('longitude'))
        
        # Koordinatları ters çevirerek gönder
        details = fuel_recommender.get_station_details(lon, lat)
        if details:
            return jsonify(details)
        return jsonify({'error': 'İstasyon bulunamadı'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/fuel_stations_map')
def fuel_stations_map():
    # CSV dosyasını oku
    stations_df = pd.read_csv('data/fuel_station.csv')
    
    # NaN değerleri temizle
    stations_df = stations_df.dropna(subset=['longitude', 'latitude'])
    
    # İstasyon listesini oluştur
    stations = []
    for _, row in stations_df.iterrows():
        station = {
            'name': row['adi'],
            'brand': row['akaryakit_dagitim_sirketi_tnm'],
            'type': row['is_nevi_tnm'],
            'district': row['ilce'],
            'neighborhood': row['mahalle_adi'],
            'longitude': row['longitude'],
            'latitude': row['latitude']
        }
        stations.append(station)
    
    return render_template('fuel_stations_map.html', stations=stations)

def load_fuel_stations():
    """Yakıt istasyonlarını CSV dosyasından yükler"""
    stations = []
    try:
        with open('data/fuel_station.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Koordinatları önce longitude sonra latitude olarak al
                    longitude = float(row['longitude'])
                    latitude = float(row['latitude'])
                    stations.append({
                        'name': row['adi'],
                        'brand': row['akaryakit_dagitim_sirketi_tnm'],
                        'type': row['is_nevi_tnm'],
                        'district': row['ilce'],
                        'neighborhood': row['mahalle_adi'],
                        'coordinates': (longitude, latitude)
                    })
                except (ValueError, KeyError) as e:
                    print(f"Satır okuma hatası: {e}")
                    continue
    except Exception as e:
        print(f"CSV dosyası okuma hatası: {e}")
    return stations

@app.route('/stations_map')
def show_stations_map():
    """Yakıt istasyonları haritasını gösterir"""
    return render_template('stations_map.html')

# Rota önerileri listesi
ROUTE_SUGGESTIONS = [
    {
        "title": "Boğaz Boyunca Klasik Tur",
        "description": "Beşiktaş'tan Sarıyer'e Boğaz boyunca keyifli bir sürüş.",
        "start": "Beşiktaş Meydanı",
        "end": "Sarıyer Merkez",
        "start_coords": "41.0438,29.0039",
        "end_coords": "41.1656,29.0497",
        "duration": "35 dk",
        "highlights": ["Boğaz manzarası", "Tarihi yalılar", "Köprü geçişleri"]
    },
    {
        "title": "Üsküdar'dan Beykoz'a Doğa Rotası",
        "description": "Üsküdar'dan Beykoz'a sahil ve doğa manzaralı uzun bir yolculuk.",
        "start": "Üsküdar Meydanı",
        "end": "Beykoz Korusu",
        "start_coords": "41.0220,29.0156",
        "end_coords": "41.1082,29.0634",
        "duration": "30 dk",
        "highlights": ["Sahil şeridi", "Çubuklu", "Kuzguncuk"]
    },
    {
        "title": "Kadıköy'den Pendik Marina'ya",
        "description": "Kadıköy'den başlayarak Bağdat Caddesi üzerinden Pendik sahiline varış.",
        "start": "Kadıköy Rıhtım",
        "end": "Pendik Sahil",
        "start_coords": "40.9902,29.0250",
        "end_coords": "40.8781,29.2686",
        "duration": "40 dk",
        "highlights": ["Bağdat Caddesi", "Sahil yolu", "Marina alanı"]
    },
    {
        "title": "Taksim'den Emirgan'a Şehir Kaçamağı",
        "description": "Taksim'den Emirgan Korusu'na şehir içinden doğaya kaçış.",
        "start": "Taksim Meydanı",
        "end": "Emirgan Korusu",
        "start_coords": "41.0369,28.9850",
        "end_coords": "41.1096,29.0536",
        "duration": "25 dk",
        "highlights": ["Boğaz hattı", "Şehir manzarası", "Korular"]
    },
    {
        "title": "Florya'dan Büyükçekmece Sahili'ne",
        "description": "Marmara sahilinde Florya'dan Büyükçekmece'ye uzun bir sahil sürüşü.",
        "start": "Florya",
        "end": "Büyükçekmece Sahili",
        "start_coords": "40.9740,28.7762",
        "end_coords": "41.0125,28.5850",
        "duration": "40 dk",
        "highlights": ["Deniz manzarası", "Yeşil alanlar", "Plajlar"]
    },
    {
        "title": "Bakırköy'den Tuzla Marina'ya",
        "description": "Bakırköy sahilinden başlayıp uzun bir sahil turu ile Tuzla Marina'ya varış.",
        "start": "Bakırköy Sahil",
        "end": "Tuzla Marina",
        "start_coords": "40.9761,28.8533",
        "end_coords": "40.8333,29.3106",
        "duration": "1 saat",
        "highlights": ["Deniz yolu", "Marina gezisi", "Kafe ve restoranlar"]
    },
    {
        "title": "Belgrad Ormanı Kaçışı",
        "description": "Maslak'tan Belgrad Ormanı'na doğa ile iç içe bir rota.",
        "start": "Maslak",
        "end": "Belgrad Ormanı Girişi",
        "start_coords": "41.1125,29.0200",
        "end_coords": "41.1820,28.9879",
        "duration": "20 dk",
        "highlights": ["Doğa yürüyüş yolları", "Gölet manzarası", "Piknik alanları"]
    },
    {
        "title": "Polonezköy Doğa Rotası",
        "description": "Kısıklı'dan Polonezköy'e doğanın kalbine doğru araba yolculuğu.",
        "start": "Kısıklı",
        "end": "Polonezköy Tabiat Parkı",
        "start_coords": "41.0248,29.0701",
        "end_coords": "41.0886,29.1450",
        "duration": "25 dk",
        "highlights": ["Orman yolları", "Köy havası", "Doğa parkı"]
    },
    {
        "title": "Eyüp'ten Kemerburgaz Kent Ormanı'na",
        "description": "Eyüp Sultan'dan Kemerburgaz Kent Ormanı'na doğa kaçamağı.",
        "start": "Eyüp Sultan",
        "end": "Kemerburgaz Kent Ormanı",
        "start_coords": "41.0387,28.9335",
        "end_coords": "41.1361,28.8816",
        "duration": "25 dk",
        "highlights": ["Haliç manzarası", "Orman yürüyüşleri", "Doğa aktiviteleri"]
    },
    {
        "title": "Sabiha Gökçen'den Şile'ye",
        "description": "Sabiha Gökçen Havalimanı'ndan Şile'nin sahil kasabasına uzun sürüş.",
        "start": "Sabiha Gökçen Havalimanı",
        "end": "Şile Sahili",
        "start_coords": "40.8986,29.3092",
        "end_coords": "41.1771,29.6128",
        "duration": "1 saat",
        "highlights": ["Kırsal yollar", "Şile Kalesi", "Sahil kasabası"]
    },
    {
        "title": "Yenikapı Sahili'nden Ataköy Marina'ya",
        "description": "Yenikapı sahil yolundan Ataköy Marina'ya kısa ve keyifli bir yolculuk.",
        "start": "Yenikapı Sahil",
        "end": "Ataköy Marina",
        "start_coords": "41.0016,28.9567",
        "end_coords": "40.9832,28.8724",
        "duration": "15 dk",
        "highlights": ["Deniz kenarı", "Marina", "Kafeler"]
    },
    {
        "title": "Rumeli Hisarı'ndan Kilyos Plajı'na",
        "description": "Boğaz'dan Karadeniz sahiline doğru uzanan doğal bir rota.",
        "start": "Rumeli Hisarı",
        "end": "Kilyos Plajı",
        "start_coords": "41.0847,29.0534",
        "end_coords": "41.2520,29.0225",
        "duration": "30 dk",
        "highlights": ["Boğaz manzarası", "Doğa yolları", "Karadeniz sahili"]
    },
    {
        "title": "Avcılar'dan Silivri'ye Sahil Rotası",
        "description": "Avcılar'dan Silivri'ye Marmara kıyısı boyunca huzurlu bir yolculuk.",
        "start": "Avcılar",
        "end": "Silivri Merkez",
        "start_coords": "40.9797,28.7222",
        "end_coords": "41.0733,28.2464",
        "duration": "45 dk",
        "highlights": ["Sahil şeridi", "Balık restoranları", "Küçük plajlar"]
    },
    {
        "title": "Altunizade'den Riva'ya Kaçış",
        "description": "Şehirden uzaklaşıp Riva sahiline doğa dolu bir kaçamak.",
        "start": "Altunizade",
        "end": "Riva Sahili",
        "start_coords": "41.0244,29.0423",
        "end_coords": "41.1935,29.2719",
        "duration": "40 dk",
        "highlights": ["Orman yolları", "Nehir kenarı", "Deniz keyfi"]
    },
    {
        "title": "Beylikdüzü'nden Büyükçekmece Gölü'ne",
        "description": "Beylikdüzü'nden başlayarak Büyükçekmece gölü çevresinde manzaralı bir sürüş.",
        "start": "Beylikdüzü",
        "end": "Büyükçekmece Gölü",
        "start_coords": "41.0015,28.6417",
        "end_coords": "41.0167,28.5745",
        "duration": "20 dk",
        "highlights": ["Göl manzarası", "Piknik alanları", "Sahil yolu"]
    },
     {
        "title": "Moda'dan Fenerbahçe Parkı'na",
        "description": "Kadıköy Moda sahilinden Fenerbahçe Parkı'na kısa ve keyifli bir sürüş.",
        "start": "Moda Sahili",
        "end": "Fenerbahçe Parkı",
        "start_coords": "40.9781,29.0254",
        "end_coords": "40.9639,29.0436",
        "duration": "10 dk",
        "highlights": ["Sahil yürüyüş yolları", "Deniz manzarası", "Kafeler ve park"]
    },
    {
        "title": "Ataşehir'den Ömerli Barajı'na Kaçış",
        "description": "Ataşehir'den çıkıp Ömerli Barajı çevresinde doğayla iç içe bir sürüş.",
        "start": "Ataşehir",
        "end": "Ömerli Barajı",
        "start_coords": "40.9929,29.1246",
        "end_coords": "41.0381,29.2737",
        "duration": "30 dk",
        "highlights": ["Baraj manzarası", "Doğa yolları", "Sakinlik"]
    },
    {
        "title": "Bahçeşehir'den Akbatı AVM'ye",
        "description": "Bahçeşehir çevresinden Akbatı Alışveriş Merkezi'ne alışveriş ve yemek rotası.",
        "start": "Bahçeşehir Gölet",
        "end": "Akbatı AVM",
        "start_coords": "41.0686,28.6562",
        "end_coords": "41.0706,28.6863",
        "duration": "10 dk",
        "highlights": ["Bahçeşehir manzarası", "Alışveriş molası", "Yeme içme alanları"]
    },
    {
        "title": "Maltepe Sahili'nden Dragos Tepesi'ne",
        "description": "Maltepe sahilinden Dragos Tepesi'ne doğru kısa bir manzara sürüşü.",
        "start": "Maltepe Sahili",
        "end": "Dragos Tepesi",
        "start_coords": "40.9361,29.1301",
        "end_coords": "40.9278,29.1564",
        "duration": "10 dk",
        "highlights": ["Deniz ve ada manzarası", "Dragos Yokuşu", "Sahil parkları"]
    },
    {
        "title": "Çekmeköy'den Şile Ağva'ya Uzun Kaçamak",
        "description": "Çekmeköy'den yola çıkıp Şile'nin güzel sahil kasabası Ağva'ya keyifli bir sürüş.",
        "start": "Çekmeköy Merkez",
        "end": "Ağva Sahili",
        "start_coords": "41.0247,29.2075",
        "end_coords": "41.1380,29.8564",
        "duration": "1 saat 10 dk",
        "highlights": ["Kırsal yollar", "Göksu Deresi", "Deniz ve plajlar"]
    },
    {
        "title": "Arnavutköy'den Karaburun Sahili'ne",
        "description": "Arnavutköy merkezinden Karaburun sahiline kuzeye doğru bir keşif sürüşü.",
        "start": "Arnavutköy Merkez",
        "end": "Karaburun Sahili",
        "start_coords": "41.1831,28.7382",
        "end_coords": "41.3054,28.5489",
        "duration": "35 dk",
        "highlights": ["Karadeniz sahili", "Balıkçı köyü", "Doğa yürüyüş yolları"]
    },
    {
        "title": "İstanbul Havalimanı'ndan Göktürk'e Hızlı Kaçış",
        "description": "İstanbul Havalimanı'ndan çıkıp Göktürk'e doğayla iç içe keyifli bir sürüş.",
        "start": "İstanbul Havalimanı",
        "end": "Göktürk Merkez",
        "start_coords": "41.2625,28.7276",
        "end_coords": "41.1500,28.9047",
        "duration": "25 dk",
        "highlights": ["Kırsal yollar", "Göktürk Ormanı", "Kafeler"]
    },
    {
        "title": "Sabiha Gökçen'den Viaport AVM'ye Alışveriş Rotası",
        "description": "Sabiha Gökçen Havalimanı'ndan Viaport Alışveriş Merkezi'ne kısa bir alışveriş kaçamağı.",
        "start": "Sabiha Gökçen Havalimanı",
        "end": "Viaport AVM",
        "start_coords": "40.8986,29.3092",
        "end_coords": "40.9212,29.3197",
        "duration": "10 dk",
        "highlights": ["Alışveriş merkezi", "Kafeler", "Açık alanlar"]
    },
    {
        "title": "İstanbul Havalimanı'ndan Kilyos Plajı'na Deniz Kaçamağı",
        "description": "İstanbul Havalimanı'ndan Karadeniz kıyısındaki Kilyos'a kısa bir yaz rotası.",
        "start": "İstanbul Havalimanı",
        "end": "Kilyos Plajı",
        "start_coords": "41.2625,28.7276",
        "end_coords": "41.2520,29.0225",
        "duration": "40 dk",
        "highlights": ["Kırsal yollar", "Karadeniz sahili", "Plajlar"]
    },
    {
        "title": "Sabiha Gökçen'den Aydos Ormanı'na Doğa Rotası",
        "description": "Şehirden çıkıp Aydos Ormanı'nda doğayla buluşabileceğiniz bir sürüş.",
        "start": "Sabiha Gökçen Havalimanı",
        "end": "Aydos Ormanı",
        "start_coords": "40.8986,29.3092",
        "end_coords": "40.9460,29.2486",
        "duration": "20 dk",
        "highlights": ["Orman havası", "Piknik alanları", "Yürüyüş yolları"]
    }
]

@app.route('/get_route_suggestions')
def get_route_suggestions():
    try:
        # Rastgele 5 rota seç
        suggestions = random.sample(ROUTE_SUGGESTIONS, 5)
        return jsonify({
            'success': True,
            'suggestions': suggestions
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == "__main__":
    app.run(debug=True)
