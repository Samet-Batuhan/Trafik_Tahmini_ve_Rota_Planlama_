import streamlit as st
import folium
import random
import networkx as nx
from folium.plugins import HeatMap
import osmnx as ox

# Trafik yoğunluğuna göre hız belirleme fonksiyonu
def get_average_speed_by_hour(hour):
    hourly_speeds = {
        0: 60, 1: 60, 2: 60, 3: 60, 4: 55, 5: 50, 6: 50, 7: 40, 8: 30, 9: 35,
        10: 45, 11: 50, 12: 40, 13: 45, 14: 50, 15: 40, 16: 35, 17: 25, 18: 20,
        19: 30, 20: 40, 21: 50, 22: 55, 23: 60
    }
    return hourly_speeds.get(hour, 50)

# Trafik yoğunluğunu hesaplayan fonksiyon
def calculate_traffic_density(hour_speed, max_speed=60):
    return (1 - hour_speed / max_speed) * 100

def main():
    st.title('Rota ve Trafik Hesaplama Uygulaması')

    # Kullanıcıdan girdi alma
    start_lat = st.number_input('Başlangıç Noktası Enlem (Latitude)', -90.0, 90.0, 41.0082)
    start_lon = st.number_input('Başlangıç Noktası Boylam (Longitude)', -180.0, 180.0, 28.9784)
    end_lat = st.number_input('Bitiş Noktası Enlem (Latitude)', -90.0, 90.0, 40.9911)
    end_lon = st.number_input('Bitiş Noktası Boylam (Longitude)', -180.0, 180.0, 29.0086)

    hour_of_day = st.number_input('Hangi saat için trafik yoğunluğunu hesaplayalım? (0-23)', 0, 23, 8)

    # GraphML dosyasını yükle
    G = nx.read_graphml("filtered_map.graphml")  # Dosyanın yolu

    # Graph CRS (Coordinate Reference System) ekleme
    if "crs" not in G.graph:
        G.graph["crs"] = "EPSG:4326"

    # Kullanıcıdan saati al
    average_speed_kmh = get_average_speed_by_hour(hour_of_day)

    # Koordinatları düğümlere eşleştirin
    nodes_gdf = ox.graph_to_gdfs(G, nodes=True, edges=False)
    
    # Koordinatları almak için doğru `x` ve `y` sütunları olmalı
    nodes_gdf = nodes_gdf.dropna(subset=['x', 'y'])

    start_node = ox.distance.nearest_nodes(G, X=start_lon, Y=start_lat)
    end_node = ox.distance.nearest_nodes(G, X=end_lon, Y=end_lat)

    # En kısa yolu hesaplayın
    route = nx.shortest_path(G, start_node, end_node, weight='length')
    
    # Rota koordinatlarını al
    route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]
    
    # Yol uzunluğunu hesapla (metre cinsinden)
    route_length_meters = nx.shortest_path_length(G, source=start_node, target=end_node, weight='length')
    route_length_km = route_length_meters / 1000  # km cinsinden

    # Tahmini varış süresi
    estimated_travel_time_hours = route_length_km / average_speed_kmh
    estimated_travel_time_minutes = estimated_travel_time_hours * 60

    # Harita oluşturma
    map_route = folium.Map(location=[start_lat, start_lon], zoom_start=12)
    
    # Başlangıç ve bitiş noktalarını haritaya ekle
    folium.Marker([start_lat, start_lon], popup="Başlangıç", icon=folium.Icon(color='green')).add_to(map_route)
    folium.Marker([end_lat, end_lon], popup="Bitiş", icon=folium.Icon(color='red')).add_to(map_route)

    # HeatMap için yoğunluk noktalarını oluştur
    heat_data = []
    for coord in route_coords:
        traffic_density = random.randint(1, 10)
        heat_data.append([coord[0], coord[1], traffic_density])

    # HeatMap ekle
    HeatMap(heat_data).add_to(map_route)

    # Rota çizgisini ekle
    folium.PolyLine(route_coords, color="blue", weight=6, opacity=0.7).add_to(map_route)

    # Yol uzunluğu ve tahmini varış süresi bilgilerini haritaya ekle
    info_text = f"Toplam Yol Uzunluğu: {route_length_km:.2f} km\nTahmini Varış Süresi: {estimated_travel_time_minutes:.2f} dakika"
    folium.Marker([start_lat, start_lon], 
                  popup=info_text, 
                  icon=folium.Icon(color='green')).add_to(map_route)

    # Haritayı HTML dosyasına kaydet
    map_route.save("output/secilen_guzergah.html")

    st.write(f"Tahmini varış süresi: {estimated_travel_time_minutes} dakika")

    # Haritayı göster
    st.components.v1.html(map_route._repr_html_(), height=600)

if __name__ == "__main__":
    main()
