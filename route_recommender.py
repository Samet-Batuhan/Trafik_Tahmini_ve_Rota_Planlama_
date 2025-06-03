import sqlite3
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

class RouteRecommender:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, db_path):
        """Rota öneri modelini eğit"""
        try:
            conn = sqlite3.connect(db_path)
            query = """
                SELECT start_lat, start_lon, end_lat, end_lon, hour_of_day, travel_time
                FROM routes
                WHERE travel_time IS NOT NULL
            """
            df = pd.read_sql_query(query, conn)
            conn.close()

            if df.empty:
                print("Eğitim verisi boş")
                return

            # Özellik matrisi oluştur
            X = df[['start_lat', 'start_lon', 'end_lat', 'end_lon', 'hour_of_day']].values
            y = df['travel_time'].values

            # Verileri ölçeklendir
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)

            # Modeli eğit
            self.model.fit(X_scaled, y)
            self.is_trained = True
            print("Model başarıyla eğitildi")

        except Exception as e:
            print(f"Model eğitimi sırasında hata: {str(e)}")

    def get_recommendations(self, user_id, start_lat, start_lon, end_lat, end_lon, hour_of_day, db_path):
        """Kullanıcıya özel rota önerileri üret"""
        try:
            if not self.is_trained:
                return []

            # Test verisi oluştur
            X_test = np.array([[start_lat, start_lon, end_lat, end_lon, hour_of_day]])
            X_test_scaled = self.scaler.transform(X_test)

            # Tahmini seyahat süresini hesapla
            predicted_time = self.model.predict(X_test_scaled)[0]

            # Benzer rotaları bul
            conn = sqlite3.connect(db_path)
            query = """
                SELECT r.*, 
                       ABS(r.travel_time - ?) as time_diff
                FROM routes r
                WHERE r.hour_of_day = ?
                ORDER BY time_diff ASC
                LIMIT 5
            """
            df = pd.read_sql_query(query, conn, params=(predicted_time, hour_of_day))
            conn.close()

            recommendations = []
            for _, row in df.iterrows():
                recommendations.append({
                    'start_lat': row['start_lat'],
                    'start_lon': row['start_lon'],
                    'end_lat': row['end_lat'],
                    'end_lon': row['end_lon'],
                    'travel_time': row['travel_time'],
                    'route_type': row['route_type']
                })

            return recommendations

        except Exception as e:
            print(f"Öneri üretilirken hata: {str(e)}")
            return []

    def update_preferences(self, user_id, route_id, rating, db_path):
        """Kullanıcı tercihlerini güncelle"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Değerlendirmeyi kaydet
            cursor.execute("""
                UPDATE routes 
                SET rating = ?
                WHERE id = ?
            """, (rating, route_id))
            
            conn.commit()
            conn.close()
            
            # Modeli yeniden eğit
            self.train(db_path)
            
            return True
            
        except Exception as e:
            print(f"Tercih güncellenirken hata: {str(e)}")
            return False 