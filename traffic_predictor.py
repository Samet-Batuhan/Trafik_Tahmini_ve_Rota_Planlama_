import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime
import gc

class TrafficPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_path = 'traffic_model.joblib'
        self.scaler_path = 'scaler.joblib'
        self.load_model()

    def load_model(self):
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                print("Model başarıyla yüklendi.")
            else:
                print("Model bulunamadı. Yeni model eğitilecek.")
                self.train_model()
        except Exception as e:
            print(f"Model yüklenirken hata oluştu: {e}")
            self.train_model()

    def train_model(self):
        try:
            print("Veri setleri okunuyor...")
            data_dir = 'trafik_veri'
            
            # Ay isimlerini ve sıralarını tanımla
            months = {
                'ocak': 1,
                'subat': 2,
                'mart': 3,
                'nisan': 4,
                'mayis': 5,
                'haziran': 6,
                'temmuz': 7,
                'agustos': 8,
                'eylul': 9,
                'ekim': 10,
                'kasim': 11,
                'aralik': 12
            }
            
            # CSV dosyalarını ay sırasına göre düzenle
            csv_files = []
            for f in os.listdir(data_dir):
                if f.endswith('.csv'):
                    month_name = f.split('.')[0].lower()
                    if month_name in months:
                        csv_files.append((months[month_name], f))
            
            # Ayların sırasına göre sırala
            csv_files.sort()  # Ay numarasına göre sırala
            csv_files = [f[1] for f in csv_files]  # Sadece dosya adlarını al
            
            print(f"Toplam {len(csv_files)} CSV dosyası bulundu.")
            print("İşlenecek dosyalar sırası:")
            for i, f in enumerate(csv_files, 1):
                print(f"{i}. {f}")
            
            # İlk veri örneği ile scaler'ı hazırla
            print("\nScaler hazırlanıyor...")
            first_file = os.path.join(data_dir, csv_files[0])
            first_chunk = pd.read_csv(first_file, nrows=50000)
            X_sample, _ = self.prepare_features(first_chunk)
            self.scaler = StandardScaler()
            self.scaler.fit(X_sample)
            del first_chunk, X_sample
            gc.collect()
            
            # Model parametreleri
            print("Model hazırlanıyor...")
            self.model = RandomForestRegressor(
                n_estimators=25,  # Ağaç sayısını azalttık
                max_depth=10,     # Derinliği sabit tuttuk
                min_samples_split=10,
                min_samples_leaf=4,
                n_jobs=1,         # Tek thread kullan
                random_state=42,
                warm_start=True   # Kademeli öğrenme
            )
            
            # Her dosyayı parça parça oku ve modeli güncelle
            chunk_size = 50000  # Chunk boyutunu küçülttük
            total_rows = 0
            
            for file_num, csv_file in enumerate(csv_files, 1):
                print(f"\n{file_num}/12: {csv_file} dosyası işleniyor...")
                file_path = os.path.join(data_dir, csv_file)
                
                try:
                    # Dosyayı parçalar halinde oku
                    chunks = pd.read_csv(file_path, chunksize=chunk_size)
                    for chunk_num, chunk in enumerate(chunks, 1):
                        print(f"Chunk {chunk_num} işleniyor...")
                        
                        try:
                            # Özellikleri hazırla
                            X_chunk, y_chunk = self.prepare_features(chunk)
                            
                            # Verileri ölçeklendir
                            X_chunk_scaled = self.scaler.transform(X_chunk)
                            
                            # Rastgele örnekleme yap (her chunk'ın %20'si)
                            sample_size = len(X_chunk_scaled) // 5
                            indices = np.random.choice(len(X_chunk_scaled), sample_size, replace=False)
                            X_sample = X_chunk_scaled[indices]
                            y_sample = y_chunk[indices]
                            
                            # Modeli güncelle
                            self.model.fit(X_sample, y_sample)
                            
                            # Belleği temizle
                            del X_chunk, y_chunk, X_chunk_scaled, X_sample, y_sample
                            gc.collect()
                            
                            total_rows += sample_size
                            print(f"Toplam işlenen satır: {total_rows:,}")
                            
                            # Her 3 chunk'ta bir modeli kaydet
                            if chunk_num % 3 == 0:
                                print("Ara kayıt yapılıyor...")
                                joblib.dump(self.model, self.model_path)
                                joblib.dump(self.scaler, self.scaler_path)
                        
                        except Exception as e:
                            print(f"Chunk işlenirken hata: {e}")
                            continue
                
                except Exception as e:
                    print(f"Dosya işlenirken hata: {e}")
                    continue
                
                # Her dosya sonunda kaydet
                print(f"{csv_file} tamamlandı, model kaydediliyor...")
                joblib.dump(self.model, self.model_path)
                joblib.dump(self.scaler, self.scaler_path)
            
            print("\nModel eğitimi tamamlandı!")
            print(f"Toplam işlenen satır sayısı: {total_rows:,}")
            
            # Test verisi için son dosyadan örnek al
            print("\nModel performansı değerlendiriliyor...")
            test_data = pd.read_csv(os.path.join(data_dir, csv_files[-1]), nrows=50000)
            X_test, y_test = self.prepare_features(test_data)
            X_test_scaled = self.scaler.transform(X_test)
            test_score = self.model.score(X_test_scaled, y_test)
            print(f"Test seti R² skoru: {test_score:.3f}")
            
            # Son kez kaydet
            print("\nModel kaydediliyor...")
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            print("Model başarıyla kaydedildi.")
            
        except Exception as e:
            print(f"Model eğitilirken hata oluştu: {e}")
            raise

    def prepare_features(self, df):
        # Tarih sütununu datetime'a çevir
        df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
        
        # Özellik mühendisliği
        X = np.column_stack([
            df['LONGITUDE'],
            df['LATITUDE'],
            df['DATE_TIME'].dt.hour,
            df['DATE_TIME'].dt.dayofweek,
            df['DATE_TIME'].dt.day,
            df['DATE_TIME'].dt.month
        ])
        
        y = df['AVERAGE_SPEED'].values
        
        return X, y

    def predict(self, lat, lon, hour_of_day):
        try:
            if self.model is None or self.scaler is None:
                print("Model henüz yüklenmedi veya eğitilmedi.")
                return None
            
            # Tahmin için özellik vektörü oluştur
            current_time = datetime.now()
            features = np.array([[
                lon,
                lat,
                hour_of_day,
                current_time.weekday(),
                current_time.day,
                current_time.month
            ]])
            
            # Özellikleri ölçeklendir
            features_scaled = self.scaler.transform(features)
            
            # Tahmin yap
            prediction = self.model.predict(features_scaled)[0]
            print(f"Tahmin yapıldı: {prediction:.2f} km/s")
            return prediction
            
        except Exception as e:
            print(f"Tahmin yapılırken hata oluştu: {e}")
            return None

if __name__ == "__main__":
    predictor = TrafficPredictor() 