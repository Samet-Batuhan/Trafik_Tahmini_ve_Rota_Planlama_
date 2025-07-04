# Trafik Tahmini ve Rota Planlama Projesi

Bu proje, İstanbul'daki trafik yoğunluğunu tahmin ederek kullanıcılar için en uygun rotaları planlayan, yakıt istasyonları ve otoparklar gibi önemli noktaları harita üzerinde gösteren bir web uygulamasıdır.

## 🚀 Projenin Amacı

Projenin temel amacı, anlık ve tahmini trafik verilerini kullanarak sürücülerin varış noktalarına daha hızlı ve verimli bir şekilde ulaşmalarını sağlamaktır. Uygulama, makine öğrenmesi modelleri ile trafik yoğunluğunu tahmin eder ve en kısa sürede tamamlanacak alternatif rotalar sunar.

## ✨ Özellikler

- **Trafik Yoğunluğu Tahmini:** Belirli bir tarih ve saat için trafik yoğunluğunu tahmin eden makine öğrenmesi modeli.
- **Akıllı Rota Planlama:** Başlangıç ve bitiş noktaları arasında trafik durumuna göre en hızlı rotanın çizilmesi.
- **Alternatif Rota Önerileri:** Farklı rota seçeneklerinin harita üzerinde gösterilmesi.
- **Harita Üzerinde Görselleştirme:** Rotaların, trafik sıcaklık haritasının, İSPARK otoparklarının ve yakıt istasyonlarının interaktif bir harita üzerinde gösterimi.
- **Web Arayüzü:** Kullanıcıların kolayca rota sorgulaması yapabileceği Flask tabanlı bir web uygulaması.
- **Veri Analizi Paneli:** Trafik verilerinin analiz edildiği ve görselleştirildiği Streamlit tabanlı bir arayüz.

## 🛠️ Kullanılan Teknolojiler

- **Backend:** Python, Flask, Streamlit
- **Makine Öğrenmesi:** Scikit-learn, Pandas, NumPy
- **Veritabanı:** SQLite
- **Harita ve Görselleştirme:** Folium, OSMnx
- **Frontend:** HTML, JavaScript

## ⚙️ Kurulum ve Çalıştırma

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları takip edebilirsiniz:

1.  **Projeyi klonlayın:**
    ```bash
    git clone https://github.com/Samet-Batuhan/Trafik_Tahmini_ve_Rota_Planlama_.git
    cd Trafik_Tahmini_ve_Rota_Planlama_
    ```

2.  **Gerekli kütüphaneleri yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Flask uygulamasını başlatın:**
    Ana web arayüzünü çalıştırmak için:
    ```bash
    python app.py
    ```

4.  **Streamlit panelini başlatın:**
    Veri analizi panelini görmek için:
    ```bash
    streamlit run streamlit_app.py
    ```

## 📂 Dosya Yapısı

```
.
├── app.py                      # Ana Flask uygulaması
├── streamlit_app.py            # Streamlit veri paneli
├── traffic_predictor.py        # Trafik tahmini modeli ve fonksiyonları
├── route_recommender.py        # Rota önerme motoru
├── requirements.txt            # Gerekli Python kütüphaneleri
├── traffic_model.joblib        # Eğitilmiş trafik tahmini modeli
├── models/                     # Diğer modeller ve ilgili dosyalar
├── templates/                  # HTML şablonları
├── static/                     # Statik dosyalar (CSS, JS, haritalar)
├── data/                       # Ham veri setleri (İSPARK, yakıt istasyonları vb.)
└── README.md                   # Proje açıklaması
```
