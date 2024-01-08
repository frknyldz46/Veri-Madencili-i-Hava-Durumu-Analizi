from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from datetime import datetime

# Veriyi tanımla
veri = {
    'Tarih': [
        '2023-01-01', '2023-01-10', '2023-02-01', '2023-02-10', '2023-03-01',
        '2023-03-10', '2023-04-01', '2023-04-10', '2023-05-01', '2023-05-10',
        '2023-06-01', '2023-06-10', '2023-07-01', '2023-07-10', '2023-08-01',
        '2023-08-10', '2023-09-01', '2023-09-10', '2023-10-01', '2023-10-10',
        '2023-11-01', '2023-11-10', '2023-12-01', '2023-12-10'
    ],
    'Sıcaklık': [
        5, -6, 0, 3, 15, 10, 20, 25, 32, 30, 35, 34, 39, 46, 42, 38, 28, 21,
        17, 11, 8, 5, 1, -3
    ],
    'Nem': [
        55, 60, 70, 75, 80, 85, 80, 82, 75, 78, 45, 50, 50, 55, 65, 70, 85, 88,
        40, 45, 60, 65, 75, 80
    ],
    'Hava_Durumu': [
        'Güneşli', 'Bulutlu', 'Güneşli', 'Bulutlu', 'Güneşli', 'Bulutlu',
        'Yağmurlu', 'Yağmurlu', 'Bulutlu', 'Bulutlu', 'Güneşli', 'Bulutlu',
        'Güneşli', 'Bulutlu', 'Yağmurlu', 'Yağmurlu', 'Yağmurlu', 'Yağmurlu',
        'Bulutlu', 'Bulutlu', 'Güneşli', 'Bulutlu', 'Güneşli', 'Bulutlu'
    ]
}

# Veri çerçevesini oluştur ve tarih sütununu datetime türüne dönüştür
df = pd.DataFrame(veri)
df['Tarih'] = pd.to_datetime(df['Tarih'])
df['Ay'] = df['Tarih'].dt.month

# Hava durumu sınıflandırması için etiketleri belirle
df['Hava_Durumu'] = df.apply(
    lambda x: 'Bulutlu' if (x['Sıcaklık'] > 10 and x['Nem'] > 75) else
    ('Güneşli' if (x['Sıcaklık'] > 20 and x['Nem'] <= 50) else 'Yağmurlu'),
    axis=1)

# Bağımsız değişkenleri ve bağımlı değişkeni ayır

X = df[['Sıcaklık', 'Nem', 'Ay']]
y = df['Hava_Durumu']

# Eğitim ve test veri setlerini oluştur
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# Karar ağacı sınıflandırma modelini oluştur

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Test veri setini kullanarak tahmin yap

tahminler = model.predict(X_test)

# Doğruluk oranını hesapla

dogruluk_orani = accuracy_score(y_test, tahminler)
print(f"Doğruluk Oranı: {dogruluk_orani}")

# Hava durumu tahmini yapmak için yeni veri noktası oluştur
# Kullanıcıdan veri al

sıcaklık = float(input("Sıcaklık değerini girin: "))
nem = float(input("Nem değerini girin: "))
ay = int(input("Ay değerini girin (1-12 arası): "))

# Tahmin yap
yeni_veri = pd.DataFrame({'Sıcaklık': [sıcaklık], 'Nem': [nem], 'Ay': [ay]})
tahmin = model.predict(yeni_veri)

print(f"Hava Durumu Tahmini: {tahmin[0]}")
