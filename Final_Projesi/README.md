# YOLOv8 Tabanlı Beyin Tümörü Tespiti

## İstanbul Topkapı Üniversitesi - Derin Öğrenme Final Ödevi

Bu projede, Ultralytics tarafından sağlanan **Brain Tumor Detection** veri seti kullanılarak YOLOv8 modeli ile MR görüntülerinden beyin tümörü tespiti yapılmaktadır.

## Proje Hedefi

- YOLOv8 modeliyle MR görüntülerinden beyin tümörü tespiti yapmak
- Model başarımını standart metriklerle değerlendirmek
- Veri artırma ve transfer öğrenme stratejilerini uygulamak
- Tespit sonuçlarını görselleştirmek ve modelin güvenilirliğini analiz etmek

## Veri Seti

| | Eğitim | Doğrulama |
|---|---|---|
| Görüntü Sayısı | 893 | 223 |
| Format | MR Görüntüleri | MR Görüntüleri |

**Sınıflar:**
- `0` - Negatif (tümör yok)
- `1` - Pozitif (tümör var)

**Kaynak:** [Ultralytics Brain Tumor Dataset](https://github.com/ultralytics/assets/releases/download/v0.0.0/brain-tumor.zip)

## Yöntem

- **Model:** YOLOv8n (Nano) - Transfer öğrenme ile COCO ağırlıkları kullanılarak
- **Eğitim Komutu:** `yolo detect train data=brain-tumor.yaml model=yolov8n.pt epochs=50 imgsz=640`
- **Epoch:** 50
- **Görüntü Boyutu:** 640x640

## Değerlendirme Metrikleri

- Doğruluk (Accuracy)
- mAP@0.5 ve mAP@0.5:0.95
- Kesinlik (Precision) ve Duyarlılık (Recall)
- Karışıklık Matrisi (Confusion Matrix)

## Dosya Yapısı

```
├── brain_tumor_yolov8_final.ipynb  # Ana notebook (Colab'da çalıştırılır)
├── brain_tumor_yolov8.py           # Python script versiyonu
├── brain-tumor.yaml                # Veri seti yapılandırma dosyası
├── README.md                       # Bu dosya
└── runs/                           # Eğitim sonuçları (eğitim sonrası oluşur)
    └── detect/
        └── brain_tumor_yolov8n/
            ├── weights/
            │   ├── best.pt
            │   └── last.pt
            ├── results.png
            ├── confusion_matrix.png
            ├── F1_curve.png
            └── PR_curve.png
```

## Kullanım

### Google Colab'da Çalıştırma (Önerilen)

1. `brain_tumor_yolov8_final.ipynb` dosyasını Google Colab'a yükleyin
2. **Runtime > Change runtime type > GPU (T4)** seçin
3. **Runtime > Run all** ile tüm hücreleri çalıştırın

### Yerel Çalıştırma

```bash
pip install ultralytics
python brain_tumor_yolov8.py
```

## Gereksinimler

- Python 3.8+
- ultralytics
- matplotlib
- numpy
- opencv-python
