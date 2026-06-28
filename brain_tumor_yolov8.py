# -*- coding: utf-8 -*-
"""
Istanbul Topkapi Universitesi - Derin Ogrenme Final Odevi
YOLOv8 Tabanli Derin Ogrenme Yaklasimlariyla Beyin Tumoru Tespiti

Bu script Google Colab'da calistirilmak uzere hazirlanmistir.
Colab'da calistirmak icin: Runtime > Run all
"""

# ============================================================
# 1. KUTUPHANE KURULUMU VE IMPORT
# ============================================================
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

install("ultralytics")
install("matplotlib")
install("seaborn")
install("opencv-python-headless")

import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO

# Rastgelelik sabitleme
random.seed(42)
np.random.seed(42)

print("=" * 60)
print("Kutuphaneler basariyla yuklendi!")
print("=" * 60)

# ============================================================
# 2. VERI SETI INDIRME VE HAZIRLAMA
# ============================================================
print("\n[ADIM 2] Veri seti indiriliyor...")

DATASET_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/brain-tumor.zip"
DATASET_DIR = "datasets/brain-tumor"

if not os.path.exists(DATASET_DIR):
    os.makedirs("datasets", exist_ok=True)
    import urllib.request, zipfile
    zip_path = "datasets/brain-tumor.zip"
    print(f"Indiriliyor: {DATASET_URL}")
    urllib.request.urlretrieve(DATASET_URL, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall("datasets")
    os.remove(zip_path)
    print("Veri seti basariyla indirildi ve cikarildi!")
else:
    print("Veri seti zaten mevcut.")

# Veri seti yapisini kontrol et
train_img_dir = os.path.join(DATASET_DIR, "train", "images")
val_img_dir = os.path.join(DATASET_DIR, "valid", "images")
train_lbl_dir = os.path.join(DATASET_DIR, "train", "labels")
val_lbl_dir = os.path.join(DATASET_DIR, "valid", "labels")

# Alternatif yapi kontrolu
if not os.path.exists(train_img_dir):
    train_img_dir = os.path.join(DATASET_DIR, "images", "train")
    val_img_dir = os.path.join(DATASET_DIR, "images", "val")
    train_lbl_dir = os.path.join(DATASET_DIR, "labels", "train")
    val_lbl_dir = os.path.join(DATASET_DIR, "labels", "val")

train_images = glob.glob(os.path.join(train_img_dir, "*.*"))
val_images = glob.glob(os.path.join(val_img_dir, "*.*"))

print(f"\nVeri Seti Ozeti:")
print(f"  Egitim goruntusu  : {len(train_images)}")
print(f"  Dogrulama goruntusu: {len(val_images)}")
print(f"  Siniflar: 0-negatif, 1-pozitif")

# ============================================================
# 3. YAML DOSYASI HAZIRLAMA
# ============================================================
print("\n[ADIM 3] YAML dosyasi hazirlaniyor...")

yaml_content = f"""# Brain Tumor Detection Dataset - YOLOv8
# Istanbul Topkapi Universitesi - Derin Ogrenme Final

path: {os.path.abspath(DATASET_DIR)}
train: images/train
val: images/val

names:
  0: negative
  1: positive
"""

# Alternatif yapi icin kontrol
if "train/images" in train_img_dir:
    yaml_content = f"""path: {os.path.abspath(DATASET_DIR)}
train: train/images
val: valid/images

names:
  0: negative
  1: positive
"""

yaml_path = "brain-tumor.yaml"
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"YAML dosyasi olusturuldu: {yaml_path}")
print(yaml_content)

# ============================================================
# 4. ORNEK GORUNTULERIN GORSELLESTIRMESI
# ============================================================
print("\n[ADIM 4] Ornek goruntular gorsellestiriliyor...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Egitim Veri Setinden Ornek MR Goruntuleri", fontsize=16, fontweight='bold')

sample_images = random.sample(train_images, min(8, len(train_images)))

for idx, img_path in enumerate(sample_images):
    row, col = idx // 4, idx % 4
    img = mpimg.imread(img_path)
    axes[row, col].imshow(img)
    
    # Etiket dosyasini oku
    label_path = img_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
    label = "Bilinmiyor"
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            content = f.read().strip()
            if content:
                cls = int(content.split()[0])
                label = "Pozitif (Tumor)" if cls == 1 else "Negatif"
            else:
                label = "Negatif (bos etiket)"
    
    color = 'red' if 'Pozitif' in label else 'green'
    axes[row, col].set_title(label, fontsize=11, color=color, fontweight='bold')
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig("ornek_goruntular.png", dpi=150, bbox_inches='tight')
plt.show()
print("Ornek goruntular kaydedildi: ornek_goruntular.png")

# ============================================================
# 5. SINIF DAGILIMI ANALIZI
# ============================================================
print("\n[ADIM 5] Sinif dagilimi analiz ediliyor...")

def count_classes(label_dir):
    neg_count = 0
    pos_count = 0
    empty_count = 0
    for lbl_file in glob.glob(os.path.join(label_dir, "*.txt")):
        with open(lbl_file, "r") as f:
            content = f.read().strip()
            if not content:
                empty_count += 1
                continue
            for line in content.split("\n"):
                cls = int(line.split()[0])
                if cls == 0:
                    neg_count += 1
                else:
                    pos_count += 1
    return neg_count, pos_count, empty_count

train_neg, train_pos, train_empty = count_classes(train_lbl_dir)
val_neg, val_pos, val_empty = count_classes(val_lbl_dir)

print(f"\nEgitim Seti  - Negatif: {train_neg}, Pozitif: {train_pos}, Bos: {train_empty}")
print(f"Dogrulama Seti - Negatif: {val_neg}, Pozitif: {val_pos}, Bos: {val_empty}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = ['#2ecc71', '#e74c3c']

axes[0].bar(['Negatif', 'Pozitif'], [train_neg, train_pos], color=colors)
axes[0].set_title('Egitim Seti Sinif Dagilimi', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Goruntu Sayisi')
for i, v in enumerate([train_neg, train_pos]):
    axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')

axes[1].bar(['Negatif', 'Pozitif'], [val_neg, val_pos], color=colors)
axes[1].set_title('Dogrulama Seti Sinif Dagilimi', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Goruntu Sayisi')
for i, v in enumerate([val_neg, val_pos]):
    axes[1].text(i, v + 2, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("sinif_dagilimi.png", dpi=150, bbox_inches='tight')
plt.show()
print("Sinif dagilimi grafigi kaydedildi: sinif_dagilimi.png")

# ============================================================
# 6. MODEL EGITIMI - YOLOv8n
# ============================================================
print("\n" + "=" * 60)
print("[ADIM 6] YOLOv8n modeli egitiliyor...")
print("Transfer ogrenme: yolov8n.pt onceden egitilmis agirliklar")
print("Epoch: 50 | Goruntu boyutu: 640")
print("=" * 60)

model = YOLO("yolov8n.pt")

results = model.train(
    data=yaml_path,
    epochs=50,
    imgsz=640,
    batch=16,
    name="brain_tumor_yolov8n",
    patience=10,
    save=True,
    plots=True,
    verbose=True
)

print("\nEgitim tamamlandi!")

# ============================================================
# 7. MODEL DEGERLENDIRME
# ============================================================
print("\n[ADIM 7] Model degerlendiriliyor...")

best_model_path = "runs/detect/brain_tumor_yolov8n/weights/best.pt"
if not os.path.exists(best_model_path):
    # Alternatif yollar
    possible = glob.glob("runs/detect/brain_tumor_yolov8n*/weights/best.pt")
    if possible:
        best_model_path = possible[0]

best_model = YOLO(best_model_path)
metrics = best_model.val(data=yaml_path, imgsz=640, plots=True)

print("\n" + "=" * 60)
print("DEGERLENDIRME SONUCLARI")
print("=" * 60)
print(f"  Precision (Kesinlik)  : {metrics.box.mp:.4f}")
print(f"  Recall (Duyarlilik)   : {metrics.box.mr:.4f}")
print(f"  mAP@0.5              : {metrics.box.map50:.4f}")
print(f"  mAP@0.5:0.95         : {metrics.box.map:.4f}")
print("=" * 60)

# ============================================================
# 8. EGITIM GRAFIKLERI GORSELLESTIRME
# ============================================================
print("\n[ADIM 8] Egitim grafikleri olusturuluyor...")

results_dir = os.path.dirname(os.path.dirname(best_model_path))

# results.png otomatik olusturulur, onu gosterelim
results_img = os.path.join(results_dir, "results.png")
if os.path.exists(results_img):
    fig, ax = plt.subplots(figsize=(18, 10))
    img = mpimg.imread(results_img)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title("YOLOv8n Egitim Sonuclari", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("egitim_sonuclari.png", dpi=150, bbox_inches='tight')
    plt.show()

# Confusion Matrix
cm_img = os.path.join(results_dir, "confusion_matrix.png")
if not os.path.exists(cm_img):
    cm_img = os.path.join(results_dir, "confusion_matrix_normalized.png")

if os.path.exists(cm_img):
    fig, ax = plt.subplots(figsize=(8, 8))
    img = mpimg.imread(cm_img)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title("Karisiklik Matrisi (Confusion Matrix)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("karisiklik_matrisi.png", dpi=150, bbox_inches='tight')
    plt.show()

# F1-Curve
f1_img = os.path.join(results_dir, "F1_curve.png")
if os.path.exists(f1_img):
    fig, ax = plt.subplots(figsize=(10, 6))
    img = mpimg.imread(f1_img)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title("F1-Confidence Egrisi", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("f1_egrisi.png", dpi=150, bbox_inches='tight')
    plt.show()

# PR Curve
pr_img = os.path.join(results_dir, "PR_curve.png")
if os.path.exists(pr_img):
    fig, ax = plt.subplots(figsize=(10, 6))
    img = mpimg.imread(pr_img)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title("Precision-Recall Egrisi", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("pr_egrisi.png", dpi=150, bbox_inches='tight')
    plt.show()

print("Tum grafikler kaydedildi!")

# ============================================================
# 9. ORNEK TAHMINLER (INFERENCE)
# ============================================================
print("\n[ADIM 9] Ornek tahminler yapiliyor...")

sample_val = random.sample(val_images, min(6, len(val_images)))
predict_results = best_model.predict(
    source=sample_val,
    imgsz=640,
    conf=0.25,
    save=True,
    name="brain_tumor_predictions"
)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("YOLOv8n Beyin Tumoru Tespit Sonuclari", fontsize=16, fontweight='bold')

pred_dir = "runs/detect/brain_tumor_predictions"
if not os.path.exists(pred_dir):
    possible = glob.glob("runs/detect/brain_tumor_predictions*")
    if possible:
        pred_dir = possible[-1]

pred_images = glob.glob(os.path.join(pred_dir, "*.*"))[:6]

for idx in range(6):
    row, col = idx // 3, idx % 3
    if idx < len(pred_images):
        img = mpimg.imread(pred_images[idx])
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"Tahmin {idx+1}", fontsize=12, fontweight='bold')
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig("tahmin_sonuclari.png", dpi=150, bbox_inches='tight')
plt.show()
print("Tahmin sonuclari kaydedildi: tahmin_sonuclari.png")

# ============================================================
# 10. SONUC OZETI
# ============================================================
print("\n" + "=" * 60)
print("PROJE SONUC OZETI")
print("=" * 60)
print(f"""
Proje: YOLOv8 ile Beyin Tumoru Tespiti
Model: YOLOv8n (Transfer Ogrenme - yolov8n.pt)
Veri Seti: Ultralytics Brain Tumor Detection
  - Egitim: {len(train_images)} goruntu
  - Dogrulama: {len(val_images)} goruntu
  - Siniflar: negatif (0), pozitif (1)
Egitim Parametreleri:
  - Epoch: 50
  - Goruntu Boyutu: 640x640
  - Batch Size: 16
  - Optimizer: SGD (varsayilan)

PERFORMANS METRIKLERI:
  Precision (Kesinlik)  : {metrics.box.mp:.4f}
  Recall (Duyarlilik)   : {metrics.box.mr:.4f}
  mAP@0.5              : {metrics.box.map50:.4f}
  mAP@0.5:0.95         : {metrics.box.map:.4f}

En iyi model: {best_model_path}
""")
print("=" * 60)
print("Proje basariyla tamamlandi!")
print("=" * 60)
