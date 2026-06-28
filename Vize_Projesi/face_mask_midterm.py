#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
İSTANBUL TOPKAPI ÜNİVERSİTESİ
DERİN ÖĞRENME ARASINAV ÖDEVİ

Face Mask Detection - DenseNet121 & MobileNetV1
Transfer Learning ile Yüz Maskesi Tespiti

Öğrenci: Ali Kemal Kefelioğlu
"""

# ============================================================================
# 1. GEREKLİ KÜTÜPHANELERİN YÜKLENMESİ
# ============================================================================
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Temel kütüphaneler
import numpy as np
import pandas as pd

# Görselleştirme
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn - Veri ayırma ve performans metrikleri
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, roc_auc_score
)

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121, MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Görüntü işleme
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

print("=" * 70)
print("TensorFlow Sürümü:", tf.__version__)
print("GPU Kullanılabilir mi?:", "Evet" if len(tf.config.list_physical_devices('GPU')) > 0 else "Hayır (CPU kullanılacak)")
print("=" * 70)

# ============================================================================
# 2. VERİ SETİ YÜKLEME
# ============================================================================
# Face Mask Detection Dataset
# https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
#
# Kaggle'dan indirmek için:
#   pip install kaggle
#   kaggle datasets download -d omkargurav/face-mask-dataset
#
# Veya manuel olarak indirip aşağıdaki dizine çıkartın.

# Veri seti dizini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Sonuçlar dizini oluştur
os.makedirs(RESULTS_DIR, exist_ok=True)

# Kaggle'dan veri setini indir (eğer yoksa)
DATASET_DIR = os.path.join(DATA_DIR, "face-mask-dataset")

if not os.path.exists(DATASET_DIR):
    print("\n[INFO] Veri seti indiriliyor...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    try:
        # Kaggle API ile indirme
        os.system(f'pip install -q kaggle')
        os.system(f'kaggle datasets download -d omkargurav/face-mask-dataset -p "{DATA_DIR}" --unzip')
        
        # Kaggle bazen farklı bir klasör yapısı oluşturabilir
        # Olası klasör yapılarını kontrol et
        possible_paths = [
            os.path.join(DATA_DIR, "face-mask-dataset"),
            os.path.join(DATA_DIR, "data"),
            DATA_DIR
        ]
        
        for path in possible_paths:
            if os.path.exists(os.path.join(path, "with_mask")) or \
               os.path.exists(os.path.join(path, "without_mask")):
                DATASET_DIR = path
                break
                
    except Exception as e:
        print(f"[HATA] Kaggle API ile indirme başarısız: {e}")
        print("[INFO] Lütfen veri setini manuel olarak indirin:")
        print("  1. https://www.kaggle.com/datasets/omkargurav/face-mask-dataset adresine gidin")
        print("  2. Veri setini indirin")
        print(f"  3. '{DATA_DIR}' dizinine çıkartın")
        print("  4. Klasör yapısı şöyle olmalı:")
        print(f"     {DATASET_DIR}/with_mask/")
        print(f"     {DATASET_DIR}/without_mask/")
        sys.exit(1)

# Veri seti klasör yapısını kontrol et
with_mask_dir = None
without_mask_dir = None

# Olası klasör yapılarını kontrol et
for root, dirs, files in os.walk(DATA_DIR):
    for d in dirs:
        if d.lower() in ['with_mask', 'withmask', 'with mask']:
            with_mask_dir = os.path.join(root, d)
        elif d.lower() in ['without_mask', 'withoutmask', 'without mask']:
            without_mask_dir = os.path.join(root, d)

if with_mask_dir is None or without_mask_dir is None:
    print("[HATA] Veri seti klasörleri bulunamadı!")
    print(f"[INFO] Aranan dizin: {DATA_DIR}")
    print("[INFO] Beklenen klasörler: with_mask/ ve without_mask/")
    print("\nMevcut dosya yapısı:")
    for root, dirs, files in os.walk(DATA_DIR):
        level = root.replace(DATA_DIR, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        if level < 3:
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:
                print(f'{subindent}{file}')
            if len(files) > 5:
                print(f'{subindent}... ve {len(files)-5} dosya daha')
    sys.exit(1)

print(f"\n[INFO] Maskeli yüz dizini: {with_mask_dir}")
print(f"[INFO] Maskesiz yüz dizini: {without_mask_dir}")

# ============================================================================
# Görüntüleri yükle ve etiketlerle birleştir
# ============================================================================
image_size = (224, 224)  # DenseNet121 ve MobileNet için standart giriş boyutu

def load_images_from_directory(directory, label, target_size=image_size):
    """Belirtilen dizindeki görüntüleri yükler ve etiketler."""
    images = []
    labels = []
    filenames = []
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    for filename in os.listdir(directory):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in valid_extensions:
            continue
            
        filepath = os.path.join(directory, filename)
        try:
            img = load_img(filepath, target_size=target_size)
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(label)
            filenames.append(filename)
        except Exception as e:
            print(f"  [UYARI] Yüklenemedi: {filename} - {e}")
            continue
    
    return images, labels, filenames

print("\n[INFO] Maskeli yüz görüntüleri yükleniyor...")
with_mask_images, with_mask_labels, with_mask_files = load_images_from_directory(with_mask_dir, 1)
print(f"  -> {len(with_mask_images)} maskeli yüz görüntüsü yüklendi.")

print("[INFO] Maskesiz yüz görüntüleri yükleniyor...")
without_mask_images, without_mask_labels, without_mask_files = load_images_from_directory(without_mask_dir, 0)
print(f"  -> {len(without_mask_images)} maskesiz yüz görüntüsü yüklendi.")

# NumPy dizilerine dönüştür
X = np.array(with_mask_images + without_mask_images)
y = np.array(with_mask_labels + without_mask_labels)

print(f"\n[INFO] Toplam görüntü sayısı: {len(X)}")
print(f"[INFO] Görüntü boyutu: {X[0].shape}")
print(f"[INFO] Sınıf dağılımı:")
print(f"  -> Maskeli (1): {np.sum(y == 1)}")
print(f"  -> Maskesiz (0): {np.sum(y == 0)}")

# Veri seti dağılımını görselleştir
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sınıf dağılımı
class_counts = pd.Series(y).value_counts().sort_index()
colors = ['#e74c3c', '#2ecc71']
axes[0].bar(['Maskesiz (0)', 'Maskeli (1)'], class_counts.values, color=colors, edgecolor='black')
axes[0].set_title('Veri Seti Sınıf Dağılımı', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Görüntü Sayısı')
for i, v in enumerate(class_counts.values):
    axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')

# Örnek görseller
sample_indices = np.random.choice(len(X), 8, replace=False)
for idx, sample_idx in enumerate(sample_indices[:4]):
    ax_pos = axes[1]
    
axes[1].axis('off')
axes[1].set_title('Örnek Görüntüler (Aşağıda)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'veri_seti_dagilimi.png'), dpi=150, bbox_inches='tight')
plt.show()
print("[INFO] Veri seti dağılım grafiği kaydedildi.")

# Örnek görüntüleri göster
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
fig.suptitle('Örnek Görüntüler', fontsize=16, fontweight='bold')

# 5 maskeli
mask_indices = np.where(y == 1)[0]
no_mask_indices = np.where(y == 0)[0]

for i in range(5):
    idx = mask_indices[np.random.randint(len(mask_indices))]
    axes[0, i].imshow(X[idx].astype(np.uint8))
    axes[0, i].set_title('Maskeli', color='green', fontweight='bold')
    axes[0, i].axis('off')

for i in range(5):
    idx = no_mask_indices[np.random.randint(len(no_mask_indices))]
    axes[1, i].imshow(X[idx].astype(np.uint8))
    axes[1, i].set_title('Maskesiz', color='red', fontweight='bold')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'ornek_goruntuler.png'), dpi=150, bbox_inches='tight')
plt.show()
print("[INFO] Örnek görüntüler kaydedildi.")

# ============================================================================
# 3. VERİ SETİNİN EĞİTİM/VALİDASYON VE TEST OLARAK AYRILMASI
# ============================================================================
# Toplam verinin %20'si test set, %80'i eğitim+validasyon
print("\n" + "=" * 70)
print("VERİ SETİ BÖLÜMLEME")
print("=" * 70)

# Piksel değerlerini [0, 1] aralığına normalize et
X_normalized = X / 255.0

# %20 test, %80 eğitim+validasyon
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_normalized, y, test_size=0.20, random_state=42, stratify=y
)

# Eğitim+validasyon setini %80 eğitim, %20 validasyon olarak böl
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.20, random_state=42, stratify=y_train_val
)

print(f"\nToplam veri: {len(X_normalized)}")
print(f"  Eğitim seti: {len(X_train)} ({len(X_train)/len(X_normalized)*100:.1f}%)")
print(f"  Validasyon seti: {len(X_val)} ({len(X_val)/len(X_normalized)*100:.1f}%)")
print(f"  Test seti: {len(X_test)} ({len(X_test)/len(X_normalized)*100:.1f}%)")

print(f"\nEğitim seti sınıf dağılımı:")
print(f"  Maskeli: {np.sum(y_train == 1)}, Maskesiz: {np.sum(y_train == 0)}")
print(f"Validasyon seti sınıf dağılımı:")
print(f"  Maskeli: {np.sum(y_val == 1)}, Maskesiz: {np.sum(y_val == 0)}")
print(f"Test seti sınıf dağılımı:")
print(f"  Maskeli: {np.sum(y_test == 1)}, Maskesiz: {np.sum(y_test == 0)}")

# One-hot encoding (softmax çıkışı için)
from tensorflow.keras.utils import to_categorical
y_train_cat = to_categorical(y_train, 2)
y_val_cat = to_categorical(y_val, 2)
y_test_cat = to_categorical(y_test, 2)

# ============================================================================
# 4. VERİ ARTTIRIMI (DATA AUGMENTATION)
# ============================================================================
print("\n" + "=" * 70)
print("VERİ ARTTIRIMI (DATA AUGMENTATION)")
print("=" * 70)

# Eğitim verisi için augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validasyon ve test verisi için sadece rescale (augmentation yok)
val_test_datagen = ImageDataGenerator(rescale=1.0/255)

# NOT: Veriler zaten normalize edildiği için datagen'i flow ile kullanırken
# tekrar rescale etmiyoruz. Bunun yerine direkt augmented generator oluşturuyoruz.

# Augmentation'ı eğitim verisine uygula (rescale olmadan, veriler zaten 0-1 aralığında)
train_datagen_no_rescale = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

BATCH_SIZE = 32

train_generator = train_datagen_no_rescale.flow(
    X_train, y_train_cat, batch_size=BATCH_SIZE, shuffle=True
)

print(f"Batch boyutu: {BATCH_SIZE}")
print(f"Eğitim adım sayısı (steps per epoch): {len(X_train) // BATCH_SIZE}")
print("Augmentation parametreleri:")
print("  - rotation_range=10")
print("  - width_shift_range=0.1")
print("  - height_shift_range=0.1")
print("  - shear_range=0.1")
print("  - zoom_range=0.1")
print("  - horizontal_flip=True")
print("  - fill_mode='nearest'")

# Augmented görüntüleri göster
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
fig.suptitle('Veri Arttırımı (Augmentation) Örnekleri', fontsize=16, fontweight='bold')

sample_img = X_train[0:1]  # İlk eğitim görüntüsü
temp_gen = train_datagen_no_rescale.flow(sample_img, batch_size=1)

axes[0, 0].imshow(sample_img[0])
axes[0, 0].set_title('Orijinal', fontweight='bold')
axes[0, 0].axis('off')

for i in range(1, 5):
    aug_img = next(temp_gen)[0]
    aug_img = np.clip(aug_img, 0, 1)
    axes[0, i].imshow(aug_img)
    axes[0, i].set_title(f'Augmented {i}', fontweight='bold')
    axes[0, i].axis('off')

sample_img2 = X_train[len(X_train)//2:len(X_train)//2+1]
temp_gen2 = train_datagen_no_rescale.flow(sample_img2, batch_size=1)

axes[1, 0].imshow(sample_img2[0])
axes[1, 0].set_title('Orijinal', fontweight='bold')
axes[1, 0].axis('off')

for i in range(1, 5):
    aug_img = next(temp_gen2)[0]
    aug_img = np.clip(aug_img, 0, 1)
    axes[1, i].imshow(aug_img)
    axes[1, i].set_title(f'Augmented {i}', fontweight='bold')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'augmentation_ornekleri.png'), dpi=150, bbox_inches='tight')
plt.show()
print("[INFO] Augmentation örnekleri kaydedildi.")


# ============================================================================
# 5. MODEL OLUŞTURMA FONKSİYONU
# ============================================================================
def create_model(base_model_class, model_name, image_size=(224, 224)):
    """
    Transfer learning ile model oluşturur.
    
    Args:
        base_model_class: Keras pre-trained model sınıfı (DenseNet121 veya MobileNet)
        model_name: Model adı (kayıt için)
        image_size: Giriş görüntüsü boyutu
    
    Returns:
        Derlenmiş Keras modeli
    """
    print(f"\n{'='*70}")
    print(f"MODEL OLUŞTURMA: {model_name}")
    print(f"{'='*70}")
    
    # Base model (önceden eğitilmiş ağırlıklarla)
    base_model = base_model_class(
        weights='imagenet',
        include_top=False,
        input_shape=(image_size[0], image_size[1], 3)
    )
    
    # Base model katmanlarını dondur (freeze)
    base_model.trainable = False
    
    # Yeni sınıflandırma katmanları ekle
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)                          # Dropout katmanı
    x = Dense(256, activation='relu')(x)          # 1. Dense katmanı
    x = Dense(256, activation='relu')(x)          # 2. Dense katmanı  
    x = Dense(128, activation='relu')(x)          # 3. Dense katmanı
    x = Dense(64, activation='relu')(x)           # 4. Dense katmanı
    predictions = Dense(2, activation='softmax')(x)  # Çıkış katmanı (2 sınıf)
    
    # Model oluştur
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Model derleme
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Model özeti
    print(f"\n{model_name} Modeli Özeti:")
    print(f"  Toplam parametre sayısı: {model.count_params():,}")
    trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_count = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    print(f"  Eğitilebilir parametreler: {trainable_count:,}")
    print(f"  Eğitilemez parametreler: {non_trainable_count:,}")
    
    model.summary()
    
    return model


# ============================================================================
# 6. MODEL EĞİTİM FONKSİYONU
# ============================================================================
def train_model(model, model_name, train_gen, X_val, y_val_cat, epochs=100):
    """
    Modeli eğitir ve eğitim sürecini takip eder.
    
    Args:
        model: Derlenmiş Keras modeli
        model_name: Model adı
        train_gen: Eğitim data generator
        X_val: Validasyon verileri
        y_val_cat: Validasyon etiketleri (one-hot)
        epochs: Epoch sayısı
    
    Returns:
        Eğitim geçmişi (history)
    """
    print(f"\n{'='*70}")
    print(f"MODEL EĞİTİMİ: {model_name}")
    print(f"{'='*70}")
    
    # Callback'ler
    # 1. Early Stopping - patience=25
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=25,
        verbose=1,
        restore_best_weights=True,
        mode='min'
    )
    
    # 2. ReduceLROnPlateau - factor=0.1, patience=5
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=5,
        verbose=1,
        min_lr=1e-7,
        mode='min'
    )
    
    # 3. ModelCheckpoint - En iyi modeli kaydet
    checkpoint_path = os.path.join(RESULTS_DIR, f'{model_name}_best_model.keras')
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1,
        mode='min'
    )
    
    callbacks = [early_stopping, reduce_lr, model_checkpoint]
    
    print(f"\nEğitim Parametreleri:")
    print(f"  Epoch sayısı: {epochs}")
    print(f"  Optimizer: Adam (lr=0.001)")
    print(f"  Early Stopping: patience=25")
    print(f"  ReduceLROnPlateau: factor=0.1, patience=5")
    print(f"  Batch boyutu: {BATCH_SIZE}")
    
    # Model eğitimi
    steps_per_epoch = len(X_train) // BATCH_SIZE
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=(X_val, y_val_cat),
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n[INFO] {model_name} eğitimi tamamlandı!")
    print(f"  En iyi validasyon loss: {min(history.history['val_loss']):.4f}")
    print(f"  En iyi validasyon accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"  Toplam eğitilen epoch: {len(history.history['loss'])}")
    
    return history


# ============================================================================
# 7. EĞİTİM GRAFİKLERİ FONKSİYONU
# ============================================================================
def plot_training_history(history, model_name):
    """
    Eğitim ve validasyon accuracy/loss grafiklerini çizer.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{model_name} - Eğitim Grafikleri', fontsize=16, fontweight='bold')
    
    epochs_range = range(1, len(history.history['loss']) + 1)
    
    # Accuracy grafiği
    axes[0].plot(epochs_range, history.history['accuracy'], 'b-', label='Eğitim Accuracy', linewidth=2)
    axes[0].plot(epochs_range, history.history['val_accuracy'], 'r-', label='Validasyon Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy (Doğruluk)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(loc='lower right', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    
    # Loss grafiği
    axes[1].plot(epochs_range, history.history['loss'], 'b-', label='Eğitim Loss', linewidth=2)
    axes[1].plot(epochs_range, history.history['val_loss'], 'r-', label='Validasyon Loss', linewidth=2)
    axes[1].set_title('Model Loss (Kayıp)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(loc='upper right', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{model_name}_egitim_grafikleri.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[INFO] {model_name} eğitim grafikleri kaydedildi.")


# ============================================================================
# 8. TAHMİN VE SONUÇ ANALİZİ FONKSİYONU
# ============================================================================
def evaluate_model(model, model_name, X_test, y_test, y_test_cat):
    """
    Modeli test seti üzerinde değerlendirir ve tüm metrikleri hesaplar.
    
    Returns:
        dict: Tüm metrikleri içeren sözlük
    """
    print(f"\n{'='*70}")
    print(f"TEST SETİ DEĞERLENDİRME: {model_name}")
    print(f"{'='*70}")
    
    # Tahminler
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = y_test
    
    # ---- METRİKLER ----
    
    # 1. Accuracy (Doğruluk)
    acc = accuracy_score(y_true, y_pred)
    
    # 2. Precision (Hassasiyet)
    prec = precision_score(y_true, y_pred, average='binary', pos_label=1)
    
    # 3. Recall / Sensitivity (Duyarlılık)
    rec = recall_score(y_true, y_pred, average='binary', pos_label=1)
    
    # 4. F1-Score
    f1 = f1_score(y_true, y_pred, average='binary', pos_label=1)
    
    # 5. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # 6. Specificity (Özgüllük)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # 7. AUC
    y_pred_proba_positive = y_pred_proba[:, 1]  # Pozitif sınıf olasılıkları
    auc_score = roc_auc_score(y_true, y_pred_proba_positive)
    
    # Sonuçları yazdır
    print(f"\n📊 {model_name} - Performans Metrikleri:")
    print(f"{'─'*50}")
    print(f"  Accuracy (Doğruluk):      {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision (Hassasiyet):   {prec:.4f} ({prec*100:.2f}%)")
    print(f"  Recall (Duyarlılık):      {rec:.4f} ({rec*100:.2f}%)")
    print(f"  Specificity (Özgüllük):   {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"  F1-Score:                 {f1:.4f} ({f1*100:.2f}%)")
    print(f"  AUC:                      {auc_score:.4f}")
    print(f"{'─'*50}")
    
    print(f"\n📋 Confusion Matrix:")
    print(f"  True Negatives (TN):  {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Positives (TP):  {tp}")
    
    # Detaylı sınıflandırma raporu
    print(f"\n📋 Sınıflandırma Raporu:")
    print(classification_report(y_true, y_pred, target_names=['Maskesiz', 'Maskeli']))
    
    # ---- GÖRSELLEŞTİRMELER ----
    
    # 1. Confusion Matrix Görselleştirmesi
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{model_name} - Test Sonuçları', fontsize=16, fontweight='bold')
    
    # Confusion Matrix heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Maskesiz', 'Maskeli'],
                yticklabels=['Maskesiz', 'Maskeli'],
                ax=axes[0], annot_kws={"size": 16})
    axes[0].set_title('Confusion Matrix (Karışıklık Matrisi)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Tahmin Edilen', fontsize=12)
    axes[0].set_ylabel('Gerçek', fontsize=12)
    
    # 2. ROC Eğrisi
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba_positive)
    
    axes[1].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Eğrisi (AUC = {auc_score:.4f})')
    axes[1].plot([0, 1], [0, 1], 'r--', linewidth=1, label='Rastgele Sınıflandırıcı')
    axes[1].fill_between(fpr, tpr, alpha=0.1, color='blue')
    axes[1].set_title('ROC Eğrisi', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    axes[1].set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    axes[1].legend(loc='lower right', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([-0.01, 1.01])
    axes[1].set_ylim([-0.01, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{model_name}_test_sonuclari.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[INFO] {model_name} test sonuçları kaydedildi.")
    
    # Metrikleri sözlük olarak döndür
    metrics = {
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'Specificity': specificity,
        'F1-Score': f1,
        'AUC': auc_score,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
    }
    
    return metrics


# ============================================================================
# 9. MODEL 1: DenseNet121
# ============================================================================
print("\n" + "#" * 70)
print("#" + " " * 20 + "MODEL 1: DenseNet121" + " " * 28 + "#")
print("#" * 70)

# DenseNet121 modeli oluştur
densenet_model = create_model(DenseNet121, "DenseNet121", image_size)

# DenseNet121 modeli eğit
densenet_history = train_model(
    densenet_model, "DenseNet121", 
    train_generator, X_val, y_val_cat, 
    epochs=100
)

# Eğitim grafiklerini çiz
plot_training_history(densenet_history, "DenseNet121")

# Test seti üzerinde değerlendirme
densenet_metrics = evaluate_model(densenet_model, "DenseNet121", X_test, y_test, y_test_cat)


# ============================================================================
# 10. MODEL 2: MobileNetV1
# ============================================================================
print("\n" + "#" * 70)
print("#" + " " * 20 + "MODEL 2: MobileNetV1" + " " * 28 + "#")
print("#" * 70)

# Eğitim generator'ı yeniden oluştur (iterasyonları sıfırlamak için)
train_generator_mobilenet = train_datagen_no_rescale.flow(
    X_train, y_train_cat, batch_size=BATCH_SIZE, shuffle=True
)

# MobileNet modeli oluştur
mobilenet_model = create_model(MobileNet, "MobileNetV1", image_size)

# MobileNet modeli eğit
mobilenet_history = train_model(
    mobilenet_model, "MobileNetV1",
    train_generator_mobilenet, X_val, y_val_cat,
    epochs=100
)

# Eğitim grafiklerini çiz
plot_training_history(mobilenet_history, "MobileNetV1")

# Test seti üzerinde değerlendirme
mobilenet_metrics = evaluate_model(mobilenet_model, "MobileNetV1", X_test, y_test, y_test_cat)


# ============================================================================
# 11. MODEL KARŞILAŞTIRMASI
# ============================================================================
print("\n" + "=" * 70)
print("MODEL KARŞILAŞTIRMASI")
print("=" * 70)

# Karşılaştırma tablosu
comparison_df = pd.DataFrame([densenet_metrics, mobilenet_metrics])
comparison_df = comparison_df[['Model', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'AUC']]

print("\n📊 Model Karşılaştırma Tablosu:")
print(comparison_df.to_string(index=False))

# Karşılaştırma grafiği
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('DenseNet121 vs MobileNetV1 - Model Karşılaştırması', fontsize=16, fontweight='bold')

# Metrikler karşılaştırması (bar chart)
metrics_to_compare = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'AUC']
x = np.arange(len(metrics_to_compare))
width = 0.35

densenet_values = [densenet_metrics[m] for m in metrics_to_compare]
mobilenet_values = [mobilenet_metrics[m] for m in metrics_to_compare]

bars1 = axes[0].bar(x - width/2, densenet_values, width, label='DenseNet121', color='#3498db', edgecolor='black')
bars2 = axes[0].bar(x + width/2, mobilenet_values, width, label='MobileNetV1', color='#e74c3c', edgecolor='black')

axes[0].set_xlabel('Metrikler', fontsize=12)
axes[0].set_ylabel('Değer', fontsize=12)
axes[0].set_title('Performans Metrikleri Karşılaştırması', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics_to_compare, rotation=45)
axes[0].legend(fontsize=12)
axes[0].set_ylim([0, 1.1])
axes[0].grid(True, alpha=0.3, axis='y')

# Bar üzerine değer yaz
for bar, val in zip(bars1, densenet_values):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar, val in zip(bars2, mobilenet_values):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# ROC Eğrisi karşılaştırması
# DenseNet121 ROC
y_pred_dense = densenet_model.predict(X_test, verbose=0)
fpr_dense, tpr_dense, _ = roc_curve(y_test, y_pred_dense[:, 1])
auc_dense = auc(fpr_dense, tpr_dense)

# MobileNet ROC
y_pred_mobile = mobilenet_model.predict(X_test, verbose=0)
fpr_mobile, tpr_mobile, _ = roc_curve(y_test, y_pred_mobile[:, 1])
auc_mobile = auc(fpr_mobile, tpr_mobile)

axes[1].plot(fpr_dense, tpr_dense, 'b-', linewidth=2, label=f'DenseNet121 (AUC = {auc_dense:.4f})')
axes[1].plot(fpr_mobile, tpr_mobile, 'r-', linewidth=2, label=f'MobileNetV1 (AUC = {auc_mobile:.4f})')
axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Rastgele')
axes[1].set_title('ROC Eğrisi Karşılaştırması', fontsize=14, fontweight='bold')
axes[1].set_xlabel('False Positive Rate', fontsize=12)
axes[1].set_ylabel('True Positive Rate', fontsize=12)
axes[1].legend(loc='lower right', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'model_karsilastirmasi.png'), dpi=150, bbox_inches='tight')
plt.show()
print("[INFO] Model karşılaştırma grafikleri kaydedildi.")

# Eğitim süreçleri karşılaştırması
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Eğitim Süreçleri Karşılaştırması', fontsize=16, fontweight='bold')

# Accuracy karşılaştırması
axes[0].plot(densenet_history.history['accuracy'], 'b-', label='DenseNet121 Eğitim', linewidth=1.5)
axes[0].plot(densenet_history.history['val_accuracy'], 'b--', label='DenseNet121 Validasyon', linewidth=1.5)
axes[0].plot(mobilenet_history.history['accuracy'], 'r-', label='MobileNetV1 Eğitim', linewidth=1.5)
axes[0].plot(mobilenet_history.history['val_accuracy'], 'r--', label='MobileNetV1 Validasyon', linewidth=1.5)
axes[0].set_title('Accuracy Karşılaştırması', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Loss karşılaştırması
axes[1].plot(densenet_history.history['loss'], 'b-', label='DenseNet121 Eğitim', linewidth=1.5)
axes[1].plot(densenet_history.history['val_loss'], 'b--', label='DenseNet121 Validasyon', linewidth=1.5)
axes[1].plot(mobilenet_history.history['loss'], 'r-', label='MobileNetV1 Eğitim', linewidth=1.5)
axes[1].plot(mobilenet_history.history['val_loss'], 'r--', label='MobileNetV1 Validasyon', linewidth=1.5)
axes[1].set_title('Loss Karşılaştırması', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'egitim_surecleri_karsilastirmasi.png'), dpi=150, bbox_inches='tight')
plt.show()
print("[INFO] Eğitim süreçleri karşılaştırma grafikleri kaydedildi.")


# ============================================================================
# 12. SONUÇ RAPORU
# ============================================================================
print("\n" + "=" * 70)
print("SONUÇ RAPORU")
print("=" * 70)

# En iyi modeli belirle
best_model_name = "DenseNet121" if densenet_metrics['Accuracy'] >= mobilenet_metrics['Accuracy'] else "MobileNetV1"

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    SONUÇ VE YORUMLAR                               ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Proje: Face Mask Detection (Yüz Maskesi Tespiti)                    ║
║  Veri Seti: Kaggle Face Mask Dataset                                 ║
║  Modeller: DenseNet121 ve MobileNetV1 (Transfer Learning)            ║
║                                                                      ║
║  Veri Seti Bilgileri:                                                ║
║    - Toplam görüntü: {len(X_normalized)}                                           ║
║    - Eğitim: {len(X_train)} | Validasyon: {len(X_val)} | Test: {len(X_test)}                ║
║    - Görüntü boyutu: {image_size[0]}x{image_size[1]}x3                                ║
║                                                                      ║
║  Eğitim Parametreleri:                                               ║
║    - Optimizer: Adam (lr=0.001)                                      ║
║    - Early Stopping: patience=25                                     ║
║    - ReduceLROnPlateau: factor=0.1, patience=5                       ║
║    - Epoch: 100 (max)                                                ║
║    - Batch Size: {BATCH_SIZE}                                                 ║
║    - Data Augmentation: Evet                                         ║
║                                                                      ║
║  Sonuçlar:                                                           ║
║    DenseNet121:                                                      ║
║      Accuracy: {densenet_metrics['Accuracy']:.4f} | F1: {densenet_metrics['F1-Score']:.4f} | AUC: {densenet_metrics['AUC']:.4f}           ║
║    MobileNetV1:                                                      ║
║      Accuracy: {mobilenet_metrics['Accuracy']:.4f} | F1: {mobilenet_metrics['F1-Score']:.4f} | AUC: {mobilenet_metrics['AUC']:.4f}           ║
║                                                                      ║
║  En İyi Model: {best_model_name}                                            ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# Yorumlar
print("""
📝 YORUMLAR:

1. Model Mimarisi:
   - Her iki modelde de ImageNet ağırlıkları kullanılarak transfer learning 
     uygulanmıştır.
   - Base model katmanları dondurulmuş (freeze), sadece eklenen dense ve 
     dropout katmanları eğitilmiştir.
   - Dropout (0.5) ile overfitting önlenmeye çalışılmıştır.

2. Hiperparametre Seçimleri:
   - Adam optimizer, adaptif öğrenme hızı sayesinde hızlı yakınsama sağlar.
   - ReduceLROnPlateau, validasyon kaybı iyileşmediğinde öğrenme hızını 
     otomatik olarak düşürerek daha ince ayar yapılmasını sağlar.
   - Early Stopping ile gereksiz epoch'lar önlenir ve en iyi model kaydedilir.

3. Veri Arttırımı:
   - Eğitim verisine rotation, shift, shear, zoom ve flip uygulanmıştır.
   - Bu teknikler modelin daha genel öğrenmesini ve farklı açılardan 
     görüntüleri tanımasını sağlar.

4. Model Karşılaştırması:
   - DenseNet121 daha derin bir mimari sunarak karmaşık özellikleri 
     yakalamada avantaj sağlayabilir.
   - MobileNetV1 daha hafif bir mimari olup, hızlı eğitim ve 
     inference süreleri sunar.
   - Her iki model de yüz maskesi tespitinde yüksek performans göstermiştir.
""")

# Sonuçları CSV olarak kaydet
comparison_df.to_csv(os.path.join(RESULTS_DIR, 'model_karsilastirmasi.csv'), index=False)
print(f"\n[INFO] Sonuçlar '{RESULTS_DIR}' dizinine kaydedildi.")
print("[INFO] Kaydedilen dosyalar:")
for f in os.listdir(RESULTS_DIR):
    filepath = os.path.join(RESULTS_DIR, f)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  📁 {f} ({size_mb:.2f} MB)")

print("\n" + "=" * 70)
print("PROGRAM TAMAMLANDI!")
print("=" * 70)
