"""
EuroSAT Veri Yükleme Modülü
===========================
Bu modül veri setinin indirilmesi, ön işleme ve DataLoader oluşturma işlemlerini içerir.
"""

import os
import torch
import zipfile
import urllib.request
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image


# ==================== SABITLER ====================
# ImageNet normalizasyon değerleri (Transfer Learning için standart)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# EuroSAT sınıfları
CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

NUM_CLASSES = 10


# ==================== VERİ DÖNÜŞÜM FONKSİYONLARI ====================
def get_train_transforms(img_size=224):
    """
    Eğitim için veri artırma (data augmentation) dönüşümleri.

    Args:
        img_size: Hedef görüntü boyutu (varsayılan: 224x224)

    Returns:
        torchvision.transforms.Compose: Dönüşüm pipeline'ı
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_test_transforms(img_size=224):
    """
    Test/Validation için dönüşümler (augmentation yok).

    Args:
        img_size: Hedef görüntü boyutu (varsayılan: 224x224)

    Returns:
        torchvision.transforms.Compose: Dönüşüm pipeline'ı
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


# ==================== VERİ SETİ İNDİRME ====================
def download_eurosat(data_dir='./data', extract_dir='./data'):
    """
    EuroSAT RGB veri setini indirir ve çıkarır.

    Args:
        data_dir: İndirme dizini
        extract_dir: Çıkarma dizini

    Returns:
        str: Veri seti klasör yolu
    """
    url = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"
    zip_path = os.path.join(data_dir, "EuroSAT.zip")
    dataset_path = os.path.join(extract_dir, "2750")  # RGB versiyonu

    # Klasör oluştur
    os.makedirs(data_dir, exist_ok=True)

    # Eğer zaten varsa indirme
    if os.path.exists(dataset_path):
        print(f"✓ Veri seti zaten mevcut: {dataset_path}")
        return dataset_path

    # İndir
    print(f"⬇ Veri seti indiriliyor: {url}")
    urllib.request.urlretrieve(url, zip_path)
    print("✓ İndirme tamamlandı!")

    # Zip'i çıkar
    print("📦 Zip dosyası çıkarılıyor...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("✓ Çıkarma tamamlandı!")

    # Zip dosyasını sil (opsiyonel)
    os.remove(zip_path)

    return dataset_path


# ==================== DATALOADER OLUŞTURMA ====================
def create_dataloaders(data_path, batch_size=32, train_ratio=0.8, val_ratio=0.1,
                       num_workers=2, img_size=224, seed=42):
    """
    Train, Validation ve Test DataLoader'ları oluşturur.

    Args:
        data_path: Veri seti klasör yolu
        batch_size: Batch boyutu
        train_ratio: Eğitim oranı (varsayılan: %80)
        val_ratio: Validation oranı (varsayılan: %10)
        num_workers: DataLoader worker sayısı
        img_size: Görüntü boyutu
        seed: Random seed (tekrarlanabilirlik için)

    Returns:
        tuple: (train_loader, val_loader, test_loader, dataset_info)
    """
    # Seed ayarla
    torch.manual_seed(seed)

    # Transform'ları al
    train_transform = get_train_transforms(img_size)
    test_transform = get_test_transforms(img_size)

    # Tam veri setini yükle (transform olmadan bölmek için)
    full_dataset = datasets.ImageFolder(root=data_path)

    # Veri setini böl
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Her bir subset'e uygun transform uygula
    # Not: Subset kullanırken transform'u wrapper class ile uyguluyoruz
    train_dataset = TransformSubset(train_dataset, train_transform)
    val_dataset = TransformSubset(val_dataset, test_transform)
    test_dataset = TransformSubset(test_dataset, test_transform)

    # DataLoader'ları oluştur
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Dataset bilgisi
    dataset_info = {
        'total_samples': total_size,
        'train_samples': train_size,
        'val_samples': val_size,
        'test_samples': test_size,
        'num_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES,
        'img_size': img_size
    }

    print("\n" + "="*50)
    print("📊 VERİ SETİ BİLGİLERİ")
    print("="*50)
    print(f"Toplam görüntü sayısı: {total_size:,}")
    print(f"Eğitim seti: {train_size:,} ({train_ratio*100:.0f}%)")
    print(f"Doğrulama seti: {val_size:,} ({val_ratio*100:.0f}%)")
    print(f"Test seti: {test_size:,} ({(1-train_ratio-val_ratio)*100:.0f}%)")
    print(f"Sınıf sayısı: {NUM_CLASSES}")
    print(f"Görüntü boyutu: {img_size}x{img_size}")
    print("="*50 + "\n")

    return train_loader, val_loader, test_loader, dataset_info


class TransformSubset(torch.utils.data.Dataset):
    """
    Subset'e transform uygulayan wrapper sınıf.
    random_split sonrasında her bir bölüme farklı transform uygulamak için kullanılır.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)


# ==================== GÖRÜNTÜ ÖN İZLEME ====================
def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Normalize edilmiş tensörü orijinal haline döndürür (görselleştirme için).

    Args:
        tensor: Normalize edilmiş görüntü tensörü
        mean: Normalizasyon ortalaması
        std: Normalizasyon standart sapması

    Returns:
        tensor: Denormalize edilmiş tensör
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


# ==================== TEST ====================
if __name__ == "__main__":
    # Test için veri setini indir ve DataLoader oluştur
    print("EuroSAT Data Loader Test")
    print("-" * 40)

    # Veri setini indir
    data_path = download_eurosat(data_dir='./data', extract_dir='./data')

    # DataLoader'ları oluştur
    train_loader, val_loader, test_loader, info = create_dataloaders(
        data_path=data_path,
        batch_size=32,
        img_size=224
    )

    # Bir batch kontrol et
    images, labels = next(iter(train_loader))
    print(f"\nBatch boyutu: {images.shape}")
    print(f"Label boyutu: {labels.shape}")
    print(f"İlk 5 label: {[CLASS_NAMES[l] for l in labels[:5].tolist()]}")

