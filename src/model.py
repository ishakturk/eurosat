"""
EuroSAT Model Tanımları
=======================
Bu modül Simple CNN ve ResNet-18 model mimarilerini içerir.
"""

import torch
import torch.nn as nn
from torchvision import models


# ==================== MODEL A: SIMPLE CNN ====================
class SimpleCNN(nn.Module):
    """
    Sıfırdan tasarlanan basit CNN modeli (Baseline).

    Mimari:
        - 4 Konvolüsyon katmanı (32 -> 64 -> 128 -> 256 filtre)
        - BatchNorm ve ReLU aktivasyon
        - MaxPooling
        - 2 Fully Connected katman
        - Dropout ile regularization

    Args:
        num_classes: Çıkış sınıf sayısı (varsayılan: 10)
        dropout_rate: Dropout oranı (varsayılan: 0.5)
    """

    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()

        # Konvolüsyon Blokları
        # Input: 3x224x224

        # Block 1: 3 -> 32, 224 -> 112
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2: 32 -> 64, 112 -> 56
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 3: 64 -> 128, 56 -> 28
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 4: 128 -> 256, 28 -> 14
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Global Average Pooling: 14x14 -> 1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Katmanlar
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

        # Ağırlık başlatma
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming He ağırlık başlatma"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


# ==================== MODEL B & C: RESNET-18 ====================
class ResNet18Model(nn.Module):
    """
    ResNet-18 Transfer Learning modeli.

    İki mod destekler:
        1. Feature Extraction (freeze=True): Sadece classifier eğitilir
        2. Fine-Tuning (freeze=False): Tüm model eğitilir

    Args:
        num_classes: Çıkış sınıf sayısı (varsayılan: 10)
        pretrained: ImageNet ağırlıklarını kullan (varsayılan: True)
        freeze_backbone: Backbone katmanlarını dondur (varsayılan: True)
    """

    def __init__(self, num_classes=10, pretrained=True, freeze_backbone=True):
        super(ResNet18Model, self).__init__()

        # Önceden eğitilmiş ResNet-18 yükle
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.backbone = models.resnet18(weights=weights)
            print("✓ ImageNet ağırlıkları yüklendi")
        else:
            self.backbone = models.resnet18(weights=None)
            print("⚠ Model sıfırdan başlatıldı")

        # Feature sayısını al
        num_features = self.backbone.fc.in_features

        # Son katmanı değiştir (EuroSAT için 10 sınıf)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )

        # Backbone'u dondur/aç
        if freeze_backbone:
            self._freeze_backbone()
            print("❄ Backbone donduruldu (Feature Extraction modu)")
        else:
            self._unfreeze_backbone()
            print("🔥 Backbone açıldı (Fine-Tuning modu)")

    def _freeze_backbone(self):
        """Backbone katmanlarını dondur (gradyan hesaplanmaz)"""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:  # FC katmanı hariç hepsini dondur
                param.requires_grad = False

    def _unfreeze_backbone(self):
        """Tüm katmanları aç"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_last_layers(self, num_layers=2):
        """
        Son N ResNet bloğunu aç (kademeli fine-tuning için).

        Args:
            num_layers: Açılacak layer sayısı (1-4 arası)
        """
        # Önce hepsini dondur
        self._freeze_backbone()

        # Son katmanları aç
        layers_to_unfreeze = ['layer4', 'layer3', 'layer2', 'layer1'][:num_layers]

        for name, param in self.backbone.named_parameters():
            for layer_name in layers_to_unfreeze:
                if layer_name in name:
                    param.requires_grad = True

        # FC her zaman açık
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

        print(f"🔓 Son {num_layers} blok açıldı: {layers_to_unfreeze}")

    def forward(self, x):
        return self.backbone(x)

    def get_trainable_params(self):
        """Eğitilebilir parametre sayısını döndür"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self):
        """Toplam parametre sayısını döndür"""
        return sum(p.numel() for p in self.parameters())


# ==================== MODEL FABRİKA FONKSİYONU ====================
def create_model(model_name, num_classes=10, pretrained=True, freeze_backbone=True):
    """
    Model oluşturma fabrika fonksiyonu.

    Args:
        model_name: Model adı ('simple_cnn', 'resnet18_frozen', 'resnet18_finetune')
        num_classes: Çıkış sınıf sayısı
        pretrained: Önceden eğitilmiş ağırlık kullan
        freeze_backbone: Backbone dondur

    Returns:
        nn.Module: Oluşturulan model
    """
    model_name = model_name.lower()

    if model_name == 'simple_cnn':
        model = SimpleCNN(num_classes=num_classes)
        print("\n" + "="*50)
        print("🔧 MODEL: Simple CNN (Baseline)")
        print("="*50)

    elif model_name == 'resnet18_frozen':
        model = ResNet18Model(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=True
        )
        print("\n" + "="*50)
        print("🔧 MODEL: ResNet-18 (Feature Extraction)")
        print("="*50)

    elif model_name == 'resnet18_finetune':
        model = ResNet18Model(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=False
        )
        print("\n" + "="*50)
        print("🔧 MODEL: ResNet-18 (Fine-Tuning)")
        print("="*50)

    else:
        raise ValueError(f"Bilinmeyen model: {model_name}")

    # Model bilgilerini yazdır
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Toplam parametre: {total_params:,}")
    print(f"Eğitilebilir parametre: {trainable_params:,}")
    print(f"Dondurulmuş parametre: {total_params - trainable_params:,}")
    print("="*50 + "\n")

    return model


# ==================== TEST ====================
if __name__ == "__main__":
    print("Model Test\n")

    # Test input
    x = torch.randn(4, 3, 224, 224)

    # Simple CNN
    print("1. Simple CNN Test:")
    model1 = create_model('simple_cnn')
    out1 = model1(x)
    print(f"   Output shape: {out1.shape}\n")

    # ResNet-18 Frozen
    print("2. ResNet-18 (Frozen) Test:")
    model2 = create_model('resnet18_frozen')
    out2 = model2(x)
    print(f"   Output shape: {out2.shape}\n")

    # ResNet-18 Fine-tune
    print("3. ResNet-18 (Fine-tune) Test:")
    model3 = create_model('resnet18_finetune')
    out3 = model3(x)
    print(f"   Output shape: {out3.shape}\n")

