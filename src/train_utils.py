"""
EuroSAT Eğitim ve Değerlendirme Yardımcı Fonksiyonları
======================================================
Bu modül eğitim döngüsü, değerlendirme metrikleri ve görselleştirme fonksiyonlarını içerir.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
    roc_curve,
    auc
)
from tqdm import tqdm
import time
import os


# ==================== EĞİTİM FONKSİYONLARI ====================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Bir epoch eğitim yapar.

    Args:
        model: Eğitilecek model
        dataloader: Eğitim DataLoader
        criterion: Loss fonksiyonu
        optimizer: Optimizer
        device: Cihaz (cuda/cpu)

    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # İstatistikler
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Progress bar güncelle
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    Model validasyonu yapar.

    Args:
        model: Değerlendirilecek model
        dataloader: Validation/Test DataLoader
        criterion: Loss fonksiyonu
        device: Cihaz (cuda/cpu)

    Returns:
        tuple: (val_loss, val_accuracy, all_preds, all_labels)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    val_loss = running_loss / total
    val_acc = correct / total

    return val_loss, val_acc, np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ==================== TAM EĞİTİM DÖNGÜSÜ ====================
def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler=None, num_epochs=10, device='cuda',
                save_path='./models/best_model.pth', early_stopping_patience=5):
    """
    Tam eğitim döngüsü.

    Args:
        model: Eğitilecek model
        train_loader: Eğitim DataLoader
        val_loader: Validation DataLoader
        criterion: Loss fonksiyonu
        optimizer: Optimizer
        scheduler: Learning rate scheduler (opsiyonel)
        num_epochs: Epoch sayısı
        device: Cihaz (cuda/cpu)
        save_path: En iyi model kayıt yolu
        early_stopping_patience: Early stopping için sabır değeri

    Returns:
        dict: Eğitim geçmişi (history)
    """
    # Geçmiş kayıt
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

    best_val_acc = 0.0
    patience_counter = 0
    start_time = time.time()

    print("\n" + "="*60)
    print("🚀 EĞİTİM BAŞLIYOR")
    print("="*60)
    print(f"Cihaz: {device}")
    print(f"Epoch sayısı: {num_epochs}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print("="*60 + "\n")

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Eğitim
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validation
        val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion, device)

        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Scheduler step
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Geçmişe kaydet
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        # Epoch süresi
        epoch_time = time.time() - epoch_start

        # Yazdır
        print(f"Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
        print(f"  LR: {current_lr:.6f}")

        # En iyi model kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # Model kaydet
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, save_path)
            print(f"  ✓ En iyi model kaydedildi! (Val Acc: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            print(f"  ⚠ İyileşme yok ({patience_counter}/{early_stopping_patience})")

        print()

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"⛔ Early stopping! {early_stopping_patience} epoch boyunca iyileşme olmadı.")
            break

    total_time = time.time() - start_time

    print("="*60)
    print("✅ EĞİTİM TAMAMLANDI")
    print(f"Toplam süre: {total_time/60:.2f} dakika")
    print(f"En iyi Validation Accuracy: {best_val_acc*100:.2f}%")
    print("="*60 + "\n")

    return history


# ==================== DEĞERLENDİRME METRİKLERİ ====================
def evaluate_model(model, test_loader, device, class_names):
    """
    Model performansını detaylı değerlendirir.

    Args:
        model: Değerlendirilecek model
        test_loader: Test DataLoader
        device: Cihaz
        class_names: Sınıf isimleri listesi

    Returns:
        dict: Tüm metrikler
    """
    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, preds, labels, probs = validate(
        model, test_loader, criterion, device
    )

    # Classification Report
    report = classification_report(labels, preds, target_names=class_names, output_dict=True)
    report_text = classification_report(labels, preds, target_names=class_names)

    # Precision, Recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None
    )

    # Macro & Weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro'
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )

    metrics = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'predictions': preds,
        'true_labels': labels,
        'probabilities': probs,
        'classification_report': report,
        'classification_report_text': report_text,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'support_per_class': support,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
    }

    # Sonuçları yazdır
    print("\n" + "="*60)
    print("📊 TEST SONUÇLARI")
    print("="*60)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"\nF1-Score (Macro): {f1_macro:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    print(f"Precision (Macro): {precision_macro:.4f}")
    print(f"Recall (Macro): {recall_macro:.4f}")
    print("\n" + "-"*60)
    print("SINIF BAZLI RAPOR:")
    print("-"*60)
    print(report_text)
    print("="*60 + "\n")

    return metrics


# ==================== GÖRSELLEŞTİRME FONKSİYONLARI ====================
def plot_training_history(history, save_path=None):
    """
    Eğitim geçmişini görselleştirir.

    Args:
        history: Eğitim geçmişi dict
        save_path: Kayıt yolu (opsiyonel)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss grafiği
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy grafiği
    axes[1].plot(epochs, [a*100 for a in history['train_acc']], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, [a*100 for a in history['val_acc']], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning Rate grafiği
    axes[2].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Eğitim grafikleri kaydedildi: {save_path}")

    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, normalize=True):
    """
    Confusion matrix ısı haritası çizer.

    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        class_names: Sınıf isimleri
        save_path: Kayıt yolu (opsiyonel)
        normalize: Normalize et (yüzde olarak)
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        linewidths=0.5
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix kaydedildi: {save_path}")

    plt.show()


def plot_class_performance(metrics, class_names, save_path=None):
    """
    Sınıf bazlı performans grafiği çizer.

    Args:
        metrics: evaluate_model'den dönen metrikler
        class_names: Sınıf isimleri
        save_path: Kayıt yolu (opsiyonel)
    """
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - width, metrics['precision_per_class'], width, label='Precision', color='#2ecc71')
    bars2 = ax.bar(x, metrics['recall_per_class'], width, label='Recall', color='#3498db')
    bars3 = ax.bar(x + width, metrics['f1_per_class'], width, label='F1-Score', color='#e74c3c')

    ax.set_xlabel('Sınıflar', fontsize=12)
    ax.set_ylabel('Skor', fontsize=12)
    ax.set_title('Sınıf Bazlı Performans Metrikleri', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', alpha=0.3)

    # Değerleri bar üzerine yaz
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Sınıf performans grafiği kaydedildi: {save_path}")

    plt.show()


def plot_sample_predictions(model, dataloader, class_names, device, num_samples=16, save_path=None):
    """
    Örnek tahminleri görselleştirir.

    Args:
        model: Model
        dataloader: DataLoader
        class_names: Sınıf isimleri
        device: Cihaz
        num_samples: Gösterilecek örnek sayısı
        save_path: Kayıt yolu (opsiyonel)
    """
    model.eval()

    # Bir batch al
    images, labels = next(iter(dataloader))
    images, labels = images[:num_samples].to(device), labels[:num_samples]

    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        confidences = probs.max(dim=1).values

    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images_denorm = images.cpu() * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)

    # Plot
    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    axes = axes.flatten()

    for i in range(num_samples):
        ax = axes[i]
        img = images_denorm[i].permute(1, 2, 0).numpy()
        ax.imshow(img)

        true_label = class_names[labels[i]]
        pred_label = class_names[preds[i].cpu()]
        conf = confidences[i].cpu().item() * 100

        color = 'green' if preds[i].cpu() == labels[i] else 'red'
        ax.set_title(f'True: {true_label}\nPred: {pred_label} ({conf:.1f}%)',
                    color=color, fontsize=10)
        ax.axis('off')

    # Boş subplot'ları kapat
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Örnek tahminler kaydedildi: {save_path}")

    plt.show()


# ==================== MODEL KARŞILAŞTIRMA ====================
def compare_models(results_dict, save_path=None):
    """
    Birden fazla modelin sonuçlarını karşılaştırır.

    Args:
        results_dict: {model_name: metrics} formatında sonuçlar
        save_path: Kayıt yolu (opsiyonel)
    """
    models = list(results_dict.keys())
    metrics_to_compare = ['test_accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

    # Tablo verisi hazırla
    data = []
    for model_name in models:
        row = [model_name]
        for metric in metrics_to_compare:
            value = results_dict[model_name].get(metric, 0)
            row.append(f"{value*100:.2f}%" if metric == 'test_accuracy' else f"{value:.4f}")
        data.append(row)

    # DataFrame benzeri yazdırma
    print("\n" + "="*80)
    print("📊 MODEL KARŞILAŞTIRMA TABLOSU")
    print("="*80)
    headers = ['Model', 'Test Accuracy', 'F1 (Macro)', 'Precision', 'Recall']
    print(f"{headers[0]:<25} {headers[1]:<15} {headers[2]:<12} {headers[3]:<12} {headers[4]:<12}")
    print("-"*80)
    for row in data:
        print(f"{row[0]:<25} {row[1]:<15} {row[2]:<12} {row[3]:<12} {row[4]:<12}")
    print("="*80 + "\n")

    # Bar chart
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    metric_names = ['Test Accuracy', 'F1-Score (Macro)', 'Precision (Macro)', 'Recall (Macro)']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

    for i, (metric, metric_name) in enumerate(zip(metrics_to_compare, metric_names)):
        values = [results_dict[m][metric] * 100 for m in models]
        bars = axes[i].bar(models, values, color=colors[i], alpha=0.8)
        axes[i].set_title(metric_name)
        axes[i].set_ylabel('Değer (%)')
        axes[i].set_ylim(0, 105)
        axes[i].tick_params(axis='x', rotation=15)

        # Değerleri bar üzerine yaz
        for bar, val in zip(bars, values):
            axes[i].annotate(f'{val:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Model karşılaştırma grafiği kaydedildi: {save_path}")

    plt.show()


# ==================== MODEL KAYDETME/YÜKLEME ====================
def save_checkpoint(model, optimizer, epoch, val_acc, path):
    """Model checkpoint kaydeder"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }, path)
    print(f"✓ Checkpoint kaydedildi: {path}")


def load_checkpoint(model, path, optimizer=None):
    """Model checkpoint yükler"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"✓ Checkpoint yüklendi: {path}")
    print(f"  Epoch: {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']*100:.2f}%")

    return checkpoint


# ==================== TEST ====================
if __name__ == "__main__":
    print("Train Utils Test")
    print("-" * 40)

    # Dummy test
    history = {
        'train_loss': [0.8, 0.5, 0.3, 0.2, 0.15],
        'train_acc': [0.6, 0.75, 0.85, 0.9, 0.93],
        'val_loss': [0.7, 0.45, 0.35, 0.28, 0.25],
        'val_acc': [0.65, 0.78, 0.82, 0.88, 0.90],
        'lr': [0.001, 0.001, 0.0005, 0.0005, 0.00025]
    }

    print("Plotting training history...")
    plot_training_history(history)

