"""
EuroSAT Proje Modülleri
=======================

Bu paket aşağıdaki modülleri içerir:
- data_loader: Veri yükleme ve ön işleme
- model: CNN ve ResNet model tanımları
- train_utils: Eğitim ve değerlendirme fonksiyonları
"""

from .data_loader import (
    download_eurosat,
    create_dataloaders,
    get_train_transforms,
    get_test_transforms,
    CLASS_NAMES,
    NUM_CLASSES,
    denormalize
)

from .model import (
    SimpleCNN,
    ResNet18Model,
    create_model
)

from .train_utils import (
    train_model,
    train_one_epoch,
    validate,
    evaluate_model,
    plot_training_history,
    plot_confusion_matrix,
    plot_class_performance,
    plot_sample_predictions,
    compare_models,
    save_checkpoint,
    load_checkpoint
)

__version__ = "1.0.0"
__author__ = "EuroSAT Project Team"

