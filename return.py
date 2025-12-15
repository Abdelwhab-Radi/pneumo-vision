@dataclass
class TrainingConfig:
    """Centralized configuration management for local training"""
    # Data paths - UPDATE THIS TO YOUR LOCAL PATH
    data_root: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chest_xray", "chest_xray")
    cache_dir: str = "./cache"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    results_dir: str = "./results"

    # Data settings
    img_size: int = 224
    batch_size: int = 32
    val_ratio: float = 0.2
    color_mode: str = "rgb"
    allowed_exts: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    # Model settings
    model_variant: str = "B0"  # B0, B3, B4, B7
    dropout_rate: float = 0.3
    use_mixed_precision: bool = True
    fine_tune_layers: int = 40

    # Training settings
    initial_epochs: int = 20
    fine_tune_epochs: int = 15
    initial_lr: float = 1e-3
    fine_tune_lr: float = 1e-5
    min_lr: float = 1e-7

    # Advanced settings
    use_tta: bool = True
    tta_augmentations: int = 5
    use_cross_validation: bool = False
    cv_folds: int = 5
    use_class_weights: bool = True
    label_smoothing: float = 0.1

    # Augmentation settings
    augmentation_strength: float = 1.0
    use_advanced_aug: bool = True
    use_albumentations: bool = ALBUMENTATIONS_AVAILABLE

    # System
    seed: int = 42
    mixed_precision_policy: str = "mixed_float16"
    force_resplit: bool = False
    split_move_files: bool = True

    # Local execution settings
    save_model_plots: bool = True
    save_predictions: bool = True
    verbose: int = 1
