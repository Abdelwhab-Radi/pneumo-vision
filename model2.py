from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import shutil
import datetime
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.applications import efficientnet
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not installed. Install with 'pip install albumentations' for advanced augmentations")

# ------------------------
# Configuration Management
# ------------------------
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
    
    # Deployment settings
    auto_optimize_for_deployment: bool = True
    save_deployment_package: bool = True
    deployment_formats: tuple = ("keras", "tflite")
    tflite_quantize: bool = True

    def to_dict(self):
        return asdict(self)

    def save(self, path: str):
        """Save configuration to JSON file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            return cls(**json.load(f))

    def print_config(self):
        """Print configuration in a readable format"""
        print("\n" + "="*50)
        print("TRAINING CONFIGURATION")
        print("="*50)
        for key, value in self.to_dict().items():
            print(f"{key:25s}: {value}")
        print("="*50 + "\n")

# ------------------------
# System Setup
# ------------------------
class SystemSetup:
    """Handle system configuration and GPU setup"""

    @staticmethod
    def set_seeds(seed: int):
        """Set random seeds for reproducibility"""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        print(f"✓ Random seeds set to {seed}")

    @staticmethod
    def configure_gpu():
        """Configure GPU memory growth and settings"""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✓ Configured {len(gpus)} GPU(s) for training")
                # Print GPU details
                for i, gpu in enumerate(gpus):
                    print(f"  GPU {i}: {gpu.name}")
            except RuntimeError as e:
                print(f"✗ GPU configuration failed: {e}")
        else:
            print("⚠ No GPUs found. Training will use CPU (slower)")

    @staticmethod
    def enable_mixed_precision(policy: str = "mixed_float16"):
        """Enable mixed precision training for faster computation"""
        try:
            policy_obj = mixed_precision.Policy(policy)
            mixed_precision.set_global_policy(policy_obj)
            print(f"✓ Mixed precision enabled: {policy}")
            print(f"  Compute dtype: {policy_obj.compute_dtype}")
            print(f"  Variable dtype: {policy_obj.variable_dtype}")
        except Exception as e:
            print(f"✗ Mixed precision setup failed: {e}")
            print("  Continuing with default precision")

    @staticmethod
    def check_tensorflow_version():
        """Check TensorFlow version and capabilities"""
        print(f"✓ TensorFlow version: {tf.__version__}")
        print(f"  Built with CUDA: {tf.test.is_built_with_cuda()}")
        physical_gpus = tf.config.list_physical_devices('GPU')
        print(f"  GPU available: {len(physical_gpus) > 0}")
        if not physical_gpus:
             print("\n" + "!"*50)
             print("WARNING: No GPU detected. Running on CPU.")
             print("To use your RTX 1070, ensure you have:")
             print("1. NVIDIA Drivers installed")
             print("2. CUDA Toolkit and cuDNN compatible with your TensorFlow version")
             print("!"*50 + "\n")

# ------------------------
# Local File Manager
# ------------------------
class LocalFileManager:
    """Manage local files and directories"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories for local storage"""
        dirs = [
            self.config.cache_dir,
            self.config.checkpoint_dir,
            self.config.log_dir,
            self.config.results_dir,
            Path(self.config.results_dir) / "plots",
            Path(self.config.results_dir) / "models",
            Path(self.config.results_dir) / "predictions"
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created local directories for results")

    def save_plot(self, fig, name: str):
        """Save matplotlib figure locally"""
        if self.config.save_model_plots:
            path = Path(self.config.results_dir) / "plots" / f"{name}.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  Saved plot: {path}")

    def save_model(self, model, name: str):
        """Save model locally"""
        path = Path(self.config.results_dir) / "models" / f"{name}.keras"
        model.save(path)
        print(f"✓ Saved model: {path}")
        return path

# ------------------------
# Data Management
# ------------------------
class DataManager:
    """Handle data loading, splitting, and preprocessing"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.file_manager = LocalFileManager(config)

    @staticmethod
    def is_image_file(path: Path, allowed_exts: tuple) -> bool:
        """Check if file is a valid image"""
        return path.is_file() and path.suffix.lower() in allowed_exts

    @staticmethod
    def count_images_in_tree(root: Path, allowed_exts: tuple) -> int:
        """Count all images in directory tree"""
        if not root.exists():
            return 0
        return sum(1 for p in root.rglob("*") if DataManager.is_image_file(p, allowed_exts))

    def get_dataset_info(self, directory: Path) -> Dict:
        """Get detailed information about dataset"""
        info = {
            'path': str(directory),
            'exists': directory.exists(),
            'total_images': 0,
            'classes': {}
        }

        if directory.exists():
            for class_dir in directory.iterdir():
                if class_dir.is_dir():
                    count = len([p for p in class_dir.iterdir()
                               if self.is_image_file(p, self.config.allowed_exts)])
                    info['classes'][class_dir.name] = count
                    info['total_images'] += count

        return info

    def print_dataset_summary(self):
        """Print summary of all datasets"""
        print("\n" + "="*50)
        print("DATASET SUMMARY")
        print("="*50)

        data_root = Path(self.config.data_root)
        for split in ['train', 'val', 'test']:
            dir_path = data_root / split
            info = self.get_dataset_info(dir_path)

            print(f"\n{split.upper()} Dataset:")
            if info['exists']:
                print(f"  Path: {info['path']}")
                print(f"  Total images: {info['total_images']}")
                for class_name, count in info['classes'].items():
                    print(f"    {class_name}: {count} images")
            else:
                print(f"  Not found at: {info['path']}")

    def validate_dataset(self, ds: tf.data.Dataset, expected_classes: int = 2) -> bool:
        """Validate dataset integrity"""
        try:
            # Check if dataset is not empty
            sample = next(iter(ds))
            if sample is None:
                raise ValueError("Dataset is empty")

            # Check image shape
            images, labels = sample
            if len(images.shape) != 4:  # (batch, height, width, channels)
                raise ValueError(f"Invalid image shape: {images.shape}")

            # Check label distribution
            unique_labels = set()
            for _, labels_batch in ds.take(10):
                unique_labels.update(labels_batch.numpy().flatten().tolist())

            if len(unique_labels) > expected_classes:
                raise ValueError(f"Expected max {expected_classes} classes, found {len(unique_labels)}")

            print(f"✓ Dataset validation passed - Found {len(unique_labels)} classes")
            return True

        except Exception as e:
            print(f"✗ Dataset validation failed: {str(e)}")
            return False

    def stratified_split_train_to_val(self, train_dir: Path, val_dir: Path):
        """Create stratified validation split from training data"""
        if not train_dir.exists():
            raise FileNotFoundError(f"Train directory not found: {train_dir}")

        class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
        if not class_dirs:
            raise RuntimeError(f"No class subfolders found in: {train_dir}")

        total_moved = 0
        split_info = {}

        for class_dir in class_dirs:
            class_name = class_dir.name
            files = [p for p in class_dir.iterdir() if self.is_image_file(p, self.config.allowed_exts)]
            n_total = len(files)

            if n_total == 0:
                print(f"[WARN] No images in {class_dir}")
                continue

            n_val = max(1, int(round(n_total * self.config.val_ratio)))
            rng = np.random.default_rng(self.config.seed)
            val_files = rng.choice(files, size=n_val, replace=False)

            # Create class subfolder in val/
            dest_class_dir = val_dir / class_name
            dest_class_dir.mkdir(parents=True, exist_ok=True)

            # Move or copy files
            for src_file in val_files:
                dst_file = dest_class_dir / src_file.name
                if dst_file.exists():
                    continue

                if self.config.split_move_files:
                    shutil.move(str(src_file), str(dst_file))
                else:
                    shutil.copy2(str(src_file), str(dst_file))
                total_moved += 1

            split_info[class_name] = {
                'total': n_total,
                'train': n_total - n_val,
                'val': n_val
            }

            print(f"[Split] {class_name}: Total={n_total}, Train={n_total-n_val}, Val={n_val}")

        print(f"[Split] Completed. Files {'moved' if self.config.split_move_files else 'copied'}: {total_moved}")

        # Save split info locally
        split_info_path = Path(self.config.results_dir) / "split_info.json"
        with open(split_info_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        print(f"✓ Split info saved to: {split_info_path}")

        return split_info

    def create_optimized_dataset(self, directory: Path, shuffle: bool,
                                batch_size: Optional[int] = None) -> Tuple[tf.data.Dataset, List[str]]:
        """Create optimized data pipeline with caching and prefetching"""
        if batch_size is None:
            batch_size = self.config.batch_size

        # Create base dataset
        ds = keras.utils.image_dataset_from_directory(
            directory,
            labels="inferred",
            label_mode="int",
            color_mode=self.config.color_mode,
            image_size=(self.config.img_size, self.config.img_size),
            batch_size=batch_size,
            shuffle=shuffle,
            seed=self.config.seed,
        )

        # Get class names before optimization
        class_names = ds.class_names

        # Optimize pipeline
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        options.experimental_optimization.map_parallelization = True
        options.threading.private_threadpool_size = 8
        ds = ds.with_options(options)

        # Cache to disk for large datasets
        cache_path = Path(self.config.cache_dir) / f"{directory.name}_cache"
        ds = ds.cache(str(cache_path))

        if shuffle:
            ds = ds.shuffle(buffer_size=1000, seed=self.config.seed)

        # Prefetch with dynamic buffer size
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds, class_names

    def compute_class_weights(self, ds: tf.data.Dataset, num_classes: int) -> Tuple[Dict, np.ndarray]:
        """Compute class weights for imbalanced dataset"""
        counts = np.zeros(num_classes, dtype=np.int64)
        for _, y in ds.unbatch():
            counts[int(y.numpy())] += 1

        total = counts.sum()
        class_weights = {
            i: float(total / (num_classes * count))
            for i, count in enumerate(counts) if count > 0
        }

        return class_weights, counts

# ------------------------
# Augmentation Pipeline
# ------------------------
class AugmentationPipeline:
    """Advanced augmentation pipeline for medical images"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.strength = config.augmentation_strength

    def get_basic_augmentation(self) -> keras.Sequential:
        """Basic Keras augmentations"""
        return keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1 * self.strength),
            layers.RandomZoom(0.15 * self.strength),
            layers.RandomTranslation(0.1 * self.strength, 0.1 * self.strength),
            layers.RandomContrast(0.2 * self.strength),
            layers.RandomBrightness(0.2 * self.strength),
        ], name="basic_augmentation")

    def get_advanced_augmentation(self) -> keras.Sequential:
        """Advanced medical image-specific augmentations"""
        return keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15 * self.strength),
            layers.RandomZoom(0.2 * self.strength),
            layers.RandomTranslation(0.1 * self.strength, 0.1 * self.strength),
            layers.RandomContrast(0.25 * self.strength),
            layers.RandomBrightness(0.25 * self.strength),
            layers.GaussianNoise(0.02 * self.strength),
        ], name="advanced_augmentation")

    @staticmethod
    def get_albumentations_pipeline():
        """Albumentations pipeline for X-ray specific augmentations"""
        if not ALBUMENTATIONS_AVAILABLE:
            return None

        return A.Compose([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.GridDistortion(p=0.2),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=0.1, p=0.3),
        ])

    def apply_tta_augmentation(self, image: tf.Tensor) -> tf.Tensor:
        """Apply Test Time Augmentation"""
        augmentations = [
            lambda x: x,  # Original
            lambda x: tf.image.flip_left_right(x),
            lambda x: tf.image.adjust_brightness(x, 0.1),
            lambda x: tf.image.adjust_contrast(x, 1.1),
            lambda x: tf.image.rot90(x, k=1),
        ]

        idx = np.random.randint(0, len(augmentations))
        return augmentations[idx](image)

# ------------------------
# Model Architecture
# ------------------------
class ModelBuilder:
    """Build and compile various model architectures"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.augmentation = AugmentationPipeline(config)

    @staticmethod
    def get_effnet_class_and_preprocess(variant: str):
        """Get EfficientNet class and preprocessing function"""
        variant_map = {
            "B0": (efficientnet.EfficientNetB0, efficientnet.preprocess_input),
            "B3": (efficientnet.EfficientNetB3, efficientnet.preprocess_input),
            "B4": (efficientnet.EfficientNetB4, efficientnet.preprocess_input),
            "B7": (efficientnet.EfficientNetB7, efficientnet.preprocess_input),
        }

        variant = variant.upper()
        if variant not in variant_map:
            raise ValueError(f"Unsupported EfficientNet variant: {variant}")

        return variant_map[variant]

    def build_enhanced_model(self, input_shape: Tuple[int, int, int],
                           num_classes: int = 1) -> Tuple[keras.Model, keras.Model]:
        """Build enhanced model with improved architecture"""

        # Get EfficientNet components
        EffNetClass, preprocess_fn = self.get_effnet_class_and_preprocess(self.config.model_variant)

        # Input layer
        inputs = keras.Input(shape=input_shape, name="input_image")

        # Augmentation block (only applied during training)
        if self.config.use_advanced_aug:
            augmented = self.augmentation.get_advanced_augmentation()(inputs)
        else:
            augmented = self.augmentation.get_basic_augmentation()(inputs)

        # Preprocessing
        preprocessed = preprocess_fn(augmented)

        # Base model (transfer learning)
        base_model = EffNetClass(
            include_top=False,
            weights="imagenet",
            input_tensor=preprocessed,
            pooling=None
        )
        base_model.trainable = False  # Initially frozen

        # Feature extraction
        features = base_model(preprocessed, training=False)

        # Multi-scale pooling
        gap = layers.GlobalAveragePooling2D(name="gap")(features)
        gmp = layers.GlobalMaxPooling2D(name="gmp")(features)

        # Concatenate pooled features
        concat = layers.Concatenate(name="concat_pool")([gap, gmp])

        # Dense layers with batch normalization
        x = layers.Dense(512, use_bias=False, name="dense_1")(concat)
        x = layers.BatchNormalization(name="bn_1")(x)
        x = layers.Activation("relu", name="relu_1")(x)
        x = layers.Dropout(self.config.dropout_rate, name="dropout_1")(x)

        x = layers.Dense(256, use_bias=False, name="dense_2")(x)
        x = layers.BatchNormalization(name="bn_2")(x)
        x = layers.Activation("relu", name="relu_2")(x)
        x = layers.Dropout(self.config.dropout_rate * 0.7, name="dropout_2")(x)

        x = layers.Dense(128, use_bias=False, name="dense_3")(x)
        x = layers.BatchNormalization(name="bn_3")(x)
        x = layers.Activation("relu", name="relu_3")(x)
        x = layers.Dropout(self.config.dropout_rate * 0.5, name="dropout_3")(x)

        # Output layer
        if num_classes == 1:
            outputs = layers.Dense(1, activation="sigmoid", dtype="float32", name="output")(x)
        else:
            outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="output")(x)

        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name=f"pneumonia_enhanced_{self.config.model_variant}")

        return model, base_model

    def compile_model(self, model: keras.Model, learning_rate: float,
                     num_classes: int = 1) -> keras.Model:
        """Compile model with appropriate loss and metrics"""

        # Optimizer
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        # Loss function
        if num_classes == 1:
            loss = keras.losses.BinaryCrossentropy(label_smoothing=self.config.label_smoothing)

            metrics = [
                keras.metrics.BinaryAccuracy(name="accuracy"),
                keras.metrics.AUC(name="auc"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
            ]
        else:
            loss = keras.losses.SparseCategoricalCrossentropy(label_smoothing=self.config.label_smoothing)
            metrics = [
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.AUC(name="auc", multi_label=True),
            ]

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

        return model

    @staticmethod
    def unfreeze_layers(base_model: keras.Model, num_layers: int):
        """Unfreeze top layers for fine-tuning"""
        # Freeze all layers first
        for layer in base_model.layers:
            layer.trainable = False

        # Unfreeze top layers
        for layer in base_model.layers[-num_layers:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

# ------------------------
# Custom Callbacks
# ------------------------
class MedicalMetricsCallback(keras.callbacks.Callback):
    """Calculate medical-specific metrics during training"""

    def __init__(self, validation_data: tf.data.Dataset, config: TrainingConfig):
        super().__init__()
        self.validation_data = validation_data
        self.config = config
        self.metrics_history = []

    def on_epoch_end(self, epoch, logs=None):
        """Calculate sensitivity and specificity at epoch end"""
        y_true = []
        y_pred = []

        for x_batch, y_batch in self.validation_data:
            predictions = self.model.predict(x_batch, verbose=0)
            y_pred.extend(predictions.ravel())
            y_true.extend(y_batch.numpy().flatten())

        y_pred_binary = (np.array(y_pred) >= 0.5).astype(int)
        y_true = np.array(y_true)

        # Calculate confusion matrix components
        tn = np.sum((y_true == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true == 1) & (y_pred_binary == 0))
        tp = np.sum((y_true == 1) & (y_pred_binary == 1))

        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

        # Store metrics
        metrics = {
            'epoch': epoch + 1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv
        }
        self.metrics_history.append(metrics)

        print(f"\n[Medical Metrics] Epoch {epoch + 1}:")
        print(f"  Sensitivity (Recall): {sensitivity:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  PPV (Precision): {ppv:.4f}")
        print(f"  NPV: {npv:.4f}")

        # Add to logs
        if logs is not None:
            logs['sensitivity'] = sensitivity
            logs['specificity'] = specificity
            logs['ppv'] = ppv
            logs['npv'] = npv

    def save_metrics(self, path: str):
        """Save metrics history to JSON"""
        with open(path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

class CosineAnnealingScheduler(keras.callbacks.Callback):
    """Cosine annealing learning rate scheduler"""

    def __init__(self, initial_lr: float, min_lr: float, epochs: int):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.epochs = epochs

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.epochs:
            lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (
                1 + np.cos(np.pi * epoch / self.epochs)
            )
        else:
            lr = self.min_lr

        keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        if epoch % 5 == 0:  # Print every 5 epochs to reduce verbosity
            print(f"\nEpoch {epoch + 1}: Learning rate = {lr:.2e}")

def get_callbacks(config: TrainingConfig, stage: str, validation_data: tf.data.Dataset) -> List:
    """Get comprehensive callbacks for training"""

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = []

    # TensorBoard
    log_dir = Path(config.log_dir) / f"{stage}_{timestamp}"
    callbacks.append(keras.callbacks.TensorBoard(
        log_dir=str(log_dir),
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch',
        profile_batch=(10, 20)
    ))

    # Model checkpoint
    checkpoint_path = Path(config.checkpoint_dir) / f"best_model_{stage}_{timestamp}.keras"
    callbacks.append(keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    ))

    # Early stopping
    callbacks.append(keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=7,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001
    ))

    # Learning rate scheduler
    if stage == "initial":
        callbacks.append(CosineAnnealingScheduler(
            initial_lr=config.initial_lr,
            min_lr=config.min_lr,
            epochs=config.initial_epochs
        ))
    else:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=config.min_lr,
            verbose=1
        ))

    # Medical metrics
    medical_callback = MedicalMetricsCallback(validation_data, config)
    callbacks.append(medical_callback)

    # CSV logger
    csv_path = Path(config.log_dir) / f"training_log_{stage}_{timestamp}.csv"
    callbacks.append(keras.callbacks.CSVLogger(str(csv_path)))

    return callbacks

# ------------------------
# Model Evaluation & Visualization
# ------------------------
class ModelEvaluator:
    """Comprehensive model evaluation and visualization"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.file_manager = LocalFileManager(config)

    def plot_training_history(self, history, title_suffix=""):
        """Plot training history and save locally"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        metrics = ['loss', 'accuracy', 'auc']
        titles = ['Loss', 'Accuracy', 'AUC']

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[0, idx]
            if metric in history.history:
                epochs = range(1, len(history.history[metric]) + 1)
                ax.plot(epochs, history.history[metric], 'b-', label='Train')
                if f'val_{metric}' in history.history:
                    ax.plot(epochs, history.history[f'val_{metric}'], 'r-', label='Val')
                ax.set_title(f'{title} {title_suffix}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(title)
                ax.legend()
                ax.grid(True, alpha=0.3)

        # Plot medical metrics if available
        medical_metrics = ['sensitivity', 'specificity', 'ppv']
        medical_titles = ['Sensitivity', 'Specificity', 'PPV']

        for idx, (metric, title) in enumerate(zip(medical_metrics, medical_titles)):
            ax = axes[1, idx]
            if metric in history.history:
                epochs = range(1, len(history.history[metric]) + 1)
                ax.plot(epochs, history.history[metric], 'g-', label=title)
                ax.set_title(f'{title} {title_suffix}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(title)
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot locally
        self.file_manager.save_plot(fig, f"training_history_{title_suffix.replace(' ', '_')}")
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, class_names, title="Confusion Matrix"):
        """Plot confusion matrix and save locally"""
        cm = confusion_matrix(y_true, y_pred)

        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()

        # Save plot locally
        self.file_manager.save_plot(fig, f"confusion_matrix_{title.replace(' ', '_')}")
        plt.show()

        return cm

    def plot_roc_curve(self, y_true, y_prob, title="ROC Curve"):
        """Plot ROC curve and save locally"""
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        fig = plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot locally
        self.file_manager.save_plot(fig, f"roc_curve_{title.replace(' ', '_')}")
        plt.show()

        return auc

    def comprehensive_evaluation(self, model: keras.Model, test_ds: tf.data.Dataset,
                               class_names: List[str]) -> Dict:
        """Perform comprehensive model evaluation"""
        print("\n" + "="*50)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*50)

        # Basic evaluation
        results = model.evaluate(test_ds, verbose=1, return_dict=True)

        # Get predictions
        y_true = []
        y_prob = []

        for x_batch, y_batch in test_ds:
            predictions = model.predict(x_batch, verbose=0)
            y_prob.extend(predictions.ravel())
            y_true.extend(y_batch.numpy().flatten())

        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        y_pred = (y_prob >= 0.5).astype(int)

        # Classification report
        print("\nClassification Report:")
        print("="*50)
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        print(classification_report(y_true, y_pred, target_names=class_names))

        # Save classification report
        report_path = Path(self.config.results_dir) / "classification_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Plot confusion matrix
        cm = self.plot_confusion_matrix(y_true, y_pred, class_names)

        # Plot ROC curve
        auc = self.plot_roc_curve(y_true, y_prob)

        # Calculate additional metrics
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        evaluation_results = {
            **results,
            'confusion_matrix': cm.tolist(),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'roc_auc': auc,
            'classification_report': report
        }

        print("\nDetailed Metrics:")
        print("="*50)
        for metric, value in evaluation_results.items():
            if not isinstance(value, (list, dict)):
                print(f"{metric:20s}: {value:.4f}")

        # Save evaluation results locally
        results_path = Path(self.config.results_dir) / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        print(f"\n✓ Evaluation results saved to: {results_path}")

        return evaluation_results

    def predict_with_tta(self, model: keras.Model, test_ds: tf.data.Dataset,
                        num_augmentations: int = 5) -> np.ndarray:
        """Test Time Augmentation for improved predictions"""
        print(f"\nApplying Test Time Augmentation ({num_augmentations} augmentations)...")

        predictions = []
        aug_pipeline = AugmentationPipeline(self.config)

        for i in range(num_augmentations):
            print(f"  Augmentation {i+1}/{num_augmentations}")

            if i == 0:
                # Original predictions
                y_pred = model.predict(test_ds, verbose=0)
            else:
                # Apply augmentation
                augmented_ds = test_ds.map(
                    lambda x, y: (aug_pipeline.apply_tta_augmentation(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE
                )
                y_pred = model.predict(augmented_ds, verbose=0)

            predictions.append(y_pred)

        # Average predictions
        final_predictions = np.mean(predictions, axis=0)
        print("✓ TTA completed")

        # Save predictions if configured
        if self.config.save_predictions:
            pred_path = Path(self.config.results_dir) / "predictions" / "tta_predictions.npy"
            np.save(pred_path, final_predictions)
            print(f"✓ TTA predictions saved to: {pred_path}")

        return final_predictions

# ------------------------
# Model Optimization
# ------------------------
class ModelOptimizer:
    """Optimize model for deployment"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.file_manager = LocalFileManager(config)

    def convert_to_tflite(self, model: keras.Model, save_name: str,
                          quantize: bool = True, representative_data=None):
        """Convert model to TensorFlow Lite format"""
        print("\nConverting model to TensorFlow Lite...")

        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        if quantize:
            # Enable optimizations
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            if representative_data is not None:
                # Full integer quantization
                def representative_dataset():
                    for data in representative_data.take(100):
                        yield [tf.cast(data[0], tf.float32)]

                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                ]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8

        # Convert model
        tflite_model = converter.convert()

        # Save model
        tflite_path = Path(self.config.results_dir) / "models" / f"{save_name}.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        # Calculate model sizes
        keras_path = Path(self.config.results_dir) / "models" / f"{save_name}.keras"
        if keras_path.exists():
            original_size = os.path.getsize(keras_path) / (1024 * 1024)
            optimized_size = os.path.getsize(tflite_path) / (1024 * 1024)

            print(f"✓ Model converted to TFLite")
            print(f"  Original size: {original_size:.2f} MB")
            print(f"  Optimized size: {optimized_size:.2f} MB")
            print(f"  Compression ratio: {original_size/optimized_size:.2f}x")

        return str(tflite_path)

# ------------------------
# Main Training Pipeline
# ------------------------
class TrainingPipeline:
    """Main training pipeline orchestrator"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_environment()
        self.initialize_components()

    def setup_environment(self):
        """Setup system environment"""
        print("\n" + "="*50)
        print("SYSTEM SETUP")
        print("="*50)

        SystemSetup.check_tensorflow_version()
        SystemSetup.set_seeds(self.config.seed)
        SystemSetup.configure_gpu()

        if self.config.use_mixed_precision:
            SystemSetup.enable_mixed_precision(self.config.mixed_precision_policy)

    def initialize_components(self):
        """Initialize pipeline components"""
        self.data_manager = DataManager(self.config)
        self.model_builder = ModelBuilder(self.config)
        self.evaluator = ModelEvaluator(self.config)
        self.optimizer = ModelOptimizer(self.config)
        self.file_manager = LocalFileManager(self.config)

    def prepare_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
        """Prepare datasets"""
        print("\n" + "="*50)
        print("DATA PREPARATION")
        print("="*50)

        # Print dataset summary
        self.data_manager.print_dataset_summary()

        # Define paths
        data_root = Path(self.config.data_root)
        train_dir = data_root / "train"
        val_dir = data_root / "val"
        test_dir = data_root / "test"

        # Create validation split if needed
        if not val_dir.exists() or self.config.force_resplit or \
           self.data_manager.count_images_in_tree(val_dir, self.config.allowed_exts) == 0:
            print("\n[Info] Creating validation split from training data...")
            val_dir.mkdir(parents=True, exist_ok=True)
            split_info = self.data_manager.stratified_split_train_to_val(train_dir, val_dir)
        else:
            print("\n[Info] Using existing validation directory")

        # Check test directory
        use_val_as_test = False
        if not test_dir.exists() or \
           self.data_manager.count_images_in_tree(test_dir, self.config.allowed_exts) == 0:
            print("[Warning] Test directory not found or empty. Using validation set for testing.")
            use_val_as_test = True

        # Create datasets
        print("\nCreating optimized datasets...")
        train_ds, train_class_names = self.data_manager.create_optimized_dataset(train_dir, shuffle=True)
        val_ds, _ = self.data_manager.create_optimized_dataset(val_dir, shuffle=False)

        if use_val_as_test:
            test_ds = val_ds
        else:
            test_ds, _ = self.data_manager.create_optimized_dataset(test_dir, shuffle=False)

        # Use class names from training dataset
        class_names = train_class_names
        print(f"\nClass mapping: {dict(enumerate(class_names))}")

        # Validate datasets
        print("\nValidating datasets...")
        self.data_manager.validate_dataset(train_ds, len(class_names))
        self.data_manager.validate_dataset(val_ds, len(class_names))

        # Compute class weights
        if self.config.use_class_weights:
            class_weights, counts = self.data_manager.compute_class_weights(train_ds, len(class_names))
            print(f"\nClass distribution: {dict(zip(class_names, counts))}")
            print(f"Class weights: {class_weights}")
            self.class_weights = class_weights
        else:
            self.class_weights = None

        return train_ds, val_ds, test_ds, class_names

    def train_model(self, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset) -> Tuple[keras.Model, Dict]:
        """Train model with two-stage approach"""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)

        # Build model
        print("\nBuilding enhanced model...")
        input_shape = (self.config.img_size, self.config.img_size, 3)
        model, base_model = self.model_builder.build_enhanced_model(input_shape)

        # Stage 1: Train with frozen base
        print("\n" + "-"*50)
        print("STAGE 1: Training with frozen base")
        print("-"*50)

        model = self.model_builder.compile_model(model, self.config.initial_lr)

        if self.config.verbose > 0:
            model.summary()

        # Get callbacks
        callbacks_stage1 = get_callbacks(self.config, "stage1_frozen", val_ds)

        # Train
        history_stage1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.initial_epochs,
            class_weight=self.class_weights,
            callbacks=callbacks_stage1,
            verbose=self.config.verbose
        )

        # Plot history
        self.evaluator.plot_training_history(history_stage1, "Stage 1 (Frozen)")

        # Save intermediate model
        self.file_manager.save_model(model, "model_stage1_frozen")

        # Stage 2: Fine-tuning
        print("\n" + "-"*50)
        print("STAGE 2: Fine-tuning")
        print("-"*50)

        # Unfreeze top layers
        self.model_builder.unfreeze_layers(base_model, self.config.fine_tune_layers)

        # Recompile with lower learning rate
        model = self.model_builder.compile_model(model, self.config.fine_tune_lr)

        # Get callbacks
        callbacks_stage2 = get_callbacks(self.config, "stage2_finetune", val_ds)

        # Train
        history_stage2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.fine_tune_epochs,
            class_weight=self.class_weights,
            callbacks=callbacks_stage2,
            verbose=self.config.verbose
        )

        # Plot history
        self.evaluator.plot_training_history(history_stage2, "Stage 2 (Fine-tuning)")

        # Save final model
        self.file_manager.save_model(model, "model_final")

        # Combine histories
        combined_history = {
            'stage1': history_stage1.history,
            'stage2': history_stage2.history
        }

        # Save training history
        history_path = Path(self.config.results_dir) / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(combined_history, f, indent=2, default=str)

        return model, combined_history

    def evaluate_model(self, model: keras.Model, test_ds: tf.data.Dataset,
                      class_names: List[str]) -> Dict:
        """Evaluate model performance"""

        # Standard evaluation
        eval_results = self.evaluator.comprehensive_evaluation(model, test_ds, class_names)

        # TTA evaluation if enabled
        if self.config.use_tta:
            print("\n" + "="*50)
            print("TEST TIME AUGMENTATION")
            print("="*50)

            # Get TTA predictions
            y_true = []
            for _, y_batch in test_ds:
                y_true.extend(y_batch.numpy().flatten())
            y_true = np.array(y_true)

            y_prob_tta = self.evaluator.predict_with_tta(
                model, test_ds, self.config.tta_augmentations
            ).ravel()

            y_pred_tta = (y_prob_tta >= 0.5).astype(int)

            # Calculate TTA metrics
            print("\nTTA Classification Report:")
            print("="*50)
            tta_report = classification_report(y_true, y_pred_tta, target_names=class_names, output_dict=True)
            print(classification_report(y_true, y_pred_tta, target_names=class_names))

            # Save TTA results
            tta_results_path = Path(self.config.results_dir) / "tta_results.json"
            with open(tta_results_path, 'w') as f:
                json.dump(tta_report, f, indent=2)

            # Update results
            eval_results['tta_predictions'] = y_prob_tta.tolist()
            eval_results['tta_report'] = tta_report

        return eval_results

    def optimize_for_deployment(self, model: keras.Model, train_ds: tf.data.Dataset):
        """Optimize model for deployment"""
        print("\n" + "="*50)
        print("MODEL OPTIMIZATION")
        print("="*50)

        # Convert to TFLite
        tflite_path = self.optimizer.convert_to_tflite(
            model, "model_optimized", quantize=True, representative_data=train_ds
        )

        return tflite_path
    
    def create_deployment_manifest(self, model: keras.Model, eval_results: Dict, tflite_path: str) -> str:
        """Create deployment manifest with model info and metrics"""
        manifest_path = Path(self.config.results_dir) / "models" / "deployment_manifest.json"
        
        manifest = {
            "deployment_info": {
                "created_at": datetime.datetime.now().isoformat(),
                "model_variant": self.config.model_variant,
                "img_size": self.config.img_size,
                "class_names": ["NORMAL", "PNEUMONIA"]
            },
            "model_files": {
                "keras_model": str(Path(self.config.results_dir) / "models" / "final_model.keras"),
                "tflite_model": str(tflite_path),
                "config_file": str(Path(self.config.results_dir) / "training_config.json")
            },
            "performance_metrics": {
                "accuracy": float(eval_results.get('accuracy', 0)),
                "auc": float(eval_results.get('auc', 0)),
                "sensitivity": float(eval_results.get('sensitivity', 0)),
                "specificity": float(eval_results.get('specificity', 0)),
                "precision": float(eval_results.get('precision', 0)),
                "recall": float(eval_results.get('recall', 0))
            },
            "preprocessing": {
                "input_shape": [self.config.img_size, self.config.img_size, 3],
                "preprocessing_function": f"efficientnet.preprocess_input (variant: {self.config.model_variant})",
                "color_mode": self.config.color_mode
            },
            "deployment_notes": [
                "Use the keras model for standard deployment with TensorFlow",
                "Use the tflite model for mobile/edge deployment",
                "TFLite model includes quantization for faster inference",
                f"Expected input size: {self.config.img_size}x{self.config.img_size}",
                "Normalize images using EfficientNet preprocessing before inference"
            ]
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return str(manifest_path)

    def run(self):
        """Run complete training pipeline"""
        print("\n" + "="*70)
        print(" PNEUMONIA DETECTION - ENHANCED TRAINING PIPELINE ")
        print("="*70)

        # Print and save configuration
        self.config.print_config()
        config_path = Path(self.config.results_dir) / "training_config.json"
        self.config.save(str(config_path))
        print(f"✓ Configuration saved to: {config_path}")

        # Prepare data
        train_ds, val_ds, test_ds, class_names = self.prepare_data()

        # Train model
        model, history = self.train_model(train_ds, val_ds)

        # Evaluate model
        eval_results = self.evaluate_model(model, test_ds, class_names)

        # Optimize for deployment (automatic if enabled in config)
        if self.config.auto_optimize_for_deployment:
            print("\n" + "="*50)
            print("DEPLOYMENT OPTIMIZATION")
            print("="*50)
            print("📦 Automatically optimizing model for deployment...")
            
            tflite_path = self.optimize_for_deployment(model, train_ds)
            print(f"✓ Optimized TFLite model saved to: {tflite_path}")
            
            # Create deployment manifest
            deployment_manifest = self.create_deployment_manifest(model, eval_results, tflite_path)
            print(f"✓ Deployment manifest created: {deployment_manifest}")
        else:
            print("\n⚠ Automatic deployment optimization is disabled in config")
            user_input = input("Optimize model for deployment manually? (y/n): ").lower().strip()
            if user_input == 'y':
                tflite_path = self.optimize_for_deployment(model, train_ds)
                print(f"\n✓ Optimized model saved to: {tflite_path}")

        # Print final summary
        print("\n" + "="*70)
        print(" TRAINING COMPLETED SUCCESSFULLY ")
        print("="*70)
        print(f"\nAll results saved to: {Path(self.config.results_dir).absolute()}")
        print("\nFinal Model Performance:")
        print(f"  Accuracy: {eval_results.get('accuracy', 0):.4f}")
        print(f"  AUC: {eval_results.get('auc', 0):.4f}")
        print(f"  Sensitivity: {eval_results.get('sensitivity', 0):.4f}")
        print(f"  Specificity: {eval_results.get('specificity', 0):.4f}")

        return model, eval_results

# ------------------------
# Main Execution
# ------------------------
def main():
    """Main execution function"""

    # Create configuration
    # config = TrainingConfig()
    config = TrainingConfig.load('results/training_config_safe.json')

    # Create and run pipeline
    pipeline = TrainingPipeline(config)
    model, results = pipeline.run()

    print("\n✓ All processes completed successfully!")
    print(f"✓ Check the '{config.results_dir}' folder for all outputs")

    return model, results

if __name__ == "__main__":
    model, results = main()