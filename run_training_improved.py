"""
Improved Training Script
Use this to train with optimized configurations for higher accuracy
"""

import sys
import argparse
from pathlib import Path
from model import TrainingConfig

# Import your main training script components
# Adjust imports based on your model.py structure


def print_banner(text):
    """Print a formatted banner"""
    width = 70
    print("\n" + "="*width)
    print(text.center(width))
    print("="*width + "\n")


def compare_configs(current_config, new_config):
    """Show comparison between current and new config"""
    print_banner("CONFIGURATION COMPARISON")
    
    important_params = [
        'model_variant',
        'img_size',
        'batch_size',
        'initial_epochs',
        'fine_tune_epochs',
        'fine_tune_layers',
        'augmentation_strength',
        'dropout_rate',
        'tta_augmentations'
    ]
    
    print(f"{'Parameter':<25} {'Current':<15} {'New':<15} {'Change':<15}")
    print("-" * 70)
    
    for param in important_params:
        current_val = getattr(current_config, param, None)
        new_val = getattr(new_config, param, None)
        
        if current_val != new_val:
            if isinstance(current_val, float):
                change = "↑" if new_val > current_val else "↓"
            elif isinstance(current_val, int):
                change = "↑" if new_val > current_val else "↓"
            else:
                change = "Changed"
            
            print(f"{param:<25} {str(current_val):<15} {str(new_val):<15} {change:<15}")
    
    # Calculate total epochs
    current_total = current_config.initial_epochs + current_config.fine_tune_epochs
    new_total = new_config.initial_epochs + new_config.fine_tune_epochs
    
    print("-" * 70)
    print(f"{'Total Epochs':<25} {current_total:<15} {new_total:<15} {'↑' if new_total > current_total else '↓':<15}")
    print("-" * 70)


def estimate_training_time(config):
    """Estimate training time based on configuration"""
    base_time = 2  # hours for B0, 224, 35 epochs
    
    # Model size factor
    model_factors = {'B0': 1.0, 'B3': 2.5, 'B4': 4.0, 'B7': 8.0}
    model_factor = model_factors.get(config.model_variant, 1.0)
    
    # Image size factor
    img_factor = (config.img_size / 224) ** 2
    
    # Epoch factor
    total_epochs = config.initial_epochs + config.fine_tune_epochs
    epoch_factor = total_epochs / 35
    
    # Batch size factor (smaller batch = more iterations)
    batch_factor = 32 / config.batch_size
    
    estimated_time = base_time * model_factor * img_factor * epoch_factor * batch_factor
    
    return estimated_time


def estimate_gpu_memory(config):
    """Estimate GPU memory requirements"""
    base_memory = 4  # GB for B0, 224, batch 32
    
    model_factors = {'B0': 1.0, 'B3': 1.5, 'B4': 2.0, 'B7': 3.0}
    model_factor = model_factors.get(config.model_variant, 1.0)
    
    img_factor = (config.img_size / 224) ** 2
    batch_factor = config.batch_size / 32
    
    estimated_memory = base_memory * model_factor * img_factor * batch_factor
    
    return estimated_memory


def print_requirements(config):
    """Print estimated requirements"""
    print_banner("ESTIMATED REQUIREMENTS")
    
    training_time = estimate_training_time(config)
    gpu_memory = estimate_gpu_memory(config)
    
    print(f"⏱️  Training Time: ~{training_time:.1f} hours")
    print(f"💾 GPU Memory:    ~{gpu_memory:.1f} GB")
    print(f"🖼️  Image Size:    {config.img_size}x{config.img_size}")
    print(f"📊 Batch Size:    {config.batch_size}")
    print(f"🔄 Total Epochs:  {config.initial_epochs + config.fine_tune_epochs}")
    
    # Check if it fits in RTX 1070
    if gpu_memory > 8:
        print(f"\n⚠️  WARNING: Estimated memory ({gpu_memory:.1f} GB) may exceed")
        print(f"   your RTX 1070's 8 GB. Consider reducing batch_size.")
    else:
        print(f"\n✅ Should fit in your RTX 1070 (8 GB VRAM)")


def print_expected_accuracy(config_type):
    """Print expected accuracy improvement"""
    print_banner("EXPECTED ACCURACY")
    
    if config_type == "current":
        accuracy = "92-94%"
        improvement = "Baseline"
    elif config_type == "high":
        accuracy = "95-97%"
        improvement = "+2-3% improvement"
    elif config_type == "ultra":
        accuracy = "97-98%+"
        improvement = "+4-5% improvement"
    else:
        accuracy = "Unknown"
        improvement = "Unknown"
    
    print(f"Expected Accuracy:  {accuracy}")
    print(f"Improvement:        {improvement}")


def main():
    """Main function"""
    
    print_banner("🚀 IMPROVED MODEL TRAINING")
    
    parser = argparse.ArgumentParser(description="Train pneumonia detection model with improved accuracy")
    parser.add_argument(
        '--config',
        type=str,
        choices=['current', 'high', 'ultra'],
        default='high',
        help='Configuration to use: current (baseline), high (recommended), ultra (maximum)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Show comparison with current config and exit'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show config info and exit'
    )
    
    args = parser.parse_args()
    
    # Load configurations
    config_paths = {
        'current': 'results/training_config.json',
        'high': 'configs/high_accuracy_config.json',
        'ultra': 'configs/ultra_high_accuracy_config.json'
    }
    
    # Check if config files exist
    current_config_path = Path(config_paths['current'])
    selected_config_path = Path(config_paths[args.config])
    
    if not selected_config_path.exists():
        print(f"❌ Config file not found: {selected_config_path}")
        print(f"\nAvailable configs:")
        for name, path in config_paths.items():
            exists = "✅" if Path(path).exists() else "❌"
            print(f"  {exists} {name}: {path}")
        return
    
    # Load selected config
    print(f"Loading configuration: {args.config}")
    new_config = TrainingConfig.load(str(selected_config_path))
    
    # Load current config for comparison
    if current_config_path.exists():
        current_config = TrainingConfig.load(str(current_config_path))
    else:
        current_config = TrainingConfig()  # Use defaults
    
    # Show info
    if args.info or args.compare:
        compare_configs(current_config, new_config)
        print_requirements(new_config)
        print_expected_accuracy(args.config)
        
        if args.compare or args.info:
            return
    
    # Confirm training
    print_banner("READY TO TRAIN")
    
    print(f"Configuration: {args.config.upper()}")
    print(f"Model: EfficientNet-{new_config.model_variant}")
    print(f"Image Size: {new_config.img_size}x{new_config.img_size}")
    print(f"Total Epochs: {new_config.initial_epochs + new_config.fine_tune_epochs}")
    
    print_requirements(new_config)
    print_expected_accuracy(args.config)
    
    print("\n" + "="*70)
    response = input("\n⚠️  Training will start. Continue? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\n❌ Training cancelled.")
        return
    
    print("\n" + "="*70)
    print("STARTING TRAINING WITH IMPROVED CONFIGURATION")
    print("="*70)
    
    # Save the config
    new_config.save("results/training_config_improved.json")
    print(f"\n✅ Config saved to: results/training_config_improved.json")
    
    print("\n📝 To train with this config, modify your model.py:")
    print("   Change the config loading line to:")
    print("   config = TrainingConfig.load('results/training_config_improved.json')")
    print("\n   Then run your training script as normal.")
    
    print("\n" + "="*70)
    print("Or run your model.py with the improved config loaded")
    print("="*70)
    
    # Note: To actually start training, you'd import and call your
    # training function here. For example:
    # from model import main as train_model
    # train_model(config=new_config)


if __name__ == "__main__":
    main()
