"""
Model Evaluation Script
This script loads the trained model and evaluates its accuracy on the test set.
It provides comprehensive metrics including accuracy, precision, recall, F1-score, etc.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from model import TrainingConfig

class ModelEvaluator:
    """Evaluate trained model and display comprehensive metrics"""
    
    def __init__(self, model_path: str, config_path: str = None):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained Keras model (.keras file)
            config_path: Path to training config JSON (optional)
        """
        self.model_path = model_path
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            self.config = TrainingConfig.load(config_path)
        else:
            # Use default config
            self.config = TrainingConfig()
        
        # Load model
        print(f"\nLoading model from: {model_path}")
        self.model = keras.models.load_model(model_path)
        print("✓ Model loaded successfully")
        
        # Display model summary
        print("\n" + "="*60)
        print("MODEL ARCHITECTURE SUMMARY")
        print("="*60)
        self.model.summary()
    
    def load_test_data(self):
        """Load test dataset"""
        test_dir = Path(self.config.data_root) / "test"
        
        if not test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {test_dir}")
        
        print(f"\nLoading test data from: {test_dir}")
        
        # Create test dataset
        test_ds = keras.utils.image_dataset_from_directory(
            test_dir,
            labels="inferred",
            label_mode="int",
            color_mode=self.config.color_mode,
            image_size=(self.config.img_size, self.config.img_size),
            batch_size=self.config.batch_size,
            shuffle=False,  # Don't shuffle for evaluation
            seed=self.config.seed,
        )
        
        # Get class names
        self.class_names = test_ds.class_names
        print(f"✓ Test data loaded - Classes: {self.class_names}")
        
        # Optimize pipeline
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
        
        return test_ds
    
    def evaluate(self, test_ds: tf.data.Dataset):
        """
        Perform comprehensive evaluation
        
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        print("\n" + "="*60)
        print("EVALUATING MODEL ON TEST DATA")
        print("="*60)
        
        # Get predictions
        print("\nGenerating predictions...")
        y_true = []
        y_prob = []
        
        for x_batch, y_batch in test_ds:
            predictions = self.model.predict(x_batch, verbose=0)
            y_prob.extend(predictions.ravel())
            y_true.extend(y_batch.numpy().flatten())
        
        y_true = np.array(y_true, dtype=int)
        y_prob = np.array(y_prob)
        y_pred = (y_prob >= 0.5).astype(int)
        
        # Calculate all metrics
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary')
        metrics['recall'] = recall_score(y_true, y_pred, average='binary')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='binary')
        
        # ROC AUC
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Medical metrics (for binary classification)
        tn, fp, fn, tp = cm.ravel()
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        # Sensitivity and Specificity
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Positive Predictive Value (PPV) and Negative Predictive Value (NPV)
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Display results
        self._display_metrics(metrics)
        
        # Generate visualizations
        self._plot_confusion_matrix(cm)
        self._plot_roc_curve(y_true, y_prob, metrics['roc_auc'])
        
        # Generate classification report
        print("\n" + "="*60)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*60)
        report = classification_report(y_true, y_pred, target_names=self.class_names)
        print(report)
        
        # Save results
        self._save_results(metrics, report)
        
        return metrics
    
    def _display_metrics(self, metrics):
        """Display metrics in a readable format"""
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        
        print(f"\n📊 Overall Performance:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1_score']:.4f}")
        print(f"   ROC AUC:   {metrics['roc_auc']:.4f}")
        
        print(f"\n🏥 Medical Metrics:")
        print(f"   Sensitivity (True Positive Rate): {metrics['sensitivity']:.4f} ({metrics['sensitivity']*100:.2f}%)")
        print(f"   Specificity (True Negative Rate): {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
        print(f"   PPV (Positive Predictive Value):  {metrics['ppv']:.4f}")
        print(f"   NPV (Negative Predictive Value):  {metrics['npv']:.4f}")
        
        print(f"\n📈 Confusion Matrix:")
        print(f"   True Positives:  {metrics['true_positives']}")
        print(f"   True Negatives:  {metrics['true_negatives']}")
        print(f"   False Positives: {metrics['false_positives']}")
        print(f"   False Negatives: {metrics['false_negatives']}")
        
        # Interpretation
        print(f"\n💡 Interpretation:")
        if metrics['accuracy'] >= 0.95:
            print("   ✅ Excellent accuracy!")
        elif metrics['accuracy'] >= 0.90:
            print("   ✅ Very good accuracy!")
        elif metrics['accuracy'] >= 0.85:
            print("   ⚠️  Good accuracy, but could be improved")
        else:
            print("   ❌ Accuracy needs improvement")
        
        if metrics['sensitivity'] >= 0.90:
            print("   ✅ High sensitivity - good at detecting positive cases")
        else:
            print("   ⚠️  Lower sensitivity - may miss some positive cases")
        
        if metrics['specificity'] >= 0.90:
            print("   ✅ High specificity - good at identifying negative cases")
        else:
            print("   ⚠️  Lower specificity - may produce false alarms")
    
    def _plot_confusion_matrix(self, cm):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        # Save
        output_path = Path(self.config.results_dir) / "evaluation_confusion_matrix.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Confusion matrix saved to: {output_path}")
        plt.show()
    
    def _plot_roc_curve(self, y_true, y_prob, auc_score):
        """Plot and save ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save
        output_path = Path(self.config.results_dir) / "evaluation_roc_curve.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ ROC curve saved to: {output_path}")
        plt.show()
    
    def _save_results(self, metrics, report):
        """Save evaluation results to JSON"""
        results = {
            'model_path': str(self.model_path),
            'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in metrics.items()},
            'classification_report': report
        }
        
        output_path = Path(self.config.results_dir) / "evaluation_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✓ Evaluation results saved to: {output_path}")


def main():
    """Main evaluation function"""
    
    # Set paths - MODIFY THESE TO MATCH YOUR FILES
    model_path = "a:/project/results/models/model_stage1_frozen.keras"
    config_path = "a:/project/results/training_config.json"
    
    # Create evaluator
    evaluator = ModelEvaluator(model_path, config_path)
    
    # Load test data
    test_ds = evaluator.load_test_data()
    
    # Evaluate model
    metrics = evaluator.evaluate(test_ds)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print(f"\n✅ Model Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"✅ All results saved to: {evaluator.config.results_dir}")


if __name__ == "__main__":
    main()
