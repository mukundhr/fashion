import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)
from sklearn.decomposition import PCA
import joblib

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow")

# Fashion MNIST class names
FASHION_MNIST_CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class FashionMNISTClassifier:
    """Fashion MNIST image classification pipeline."""
    
    def __init__(self, output_dir: str = "reports_mnist"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self) -> tuple:
        """Load Fashion MNIST dataset."""
        print("=" * 60)
        print("LOADING FASHION MNIST DATA")
        print("=" * 60)
        
        if TF_AVAILABLE:
            # Load via TensorFlow/Keras
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        else:
            # Fallback: try to load from CSV if available
            raise ImportError("TensorFlow required for Fashion MNIST. Install: pip install tensorflow")
        
        # Normalize pixel values to [0, 1]
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Reshape for traditional ML (flatten)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # For CNN, keep 2D shape with channel dimension
        X_train_cnn = X_train.reshape(-1, 28, 28, 1)
        X_test_cnn = X_test.reshape(-1, 28, 28, 1)
        
        # One-hot encode labels for CNN
        y_train_cat = to_categorical(y_train, 10)
        y_test_cat = to_categorical(y_test, 10)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_train_flat = X_train_flat
        self.X_test_flat = X_test_flat
        self.X_train_cnn = X_train_cnn
        self.X_test_cnn = X_test_cnn
        self.y_train_cat = y_train_cat
        self.y_test_cat = y_test_cat
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Image shape: {X_train.shape[1:]}")
        print(f"Classes: {FASHION_MNIST_CLASSES}")
        print(f"Train class distribution: {np.bincount(y_train)}")
        print(f"Test class distribution: {np.bincount(y_test)}")
        
        return (X_train, y_train), (X_test, y_test)
    
    def perform_eda(self):
        """Perform EDA on Fashion MNIST."""
        print("\n" + "=" * 60)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        # 1. Class distribution
        self._plot_class_distribution()
        
        # 2. Sample images per class
        self._plot_sample_images()
        
        # 3. Pixel intensity distribution
        self._plot_pixel_distribution()
        
        # 4. Average image per class
        self._plot_average_images()
        
        # 5. PCA visualization
        self._plot_pca()
        
    def _plot_class_distribution(self):
        """Plot class distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        train_counts = np.bincount(self.y_train)
        test_counts = np.bincount(self.y_test)
        
        axes[0].bar(range(10), train_counts, color='steelblue', edgecolor='black')
        axes[0].set_xticks(range(10))
        axes[0].set_xticklabels(FASHION_MNIST_CLASSES, rotation=45, ha='right')
        axes[0].set_title('Training Set Class Distribution', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Count')
        
        axes[1].bar(range(10), test_counts, color='coral', edgecolor='black')
        axes[1].set_xticks(range(10))
        axes[1].set_xticklabels(FASHION_MNIST_CLASSES, rotation=45, ha='right')
        axes[1].set_title('Test Set Class Distribution', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mnist_class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sample_images(self):
        """Plot sample images from each class."""
        fig, axes = plt.subplots(2, 10, figsize=(20, 5))
        fig.suptitle('Sample Images per Class', fontsize=16, fontweight='bold')
        
        for class_idx in range(10):
            # Find indices for this class
            train_indices = np.where(self.y_train == class_idx)[0]
            test_indices = np.where(self.y_test == class_idx)[0]
            
            # Show one train, one test sample
            axes[0, class_idx].imshow(self.X_train[train_indices[0]], cmap='gray')
            axes[0, class_idx].set_title(f'Train: {FASHION_MNIST_CLASSES[class_idx]}', fontsize=10)
            axes[0, class_idx].axis('off')
            
            axes[1, class_idx].imshow(self.X_test[test_indices[0]], cmap='gray')
            axes[1, class_idx].set_title(f'Test: {FASHION_MNIST_CLASSES[class_idx]}', fontsize=10)
            axes[1, class_idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mnist_sample_images.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pixel_distribution(self):
        """Plot pixel intensity distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].hist(self.X_train.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].set_title('Pixel Intensity Distribution (Train)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Pixel Value')
        axes[0].set_ylabel('Frequency')
        
        axes[1].hist(self.X_test.flatten(), bins=50, color='coral', alpha=0.7, edgecolor='black')
        axes[1].set_title('Pixel Intensity Distribution (Test)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Pixel Value')
        axes[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mnist_pixel_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_average_images(self):
        """Plot average image per class."""
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle('Average Image per Class', fontsize=16, fontweight='bold')
        
        for class_idx in range(10):
            row = class_idx // 5
            col = class_idx % 5
            class_images = self.X_train[self.y_train == class_idx]
            avg_image = np.mean(class_images, axis=0)
            
            im = axes[row, col].imshow(avg_image, cmap='gray')
            axes[row, col].set_title(FASHION_MNIST_CLASSES[class_idx], fontsize=12)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mnist_average_images.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pca(self):
        """Plot PCA projection of the data."""
        print("Computing PCA... (this may take a moment)")
        
        # Sample for faster PCA
        sample_size = min(5000, len(self.X_train_flat))
        indices = np.random.choice(len(self.X_train_flat), sample_size, replace=False)
        X_sample = self.X_train_flat[indices]
        y_sample = self.y_train[indices]
        
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_sample)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_sample, cmap='tab10', alpha=0.6, s=10)
        plt.colorbar(scatter, label='Class', ticks=range(10))
        plt.title(f'PCA Projection (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})', fontsize=14, fontweight='bold')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mnist_pca.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def train_traditional_models(self) -> dict:
        """Train traditional ML models on flattened images."""
        print("\n" + "=" * 60)
        print("TRAINING TRADITIONAL ML MODELS")
        print("=" * 60)
        
        # Use PCA for dimensionality reduction (speed up training)
        print("Applying PCA for dimensionality reduction...")
        pca = PCA(n_components=0.95, random_state=42)  # Keep 95% variance
        X_train_pca = pca.fit_transform(self.X_train_flat)
        X_test_pca = pca.transform(self.X_test_flat)
        print(f"Reduced from {self.X_train_flat.shape[1]} to {X_train_pca.shape[1]} features")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_pca)
        X_test_scaled = scaler.transform(X_test_pca)
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, self.y_train, cv=3, scoring='accuracy', n_jobs=-1)
            
            # Train
            model.fit(X_train_scaled, self.y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.y_test, y_pred, average='weighted', zero_division=0
            )
            
            results[name] = {
                'model': model,
                'pca': pca,
                'scaler': scaler,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': accuracy,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': f1,
                'y_pred': y_pred
            }
            
            print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            print(f"  Test Accuracy: {accuracy:.4f}")
            print(f"  Test F1: {f1:.4f}")
        
        self.models.update(results)
        return results
    
    def build_cnn_model(self) -> models.Model:
        """Build a CNN model for Fashion MNIST."""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_cnn(self, epochs: int = 10, batch_size: int = 128) -> dict:
        """Train CNN model."""
        if not TF_AVAILABLE:
            print("TensorFlow not available. Skipping CNN training.")
            return {}
        
        print("\n" + "=" * 60)
        print("TRAINING CNN MODEL")
        print("=" * 60)
        
        # Build model
        cnn_model = self.build_cnn_model()
        cnn_model.summary()
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5, restore_best_weights=True
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
        )
        
        # Train
        history = cnn_model.fit(
            self.X_train_cnn, self.y_train_cat,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.X_test_cnn, self.y_test_cat),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate
        test_loss, test_accuracy = cnn_model.evaluate(self.X_test_cnn, self.y_test_cat, verbose=0)
        
        # Predictions
        y_pred_proba = cnn_model.predict(self.X_test_cnn, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, average='weighted', zero_division=0
        )
        
        results = {
            'model': cnn_model,
            'history': history,
            'test_accuracy': test_accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        self.models['CNN'] = results
        
        print(f"\nCNN Test Accuracy: {test_accuracy:.4f}")
        print(f"CNN Test F1: {f1:.4f}")
        
        # Plot training history
        self._plot_training_history(history)
        
        return results
    
    def _plot_training_history(self, history):
        """Plot CNN training history."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(history.history['accuracy'], label='Train')
        axes[0].plot(history.history['val_accuracy'], label='Validation')
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(history.history['loss'], label='Train')
        axes[1].plot(history.history['val_loss'], label='Validation')
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mnist_cnn_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate_all_models(self) -> pd.DataFrame:
        """Evaluate and compare all models."""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        comparison_data = []
        for name, result in self.models.items():
            comparison_data.append({
                'Model': name,
                'Test Accuracy': result['test_accuracy'],
                'Test Precision': result['test_precision'],
                'Test Recall': result['test_recall'],
                'Test F1': result['test_f1']
            })
        
        comparison_df = pd.DataFrame(comparison_data).sort_values('Test Accuracy', ascending=False)
        print("\nModel Comparison:")
        print(comparison_df.to_string(index=False))
        
        comparison_df.to_csv(self.output_dir / 'mnist_model_comparison.csv', index=False)
        
        self.best_model_name = comparison_df.iloc[0]['Model']
        self.best_model = self.models[self.best_model_name]['model']
        print(f"\nBest Model: {self.best_model_name}")
        
        # Detailed evaluation for best model
        self._detailed_evaluation(self.best_model_name)
        
        # Plot comparison
        self._plot_model_comparison(comparison_df)
        
        return comparison_df
    
    def _detailed_evaluation(self, model_name: str):
        """Detailed evaluation for a specific model."""
        result = self.models[model_name]
        y_pred = result['y_pred']
        
        print(f"\n{'='*60}")
        print(f"DETAILED EVALUATION: {model_name}")
        print(f"{'='*60}")
        
        # Classification report
        print("\nClassification Report:")
        report = classification_report(
            self.y_test, y_pred,
            target_names=FASHION_MNIST_CLASSES,
            output_dict=True
        )
        print(classification_report(self.y_test, y_pred, target_names=FASHION_MNIST_CLASSES, zero_division=0))
        
        # Save report
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(self.output_dir / f'mnist_classification_report_{model_name.replace(" ", "_").lower()}.csv')
        
        # Confusion Matrix
        self._plot_confusion_matrix(y_pred, model_name)
        
        # Per-class metrics
        self._plot_per_class_metrics(report, model_name)
        
        # Misclassified examples
        self._plot_misclassified(y_pred, model_name)
    
    def _plot_confusion_matrix(self, y_pred, model_name: str):
        """Plot confusion matrix."""
        cm = confusion_matrix(self.y_test, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=FASHION_MNIST_CLASSES, yticklabels=FASHION_MNIST_CLASSES, ax=axes[0])
        axes[0].set_title(f'Confusion Matrix (Counts) - {model_name}', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=FASHION_MNIST_CLASSES, yticklabels=FASHION_MNIST_CLASSES, ax=axes[1])
        axes[1].set_title(f'Confusion Matrix (Normalized) - {model_name}', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'mnist_confusion_matrix_{model_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_class_metrics(self, report: dict, model_name: str):
        """Plot per-class metrics."""
        classes = [c for c in FASHION_MNIST_CLASSES if c in report]
        metrics = ['precision', 'recall', 'f1-score']
        
        data = {metric: [report[c][metric] for c in classes] for metric in metrics}
        df_metrics = pd.DataFrame(data, index=classes)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        df_metrics.plot(kind='bar', ax=ax, edgecolor='black')
        ax.set_title(f'Per-Class Metrics - {model_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xlabel('Class')
        ax.legend(title='Metric')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'mnist_per_class_metrics_{model_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_misclassified(self, y_pred, model_name: str):
        """Plot misclassified examples."""
        misclassified_idx = np.where(self.y_test != y_pred)[0]
        
        if len(misclassified_idx) == 0:
            print("No misclassified examples!")
            return
        
        # Show up to 20 misclassified examples
        n_show = min(20, len(misclassified_idx))
        n_cols = 5
        n_rows = (n_show + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        fig.suptitle(f'Misclassified Examples - {model_name}', fontsize=16, fontweight='bold')
        
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, idx in enumerate(misclassified_idx[:n_show]):
            ax = axes[i]
            ax.imshow(self.X_test[idx], cmap='gray')
            true_label = FASHION_MNIST_CLASSES[self.y_test[idx]]
            pred_label = FASHION_MNIST_CLASSES[y_pred[idx]]
            ax.set_title(f'True: {true_label}\nPred: {pred_label}', fontsize=10, color='red')
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_show, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'mnist_misclassified_{model_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_comparison(self, comparison_df: pd.DataFrame):
        """Plot model comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics = ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1']
        colors = ['steelblue', 'coral', 'mediumseagreen', 'gold']
        
        for idx, (metric, color) in enumerate(zip(metrics, colors)):
            ax = axes[idx // 2, idx % 2]
            bars = ax.bar(comparison_df['Model'], comparison_df[metric], color=color, edgecolor='black')
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylim(0, 1.05)
            
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Fashion MNIST Model Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mnist_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_models(self):
        """Save all trained models."""
        models_dir = self.output_dir.parent / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        for name, result in self.models.items():
            if name == 'CNN' and TF_AVAILABLE:
                # Save Keras model
                model_path = models_dir / f'mnist_{name.lower()}_model.h5'
                result['model'].save(model_path)
                print(f"CNN model saved to: {model_path}")
            else:
                # Save sklearn model with preprocessing
                model_data = {
                    'model': result['model'],
                    'pca': result.get('pca'),
                    'scaler': result.get('scaler'),
                    'class_names': FASHION_MNIST_CLASSES
                }
                model_path = models_dir / f'mnist_{name.lower().replace(" ", "_")}.joblib'
                joblib.dump(model_data, model_path)
                print(f"{name} model saved to: {model_path}")


def main():
    """Main execution pipeline for Fashion MNIST."""
    print("[START] Starting Fashion MNIST Classification Pipeline")
    print("=" * 60)
    
    classifier = FashionMNISTClassifier(output_dir="fashion_classification/reports_mnist")
    
    # 1. Load data
    classifier.load_data()
    
    # 2. Perform EDA
    classifier.perform_eda()
    
    # 3. Train traditional ML models
    classifier.train_traditional_models()
    
    # 4. Train CNN (if TensorFlow available)
    if TF_AVAILABLE:
        classifier.train_cnn(epochs=15, batch_size=128)
    
    # 5. Evaluate all models
    classifier.evaluate_all_models()
    
    # 6. Save models
    classifier.save_models()
    
    print("\n" + "=" * 60)
    print("[SUCCESS] FASHION MNIST PIPELINE COMPLETED!")
    print("=" * 60)
    print(f"Best Model: {classifier.best_model_name}")
    print(f"Test Accuracy: {classifier.models[classifier.best_model_name]['test_accuracy']:.4f}")
    print(f"Reports saved to: fashion_classification/reports_mnist/")
    print(f"Models saved to: fashion_classification/models/")
    
    return classifier


if __name__ == "__main__":
    classifier = main()