import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# For visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


class FashionClassifier:
    """Complete fashion product classification pipeline."""
    
    def __init__(self, data_path: str, output_dir: str = "reports"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self, sample_size: int = 10000) -> pd.DataFrame:
        """Load and perform initial inspection of the dataset."""
        print("=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        
        self.df = pd.read_csv(self.data_path)
        
        # Sample data for faster processing (remove or increase for full dataset)
        if sample_size and len(self.df) > sample_size:
            self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            print(f"Sampled {sample_size} rows for faster processing")
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nColumns: {self.df.columns.tolist()}")
        print(f"\nData types:\n{self.df.dtypes}")
        print(f"\nFirst 5 rows:\n{self.df.head()}")
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        print(f"\nLabel distribution:\n{self.df['label'].value_counts()}")
        
        return self.df
    
    def perform_eda(self) -> dict:
        """Perform comprehensive Exploratory Data Analysis."""
        print("\n" + "=" * 60)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        eda_results = {}
        
        # 1. Basic statistics
        eda_results['basic_stats'] = {
            'total_samples': len(self.df),
            'num_classes': self.df['label'].nunique(),
            'classes': self.df['label'].unique().tolist(),
            'class_distribution': self.df['label'].value_counts().to_dict(),
            'kids_distribution': self.df['kids'].value_counts().to_dict(),
            'unique_senders': self.df['sender_id'].nunique()
        }
        
        print(f"\nTotal samples: {eda_results['basic_stats']['total_samples']}")
        print(f"Number of classes: {eda_results['basic_stats']['num_classes']}")
        print(f"Classes: {eda_results['basic_stats']['classes']}")
        print(f"Kids distribution: {eda_results['basic_stats']['kids_distribution']}")
        print(f"Unique senders: {eda_results['basic_stats']['unique_senders']}")
        
        # 2. Class distribution visualization
        self._plot_class_distribution()
        
        # 3. Kids vs Adults distribution per class
        self._plot_kids_distribution()
        
        # 4. Sender analysis
        self._plot_sender_analysis()
        
        # 5. Class imbalance check
        eda_results['class_imbalance'] = self._check_class_imbalance()
        
        return eda_results
    
    def _plot_class_distribution(self):
        """Plot class distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Count plot
        class_counts = self.df['label'].value_counts()
        axes[0].bar(range(len(class_counts)), class_counts.values, color='steelblue', edgecolor='black')
        axes[0].set_xticks(range(len(class_counts)))
        axes[0].set_xticklabels(class_counts.index, rotation=45, ha='right')
        axes[0].set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Number of Samples')
        axes[0].set_xlabel('Class')
        
        # Percentage plot
        class_pct = (class_counts / len(self.df) * 100).round(2)
        axes[1].bar(range(len(class_pct)), class_pct.values, color='coral', edgecolor='black')
        axes[1].set_xticks(range(len(class_pct)))
        axes[1].set_xticklabels(class_pct.index, rotation=45, ha='right')
        axes[1].set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Percentage (%)')
        axes[1].set_xlabel('Class')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create interactive plotly version
        fig_plotly = px.bar(
            x=class_counts.index, y=class_counts.values,
            labels={'x': 'Class', 'y': 'Count'},
            title='Fashion Product Class Distribution',
            color=class_counts.values,
            color_continuous_scale='Viridis'
        )
        fig_plotly.write_html(self.output_dir / 'class_distribution_interactive.html')
    
    def _plot_kids_distribution(self):
        """Plot kids vs adults distribution across classes."""
        kids_crosstab = pd.crosstab(self.df['label'], self.df['kids'], normalize='index') * 100
        
        fig, ax = plt.subplots(figsize=(12, 6))
        kids_crosstab.plot(kind='bar', stacked=True, ax=ax, color=['skyblue', 'lightcoral'], edgecolor='black')
        ax.set_title('Kids vs Adults Distribution by Class (%)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Percentage')
        ax.set_xlabel('Class')
        ax.legend(title='Kids', labels=['Adults (False)', 'Kids (True)'])
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'kids_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sender_analysis(self):
        """Analyze sender distribution."""
        sender_counts = self.df['sender_id'].value_counts()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Top 20 senders
        top_senders = sender_counts.head(20)
        axes[0].bar(range(len(top_senders)), top_senders.values, color='mediumseagreen', edgecolor='black')
        axes[0].set_xticks(range(len(top_senders)))
        axes[0].set_xticklabels(top_senders.index.astype(str), rotation=45, ha='right')
        axes[0].set_title('Top 20 Senders by Sample Count', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Number of Samples')
        axes[0].set_xlabel('Sender ID')
        
        # Sender count distribution
        axes[1].hist(sender_counts.values, bins=50, color='gold', edgecolor='black', alpha=0.7)
        axes[1].set_title('Distribution of Samples per Sender', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Number of Samples')
        axes[1].set_ylabel('Number of Senders')
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sender_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _check_class_imbalance(self) -> dict:
        """Check for class imbalance."""
        class_counts = self.df['label'].value_counts()
        imbalance_ratio = class_counts.max() / class_counts.min()
        
        print(f"\nClass Imbalance Ratio (max/min): {imbalance_ratio:.2f}")
        print(f"Majority class: {class_counts.index[0]} ({class_counts.iloc[0]} samples)")
        print(f"Minority class: {class_counts.index[-1]} ({class_counts.iloc[-1]} samples)")
        
        return {
            'imbalance_ratio': imbalance_ratio,
            'majority_class': class_counts.index[0],
            'minority_class': class_counts.index[-1],
            'is_imbalanced': imbalance_ratio > 1.5
        }
    
    def prepare_features(self) -> tuple:
        """Prepare features for modeling."""
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING")
        print("=" * 60)
        
        # Create features
        df_features = self.df.copy()
        
        # Encode target
        df_features['label_encoded'] = self.label_encoder.fit_transform(df_features['label'])
        self.class_names = self.label_encoder.classes_
        print(f"Classes encoded: {dict(zip(self.class_names, range(len(self.class_names))))}")
        
        # Feature engineering
        # 1. Sender statistics
        sender_stats = df_features.groupby('sender_id').agg(
            sender_total_samples=('label', 'count'),
            sender_unique_classes=('label', 'nunique'),
            sender_kids_ratio=('kids', lambda x: (x == True).mean())
        ).reset_index()
        
        df_features = df_features.merge(sender_stats, on='sender_id', how='left')
        
        # 2. Class-level statistics (global)
        class_stats = df_features.groupby('label').agg(
            class_total_samples=('label', 'count'),
            class_kids_ratio=('kids', lambda x: (x == True).mean())
        ).reset_index()
        class_stats.columns = ['label', 'class_total_samples', 'class_kids_ratio']
        df_features = df_features.merge(class_stats, on='label', how='left')
        
        # 3. Kids as numeric
        df_features['kids_numeric'] = df_features['kids'].astype(int)
        
        # 4. Image ID features (hash-based for pseudo-random features)
        df_features['image_hash'] = df_features['image'].apply(lambda x: hash(x) % 10000)
        df_features['image_hash_norm'] = df_features['image_hash'] / 10000.0
        
        # Select features for modeling
        feature_cols = [
            'sender_id', 'kids_numeric', 'sender_total_samples', 
            'sender_unique_classes', 'sender_kids_ratio',
            'class_total_samples', 'class_kids_ratio',
            'image_hash_norm'
        ]
        
        X = df_features[feature_cols].copy()
        y = df_features['label_encoded'].copy()
        
        print(f"Feature columns: {feature_cols}")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Train-test split with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTrain set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self) -> dict:
        """Train multiple models and compare performance."""
        print("\n" + "=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)
        
        # Define models to test (reduced for speed)
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        }
        
        # Scale features for models that need it
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # For tree-based models, use original features
        X_train_orig = self.X_train.values
        X_test_orig = self.X_test.values
        
        results = {}
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced CV folds
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled features for linear models, original for tree-based
            if name in ['Logistic Regression']:
                X_tr, X_te = X_train_scaled, X_test_scaled
            else:
                X_tr, X_te = X_train_orig, X_test_orig
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_tr, self.y_train, cv=cv, scoring='accuracy', n_jobs=-1)
            
            # Train on full training set
            model.fit(X_tr, self.y_train)
            
            # Predictions
            y_pred = model.predict(X_te)
            y_pred_proba = model.predict_proba(X_te) if hasattr(model, 'predict_proba') else None
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.y_test, y_pred, average='weighted', zero_division=0
            )
            
            results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': accuracy,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': f1,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'scaler': scaler if name in ['Logistic Regression'] else None
            }
            
            print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            print(f"  Test Accuracy: {accuracy:.4f}")
            print(f"  Test F1 (weighted): {f1:.4f}")
        
        self.models = results
        return results
    
    def evaluate_models(self) -> pd.DataFrame:
        """Comprehensive model evaluation."""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        # Create comparison dataframe
        comparison_data = []
        for name, result in self.models.items():
            comparison_data.append({
                'Model': name,
                'CV Accuracy (Mean)': result['cv_mean'],
                'CV Accuracy (Std)': result['cv_std'],
                'Test Accuracy': result['test_accuracy'],
                'Test Precision': result['test_precision'],
                'Test Recall': result['test_recall'],
                'Test F1': result['test_f1']
            })
        
        comparison_df = pd.DataFrame(comparison_data).sort_values('Test Accuracy', ascending=False)
        print("\nModel Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_df.to_csv(self.output_dir / 'model_comparison.csv', index=False)
        
        # Select best model
        self.best_model_name = comparison_df.iloc[0]['Model']
        self.best_model = self.models[self.best_model_name]['model']
        print(f"\nBest Model: {self.best_model_name}")
        
        # Detailed evaluation for best model
        self._detailed_evaluation(self.best_model_name)
        
        # Plot model comparison
        self._plot_model_comparison(comparison_df)
        
        return comparison_df
    
    def _detailed_evaluation(self, model_name: str):
        """Detailed evaluation for a specific model."""
        result = self.models[model_name]
        y_pred = result['y_pred']
        y_pred_proba = result['y_pred_proba']
        
        print(f"\n{'='*60}")
        print(f"DETAILED EVALUATION: {model_name}")
        print(f"{'='*60}")
        
        # Classification report
        print("\nClassification Report:")
        report = classification_report(
            self.y_test, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        print(classification_report(self.y_test, y_pred, target_names=self.class_names, zero_division=0))
        
        # Save classification report
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(self.output_dir / f'classification_report_{model_name.replace(" ", "_").lower()}.csv')
        
        # Confusion Matrix
        self._plot_confusion_matrix(y_pred, model_name)
        
        # Per-class metrics visualization
        self._plot_per_class_metrics(report, model_name)
        
        # ROC Curves (if probabilities available)
        if y_pred_proba is not None:
            self._plot_roc_curves(y_pred_proba, model_name)
        
        # Feature importance (for tree-based models)
        if hasattr(result['model'], 'feature_importances_'):
            self._plot_feature_importance(result['model'], model_name)
    
    def _plot_confusion_matrix(self, y_pred, model_name: str):
        """Plot confusion matrix."""
        cm = confusion_matrix(self.y_test, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names, ax=axes[0])
        axes[0].set_title(f'Confusion Matrix (Counts) - {model_name}', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names, ax=axes[1])
        axes[1].set_title(f'Confusion Matrix (Normalized) - {model_name}', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Interactive plotly version
        fig_plotly = px.imshow(
            cm_normalized, 
            labels=dict(x="Predicted", y="Actual", color="Proportion"),
            x=self.class_names, y=self.class_names,
            title=f'Normalized Confusion Matrix - {model_name}',
            color_continuous_scale='Blues',
            text_auto='.2f'
        )
        fig_plotly.write_html(self.output_dir / f'confusion_matrix_{model_name.replace(" ", "_").lower()}_interactive.html')
    
    def _plot_per_class_metrics(self, report: dict, model_name: str):
        """Plot per-class precision, recall, F1."""
        classes = [c for c in self.class_names if c in report]
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
        plt.savefig(self.output_dir / f'per_class_metrics_{model_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self, y_pred_proba, model_name: str):
        """Plot ROC curves for each class."""
        n_classes = len(self.class_names)
        
        # Binarize the output
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(self.y_test, classes=range(n_classes))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])
            ax.plot(fpr, tpr, label=f'{self.class_names[i]} (AUC = {roc_auc:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'roc_curves_{model_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, model, model_name: str):
        """Plot feature importance for tree-based models."""
        feature_names = self.X_train.columns
        importances = model.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(importances)), importances[indices], color='teal', edgecolor='black')
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        ax.set_title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Importance')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'feature_importance_{model_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_comparison(self, comparison_df: pd.DataFrame):
        """Plot model comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics = ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1']
        colors = ['steelblue', 'coral', 'mediumseagreen', 'gold']
        
        for idx, (metric, color) in enumerate(zip(metrics, colors)):
            ax = axes[idx // 2, idx % 2]
            bars = ax.bar(comparison_df['Model'], comparison_df[metric], color=color, edgecolor='black')
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylim(0, 1.05)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Model Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def hyperparameter_tuning(self, model_name: str = None) -> dict:
        """Perform hyperparameter tuning for the best model (simplified for speed)."""
        if model_name is None:
            model_name = self.best_model_name
        
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER TUNING: {model_name}")
        print(f"{'='*60}")
        
        # Simplified parameter grids for speed
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, None],
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.2],
                'max_depth': [3, 5],
            },
            'Logistic Regression': {
                'C': [1, 10],
                'solver': ['lbfgs'],
            },
            'SVM': {
                'C': [1, 10],
                'kernel': ['rbf'],
            }
        }
        
        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}. Skipping tuning.")
            return {}
        
        # Get the base model
        base_model = self.models[model_name]['model']
        scaler = self.models[model_name]['scaler']
        
        # Prepare data
        if scaler is not None:
            X_train_use = scaler.transform(self.X_train)
            X_test_use = scaler.transform(self.X_test)
        else:
            X_train_use = self.X_train.values
            X_test_use = self.X_test.values
        
        # Grid search - use fewer CV folds for speed
        grid_search = GridSearchCV(
            base_model.__class__(), 
            param_grids[model_name],
            cv=2,  # Minimal for speed
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train_use, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Evaluate tuned model
        tuned_model = grid_search.best_estimator_
        y_pred_tuned = tuned_model.predict(X_test_use)
        tuned_accuracy = accuracy_score(self.y_test, y_pred_tuned)
        
        print(f"Tuned test accuracy: {tuned_accuracy:.4f}")
        print(f"Original test accuracy: {self.models[model_name]['test_accuracy']:.4f}")
        print(f"Improvement: {tuned_accuracy - self.models[model_name]['test_accuracy']:.4f}")
        
        # Update best model if improved
        if tuned_accuracy > self.models[model_name]['test_accuracy']:
            self.models[model_name]['model'] = tuned_model
            self.models[model_name]['test_accuracy'] = tuned_accuracy
            self.best_model = tuned_model
            print("Best model updated with tuned version!")
        
        return grid_search.best_params_
    
    def save_model(self, model_name: str = None):
        """Save the best model and preprocessing objects."""
        if model_name is None:
            model_name = self.best_model_name
        
        model_data = {
            'model': self.models[model_name]['model'],
            'label_encoder': self.label_encoder,
            'scaler': self.models[model_name]['scaler'],
            'feature_names': self.X_train.columns.tolist(),
            'class_names': self.class_names.tolist(),
            'model_name': model_name
        }
        
        model_path = self.output_dir.parent / 'models' / f'best_fashion_classifier_{model_name.replace(" ", "_").lower()}.joblib'
        joblib.dump(model_data, model_path)
        print(f"\nModel saved to: {model_path}")
        
        return model_path
    
    def predict_sample(self, sample_data: dict, model_name: str = None) -> dict:
        """Make prediction on a single sample."""
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models[model_name]['model']
        scaler = self.models[model_name]['scaler']
        
        # Create feature vector (simplified - in practice you'd need full feature engineering)
        # This is a placeholder for demonstration
        feature_vector = np.array([[
            sample_data.get('sender_id', 0),
            sample_data.get('kids_numeric', 0),
            sample_data.get('sender_total_samples', 1),
            sample_data.get('sender_unique_classes', 1),
            sample_data.get('sender_kids_ratio', 0),
            sample_data.get('class_total_samples', 100),
            sample_data.get('class_kids_ratio', 0),
            sample_data.get('image_hash_norm', 0.5)
        ]])
        
        if scaler is not None:
            feature_vector = scaler.transform(feature_vector)
        
        pred = model.predict(feature_vector)[0]
        pred_proba = model.predict_proba(feature_vector)[0] if hasattr(model, 'predict_proba') else None
        
        result = {
            'predicted_class': self.class_names[pred],
            'predicted_class_id': int(pred)
        }
        
        if pred_proba is not None:
            result['probabilities'] = dict(zip(self.class_names, pred_proba.tolist()))
            result['confidence'] = float(pred_proba.max())
        
        return result
    
    def generate_report(self):
        """Generate a comprehensive HTML report."""
        print("\n" + "=" * 60)
        print("GENERATING REPORT")
        print("=" * 60)
        
        report_path = self.output_dir / 'fashion_classification_report.html'
        
        # Create comparison table HTML
        comparison_data = []
        for name, result in self.models.items():
            comparison_data.append({
                'Model': name,
                'CV Accuracy': f"{result['cv_mean']:.4f} (±{result['cv_std']:.4f})",
                'Test Accuracy': f"{result['test_accuracy']:.4f}",
                'Test Precision': f"{result['test_precision']:.4f}",
                'Test Recall': f"{result['test_recall']:.4f}",
                'Test F1': f"{result['test_f1']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fashion Product Classification Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metric {{ display: inline-block; background: #ecf0f1; padding: 15px; margin: 10px; border-radius: 5px; min-width: 150px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ font-size: 14px; color: #7f8c8d; }}
                .best-model {{ background: #27ae60; color: white; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>[REPORT] Fashion Product Classification Report</h1>
                
                <h2>Dataset Overview</h2>
                <div class="metric">
                    <div class="metric-value">{len(self.df)}</div>
                    <div class="metric-label">Total Samples</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{self.df['label'].nunique()}</div>
                    <div class="metric-label">Classes</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{self.df['sender_id'].nunique()}</div>
                    <div class="metric-label">Unique Senders</div>
                </div>
                
                <h2>Class Distribution</h2>
                <table>
                    <tr><th>Class</th><th>Count</th><th>Percentage</th></tr>
        """
        
        class_counts = self.df['label'].value_counts()
        for cls, count in class_counts.items():
            pct = count / len(self.df) * 100
            html_report += f"<tr><td>{cls}</td><td>{count}</td><td>{pct:.1f}%</td></tr>"
        
        html_report += f"""
                </table>
                
                <h2>Model Comparison</h2>
                <div class="best-model">
                    <strong>[BEST] Best Model: {self.best_model_name}</strong> with Test Accuracy: {self.models[self.best_model_name]['test_accuracy']:.4f}
                </div>
                {comparison_df.to_html(index=False, classes='comparison-table')}
                
                <h2>Best Model Details: {self.best_model_name}</h2>
                <div class="metric">
                    <div class="metric-value">{self.models[self.best_model_name]['test_accuracy']:.4f}</div>
                    <div class="metric-label">Test Accuracy</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{self.models[self.best_model_name]['test_f1']:.4f}</div>
                    <div class="metric-label">Weighted F1 Score</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{self.models[self.best_model_name]['cv_mean']:.4f}</div>
                    <div class="metric-label">CV Accuracy (Mean)</div>
                </div>
                
                <h2>Visualizations</h2>
                <p>Generated visualizations are available in the reports directory:</p>
                <ul>
                    <li>Class Distribution: <code>class_distribution.png</code></li>
                    <li>Kids Distribution: <code>kids_distribution.png</code></li>
                    <li>Sender Analysis: <code>sender_analysis.png</code></li>
                    <li>Model Comparison: <code>model_comparison.png</code></li>
                    <li>Confusion Matrix: <code>confusion_matrix_{self.best_model_name.replace(" ", "_").lower()}.png</code></li>
                    <li>Per-Class Metrics: <code>per_class_metrics_{self.best_model_name.replace(" ", "_").lower()}.png</code></li>
        """
        
        if hasattr(self.models[self.best_model_name]['model'], 'feature_importances_'):
            html_report += f'<li>Feature Importance: <code>feature_importance_{self.best_model_name.replace(" ", "_").lower()}.png</code></li>'
        
        if self.models[self.best_model_name]['y_pred_proba'] is not None:
            html_report += f'<li>ROC Curves: <code>roc_curves_{self.best_model_name.replace(" ", "_").lower()}.png</code></li>'
        
        html_report += """
                </ul>
                
                <h2>Conclusion</h2>
                <p>This report summarizes the fashion product classification pipeline including data exploration, 
                feature engineering, model training, and evaluation. The best performing model has been saved 
                and can be used for predictions on new data.</p>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_report)
        
        print(f"Report saved to: {report_path}")
        
        return report_path


def main():
    """Main execution pipeline."""
    print("[START] Starting Fashion Product Classification Pipeline")
    print("=" * 60)
    
    # Initialize classifier - use absolute path or relative to working directory
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "clothing_dataset.csv")
    output_dir = os.path.join(base_dir, "reports")
    
    classifier = FashionClassifier(
        data_path=data_path,
        output_dir=output_dir
    )
    
    # 1. Load data (sample 10000 for speed)
    classifier.load_data(sample_size=10000)
    
    # 2. Perform EDA
    eda_results = classifier.perform_eda()
    
    # 3. Prepare features
    classifier.prepare_features()
    
    # 4. Train models
    classifier.train_models()
    
    # 5. Evaluate models
    comparison_df = classifier.evaluate_models()
    
    # 6. Hyperparameter tuning for best model (simplified)
    classifier.hyperparameter_tuning()
    
    # 7. Save best model
    classifier.save_model()
    
    # 8. Generate report
    classifier.generate_report()
    
    print("\n" + "=" * 60)
    print("[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nBest Model: {classifier.best_model_name}")
    print(f"Test Accuracy: {classifier.models[classifier.best_model_name]['test_accuracy']:.4f}")
    print(f"Reports saved to: fashion_classification/reports/")
    print(f"Model saved to: fashion_classification/models/")
    
    return classifier


if __name__ == "__main__":
    classifier = main()