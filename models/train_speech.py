"""
Speech Model Training Script
Czech UDPR Dataset - 3-Class Classification (HC vs RBD vs PD)

Model: Ensemble (Random Forest + XGBoost + SVM + Gradient Boosting)
Task:  3-class classification
       0 = HC  (Healthy Controls)
       1 = RBD (At-Risk / REM Sleep Behavior Disorder)
       2 = PD  (Parkinson's Disease)

Why Ensemble instead of Deep Learning?
- Only 130 subjects (too small for deep learning)
- Ensemble models excel on small tabular data
- More robust, less overfitting
- Faster to train and evaluate

Validation Strategy:
- Stratified 5-Fold Cross Validation
- Ensures each fold has same class distribution
- Subject-level split (each subject in exactly one fold)

Author: Your Name
Date: 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
from pathlib import Path

# ML Libraries
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    cross_validate
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score
)
from sklearn.preprocessing import label_binarize
import xgboost as xgb

warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    'data_path':    'data/speech/processed/speech_features_scaled.csv',
    'models_dir':   'models/saved',
    'results_dir':  'results',
    'random_state': 42,
    'n_folds':      5,
    'n_classes':    3,
    'class_names':  ['HC (Healthy)', 'RBD (At-Risk)', 'PD (Parkinson\'s)'],
    'class_short':  ['HC', 'RBD', 'PD'],
}


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────
def load_data(data_path):
    """
    Load preprocessed speech features

    Returns:
        X: Feature matrix (130, 24)
        y: Labels (0=HC, 1=RBD, 2=PD)
        feature_names: List of feature names
    """
    print("Loading preprocessed speech data...")

    df = pd.read_csv(data_path)

    # Separate features from metadata
    meta_cols = ['participant_code', 'label', 'label_name']
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].values
    y = df['label'].values
    feature_names = feature_cols

    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {np.unique(y)}")

    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        name = CONFIG['class_names'][label]
        print(f"  {name}: {count} ({count/len(y)*100:.1f}%)")

    return X, y, feature_names


# ─────────────────────────────────────────────
#  MODEL DEFINITIONS
# ─────────────────────────────────────────────
def build_models():
    rs = CONFIG['random_state']

    models = {

        # Simpler RF - less overfitting
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=5,          # ← Limit depth to prevent memorizing
            min_samples_split=10, # ← Need more samples to split
            min_samples_leaf=5,   # ← Need more samples in leaves
            max_features='sqrt',
            class_weight='balanced',
            random_state=rs,
            n_jobs=-1
        ),

        # XGBoost with stronger regularization
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,     # ← Fewer trees
            learning_rate=0.05,
            max_depth=3,          # ← Shallower trees
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,        # ← L1 regularization
            reg_lambda=2.0,       # ← L2 regularization
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=rs,
            n_jobs=-1,
            verbosity=0
        ),

        # SVM - naturally good with small data
        'SVM': SVC(
            kernel='rbf',
            C=1.0,                # ← Reduced from 10 to 1
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=rs
        ),

        # Logistic Regression - simple linear model
        'Logistic Regression': LogisticRegression(
            C=0.1,                # ← Strong regularization
            class_weight='balanced',
            max_iter=2000,
            random_state=rs,
            solver='lbfgs'
        ),
    }

    return models

def build_ensemble(models):
    """
    Build soft voting ensemble from individual models

    Soft voting: averages predicted probabilities
    Better than hard voting for calibrated models

    Args:
        models: dict of model_name: model_object

    Returns:
        VotingClassifier ensemble
    """
    estimators = [(name, model) for name, model in models.items()]

    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',
        n_jobs=-1
    )

    return ensemble


# ─────────────────────────────────────────────
#  CROSS VALIDATION
# ─────────────────────────────────────────────
def run_cross_validation(models, ensemble, X, y):
    """
    Run stratified k-fold cross validation

    Evaluates each individual model AND the ensemble
    to show that fusion improves performance

    Args:
        models: dict of individual models
        ensemble: VotingClassifier
        X: Feature matrix
        y: Labels

    Returns:
        results_df: DataFrame with CV results
    """
    print(f"\nRunning {CONFIG['n_folds']}-Fold Stratified Cross Validation...")
    print("-" * 60)

    cv = StratifiedKFold(
        n_splits=CONFIG['n_folds'],
        shuffle=True,
        random_state=CONFIG['random_state']
    )

    results = {}

    # Evaluate each individual model
    all_models = dict(models)
    all_models['ENSEMBLE'] = ensemble

    for name, model in all_models.items():
        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            scoring=['accuracy', 'f1_macro'],
            return_train_score=True,
            n_jobs=-1
        )

        results[name] = {
            'CV Accuracy (mean)': cv_results['test_accuracy'].mean(),
            'CV Accuracy (std)':  cv_results['test_accuracy'].std(),
            'CV F1 Macro (mean)': cv_results['test_f1_macro'].mean(),
            'CV F1 Macro (std)':  cv_results['test_f1_macro'].std(),
            'Train Accuracy':     cv_results['train_accuracy'].mean(),
        }

        marker = ' ← ENSEMBLE' if name == 'ENSEMBLE' else ''
        print(f"  {name:<25} "
              f"Acc: {results[name]['CV Accuracy (mean)']:.4f} "
              f"(±{results[name]['CV Accuracy (std)']:.4f}) "
              f"F1: {results[name]['CV F1 Macro (mean)']:.4f}"
              f"{marker}")

    results_df = pd.DataFrame(results).T
    return results_df


# ─────────────────────────────────────────────
#  FINAL TRAINING & EVALUATION
# ─────────────────────────────────────────────
def train_and_evaluate(ensemble, X, y):
    """
    Train final ensemble on all data and evaluate

    Note: We use CV for unbiased evaluation.
    Final model trained on ALL data for deployment.

    Args:
        ensemble: VotingClassifier
        X: Feature matrix
        y: Labels

    Returns:
        trained ensemble
        predictions and probabilities
    """
    print("\nTraining final ensemble on all data...")

    ensemble.fit(X, y)

    y_pred = ensemble.predict(X)
    y_prob = ensemble.predict_proba(X)

    # Metrics
    accuracy = accuracy_score(y, y_pred)

    # ROC-AUC (one-vs-rest, macro average)
    y_bin = label_binarize(y, classes=[0, 1, 2])
    roc_auc = roc_auc_score(
        y_bin, y_prob,
        multi_class='ovr',
        average='macro'
    )

    f1 = f1_score(y, y_pred, average='macro')

    print(f"  Training Accuracy: {accuracy:.4f}")
    print(f"  ROC-AUC (macro):   {roc_auc:.4f}")
    print(f"  F1 Score (macro):  {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y, y_pred,
        target_names=CONFIG['class_names']
    ))

    return ensemble, y_pred, y_prob


# ─────────────────────────────────────────────
#  VISUALIZATIONS
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, save_dir):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d',
        cmap='Blues',
        xticklabels=CONFIG['class_short'],
        yticklabels=CONFIG['class_short'],
        annot_kws={'size': 14}
    )
    plt.title('Speech Model - Confusion Matrix (3-Class)', fontsize=14, pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    path = os.path.join(save_dir, 'speech_confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Confusion matrix: {path}")


def plot_roc_curves(y_true, y_prob, save_dir):
    """Plot ROC curves for each class (one-vs-rest)"""
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    colors = ['#2196F3', '#FF9800', '#F44336']

    plt.figure(figsize=(9, 6))

    for i, (class_name, color) in enumerate(
        zip(CONFIG['class_names'], colors)
    ):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{class_name} (AUC = {auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Speech Model - ROC Curves (One-vs-Rest)', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, 'speech_roc_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ ROC curves: {path}")


def plot_feature_importance(ensemble, feature_names, save_dir):
    """Plot top 15 feature importances from Random Forest"""

    # Extract RF from ensemble
    rf_model = None
    for name, estimator in ensemble.named_estimators_.items():
        if 'random' in name.lower() or 'forest' in name.lower():
            rf_model = estimator
            break

    if rf_model is None or not hasattr(rf_model, 'feature_importances_'):
        print("  ⚠️  Could not extract feature importances")
        return

    importances = rf_model.feature_importances_
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Clean feature names for display
    feat_imp['display_name'] = feat_imp['feature'].apply(
        lambda x: x.replace('reading_', 'R: ')
                   .replace('monologue_', 'M: ')[:45]
    )

    plt.figure(figsize=(10, 8))
    colors = ['#1976D2' if 'R:' in n else '#388E3C'
              for n in feat_imp['display_name'].head(15)]
    bars = plt.barh(
        feat_imp['display_name'].head(15)[::-1],
        feat_imp['importance'].head(15)[::-1],
        color=colors[::-1], alpha=0.85
    )
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title('Top 15 Speech Features\n(Blue=Reading, Green=Monologue)',
              fontsize=13)
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, 'speech_feature_importance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Feature importance: {path}")

    # Save as CSV too
    feat_imp_path = os.path.join(
        save_dir.replace('figures', 'tables'),
        'speech_feature_importance.csv'
    )
    os.makedirs(os.path.dirname(feat_imp_path), exist_ok=True)
    feat_imp.to_csv(feat_imp_path, index=False)
    print(f"  ✅ Feature importance CSV: {feat_imp_path}")


def plot_class_probabilities(y_true, y_prob, save_dir):
    """Plot predicted probability distributions per class"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = ['#2196F3', '#FF9800', '#F44336']

    for i, (class_name, color) in enumerate(
        zip(CONFIG['class_names'], colors)
    ):
        ax = axes[i]
        for true_class in range(3):
            mask = y_true == true_class
            ax.hist(
                y_prob[mask, i],
                bins=15, alpha=0.6,
                label=CONFIG['class_short'][true_class],
                density=True
            )

        ax.set_title(f'P({CONFIG["class_short"][i]})', fontsize=12)
        ax.set_xlabel('Predicted Probability', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Speech Model - Probability Distributions per Class',
                 fontsize=13, y=1.02)
    plt.tight_layout()

    path = os.path.join(save_dir, 'speech_probability_distributions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Probability distributions: {path}")


def plot_cv_comparison(cv_results_df, save_dir):
    """Plot cross-validation accuracy comparison across models"""
    models_list = cv_results_df.index.tolist()
    accs = cv_results_df['CV Accuracy (mean)'].values
    stds = cv_results_df['CV Accuracy (std)'].values

    colors = ['#F44336' if m == 'ENSEMBLE' else '#90CAF9'
              for m in models_list]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(models_list, accs, yerr=stds,
                   color=colors, alpha=0.85,
                   capsize=5, edgecolor='white', linewidth=0.5)

    plt.ylabel('CV Accuracy', fontsize=12)
    plt.title('Speech Model - Cross-Validation Comparison', fontsize=13)
    plt.xticks(rotation=15, ha='right', fontsize=10)
    plt.ylim(0, 1.0)
    plt.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, acc in zip(bars, accs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{acc:.3f}',
            ha='center', va='bottom', fontsize=9
        )

    plt.tight_layout()
    path = os.path.join(save_dir, 'speech_cv_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ CV comparison: {path}")


# ─────────────────────────────────────────────
#  SAVE RESULTS
# ─────────────────────────────────────────────
def save_results(cv_results_df, y_true, y_pred, y_prob, save_dir):
    """Save all results to CSV tables"""
    os.makedirs(save_dir, exist_ok=True)

    # CV results table
    cv_path = os.path.join(save_dir, 'speech_cv_results.csv')
    cv_results_df.to_csv(cv_path)
    print(f"  ✅ CV results: {cv_path}")

    # Per-class metrics
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    per_class = []
    for i, name in enumerate(CONFIG['class_names']):
        auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
        per_class.append({
            'Class': name,
            'ROC-AUC': round(auc, 4),
            'Samples': int((y_true == i).sum())
        })

    per_class_df = pd.DataFrame(per_class)
    per_class_path = os.path.join(save_dir, 'speech_per_class_metrics.csv')
    per_class_df.to_csv(per_class_path, index=False)
    print(f"  ✅ Per-class metrics: {per_class_path}")

    # Summary
    summary = {
    'Metric': [
        'CV Accuracy (mean)',
        'CV Accuracy (std)',
        'CV F1 Macro (mean)',
        'CV F1 Macro (std)',
        'ROC-AUC (macro OvR)',
    ],
    'Value': [
        round(cv_results_df.loc['ENSEMBLE', 'CV Accuracy (mean)'], 4),
        round(cv_results_df.loc['ENSEMBLE', 'CV Accuracy (std)'], 4),
        round(cv_results_df.loc['ENSEMBLE', 'CV F1 Macro (mean)'], 4),
        round(cv_results_df.loc['ENSEMBLE', 'CV F1 Macro (std)'], 4),
        round(cv_results_df.loc['ENSEMBLE', 'CV F1 Macro (mean)'], 4),
    ]
}

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(save_dir, 'speech_model_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"  ✅ Model summary: {summary_path}")


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    print("="*80)
    print("SPEECH MODEL TRAINING - 3-Class Classification (HC / RBD / PD)")
    print("="*80)

    # Create output directories
    figures_dir = os.path.join(CONFIG['results_dir'], 'figures')
    tables_dir  = os.path.join(CONFIG['results_dir'], 'tables')
    models_dir  = CONFIG['models_dir']

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # ── Step 1: Load Data ──────────────────────────────────────
    print("\n" + "─"*40)
    print("STEP 1: Loading Data")
    print("─"*40)
    X, y, feature_names = load_data(CONFIG['data_path'])

    # ── Step 2: Build Models ───────────────────────────────────
    print("\n" + "─"*40)
    print("STEP 2: Building Models")
    print("─"*40)
    models = build_models()
    ensemble = build_ensemble(models)
    print(f"  Individual models: {list(models.keys())}")
    print(f"  Ensemble: Soft Voting Classifier")

    # ── Step 3: Cross Validation ───────────────────────────────
    print("\n" + "─"*40)
    print("STEP 3: Cross Validation")
    print("─"*40)
    cv_results_df = run_cross_validation(models, ensemble, X, y)

    # ── Step 4: Train Final Model ──────────────────────────────
    print("\n" + "─"*40)
    print("STEP 4: Training Final Model")
    print("─"*40)
    ensemble, y_pred, y_prob = train_and_evaluate(ensemble, X, y)

    # ── Step 5: Visualizations ─────────────────────────────────
    print("\n" + "─"*40)
    print("STEP 5: Generating Visualizations")
    print("─"*40)
    plot_confusion_matrix(y, y_pred, figures_dir)
    plot_roc_curves(y, y_prob, figures_dir)
    plot_feature_importance(ensemble, feature_names, figures_dir)
    plot_class_probabilities(y, y_prob, figures_dir)
    plot_cv_comparison(cv_results_df, figures_dir)

    # ── Step 6: Save Results ───────────────────────────────────
    print("\n" + "─"*40)
    print("STEP 6: Saving Results")
    print("─"*40)
    save_results(cv_results_df, y, y_pred, y_prob, tables_dir)

    # ── Step 7: Save Model ─────────────────────────────────────
    print("\n" + "─"*40)
    print("STEP 7: Saving Model")
    print("─"*40)
    model_path = os.path.join(models_dir, 'speech_model.pkl')
    joblib.dump(ensemble, model_path)
    print(f"  ✅ Model saved: {model_path}")

    # ── Final Summary ──────────────────────────────────────────
    print("\n" + "="*80)
    print("✅ SPEECH MODEL TRAINING COMPLETE!")
    print("="*80)

    y_bin = label_binarize(y, classes=[0, 1, 2])
    roc_auc = roc_auc_score(y_bin, y_prob, multi_class='ovr', average='macro')

    print(f"""
  Dataset:          Czech UDPR (130 subjects)
  Task:             3-Class (HC / RBD / PD)
  Model:            Soft Voting Ensemble

  CV Accuracy:      {cv_results_df.loc['ENSEMBLE', 'CV Accuracy (mean)']:.4f} \
(±{cv_results_df.loc['ENSEMBLE', 'CV Accuracy (std)']:.4f})
  CV F1 Macro:      {cv_results_df.loc['ENSEMBLE', 'CV F1 Macro (mean)']:.4f}
  ROC-AUC (macro):  {roc_auc:.4f}

  Model saved:      {model_path}
  Figures saved:    {figures_dir}/
  Tables saved:     {tables_dir}/

  Next steps:
    → Run models/train_gait.py
    → Run models/train_handwriting.py
    """)


if __name__ == "__main__":
    main()