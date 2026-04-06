"""
Gait Model Training Script
PhysioNet Gait Dataset - Binary Classification (HC vs PD)

Model: Ensemble (Random Forest + XGBoost + SVM)
Task:  Binary classification
       0 = HC (Healthy Controls)
       1 = PD (Parkinson's Disease)

Validation Strategy:
- Stratified 5-Fold Cross Validation
- Subject-wise split (no data leakage)

Author: Your Name
Date: 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score
)
import xgboost as xgb

warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    'data_path':    'data/gait/processed/gait_features_scaled.csv',
    'models_dir':   'models/saved',
    'results_dir':  'results',
    'random_state': 42,
    'n_folds':      5,
    'class_names':  ['HC (Healthy)', 'PD (Parkinson\'s)'],
    'class_short':  ['HC', 'PD'],
}


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────
def load_data(data_path):
    """
    Load preprocessed gait features

    Returns:
        X: Feature matrix (165, 24)
        y: Labels (0=HC, 1=PD)
        feature_names: List of feature names
    """
    print("Loading preprocessed gait data...")

    df = pd.read_csv(data_path)

    # Separate features from metadata
    meta_cols = ['subject_id', 'label', 'label_name']
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].values
    y = df['label'].values
    feature_names = feature_cols

    print(f"  Subjects:  {X.shape[0]}")
    print(f"  Features:  {X.shape[1]}")

    healthy = (y == 0).sum()
    pd_count = (y == 1).sum()
    print(f"  HC (Healthy):       {healthy} ({healthy/len(y)*100:.1f}%)")
    print(f"  PD (Parkinson's):   {pd_count} ({pd_count/len(y)*100:.1f}%)")

    return X, y, feature_names


# ─────────────────────────────────────────────
#  MODEL DEFINITIONS
# ─────────────────────────────────────────────
def build_models():
    """
    Build individual models for the ensemble

    Tuned for:
    - Binary classification
    - 165 subjects (small-medium dataset)
    - Slightly imbalanced (43% HC / 57% PD)
    """
    rs = CONFIG['random_state']

    models = {

        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced',
            random_state=rs,
            n_jobs=-1
        ),

        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            scale_pos_weight=72/93,  # handle class imbalance
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=rs,
            n_jobs=-1,
            verbosity=0
        ),

        'SVM': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=rs
        ),

        'Logistic Regression': LogisticRegression(
            C=0.1,
            class_weight='balanced',
            max_iter=2000,
            random_state=rs,
            solver='lbfgs'
        ),
    }

    return models


def build_ensemble(models):
    """Build soft voting ensemble"""
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
    Run stratified 5-fold cross validation

    Evaluates each individual model AND the ensemble
    """
    print(f"\nRunning {CONFIG['n_folds']}-Fold Stratified Cross Validation...")
    print("-" * 60)

    cv = StratifiedKFold(
        n_splits=CONFIG['n_folds'],
        shuffle=True,
        random_state=CONFIG['random_state']
    )

    results = {}
    all_models = dict(models)
    all_models['ENSEMBLE'] = ensemble

    for name, model in all_models.items():
        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            scoring=['accuracy', 'f1', 'roc_auc'],
            return_train_score=True,
            n_jobs=-1
        )

        results[name] = {
            'CV Accuracy (mean)': cv_results['test_accuracy'].mean(),
            'CV Accuracy (std)':  cv_results['test_accuracy'].std(),
            'CV F1 (mean)':       cv_results['test_f1'].mean(),
            'CV F1 (std)':        cv_results['test_f1'].std(),
            'CV ROC-AUC (mean)':  cv_results['test_roc_auc'].mean(),
            'CV ROC-AUC (std)':   cv_results['test_roc_auc'].std(),
            'Train Accuracy':     cv_results['train_accuracy'].mean(),
        }

        marker = ' ← ENSEMBLE' if name == 'ENSEMBLE' else ''
        print(f"  {name:<25} "
              f"Acc: {results[name]['CV Accuracy (mean)']:.4f} "
              f"(±{results[name]['CV Accuracy (std)']:.4f})  "
              f"ROC-AUC: {results[name]['CV ROC-AUC (mean)']:.4f}"
              f"{marker}")

    results_df = pd.DataFrame(results).T
    return results_df


# ─────────────────────────────────────────────
#  FINAL TRAINING & EVALUATION
# ─────────────────────────────────────────────
def train_and_evaluate(ensemble, X, y):
    """Train final ensemble and evaluate on training data"""
    print("\nTraining final ensemble on all data...")

    ensemble.fit(X, y)

    y_pred = ensemble.predict(X)
    y_prob = ensemble.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    roc_auc  = roc_auc_score(y, y_prob)
    f1       = f1_score(y, y_pred)

    print(f"  Training Accuracy: {accuracy:.4f}")
    print(f"  ROC-AUC:           {roc_auc:.4f}")
    print(f"  F1 Score:          {f1:.4f}")

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

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt='d',
        cmap='Blues',
        xticklabels=CONFIG['class_short'],
        yticklabels=CONFIG['class_short'],
        annot_kws={'size': 16}
    )
    plt.title('Gait Model - Confusion Matrix', fontsize=14, pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    path = os.path.join(save_dir, 'gait_confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Confusion matrix: {path}")


def plot_roc_curve(y_true, y_prob, cv_auc, save_dir):
    """Plot ROC curve with CV AUC reference"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    train_auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#2196F3', lw=2,
             label=f'Train ROC (AUC = {train_auc:.3f})')
    plt.axhline(y=cv_auc, color='#FF9800', lw=2, linestyle='--',
                label=f'CV ROC-AUC = {cv_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Gait Model - ROC Curve (HC vs PD)', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, 'gait_roc_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ ROC curve: {path}")


def plot_feature_importance(ensemble, feature_names, save_dir):
    """Plot top 15 feature importances from Random Forest"""
    rf_model = None
    for name, estimator in ensemble.named_estimators_.items():
        if 'random' in name.lower():
            rf_model = estimator
            break

    if rf_model is None:
        return

    importances = rf_model.feature_importances_
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 8))
    plt.barh(
        feat_imp['feature'].head(15)[::-1],
        feat_imp['importance'].head(15)[::-1],
        color='#1976D2', alpha=0.85
    )
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title('Top 15 Gait Features (Random Forest)', fontsize=13)
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, 'gait_feature_importance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Feature importance: {path}")

    # Save CSV
    tables_dir = save_dir.replace('figures', 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    feat_imp.to_csv(
        os.path.join(tables_dir, 'gait_feature_importance.csv'),
        index=False
    )
    print(f"  ✅ Feature importance CSV saved")


def plot_cv_comparison(cv_results_df, save_dir):
    """Plot CV ROC-AUC comparison across models"""
    models_list = cv_results_df.index.tolist()
    aucs = cv_results_df['CV ROC-AUC (mean)'].values
    stds = cv_results_df['CV ROC-AUC (std)'].values

    colors = ['#F44336' if m == 'ENSEMBLE' else '#90CAF9'
              for m in models_list]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(
        models_list, aucs,
        yerr=stds, color=colors,
        alpha=0.85, capsize=5,
        edgecolor='white', linewidth=0.5
    )

    plt.ylabel('CV ROC-AUC', fontsize=12)
    plt.title('Gait Model - CV ROC-AUC Comparison', fontsize=13)
    plt.xticks(rotation=15, ha='right', fontsize=10)
    plt.ylim(0, 1.0)
    plt.grid(True, axis='y', alpha=0.3)

    for bar, auc in zip(bars, aucs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{auc:.3f}',
            ha='center', va='bottom', fontsize=9
        )

    plt.tight_layout()
    path = os.path.join(save_dir, 'gait_cv_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ CV comparison: {path}")


def plot_probability_distribution(y_true, y_prob, save_dir):
    """Plot predicted probability distribution for each class"""
    plt.figure(figsize=(9, 5))

    plt.hist(y_prob[y_true == 0], bins=20, alpha=0.6,
             color='#2196F3', label='HC (Healthy)', density=True)
    plt.hist(y_prob[y_true == 1], bins=20, alpha=0.6,
             color='#F44336', label='PD (Parkinson\'s)', density=True)

    plt.axvline(x=0.5, color='black', linestyle='--',
                lw=1.5, label='Decision boundary (0.5)')
    plt.xlabel('Predicted P(PD)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Gait Model - Predicted Probability Distribution', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, 'gait_probability_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Probability distribution: {path}")


# ─────────────────────────────────────────────
#  SAVE RESULTS
# ─────────────────────────────────────────────
def save_results(cv_results_df, y_true, y_pred, y_prob, save_dir):
    """Save all results to CSV tables"""
    os.makedirs(save_dir, exist_ok=True)

    # CV results
    cv_path = os.path.join(save_dir, 'gait_cv_results.csv')
    cv_results_df.to_csv(cv_path)
    print(f"  ✅ CV results: {cv_path}")

    # Model summary
    ensemble_row = cv_results_df.loc['ENSEMBLE']
    summary = {
        'Metric': [
            'CV Accuracy (mean)',
            'CV Accuracy (std)',
            'CV F1 (mean)',
            'CV ROC-AUC (mean)',
            'CV ROC-AUC (std)',
        ],
        'Value': [
            round(ensemble_row['CV Accuracy (mean)'], 4),
            round(ensemble_row['CV Accuracy (std)'], 4),
            round(ensemble_row['CV F1 (mean)'], 4),
            round(ensemble_row['CV ROC-AUC (mean)'], 4),
            round(ensemble_row['CV ROC-AUC (std)'], 4),
        ]
    }

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(save_dir, 'gait_model_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"  ✅ Model summary: {summary_path}")


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    print("="*80)
    print("GAIT MODEL TRAINING - Binary Classification (HC vs PD)")
    print("="*80)

    # Create output directories
    figures_dir = os.path.join(CONFIG['results_dir'], 'figures')
    tables_dir  = os.path.join(CONFIG['results_dir'], 'tables')
    models_dir  = CONFIG['models_dir']

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir,  exist_ok=True)
    os.makedirs(models_dir,  exist_ok=True)

    # ── Step 1: Load Data ──────────────────────────────────────
    print("\n" + "─"*40)
    print("STEP 1: Loading Data")
    print("─"*40)
    X, y, feature_names = load_data(CONFIG['data_path'])

    # ── Step 2: Build Models ───────────────────────────────────
    print("\n" + "─"*40)
    print("STEP 2: Building Models")
    print("─"*40)
    models   = build_models()
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
    cv_auc = cv_results_df.loc['ENSEMBLE', 'CV ROC-AUC (mean)']
    plot_confusion_matrix(y, y_pred, figures_dir)
    plot_roc_curve(y, y_prob, cv_auc, figures_dir)
    plot_feature_importance(ensemble, feature_names, figures_dir)
    plot_cv_comparison(cv_results_df, figures_dir)
    plot_probability_distribution(y, y_prob, figures_dir)

    # ── Step 6: Save Results ───────────────────────────────────
    print("\n" + "─"*40)
    print("STEP 6: Saving Results")
    print("─"*40)
    save_results(cv_results_df, y, y_pred, y_prob, tables_dir)

    # ── Step 7: Save Model ─────────────────────────────────────
    print("\n" + "─"*40)
    print("STEP 7: Saving Model")
    print("─"*40)
    model_path = os.path.join(models_dir, 'gait_model.pkl')
    joblib.dump(ensemble, model_path)
    print(f"  ✅ Model saved: {model_path}")

    # ── Final Summary ──────────────────────────────────────────
    ensemble_row = cv_results_df.loc['ENSEMBLE']

    print("\n" + "="*80)
    print("✅ GAIT MODEL TRAINING COMPLETE!")
    print("="*80)
    print(f"""
  Dataset:          PhysioNet Gait (165 subjects)
  Task:             Binary (HC vs PD)
  Model:            Soft Voting Ensemble

  CV Accuracy:      {ensemble_row['CV Accuracy (mean)']:.4f} \
(±{ensemble_row['CV Accuracy (std)']:.4f})
  CV F1 Score:      {ensemble_row['CV F1 (mean)']:.4f}
  CV ROC-AUC:       {ensemble_row['CV ROC-AUC (mean)']:.4f} \
(±{ensemble_row['CV ROC-AUC (std)']:.4f})

  Model saved:      {model_path}
  Figures saved:    {figures_dir}/
  Tables saved:     {tables_dir}/

  Next steps:
    → Run models/train_handwriting.py
    """)


if __name__ == "__main__":
    main()