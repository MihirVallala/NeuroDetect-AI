"""
Handwriting Model Training Script
Kaggle Spiral/Wave Dataset - Binary Classification (HC vs PD)

Model: EfficientNetB0 with Transfer Learning (PyTorch)
Task:  Binary classification
       0 = HC (Healthy Controls)
       1 = PD (Parkinson's Disease)

Why Transfer Learning?
- Only 144 training images (very small!)
- EfficientNetB0 pretrained on ImageNet
- Already knows edges, curves, shapes
- We just fine-tune for spiral/wave patterns

Training Strategy:
- Phase 1 (epochs 1-10):  Freeze backbone, train head only
- Phase 2 (epochs 11-30): Unfreeze all, fine-tune with small LR
- Data augmentation to artificially increase dataset size
- Best model saved based on validation ROC-AUC

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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score
)
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    'train_images':  'data/handwriting/processed/X_train.npy',
    'train_labels':  'data/handwriting/processed/y_train.npy',
    'test_images':   'data/handwriting/processed/X_test.npy',
    'test_labels':   'data/handwriting/processed/y_test.npy',
    'models_dir':    'models/saved',
    'results_dir':   'results',
    'random_state':  42,
    'batch_size':    16,
    'epochs':        30,
    'lr_phase1':     0.001,   # Head only
    'lr_phase2':     0.0001,  # Fine-tuning all layers
    'phase2_start':  10,      # Epoch to unfreeze backbone
    'class_names':   ['HC (Healthy)', 'PD (Parkinson\'s)'],
    'class_short':   ['HC', 'PD'],
}


# ─────────────────────────────────────────────
#  DATASET CLASS
# ─────────────────────────────────────────────
class HandwritingDataset(Dataset):
    """
    PyTorch Dataset for handwriting images

    Applies augmentation during training,
    only normalization during testing
    """

    def __init__(self, images, labels, mode='train'):
        """
        Args:
            images: numpy array (N, 224, 224, 3) float32 [0,1]
            labels: numpy array (N,) int64
            mode:   'train' applies augmentation, 'test' does not
        """
        self.images = images
        self.labels = labels
        self.mode   = mode

        # ImageNet normalization
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.2
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert float32 [0,1] back to uint8 [0,255] for PIL
        img = (self.images[idx] * 255).astype(np.uint8)
        label = int(self.labels[idx])

        img_tensor = self.transform(img)

        return img_tensor, label


# ─────────────────────────────────────────────
#  MODEL DEFINITION
# ─────────────────────────────────────────────
class HandwritingCNN(nn.Module):
    """
    EfficientNetB0 with custom classification head

    Architecture:
    - EfficientNetB0 backbone (pretrained on ImageNet)
    - Custom head: Dropout → Linear(256) → ReLU → Dropout → Linear(2)
    """

    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()

        # Load pretrained EfficientNetB0
        self.backbone = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )

        # Freeze backbone initially
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────
def load_data():
    """Load preprocessed numpy arrays"""
    print("Loading preprocessed handwriting data...")

    X_train = np.load(CONFIG['train_images'])
    y_train = np.load(CONFIG['train_labels'])
    X_test  = np.load(CONFIG['test_images'])
    y_test  = np.load(CONFIG['test_labels'])

    print(f"  Training images: {X_train.shape}")
    print(f"  Testing images:  {X_test.shape}")
    print(f"  Train - HC: {(y_train==0).sum()}, PD: {(y_train==1).sum()}")
    print(f"  Test  - HC: {(y_test==0).sum()},  PD: {(y_test==1).sum()}")

    return X_train, y_train, X_test, y_test


def create_dataloaders(X_train, y_train, X_test, y_test):
    """
    Create PyTorch DataLoaders with weighted sampling

    Weighted sampling ensures each batch has
    balanced classes even if dataset is imbalanced
    """
    train_dataset = HandwritingDataset(X_train, y_train, mode='train')
    test_dataset  = HandwritingDataset(X_test,  y_test,  mode='test')

    # Weighted sampler for balanced batches
    class_counts  = np.bincount(y_train)
    weights       = 1.0 / class_counts[y_train]
    sampler       = WeightedRandomSampler(
        weights=torch.FloatTensor(weights),
        num_samples=len(weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        sampler=sampler,
        num_workers=0,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return train_loader, test_loader


# ─────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch"""
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model on a dataloader"""
    model.eval()
    total_loss = 0
    all_preds  = []
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)
            probs   = torch.softmax(outputs, dim=1)

            total_loss += loss.item()
            _, predicted = probs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    acc     = accuracy_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    loss    = total_loss / len(loader)

    return loss, acc, roc_auc, all_preds, all_probs, all_labels


# ─────────────────────────────────────────────
#  FULL TRAINING PIPELINE
# ─────────────────────────────────────────────
def train_model(train_loader, test_loader, device):
    """
    Train EfficientNetB0 with 2-phase strategy

    Phase 1 (epochs 1-10):
    - Backbone frozen
    - Only head layers train
    - LR = 0.001

    Phase 2 (epochs 11-30):
    - All layers unfrozen
    - Fine-tune entire network
    - LR = 0.0001 (smaller to avoid destroying pretrained weights)
    """
    print(f"\nUsing device: {device}")

    # Initialize model
    model = HandwritingCNN(num_classes=2, freeze_backbone=True)
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Phase 1 optimizer (head only)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['lr_phase1'],
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs']
    )

    # Tracking
    history = {
        'train_loss': [], 'test_loss': [],
        'train_acc':  [], 'test_acc':  [],
        'test_auc':   []
    }

    best_auc       = 0
    best_model_path = os.path.join(
        CONFIG['models_dir'], 'handwriting_model_best.pth'
    )
    os.makedirs(CONFIG['models_dir'], exist_ok=True)

    print(f"\nTraining for {CONFIG['epochs']} epochs...")
    print(f"  Phase 1 (epochs 1-{CONFIG['phase2_start']}): "
          f"Head only, LR={CONFIG['lr_phase1']}")
    print(f"  Phase 2 (epochs {CONFIG['phase2_start']+1}-"
          f"{CONFIG['epochs']}): "
          f"Full fine-tune, LR={CONFIG['lr_phase2']}")
    print("-" * 60)

    for epoch in range(1, CONFIG['epochs'] + 1):

        # Switch to Phase 2
        if epoch == CONFIG['phase2_start'] + 1:
            print(f"\n  → Epoch {epoch}: Unfreezing backbone for fine-tuning...")
            model.unfreeze_backbone()
            optimizer = optim.Adam(
                model.parameters(),
                lr=CONFIG['lr_phase2'],
                weight_decay=1e-4
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=CONFIG['epochs'] - CONFIG['phase2_start']
            )

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Evaluate
        test_loss, test_acc, test_auc, _, _, _ = evaluate(
            model, test_loader, criterion, device
        )

        scheduler.step()

        # Track history
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['test_auc'].append(test_auc)

        # Save best model
        if test_auc > best_auc:
            best_auc = test_auc
            torch.save(model.state_dict(), best_model_path)

        # Print every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{CONFIG['epochs']} | "
                  f"Train Loss: {train_loss:.4f} "
                  f"Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} "
                  f"Acc: {test_acc:.4f} "
                  f"AUC: {test_auc:.4f}"
                  f"{'  ← BEST' if test_auc == best_auc else ''}")

    print(f"\n  Best Test ROC-AUC: {best_auc:.4f}")
    print(f"  Best model saved: {best_model_path}")

    return model, history, best_auc, best_model_path


# ─────────────────────────────────────────────
#  FINAL EVALUATION
# ─────────────────────────────────────────────
def final_evaluation(best_model_path, test_loader, device):
    """Load best model and run final evaluation"""
    print("\nLoading best model for final evaluation...")

    model = HandwritingCNN(num_classes=2, freeze_backbone=False)
    model.load_state_dict(
        torch.load(best_model_path, map_location=device)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    _, acc, auc, y_pred, y_prob, y_true = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\nFinal Test Results:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  F1 Score:  {f1_score(y_true, y_pred):.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=CONFIG['class_names']
    ))

    return y_true, y_pred, y_prob, acc, auc


# ─────────────────────────────────────────────
#  VISUALIZATIONS
# ─────────────────────────────────────────────
def plot_training_curves(history, save_dir):
    """Plot loss and accuracy curves"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(epochs, history['train_loss'],
                 'b-', lw=2, label='Train')
    axes[0].plot(epochs, history['test_loss'],
                 'r-', lw=2, label='Test')
    axes[0].axvline(x=CONFIG['phase2_start'], color='gray',
                    linestyle='--', alpha=0.7, label='Phase 2 start')
    axes[0].set_title('Loss Curves', fontsize=12)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history['train_acc'],
                 'b-', lw=2, label='Train')
    axes[1].plot(epochs, history['test_acc'],
                 'r-', lw=2, label='Test')
    axes[1].axvline(x=CONFIG['phase2_start'], color='gray',
                    linestyle='--', alpha=0.7, label='Phase 2 start')
    axes[1].set_title('Accuracy Curves', fontsize=12)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # ROC-AUC
    axes[2].plot(epochs, history['test_auc'],
                 'g-', lw=2, label='Test AUC')
    axes[2].axvline(x=CONFIG['phase2_start'], color='gray',
                    linestyle='--', alpha=0.7, label='Phase 2 start')
    axes[2].axhline(y=max(history['test_auc']), color='orange',
                    linestyle=':', alpha=0.7,
                    label=f"Best: {max(history['test_auc']):.3f}")
    axes[2].set_title('Test ROC-AUC', fontsize=12)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('ROC-AUC')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Handwriting Model - Training Curves',
                 fontsize=13, y=1.02)
    plt.tight_layout()

    path = os.path.join(save_dir, 'handwriting_training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Training curves: {path}")


def plot_confusion_matrix(y_true, y_pred, save_dir):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt='d',
        cmap='Blues',
        xticklabels=CONFIG['class_short'],
        yticklabels=CONFIG['class_short'],
        annot_kws={'size': 16}
    )
    plt.title('Handwriting Model - Confusion Matrix', fontsize=14, pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    path = os.path.join(save_dir, 'handwriting_confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Confusion matrix: {path}")


def plot_roc_curve(y_true, y_prob, save_dir):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#9C27B0', lw=2,
             label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1,
             alpha=0.5, label='Random')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Handwriting Model - ROC Curve (HC vs PD)', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, 'handwriting_roc_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ ROC curve: {path}")


def plot_probability_distribution(y_true, y_prob, save_dir):
    """Plot predicted probability distribution"""
    plt.figure(figsize=(9, 5))

    plt.hist(y_prob[y_true == 0], bins=15, alpha=0.6,
             color='#2196F3', label='HC (Healthy)', density=True)
    plt.hist(y_prob[y_true == 1], bins=15, alpha=0.6,
             color='#F44336', label="PD (Parkinson's)", density=True)

    plt.axvline(x=0.5, color='black', linestyle='--',
                lw=1.5, label='Decision boundary')
    plt.xlabel('Predicted P(PD)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Handwriting Model - Probability Distribution', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(
        save_dir, 'handwriting_probability_distribution.png'
    )
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Probability distribution: {path}")


# ─────────────────────────────────────────────
#  SAVE RESULTS
# ─────────────────────────────────────────────
def save_results(history, y_true, y_pred, y_prob,
                 best_auc, save_dir):
    """Save training history and metrics"""
    os.makedirs(save_dir, exist_ok=True)

    # Training history
    history_df = pd.DataFrame(history)
    history_df.index += 1
    history_df.index.name = 'epoch'
    history_df.to_csv(
        os.path.join(save_dir, 'handwriting_training_history.csv')
    )
    print(f"  ✅ Training history saved")

    # Model summary
    summary = {
        'Metric': [
            'Test Accuracy',
            'Test ROC-AUC (best)',
            'Test F1 Score',
            'Total Epochs',
            'Architecture',
        ],
        'Value': [
            round(accuracy_score(y_true, y_pred), 4),
            round(best_auc, 4),
            round(f1_score(y_true, y_pred), 4),
            CONFIG['epochs'],
            'EfficientNetB0 (Transfer Learning)',
        ]
    }

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(
        save_dir, 'handwriting_model_summary.csv'
    )
    summary_df.to_csv(summary_path, index=False)
    print(f"  ✅ Model summary: {summary_path}")


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    print("="*80)
    print("HANDWRITING MODEL TRAINING - Binary Classification (HC vs PD)")
    print("="*80)

    # Set random seeds for reproducibility
    torch.manual_seed(CONFIG['random_state'])
    np.random.seed(CONFIG['random_state'])

    # Create output directories
    figures_dir = os.path.join(CONFIG['results_dir'], 'figures')
    tables_dir  = os.path.join(CONFIG['results_dir'], 'tables')

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir,  exist_ok=True)
    os.makedirs(CONFIG['models_dir'], exist_ok=True)

    # Device
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    # ── Step 1: Load Data ──────────────────────────────────────
    print("\n" + "─"*40)
    print("STEP 1: Loading Data")
    print("─"*40)
    X_train, y_train, X_test, y_test = load_data()

    # ── Step 2: Create DataLoaders ─────────────────────────────
    print("\n" + "─"*40)
    print("STEP 2: Creating DataLoaders")
    print("─"*40)
    train_loader, test_loader = create_dataloaders(
        X_train, y_train, X_test, y_test
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    print(f"  Augmentation:  Flip, Rotation, ColorJitter, Affine")

    # ── Step 3: Train Model ────────────────────────────────────
    print("\n" + "─"*40)
    print("STEP 3: Training EfficientNetB0")
    print("─"*40)
    model, history, best_auc, best_model_path = train_model(
        train_loader, test_loader, device
    )

    # ── Step 4: Final Evaluation ───────────────────────────────
    print("\n" + "─"*40)
    print("STEP 4: Final Evaluation")
    print("─"*40)
    y_true, y_pred, y_prob, acc, auc = final_evaluation(
        best_model_path, test_loader, device
    )

    # ── Step 5: Visualizations ─────────────────────────────────
    print("\n" + "─"*40)
    print("STEP 5: Generating Visualizations")
    print("─"*40)
    plot_training_curves(history, figures_dir)
    plot_confusion_matrix(y_true, y_pred, figures_dir)
    plot_roc_curve(y_true, y_prob, figures_dir)
    plot_probability_distribution(y_true, y_prob, figures_dir)

    # ── Step 6: Save Results ───────────────────────────────────
    print("\n" + "─"*40)
    print("STEP 6: Saving Results")
    print("─"*40)
    save_results(history, y_true, y_pred, y_prob, best_auc, tables_dir)

    # ── Step 7: Save Final Model ───────────────────────────────
    print("\n" + "─"*40)
    print("STEP 7: Saving Final Model")
    print("─"*40)
    final_path = os.path.join(
        CONFIG['models_dir'], 'handwriting_model_final.pth'
    )
    torch.save(model.state_dict(), final_path)
    print(f"  ✅ Final model: {final_path}")
    print(f"  ✅ Best model:  {best_model_path}")

    # ── Final Summary ──────────────────────────────────────────
    print("\n" + "="*80)
    print("✅ HANDWRITING MODEL TRAINING COMPLETE!")
    print("="*80)
    print(f"""
  Dataset:       Kaggle Spirals/Waves (204 images)
  Architecture:  EfficientNetB0 (Transfer Learning)
  Task:          Binary (HC vs PD)

  Test Accuracy: {acc:.4f}
  Test ROC-AUC:  {best_auc:.4f}
  Test F1 Score: {f1_score(y_true, y_pred):.4f}

  Best model:    {best_model_path}
  Final model:   {final_path}
  Figures:       {figures_dir}/
  Tables:        {tables_dir}/

  Next steps:
    → All 3 models trained!
    → Build fusion model
    → Build FastAPI backend
    """)


if __name__ == "__main__":
    main()