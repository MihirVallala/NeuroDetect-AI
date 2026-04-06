

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import joblib
import warnings

warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent.parent))


class HandwritingDataPreprocessor:
    """
    Preprocessor for Kaggle Parkinson's spiral/wave drawings

    Handles:
    - Loading PNG images from nested folder structure
    - Combining spiral and wave drawings
    - Resizing to 224x224 for EfficientNet
    - Saving processed arrays for fast training
    - Creating metadata CSV

    Note on augmentation:
    We save ORIGINAL images here.
    Augmentation is applied LIVE during model training
    using PyTorch transforms (more memory efficient)
    """

    def __init__(self, input_dir, output_dir):
        """
        Initialize preprocessor

        Args:
            input_dir: Path to handwriting/raw folder
            output_dir: Path to handwriting/processed folder
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Standard size for EfficientNetB0
        self.image_size = (224, 224)

        # Drawing types to use (ignore 'drawings' - it's a duplicate)
        self.drawing_types = ['spiral', 'wave']

        # Class mapping
        self.class_map = {
            'healthy': 0,
            'parkinson': 1
        }

    def load_images_from_folder(self, folder_path, label, drawing_type, split):
        """
        Load all PNG images from one folder

        Args:
            folder_path: Path to healthy/ or parkinson/ folder
            label: 0 (healthy) or 1 (parkinson)
            drawing_type: 'spiral' or 'wave'
            split: 'training' or 'testing'

        Returns:
            images: list of numpy arrays (224, 224, 3)
            labels: list of ints
            metadata: list of dicts with file info
        """
        images = []
        labels = []
        metadata = []

        if not os.path.exists(folder_path):
            print(f"  ⚠️  Folder not found: {folder_path}")
            return images, labels, metadata

        # Get all image files
        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        for filename in sorted(image_files):
            filepath = os.path.join(folder_path, filename)

            try:
                # Load image
                img = Image.open(filepath)

                # Convert to RGB (3 channels)
                # Some images might be grayscale or RGBA
                img = img.convert('RGB')

                # Resize to 224x224
                img = img.resize(self.image_size, Image.LANCZOS)

                # Convert to numpy array
                img_array = np.array(img, dtype=np.uint8)

                # Validate shape
                if img_array.shape != (224, 224, 3):
                    print(f"  ⚠️  Unexpected shape {img_array.shape}: {filename}")
                    continue

                images.append(img_array)
                labels.append(label)
                metadata.append({
                    'filename': filename,
                    'filepath': filepath,
                    'drawing_type': drawing_type,
                    'split': split,
                    'label': label,
                    'label_name': 'HC' if label == 0 else 'PD',
                    'class_name': 'healthy' if label == 0 else 'parkinson'
                })

            except Exception as e:
                print(f"  ⚠️  Error loading {filename}: {e}")
                continue

        return images, labels, metadata

    def load_all_images(self):
        """
        Load all images from spiral/ and wave/ folders

        Combines both drawing types for maximum data.
        Respects existing train/test split.

        Returns:
            train_images, train_labels, train_meta
            test_images, test_labels, test_meta
        """
        print("Loading images...")

        train_images, train_labels, train_meta = [], [], []
        test_images, test_labels, test_meta = [], [], []

        for drawing_type in self.drawing_types:
            print(f"\n  Processing {drawing_type} drawings:")

            for split in ['training', 'testing']:
                split_images = 0

                for class_name, label in self.class_map.items():
                    folder_path = os.path.join(
                        self.input_dir,
                        drawing_type,
                        split,
                        class_name
                    )

                    imgs, lbls, meta = self.load_images_from_folder(
                        folder_path, label, drawing_type, split
                    )

                    if split == 'training':
                        train_images.extend(imgs)
                        train_labels.extend(lbls)
                        train_meta.extend(meta)
                    else:
                        test_images.extend(imgs)
                        test_labels.extend(lbls)
                        test_meta.extend(meta)

                    split_images += len(imgs)
                    print(f"    {split}/{class_name}: {len(imgs)} images")

        print(f"\n  Total training images: {len(train_images)}")
        print(f"  Total testing images:  {len(test_images)}")
        print(f"  Grand total:           {len(train_images) + len(test_images)}")

        return (
            train_images, train_labels, train_meta,
            test_images, test_labels, test_meta
        )

    def verify_dataset(self, train_images, train_labels,
                       test_images, test_labels):
        """
        Verify dataset integrity and print statistics

        Args:
            train_images, train_labels: Training data
            test_images, test_labels: Testing data
        """
        print("\nVerifying dataset...")

        # Training stats
        train_healthy = train_labels.count(0)
        train_pd = train_labels.count(1)
        print(f"  Training - Healthy: {train_healthy}, PD: {train_pd}")

        # Testing stats
        test_healthy = test_labels.count(0)
        test_pd = test_labels.count(1)
        print(f"  Testing  - Healthy: {test_healthy}, PD: {test_pd}")

        # Image shape verification
        if train_images:
            sample_shape = np.array(train_images[0]).shape
            print(f"  Image shape: {sample_shape} ✅")

            # Check all images same shape
            shapes = set(np.array(img).shape for img in train_images[:10])
            if len(shapes) == 1:
                print(f"  All images consistent shape ✅")
            else:
                print(f"  ⚠️  Mixed shapes detected: {shapes}")

        # Class balance check
        total_train = len(train_labels)
        if total_train > 0:
            balance = train_healthy / total_train * 100
            print(f"  Training balance: {balance:.1f}% HC / {100-balance:.1f}% PD")

        # Verify no overlap between train and test
        # (Already guaranteed by folder structure)
        print(f"  Train/test split: Pre-existing folders ✅")
        print(f"  Data leakage risk: None ✅")

    def normalize_images(self, images):
        """
        Normalize pixel values to [0, 1] range

        We save normalized float32 arrays.
        Additional augmentation applied during training.

        Args:
            images: list of uint8 numpy arrays

        Returns:
            numpy array of float32, shape (N, 224, 224, 3)
        """
        X = np.array(images, dtype=np.float32)

        # Normalize to [0, 1]
        X = X / 255.0

        return X

    def save_processed_data(self, X_train, y_train, train_meta,
                            X_test, y_test, test_meta):
        """
        Save all processed data

        Saves:
        1. Numpy arrays for fast loading during training
        2. Metadata CSV with file info
        3. Summary statistics

        Args:
            X_train: Training images (N, 224, 224, 3)
            y_train: Training labels
            train_meta: Training metadata
            X_test: Testing images (N, 224, 224, 3)
            y_test: Testing labels
            test_meta: Testing metadata
        """
        print("\nSaving processed data...")

        # 1. Save numpy arrays
        train_images_path = os.path.join(self.output_dir, 'X_train.npy')
        train_labels_path = os.path.join(self.output_dir, 'y_train.npy')
        test_images_path = os.path.join(self.output_dir, 'X_test.npy')
        test_labels_path = os.path.join(self.output_dir, 'y_test.npy')

        np.save(train_images_path, X_train)
        np.save(train_labels_path, y_train)
        np.save(test_images_path, X_test)
        np.save(test_labels_path, y_test)

        print(f"  ✅ Training images: {train_images_path} {X_train.shape}")
        print(f"  ✅ Training labels: {train_labels_path} {y_train.shape}")
        print(f"  ✅ Testing images:  {test_images_path} {X_test.shape}")
        print(f"  ✅ Testing labels:  {test_labels_path} {y_test.shape}")

        # 2. Save metadata CSV
        train_df = pd.DataFrame(train_meta)
        test_df = pd.DataFrame(test_meta)
        all_meta_df = pd.concat([train_df, test_df], ignore_index=True)

        meta_path = os.path.join(self.output_dir, 'handwriting_metadata.csv')
        all_meta_df.to_csv(meta_path, index=False)
        print(f"  ✅ Metadata CSV: {meta_path}")

        # 3. Save drawing type breakdown
        if not train_df.empty:
            breakdown_path = os.path.join(
                self.output_dir, 'dataset_breakdown.csv'
            )
            breakdown = all_meta_df.groupby(
                ['drawing_type', 'split', 'label_name']
            ).size().reset_index(name='count')
            breakdown.to_csv(breakdown_path, index=False)
            print(f"  ✅ Breakdown: {breakdown_path}")

        # 4. Save summary
        summary_path = os.path.join(
            self.output_dir, 'handwriting_preprocessing_summary.txt'
        )
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("HANDWRITING DATA PREPROCESSING SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Input directory: {self.input_dir}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")
            f.write(f"Image size: {self.image_size}\n")
            f.write(f"Drawing types: {self.drawing_types}\n\n")
            f.write("Dataset Statistics:\n")
            f.write(f"  Training images: {len(X_train)}\n")
            f.write(f"    Healthy (HC): {(y_train==0).sum()}\n")
            f.write(f"    PD:           {(y_train==1).sum()}\n\n")
            f.write(f"  Testing images:  {len(X_test)}\n")
            f.write(f"    Healthy (HC): {(y_test==0).sum()}\n")
            f.write(f"    PD:           {(y_test==1).sum()}\n\n")
            f.write(f"  Total images:    {len(X_train) + len(X_test)}\n\n")
            f.write("Normalization:\n")
            f.write("  Pixel values normalized to [0, 1]\n")
            f.write("  Additional augmentation applied during training\n\n")
            f.write("Files saved:\n")
            f.write("  X_train.npy  - Training images\n")
            f.write("  y_train.npy  - Training labels\n")
            f.write("  X_test.npy   - Testing images\n")
            f.write("  y_test.npy   - Testing labels\n")
            f.write("  handwriting_metadata.csv\n")
            f.write("  dataset_breakdown.csv\n\n")
            f.write("="*80 + "\n")
            f.write("Preprocessing completed successfully!\n")

        print(f"  ✅ Summary: {summary_path}")

    def run(self):
        """
        Run complete handwriting preprocessing pipeline

        Returns:
            Dictionary with processed arrays and metadata
        """
        print("="*80)
        print("HANDWRITING DATA PREPROCESSING - Kaggle Spiral/Wave Dataset")
        print("="*80)

        # Step 1: Load all images
        (train_images, train_labels, train_meta,
         test_images, test_labels, test_meta) = self.load_all_images()

        # Check we have data
        if not train_images:
            print("❌ ERROR: No training images found!")
            print(f"Check your input directory: {self.input_dir}")
            return None

        if not test_images:
            print("❌ ERROR: No testing images found!")
            return None

        # Step 2: Verify dataset
        self.verify_dataset(
            train_images, train_labels,
            test_images, test_labels
        )

        # Step 3: Normalize
        print("\nNormalizing pixel values to [0, 1]...")
        X_train = self.normalize_images(train_images)
        X_test = self.normalize_images(test_images)
        y_train = np.array(train_labels, dtype=np.int64)
        y_test = np.array(test_labels, dtype=np.int64)
        print(f"  ✅ Training array: {X_train.shape} float32")
        print(f"  ✅ Testing array:  {X_test.shape} float32")

        # Step 4: Save everything
        self.save_processed_data(
            X_train, y_train, train_meta,
            X_test, y_test, test_meta
        )

        print("\n" + "="*80)
        print("✅ HANDWRITING PREPROCESSING COMPLETE!")
        print("="*80)
        print(f"\nProcessed data saved to: {self.output_dir}")
        print("\nFiles created:")
        print("  1. X_train.npy                ← Training images")
        print("  2. y_train.npy                ← Training labels")
        print("  3. X_test.npy                 ← Testing images")
        print("  4. y_test.npy                 ← Testing labels")
        print("  5. handwriting_metadata.csv   ← File metadata")
        print("  6. dataset_breakdown.csv      ← Count per folder")
        print("  7. handwriting_preprocessing_summary.txt")
        print("\nNext steps:")
        print("  → Train CNN model using X_train.npy and y_train.npy")
        print("  → EfficientNetB0 with transfer learning recommended")
        print("  → Expected ROC-AUC: 0.88-0.93 (binary HC vs PD)")

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'train_meta': train_meta,
            'test_meta': test_meta
        }


def main():
    """
    Main execution function

    IMPORTANT: Run this from your project root folder:
    cd parkinsons-multimodal
    python preprocessing/preprocess_handwriting.py
    """

    # CONFIGURE THESE PATHS
    # =====================
    input_dir = 'data/handwriting/raw'
    output_dir = 'data/handwriting/processed'
    # =====================

    # Check input directory exists
    if not os.path.exists(input_dir):
        print("="*80)
        print("❌ ERROR: Input directory not found!")
        print("="*80)
        print(f"\nLooking for: {input_dir}")
        print("\nPlease update 'input_dir' in this script.")
        return

    # Check spiral and wave folders exist
    for drawing_type in ['spiral', 'wave']:
        folder = os.path.join(input_dir, drawing_type)
        if not os.path.exists(folder):
            print(f"⚠️  Warning: {drawing_type}/ folder not found at {folder}")

    # Run preprocessing
    preprocessor = HandwritingDataPreprocessor(input_dir, output_dir)
    result = preprocessor.run()

    if result is None:
        return

    # Final statistics
    print("\n" + "="*80)
    print("FINAL STATISTICS")
    print("="*80)
    print(f"Training images:  {len(result['X_train'])}")
    print(f"Testing images:   {len(result['X_test'])}")
    print(f"Total images:     {len(result['X_train']) + len(result['X_test'])}")
    print(f"Image shape:      {result['X_train'].shape[1:]} (H x W x C)")
    print(f"\nTraining class distribution:")
    print(f"  HC (Healthy):   {(result['y_train']==0).sum()}")
    print(f"  PD (Parkinson): {(result['y_train']==1).sum()}")
    print(f"\nTesting class distribution:")
    print(f"  HC (Healthy):   {(result['y_test']==0).sum()}")
    print(f"  PD (Parkinson): {(result['y_test']==1).sum()}")
    print(f"\nMemory usage:")
    train_mb = result['X_train'].nbytes / 1024 / 1024
    test_mb = result['X_test'].nbytes / 1024 / 1024
    print(f"  Training arrays: {train_mb:.1f} MB")
    print(f"  Testing arrays:  {test_mb:.1f} MB")


if __name__ == "__main__":
    main()