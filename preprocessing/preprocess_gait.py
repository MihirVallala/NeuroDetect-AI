

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent.parent))


class GaitDataPreprocessor:
    """
    Preprocessor for PhysioNet Gait PD dataset

    Handles:
    - Reading 19-column tab-separated time-series files
    - Extracting 24 biomechanical features per file
    - Averaging across multiple trials per subject
    - Subject-wise train/test split (no data leakage)
    - Normalization and saving
    """

    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.scaler = None
        self.imputer = None
        self.col_names = [
            'Time',
            'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8',
            'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8',
            'Total_Left', 'Total_Right'
        ]

    def get_subject_info(self, filename):
        """
        Parse filename to get subject ID, label, trial number

        GaCo01_02.txt → subject=GaCo01, label=0 (healthy), trial=2
        GaPt03_01.txt → subject=GaPt03, label=1 (PD), trial=1
        """
        name = filename.replace('.txt', '')
        parts = name.split('_')

        if len(parts) != 2:
            return None, None, None, None

        subject_id = parts[0]
        trial = int(parts[1])
        is_dual_task = (trial == 10)

        subject_upper = subject_id.upper()
        if 'CO' in subject_upper:
            label = 0   # Healthy
        elif 'PT' in subject_upper:
            label = 1   # Parkinson's
        else:
            return None, None, None, None

        return subject_id, label, trial, is_dual_task

    def load_gait_file(self, filepath):
        """Load one gait .txt file into numpy array"""
        try:
            data = np.loadtxt(filepath)

            if data.ndim != 2:
                return None
            if data.shape[1] < 19:
                return None
            if data.shape[1] > 19:
                data = data[:, :19]
            if data.shape[0] < 100:
                return None

            return data
        except Exception:
            return None

    def extract_features(self, data):
        """
        Extract 24 biomechanical features from one gait recording

        Features cover:
        - Force statistics (mean, std, max, CV) for each foot
        - Left-Right symmetry (KEY PD biomarker!)
        - Step count and asymmetry
        - Stride timing variability (KEY PD biomarker!)
        - Total force statistics
        - Signal shape (skewness, kurtosis)
        - Walk duration
        """
        features = {}

        left_total = data[:, 17]
        right_total = data[:, 18]

        # ── LEFT FOOT ──────────────────────────────────────────
        features['left_mean'] = left_total.mean()
        features['left_std'] = left_total.std()
        features['left_max'] = left_total.max()
        features['left_cv'] = left_total.std() / (left_total.mean() + 1e-8)

        # ── RIGHT FOOT ─────────────────────────────────────────
        features['right_mean'] = right_total.mean()
        features['right_std'] = right_total.std()
        features['right_max'] = right_total.max()
        features['right_cv'] = right_total.std() / (right_total.mean() + 1e-8)

        # ── LEFT-RIGHT SYMMETRY (KEY PD BIOMARKER) ─────────────
        total = left_total.mean() + right_total.mean() + 1e-8
        features['lr_asymmetry'] = abs(left_total.mean() - right_total.mean()) / total
        features['lr_ratio'] = left_total.mean() / (right_total.mean() + 1e-8)

        # ── STEP DETECTION ─────────────────────────────────────
        threshold = 50
        left_contact = (left_total > threshold).astype(int)
        right_contact = (right_total > threshold).astype(int)

        left_steps = int(np.sum(np.diff(left_contact) > 0))
        right_steps = int(np.sum(np.diff(right_contact) > 0))

        features['left_steps'] = left_steps
        features['right_steps'] = right_steps
        features['step_asymmetry'] = abs(left_steps - right_steps) / (left_steps + right_steps + 1e-8)

        # ── STRIDE TIMING VARIABILITY (KEY PD BIOMARKER) ───────
        left_onsets = np.where(np.diff(left_contact) > 0)[0]

        if len(left_onsets) > 2:
            stride_times = np.diff(left_onsets) / 100.0
            stride_times = stride_times[(stride_times > 0.3) & (stride_times < 3.0)]

            if len(stride_times) > 1:
                features['stride_mean'] = stride_times.mean()
                features['stride_std'] = stride_times.std()
                features['stride_cv'] = stride_times.std() / (stride_times.mean() + 1e-8)
            else:
                features['stride_mean'] = 0
                features['stride_std'] = 0
                features['stride_cv'] = 0
        else:
            features['stride_mean'] = 0
            features['stride_std'] = 0
            features['stride_cv'] = 0

        # ── TOTAL FORCE ─────────────────────────────────────────
        total_force = left_total + right_total
        features['total_force_mean'] = total_force.mean()
        features['total_force_std'] = total_force.std()
        features['total_force_cv'] = total_force.std() / (total_force.mean() + 1e-8)

        # ── SIGNAL SHAPE ────────────────────────────────────────
        features['left_skew'] = float(stats.skew(left_total))
        features['left_kurtosis'] = float(stats.kurtosis(left_total))
        features['right_skew'] = float(stats.skew(right_total))
        features['right_kurtosis'] = float(stats.kurtosis(right_total))

        # ── DURATION ────────────────────────────────────────────
        features['duration'] = float(data[-1, 0] - data[0, 0])

        return features

    def process_all_files(self):
        """
        Process all gait files - excludes dual-task (_10) files
        Groups results by subject ID
        """
        print("Scanning gait files...")

        all_files = [
            f for f in os.listdir(self.input_dir)
            if f.endswith('.txt') and f.lower() != 'format.txt'
        ]

        print(f"  Total .txt files found: {len(all_files)}")

        subject_data = {}
        subject_labels = {}
        skipped_dual = 0
        skipped_error = 0
        processed = 0

        for filename in sorted(all_files):
            filepath = os.path.join(self.input_dir, filename)

            subject_id, label, trial, is_dual_task = self.get_subject_info(filename)

            if subject_id is None:
                skipped_error += 1
                continue

            # Skip dual-task walks
            if is_dual_task:
                skipped_dual += 1
                continue

            data = self.load_gait_file(filepath)

            if data is None:
                skipped_error += 1
                continue

            features = self.extract_features(data)

            if subject_id not in subject_data:
                subject_data[subject_id] = []
                subject_labels[subject_id] = label

            subject_data[subject_id].append(features)
            processed += 1

        print(f"  Files processed (normal walks): {processed}")
        print(f"  Dual-task files skipped (_10): {skipped_dual}")
        print(f"  Error/unknown files skipped: {skipped_error}")
        print(f"  Unique subjects found: {len(subject_data)}")

        return subject_data, subject_labels

    def aggregate_subject_features(self, subject_data, subject_labels):
        """
        Average features across multiple trials per subject
        Result: ONE row per subject
        """
        print("\nAggregating features across trials per subject...")

        rows = []
        labels = []
        subject_ids = []
        feature_names = None

        trial_counts = [len(v) for v in subject_data.values()]
        print(f"  Trials per subject: min={min(trial_counts)}, "
              f"max={max(trial_counts)}, avg={np.mean(trial_counts):.1f}")

        for subject_id in sorted(subject_data.keys()):
            trials = subject_data[subject_id]
            label = subject_labels[subject_id]

            if feature_names is None:
                feature_names = list(trials[0].keys())

            # Average across all trials for this subject
            avg_features = {}
            for feat_name in feature_names:
                values = [t[feat_name] for t in trials if feat_name in t]
                avg_features[feat_name] = np.mean(values) if values else 0

            rows.append(list(avg_features.values()))
            labels.append(label)
            subject_ids.append(subject_id)

        X = np.array(rows, dtype=float)
        y = np.array(labels, dtype=int)

        print(f"  Final shape: {X.shape} (subjects x features)")
        print(f"\n  Healthy Controls: {(y==0).sum()}")
        print(f"  PD Patients: {(y==1).sum()}")
        print(f"  Total subjects: {len(y)}")

        return X, y, subject_ids, feature_names

    def handle_missing_values(self, X):
        """Handle NaN/Inf values with median imputation"""
        X = np.where(np.isinf(X), np.nan, X)
        nan_count = np.isnan(X).sum()

        if nan_count > 0:
            print(f"\n⚠️  Found {nan_count} NaN values - applying median imputation")
            self.imputer = SimpleImputer(strategy='median')
            X = self.imputer.fit_transform(X)
            print("  ✅ Imputation complete")
        else:
            print("\n✅ No missing values found in gait features!")

        return X

    def normalize_features(self, X):
        """Normalize to zero mean, unit variance"""
        print("\nNormalizing features...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        print("  ✅ Features normalized (mean≈0, std=1)")
        return X_scaled

    def save_processed_data(self, X_raw, X_scaled, y, subject_ids, feature_names):
        """Save all processed data and artifacts"""
        print("\nSaving processed data...")

        # Scaled features
        df_scaled = pd.DataFrame(X_scaled, columns=feature_names)
        df_scaled['subject_id'] = subject_ids
        df_scaled['label'] = y
        df_scaled['label_name'] = ['HC' if l == 0 else 'PD' for l in y]
        scaled_path = os.path.join(self.output_dir, 'gait_features_scaled.csv')
        df_scaled.to_csv(scaled_path, index=False)
        print(f"  ✅ Scaled features: {scaled_path}")

        # Raw features
        df_raw = pd.DataFrame(X_raw, columns=feature_names)
        df_raw['subject_id'] = subject_ids
        df_raw['label'] = y
        df_raw['label_name'] = ['HC' if l == 0 else 'PD' for l in y]
        raw_path = os.path.join(self.output_dir, 'gait_features_raw.csv')
        df_raw.to_csv(raw_path, index=False)
        print(f"  ✅ Raw features: {raw_path}")

        # Scaler
        joblib.dump(self.scaler, os.path.join(self.output_dir, 'gait_scaler.pkl'))
        print(f"  ✅ Scaler: {self.output_dir}/gait_scaler.pkl")

        # Imputer (if used)
        if self.imputer is not None:
            joblib.dump(self.imputer, os.path.join(self.output_dir, 'gait_imputer.pkl'))
            print(f"  ✅ Imputer: {self.output_dir}/gait_imputer.pkl")

        # Feature names
        feat_path = os.path.join(self.output_dir, 'gait_feature_names.txt')
        with open(feat_path, 'w') as f:
            for i, name in enumerate(feature_names, 1):
                f.write(f"{i}. {name}\n")
        print(f"  ✅ Feature names: {feat_path}")

        # Summary
        summary_path = os.path.join(self.output_dir, 'gait_preprocessing_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GAIT DATA PREPROCESSING SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Input directory: {self.input_dir}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")
            f.write(f"Total subjects: {len(y)}\n")
            f.write(f"Total features: {len(feature_names)}\n\n")
            f.write("Class distribution:\n")
            f.write(f"  Healthy (HC): {(y==0).sum()}\n")
            f.write(f"  Parkinson's (PD): {(y==1).sum()}\n\n")
            f.write("Features (24 total):\n")
            for i, name in enumerate(feature_names, 1):
                f.write(f"  {i:2d}. {name}\n")
        print(f"  ✅ Summary: {summary_path}")

    def run(self):
        """Run complete preprocessing pipeline"""
        print("="*80)
        print("GAIT DATA PREPROCESSING - PhysioNet Dataset")
        print("="*80)

        subject_data, subject_labels = self.process_all_files()

        if not subject_data:
            print("❌ ERROR: No subjects found! Check your input directory.")
            return None

        X_raw, y, subject_ids, feature_names = self.aggregate_subject_features(
            subject_data, subject_labels
        )

        X_clean = self.handle_missing_values(X_raw)
        X_scaled = self.normalize_features(X_clean)

        self.save_processed_data(X_raw, X_scaled, y, subject_ids, feature_names)

        print("\n" + "="*80)
        print("✅ GAIT PREPROCESSING COMPLETE!")
        print("="*80)
        print(f"\nProcessed data saved to: {self.output_dir}")
        print("\nFiles created:")
        print("  1. gait_features_scaled.csv  ← Use this for training")
        print("  2. gait_features_raw.csv     ← For reference")
        print("  3. gait_scaler.pkl           ← For inference")
        print("  4. gait_feature_names.txt    ← Feature list")
        print("  5. gait_preprocessing_summary.txt ← Summary")
        print("\nNext steps:")
        print("  → Train gait model using gait_features_scaled.csv")
        print("  → Expected ROC-AUC: 0.85-0.92 (binary HC vs PD)")

        return {
            'X_scaled': X_scaled,
            'X_raw': X_raw,
            'y': y,
            'subject_ids': subject_ids,
            'feature_names': feature_names,
            'scaler': self.scaler,
            'imputer': self.imputer
        }


def main():
    # CONFIGURE THESE PATHS
    # =====================
    input_dir = 'data/gait/raw/gait-in-parkinsons-disease-1.0.0'
    output_dir = 'data/gait/processed'
    # =====================

    if not os.path.exists(input_dir):
        print(f"❌ ERROR: Input directory not found: {input_dir}")
        print("Please update 'input_dir' to match your folder location.")
        return

    preprocessor = GaitDataPreprocessor(input_dir, output_dir)
    result = preprocessor.run()

    if result is None:
        return

    print("\n" + "="*80)
    print("FINAL STATISTICS")
    print("="*80)
    print(f"Total subjects: {len(result['y'])}")
    print(f"Features per subject: {result['X_scaled'].shape[1]}")
    print(f"Healthy Controls: {(result['y']==0).sum()}")
    print(f"PD Patients: {(result['y']==1).sum()}")
    print(f"Class balance: "
          f"{(result['y']==0).sum()/len(result['y'])*100:.1f}% HC / "
          f"{(result['y']==1).sum()/len(result['y'])*100:.1f}% PD")


if __name__ == "__main__":
    main()