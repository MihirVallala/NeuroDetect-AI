import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os
import sys
from pathlib import Path

# Add parent directory to path for imports if needed
sys.path.append(str(Path(__file__).parent.parent))


class SpeechDataPreprocessor:
    
    
    def __init__(self, input_path, output_dir):
       
        self.input_path = input_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        
    def load_data(self):
        
        print("Loading dataset...")
        print(f"  Input: {self.input_path}")
        
        # Load with 2-row header (row 0 = section, row 1 = column name)
        df = pd.read_csv(self.input_path, header=[0, 1])
        
        print(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        
        return df
    
    def extract_labels(self, df):
        
        print("\nExtracting labels from participant codes...")
        
        # Get participant codes (first column)
        participant_codes = df.iloc[:, 0].values
        
        # Create labels based on prefix
        labels = []
        label_names = []
        
        for code in participant_codes:
            code_str = str(code).strip()
            
            if code_str.startswith('PD'):
                labels.append(2)
                label_names.append('PD')
            elif code_str.startswith('RBD'):
                labels.append(1)
                label_names.append('RBD')
            elif code_str.startswith('HC'):
                labels.append(0)
                label_names.append('HC')
            else:
                # Unknown code - raise error
                raise ValueError(f"Unknown participant code: {code_str}")
        
        labels = np.array(labels)
        label_names = np.array(label_names)
        
        # Print distribution
        unique, counts = np.unique(labels, return_counts=True)
        label_map = {0: 'HC (Healthy)', 1: 'RBD (At Risk)', 2: 'PD (Parkinson\'s)'}
        
        print("  Label distribution:")
        for label, count in zip(unique, counts):
            print(f"    {label} - {label_map[label]}: {count} subjects")
        
        total = len(labels)
        print(f"  Total: {total} subjects")
        
        return labels, label_names, participant_codes
    
    def extract_speech_features(self, df):
       
        print("\nExtracting speech features...")
        
        # Find speech feature columns
        speech_reading_cols = [
            col for col in df.columns
            if 'reading passage' in str(col[0]).lower()
        ]
        
        speech_monologue_cols = [
            col for col in df.columns
            if 'monologue' in str(col[0]).lower()
        ]
        
        print(f"  Found {len(speech_reading_cols)} reading passage features")
        print(f"  Found {len(speech_monologue_cols)} monologue features")
        print(f"  Total: {len(speech_reading_cols) + len(speech_monologue_cols)} speech features")
        
        # Combine all speech columns
        all_speech_cols = speech_reading_cols + speech_monologue_cols
        
        # Extract feature values
        X = df[all_speech_cols].values
        
        # Create clean feature names
        feature_names = []
        
        for col in speech_reading_cols:
            # Clean up the feature name
            clean_name = col[1].replace('\n', ' ').strip()
            # Add prefix
            feature_names.append(f"reading_{clean_name}")
        
        for col in speech_monologue_cols:
            clean_name = col[1].replace('\n', ' ').strip()
            feature_names.append(f"monologue_{clean_name}")
        
        # Convert to float (in case some are stored as strings)
        X = X.astype(float)
        
        print(f"  Extracted features shape: {X.shape}")
        
        # Check for any remaining non-numeric values
        if np.isnan(X).any():
            nan_count = np.isnan(X).sum()
            print(f"  ⚠️  Found {nan_count} NaN values - will handle with imputation")
        else:
            print(f"  ✅ No missing values in speech features!")
        
        self.feature_names = feature_names
        
        return X, feature_names
    
    def handle_missing_values(self, X):
        """
        Handle missing values if any exist
        
        Strategy: Median imputation (robust to outliers)
        
        Args:
            X: Feature matrix
            
        Returns:
            X with missing values filled
        """
        if np.isnan(X).any():
            print("\nHandling missing values...")
            
            self.imputer = SimpleImputer(strategy='median')
            X_imputed = self.imputer.fit_transform(X)
            
            print(f"  ✅ Imputed {np.isnan(X).sum()} missing values with median")
            
            return X_imputed
        else:
            print("\n✅ No missing values to handle")
            return X
    
    def normalize_features(self, X):
       
        print("\nNormalizing features...")
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"  ✅ Features normalized (mean=0, std=1)")
        print(f"  Mean of scaled features: {X_scaled.mean(axis=0)[:3]} ...")
        print(f"  Std of scaled features: {X_scaled.std(axis=0)[:3]} ...")
        
        return X_scaled
    
    def save_processed_data(self, X_raw, X_scaled, labels, label_names, 
                           participant_codes, feature_names):
      
        print("\nSaving processed data...")
        
        # 1. Save scaled features (main file for training)
        df_scaled = pd.DataFrame(X_scaled, columns=feature_names)
        df_scaled['participant_code'] = participant_codes
        df_scaled['label'] = labels
        df_scaled['label_name'] = label_names
        
        scaled_path = os.path.join(self.output_dir, 'speech_features_scaled.csv')
        df_scaled.to_csv(scaled_path, index=False)
        print(f"  ✅ Scaled features: {scaled_path}")
        
        # 2. Save raw features (for reference)
        df_raw = pd.DataFrame(X_raw, columns=feature_names)
        df_raw['participant_code'] = participant_codes
        df_raw['label'] = labels
        df_raw['label_name'] = label_names
        
        raw_path = os.path.join(self.output_dir, 'speech_features_raw.csv')
        df_raw.to_csv(raw_path, index=False)
        print(f"  ✅ Raw features: {raw_path}")
        
        # 3. Save scaler (for inference)
        scaler_path = os.path.join(self.output_dir, 'speech_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"  ✅ Scaler object: {scaler_path}")
        
        # 4. Save imputer if used
        if self.imputer is not None:
            imputer_path = os.path.join(self.output_dir, 'speech_imputer.pkl')
            joblib.dump(self.imputer, imputer_path)
            print(f"  ✅ Imputer object: {imputer_path}")
        
        # 5. Save feature names
        feature_names_path = os.path.join(self.output_dir, 'feature_names.txt')
        with open(feature_names_path, 'w') as f:
            for i, name in enumerate(feature_names, 1):
                f.write(f"{i}. {name}\n")
        print(f"  ✅ Feature names: {feature_names_path}")
        
        # 6. Save summary statistics
        summary_path = os.path.join(self.output_dir, 'preprocessing_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SPEECH DATA PREPROCESSING SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Input file: {self.input_path}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")
            
            f.write(f"Total subjects: {len(labels)}\n")
            f.write(f"Total features: {len(feature_names)}\n\n")
            
            f.write("Class distribution:\n")
            unique, counts = np.unique(labels, return_counts=True)
            label_map = {0: 'HC (Healthy)', 1: 'RBD (At Risk)', 2: 'PD (Parkinson\'s)'}
            for label, count in zip(unique, counts):
                f.write(f"  {label} - {label_map[label]}: {count} ({count/len(labels)*100:.1f}%)\n")
            
            f.write(f"\nMissing values handled: {'Yes' if self.imputer else 'No'}\n")
            f.write(f"Normalization applied: {'Yes' if self.scaler else 'No'}\n")
            
            f.write("\nFeature statistics (raw data):\n")
            f.write(f"  Min: {X_raw.min(axis=0).min():.4f}\n")
            f.write(f"  Max: {X_raw.max(axis=0).max():.4f}\n")
            f.write(f"  Mean: {X_raw.mean():.4f}\n")
            f.write(f"  Std: {X_raw.std():.4f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("Preprocessing completed successfully!\n")
            f.write("="*80 + "\n")
        
        print(f"  ✅ Summary: {summary_path}")
    
    def run(self):
        """
        Run complete preprocessing pipeline
        
        Returns:
            Dictionary with processed data and metadata
        """
        print("="*80)
        print("SPEECH DATA PREPROCESSING - Czech UDPR Dataset")
        print("="*80)
        
        # Step 1: Load data
        df = self.load_data()
        
        # Step 2: Extract labels
        labels, label_names, participant_codes = self.extract_labels(df)
        
        # Step 3: Extract speech features
        X_raw, feature_names = self.extract_speech_features(df)
        
        # Step 4: Handle missing values (if any)
        X_clean = self.handle_missing_values(X_raw)
        
        # Step 5: Normalize features
        X_scaled = self.normalize_features(X_clean)
        
        # Step 6: Save everything
        self.save_processed_data(
            X_raw, X_scaled, labels, label_names,
            participant_codes, feature_names
        )
        
        print("\n" + "="*80)
        print("✅ PREPROCESSING COMPLETE!")
        print("="*80)
        print(f"\nProcessed data saved to: {self.output_dir}")
        print("\nFiles created:")
        print("  1. speech_features_scaled.csv  ← Use this for training")
        print("  2. speech_features_raw.csv     ← For reference")
        print("  3. speech_scaler.pkl           ← For inference")
        print("  4. feature_names.txt           ← Feature list")
        print("  5. preprocessing_summary.txt   ← Summary stats")
        
        print("\nNext steps:")
        print("  → Train the speech model using speech_features_scaled.csv")
        print("  → Expected ROC-AUC: 0.88-0.94 (3-class)")
        
        return {
            'X_scaled': X_scaled,
            'X_raw': X_raw,
            'labels': labels,
            'label_names': label_names,
            'participant_codes': participant_codes,
            'feature_names': feature_names,
            'scaler': self.scaler,
            'imputer': self.imputer
        }


def main():
    """
    Main execution function
    
    Modify these paths according to your folder structure:
    """
    
    # CONFIGURE THESE PATHS
    # =====================
    
    # Path to your raw dataset
    # Adjust this to match your actual file location
    input_path = 'data/speech/raw/dataset.csv'
    
    # Directory to save processed data
    output_dir = 'data/speech/processed'
    
    # =====================
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print("="*80)
        print("❌ ERROR: Input file not found!")
        print("="*80)
        print(f"\nLooking for: {input_path}")
        print("\nPlease update the 'input_path' variable in this script")
        print("to match your actual dataset location.")
        print("\nExample:")
        print("  input_path = 'data/speech/raw/dataset.csv'")
        print("  input_path = '../data/dataset.csv'")
        print("  input_path = 'C:/Users/YourName/parkinsons/data/speech/raw/dataset.csv'")
        print("\n" + "="*80)
        return
    
    # Run preprocessing
    preprocessor = SpeechDataPreprocessor(input_path, output_dir)
    result = preprocessor.run()
    
    # Print final statistics
    print("\n" + "="*80)
    print("FINAL STATISTICS")
    print("="*80)
    print(f"Samples: {result['X_scaled'].shape[0]}")
    print(f"Features: {result['X_scaled'].shape[1]}")
    print(f"Classes: {len(np.unique(result['labels']))}")
    print("\nClass distribution:")
    unique, counts = np.unique(result['labels'], return_counts=True)
    for label, count in zip(unique, counts):
        label_name = ['HC', 'RBD', 'PD'][label]
        print(f"  {label_name}: {count} ({count/len(result['labels'])*100:.1f}%)")


if __name__ == "__main__":
    main()
