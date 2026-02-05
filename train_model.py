"""
Mushroom Classification Model Training Script - ENHANCED VERSION
This script trains an optimized Gradient Boosting Classifier with advanced preprocessing
for the Partha sarathi R Mushroom Classification System.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

def acquire_dataset(filepath):
    """Load the mushroom classification dataset"""
    print("Initiating dataset acquisition...")
    df = pd.read_csv(filepath)
    print(f"Dataset successfully loaded! Dimensions: {df.shape}")
    return df

def process_features(df):
    """Prepare features and target variables with proper preprocessing"""
    print("\nProcessing feature engineering...")
    
    # Select features and target
    feature_columns = ['cap_diameter', 'cap_shape', 'gill_attachment', 'gill_color', 
                'stem_height', 'stem_width', 'stem_color', 'season']
    target_column = 'class'
    
    X = df[feature_columns]
    y = df[target_column]
    
    # Split the data first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Scale the features for better performance
    feature_scaler = StandardScaler()
    X_train_normalized = feature_scaler.fit_transform(X_train)
    X_test_normalized = feature_scaler.transform(X_test)
    
    print(f"Training subset magnitude: {X_train.shape[0]}")
    print(f"Testing subset magnitude: {X_test.shape[0]}")
    print(f"Class distribution in training subset:")
    print(f"  Edible (0): {(y_train == 0).sum()} ({(y_train == 0).sum() / len(y_train) * 100:.1f}%)")
    print(f"  Poisonous (1): {(y_train == 1).sum()} ({(y_train == 1).sum() / len(y_train) * 100:.1f}%)")
    
    return X_train_normalized, X_test_normalized, y_train, y_test, feature_columns, feature_scaler

def build_classifier(X_train, y_train):
    """Train the Gradient Boosting Classifier with optimized parameters"""
    print("\nConstructing Gradient Boosting Classifier with optimized hyperparameters...")
    
    # Use more aggressive parameters for better performance
    classifier = GradientBoostingClassifier(
        learning_rate=0.1,        # Slightly higher for faster learning
        n_estimators=200,          # More trees for better accuracy
        max_depth=5,              # Deeper trees to capture complexity
        min_samples_split=4,
        min_samples_leaf=2,
        subsample=0.8,            # Use 80% of samples for each tree
        random_state=42,
        verbose=0
    )
    
    # Train the model
    classifier.fit(X_train, y_train)
    
    print("Classifier construction completed successfully!")
    return classifier

def assess_performance(classifier, X_train, X_test, y_train, y_test):
    """Evaluate the trained model"""
    print("\n" + "="*60)
    print("PERFORMANCE ASSESSMENT REPORT")
    print("="*60)
    
    # Training accuracy
    y_train_pred = classifier.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"\nTraining Precision: {train_acc:.4f} ({train_acc*100:.2f}%)")
    
    # Test accuracy
    y_pred = classifier.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Validation Precision: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Classification report
    print("\nDetailed Classification Report (Validation Set):")
    print(classification_report(y_test, y_pred, 
                                target_names=['Edible (0)', 'Poisonous (1)']))
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"                  Predicted Edible  Predicted Poisonous")
    print(f"Actual Edible           {conf_matrix[0][0]:5d}              {conf_matrix[0][1]:5d}")
    print(f"Actual Poisonous        {conf_matrix[1][0]:5d}              {conf_matrix[1][1]:5d}")
    
    # Calculate specific metrics
    missed_danger = conf_matrix[1][0]  # Poisonous predicted as Edible (DANGEROUS!)
    false_alarms = conf_matrix[0][1]  # Edible predicted as Poisonous (Safe error)
    
    print(f"\n⚠️  Critical Misses (Poisonous → Edible): {missed_danger} ({missed_danger/len(y_test)*100:.2f}%)")
    print(f"    ^ CRITICAL: Potential safety hazard!")
    print(f"✓  False Alarms (Edible → Poisonous): {false_alarms} ({false_alarms/len(y_test)*100:.2f}%)")
    print(f"    ^ SAFE: Conservative prediction")
    
    return test_acc

def export_artifacts(classifier, feature_names, scaler, filename='mushroom_model.pkl'):
    """Save the trained model and scaler to a pickle file"""
    print(f"\nSerializing artifacts to {filename}...")
    
    artifact_bundle = {
        'model': classifier,
        'scaler': scaler,
        'features': feature_names,
        'feature_importances': dict(zip(feature_names, classifier.feature_importances_))
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(artifact_bundle, f)
    
    print(f"Artifacts successfully serialized!")
    
    # Display feature importances
    print("\nFeature Impact Analysis:")
    print("-" * 40)
    for feature, importance in sorted(
        artifact_bundle['feature_importances'].items(), 
        key=lambda x: x[1], 
        reverse=True
    ):
        print(f"{feature:20s}: {importance:.4f}")

def run_live_validation(classifier, scaler, df, feature_names):
    """Test the model with actual samples from the dataset"""
    print("\n" + "="*60)
    print("LIVE SAMPLE VALIDATION")
    print("="*60)
    
    # Test with poisonous samples
    toxic_examples = df[df['class'] == 1].head(3)
    print("\nValidating TOXIC samples:")
    for idx, row in toxic_examples.iterrows():
        sample_vector = scaler.transform(row[feature_names].values.reshape(1, -1))
        prediction = classifier.predict(sample_vector)[0]
        confidence = classifier.predict_proba(sample_vector)[0]
        result_status = "✓ VALIDATED" if prediction == 1 else "✗ FAILED"
        print(f"  Sample {idx}: Prediction={prediction} ({'POISONOUS' if prediction==1 else 'EDIBLE'}) "
              f"Conf=[E:{confidence[0]:.2f}, P:{confidence[1]:.2f}] {result_status}")
    
    # Test with edible samples
    safe_examples = df[df['class'] == 0].head(3)
    print("\nValidating SAFE samples:")
    for idx, row in safe_examples.iterrows():
        sample_vector = scaler.transform(row[feature_names].values.reshape(1, -1))
        prediction = classifier.predict(sample_vector)[0]
        confidence = classifier.predict_proba(sample_vector)[0]
        result_status = "✓ VALIDATED" if prediction == 0 else "✗ FAILED"
        print(f"  Sample {idx}: Prediction={prediction} ({'POISONOUS' if prediction==1 else 'EDIBLE'}) "
              f"Conf=[E:{confidence[0]:.2f}, P:{confidence[1]:.2f}] {result_status}")

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("MUSHROOM CLASSIFICATION SYSTEM - PARTHA SARATHI R")
    print("="*60)
    
    # Load data
    dataset = acquire_dataset('mushroom_classification.csv')
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_names, scaler = process_features(dataset)
    
    # Train model
    classifier = build_classifier(X_train, y_train)
    
    # Evaluate model
    model_accuracy = assess_performance(classifier, X_train, X_test, y_train, y_test)
    
    # Test with real samples
    run_live_validation(classifier, scaler, dataset, feature_names)
    
    # Save model
    export_artifacts(classifier, feature_names, scaler)
    
    print("\n" + "="*60)
    print("EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nFinal Validation Precision: {model_accuracy*100:.2f}%")
    print("Artifact File: mushroom_model.pkl")
    print("\n✓ System ready for deployment!")
    print("  Run: streamlit run app.py")

if __name__ == "__main__":
    main()
