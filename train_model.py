"""
Mushroom Classification Model Training Script - IMPROVED VERSION
This script trains an optimized Gradient Boosting Classifier with better preprocessing
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

def load_data(filepath):
    """Load the mushroom classification dataset"""
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    print(f"Dataset loaded successfully! Shape: {df.shape}")
    return df

def prepare_data(df):
    """Prepare features and target variables with proper preprocessing"""
    print("\nPreparing data...")
    
    # Select features and target
    features = ['cap_diameter', 'cap_shape', 'gill_attachment', 'gill_color', 
                'stem_height', 'stem_width', 'stem_color', 'season']
    target = 'class'
    
    X = df[features]
    y = df[target]
    
    # Split the data first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Scale the features for better performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Class distribution in training set:")
    print(f"  Edible (0): {(y_train == 0).sum()} ({(y_train == 0).sum() / len(y_train) * 100:.1f}%)")
    print(f"  Poisonous (1): {(y_train == 1).sum()} ({(y_train == 1).sum() / len(y_train) * 100:.1f}%)")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, features, scaler

def train_model(X_train, y_train):
    """Train the Gradient Boosting Classifier with optimized parameters"""
    print("\nTraining Gradient Boosting Classifier with optimized parameters...")
    
    # Use more aggressive parameters for better performance
    model = GradientBoostingClassifier(
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
    model.fit(X_train, y_train)
    
    print("Model training completed!")
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate the trained model"""
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    # Training accuracy
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    
    # Test accuracy
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Classification report
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred, 
                                target_names=['Edible (0)', 'Poisonous (1)']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"                  Predicted Edible  Predicted Poisonous")
    print(f"Actual Edible           {cm[0][0]:5d}              {cm[0][1]:5d}")
    print(f"Actual Poisonous        {cm[1][0]:5d}              {cm[1][1]:5d}")
    
    # Calculate specific metrics
    false_negatives = cm[1][0]  # Poisonous predicted as Edible (DANGEROUS!)
    false_positives = cm[0][1]  # Edible predicted as Poisonous (Safe error)
    
    print(f"\n⚠️  False Negatives (Poisonous → Edible): {false_negatives} ({false_negatives/len(y_test)*100:.2f}%)")
    print(f"    ^ DANGEROUS: These would poison someone!")
    print(f"✓  False Positives (Edible → Poisonous): {false_positives} ({false_positives/len(y_test)*100:.2f}%)")
    print(f"    ^ SAFE: Just being cautious")
    
    return test_accuracy

def save_model(model, features, scaler, filename='mushroom_model.pkl'):
    """Save the trained model and scaler to a pickle file"""
    print(f"\nSaving model to {filename}...")
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': features,
        'feature_importances': dict(zip(features, model.feature_importances_))
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved successfully!")
    
    # Display feature importances
    print("\nFeature Importances:")
    print("-" * 40)
    for feature, importance in sorted(
        model_data['feature_importances'].items(), 
        key=lambda x: x[1], 
        reverse=True
    ):
        print(f"{feature:20s}: {importance:.4f}")

def test_with_real_samples(model, scaler, df, features):
    """Test the model with actual samples from the dataset"""
    print("\n" + "="*60)
    print("TESTING WITH REAL DATASET SAMPLES")
    print("="*60)
    
    # Test with poisonous samples
    poisonous_samples = df[df['class'] == 1].head(3)
    print("\nTesting POISONOUS mushrooms:")
    for idx, row in poisonous_samples.iterrows():
        X_sample = scaler.transform(row[features].values.reshape(1, -1))
        pred = model.predict(X_sample)[0]
        prob = model.predict_proba(X_sample)[0]
        status = "✓ CORRECT" if pred == 1 else "✗ WRONG"
        print(f"  Sample {idx}: Predicted={pred} ({'POISONOUS' if pred==1 else 'EDIBLE'}) "
              f"Prob=[E:{prob[0]:.2f}, P:{prob[1]:.2f}] {status}")
    
    # Test with edible samples
    edible_samples = df[df['class'] == 0].head(3)
    print("\nTesting EDIBLE mushrooms:")
    for idx, row in edible_samples.iterrows():
        X_sample = scaler.transform(row[features].values.reshape(1, -1))
        pred = model.predict(X_sample)[0]
        prob = model.predict_proba(X_sample)[0]
        status = "✓ CORRECT" if pred == 0 else "✗ WRONG"
        print(f"  Sample {idx}: Predicted={pred} ({'POISONOUS' if pred==1 else 'EDIBLE'}) "
              f"Prob=[E:{prob[0]:.2f}, P:{prob[1]:.2f}] {status}")

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("MUSHROOM CLASSIFICATION MODEL TRAINING - IMPROVED")
    print("="*60)
    
    # Load data
    df = load_data('mushroom_classification.csv')
    
    # Prepare data
    X_train, X_test, y_train, y_test, features, scaler = prepare_data(df)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    accuracy = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Test with real samples
    test_with_real_samples(model, scaler, df, features)
    
    # Save model
    save_model(model, features, scaler)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nFinal Test Accuracy: {accuracy*100:.2f}%")
    print("Model file: mushroom_model.pkl")
    print("\n✓ Model is ready for use in the Streamlit app!")
    print("  Run: streamlit run app.py")

if __name__ == "__main__":
    main()
