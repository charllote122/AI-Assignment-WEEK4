import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import cross_val_score
import cv2

print("=== BREAST CANCER RESOURCE ALLOCATION - MODEL TRAINING ===")

def extract_image_features(image_path):
    """Extract features from images for machine learning"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Extract multiple features
        features = [
            np.mean(gray),           # Mean intensity
            np.std(gray),            # Standard deviation
            np.median(gray),         # Median intensity
            gray.shape[1],           # Width
            gray.shape[0],           # Height
            gray.shape[1] / gray.shape[0],  # Aspect ratio
            np.min(gray),            # Min intensity
            np.max(gray),            # Max intensity
            np.percentile(gray, 25), # 25th percentile
            np.percentile(gray, 75), # 75th percentile
        ]
        
        # Add some texture features
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        features.extend([
            np.mean(sobelx),
            np.mean(sobely),
            np.std(sobelx),
            np.std(sobely)
        ])
        
        return features
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def create_dataset():
    """Create dataset from processed images"""
    print("📊 Creating dataset from images...")
    
    base_path = "data/processed/train_val_test"
    splits = ['train', 'validation', 'test']
    
    data = []
    labels = []
    split_info = []
    
    feature_names = [
        'mean_intensity', 'std_intensity', 'median_intensity', 'width', 'height',
        'aspect_ratio', 'min_intensity', 'max_intensity', 'percentile_25', 'percentile_75',
        'sobelx_mean', 'sobely_mean', 'sobelx_std', 'sobely_std'
    ]
    
    for split in splits:
        split_path = os.path.join(base_path, split)
        
        for class_name in ['benign', 'malignant']:
            class_path = os.path.join(split_path, class_name)
            
            if not os.path.exists(class_path):
                continue
                
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
            
            print(f"   Processing {len(image_files)} {class_name} images from {split}...")
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                features = extract_image_features(img_path)
                
                if features is not None:
                    data.append(features)
                    labels.append(0 if class_name == 'benign' else 1)  # 0=benign, 1=malignant
                    split_info.append(split)
    
    # Convert to numpy arrays
    X = np.array(data)
    y = np.array(labels)
    splits = np.array(split_info)
    
    print(f"✅ Created dataset with {X.shape[0]} samples and {X.shape[1]} features")
    return X, y, splits, feature_names

def train_random_forest(X_train, y_train, X_val, y_val, feature_names):
    """Train and evaluate Random Forest model"""
    print("\n🌲 Training Random Forest Classifier...")
    
    # Create and train the model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'  # Important for imbalanced data
    )
    
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = rf_model.predict(X_train)
    y_val_pred = rf_model.predict(X_val)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    
    print("📈 Training Results:")
    print(f"   Training Accuracy: {train_accuracy:.4f}")
    print(f"   Validation Accuracy: {val_accuracy:.4f}")
    print(f"   Training F1-Score: {train_f1:.4f}")
    print(f"   Validation F1-Score: {val_f1:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔝 Top 5 Most Important Features:")
    for i, row in feature_importance.head(5).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    return rf_model, y_val_pred

def evaluate_model(model, X_test, y_test, feature_names):
    """Comprehensive model evaluation"""
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    print("\n" + "="*60)
    print("FINAL MODEL EVALUATION")
    print("="*60)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"🎯 Test Set Performance:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    # Detailed classification report
    print(f"\n📊 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title('Confusion Matrix - Breast Cancer Classification')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature Importance Plot
    plt.figure(figsize=(12, 8))
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.title('Feature Importance - Random Forest')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, f1

def save_model_and_results(model, scaler, accuracy, f1, feature_names):
    """Save model and results"""
    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'performance': {
            'accuracy': accuracy,
            'f1_score': f1
        }
    }
    
    joblib.dump(model_data, 'models/random_forest_model.pkl')
    
    # Save performance metrics
    metrics_df = pd.DataFrame({
        'metric': ['accuracy', 'f1_score'],
        'value': [accuracy, f1]
    })
    metrics_df.to_csv('results/performance_metrics.csv', index=False)
    
    print(f"\n💾 Model and results saved!")
    print(f"   📁 Model: models/random_forest_model.pkl")
    print(f"   📊 Metrics: results/performance_metrics.csv")
    print(f"   📈 Plots: results/confusion_matrix.png, results/feature_importance.png")

def main():
    # Create dataset
    X, y, splits, feature_names = create_dataset()
    
    # Split data
    train_mask = splits == 'train'
    val_mask = splits == 'validation'
    test_mask = splits == 'test'
    
    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training: {X_train.shape[0]} samples")
    print(f"   Validation: {X_val.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")
    print(f"   Features: {X_train.shape[1]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model, y_val_pred = train_random_forest(X_train_scaled, y_train, X_val_scaled, y_val, feature_names)
    
    # Final evaluation on test set
    accuracy, f1 = evaluate_model(model, X_test_scaled, y_test, feature_names)
    
    # Save everything
    save_model_and_results(model, scaler, accuracy, f1, feature_names)
    
    print(f"\n🎉 TRAINING COMPLETE!")
    print(f"📈 Final Test Accuracy: {accuracy:.4f}")
    print(f"📈 Final Test F1-Score: {f1:.4f}")
    print(f"\n🚀 Next: You can use the trained model for predictions!")

if __name__ == "__main__":
    main()