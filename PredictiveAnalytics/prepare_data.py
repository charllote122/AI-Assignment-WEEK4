# prepare_data.py
import os
import shutil
from sklearn.model_selection import train_test_split
import random

def create_organized_splits():
    """Create train/validation/test splits from your training_set"""
    
    print("=== PREPARING DATA SPLITS ===")
    
    source_dir = "data/training_set"
    output_dir = "data/processed/train_val_test"
    
    # Check if source exists
    if not os.path.exists(source_dir):
        print(f"❌ Error: {source_dir} does not exist!")
        print("Please make sure you're in the correct directory")
        return
    
    print("✅ Found training_set directory")
    
    # Create output directories
    splits = ['train', 'validation', 'test']
    classes = ['benign', 'malignant']
    
    for split in splits:
        for class_name in classes:
            dir_path = os.path.join(output_dir, split, class_name)
            os.makedirs(dir_path, exist_ok=True)
            print(f"📁 Created: {dir_path}")
    
    print("\n📊 Processing images...")
    
    # Process each class
    total_images = 0
    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        
        if not os.path.exists(class_path):
            print(f"❌ Error: {class_path} does not exist!")
            continue
            
        # Get all image files
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
        
        print(f"📸 Found {len(image_files)} {class_name} images")
        total_images += len(image_files)
        
        if len(image_files) == 0:
            print(f"⚠️  No images found in {class_path}")
            continue
        
        # Split: 70% train, 15% validation, 15% test
        train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
        
        print(f"   📋 Splitting: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
        
        # Copy files to respective directories
        copied_count = 0
        for file in train_files:
            src = os.path.join(class_path, file)
            dst = os.path.join(output_dir, 'train', class_name, file)
            shutil.copy2(src, dst)
            copied_count += 1
            
        for file in val_files:
            src = os.path.join(class_path, file)
            dst = os.path.join(output_dir, 'validation', class_name, file)
            shutil.copy2(src, dst)
            copied_count += 1
            
        for file in test_files:
            src = os.path.join(class_path, file)
            dst = os.path.join(output_dir, 'test', class_name, file)
            shutil.copy2(src, dst)
            copied_count += 1
        
        print(f"   ✅ Copied {copied_count} {class_name} images")
    
    print(f"\n🎯 Data preparation complete!")
    print(f"📁 Organized data in: {output_dir}")
    print(f"📊 Total images processed: {total_images}")

def analyze_final_structure():
    """Check the final organized structure"""
    print("\n" + "="*50)
    print("FINAL DATA STRUCTURE ANALYSIS")
    print("="*50)
    
    base_path = "data/processed/train_val_test"
    
    if not os.path.exists(base_path):
        print("❌ Processed data directory not found!")
        return
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    for split in ['train', 'validation', 'test']:
        split_path = os.path.join(base_path, split)
        if os.path.exists(split_path):
            benign_path = os.path.join(split_path, 'benign')
            malignant_path = os.path.join(split_path, 'malignant')
            
            benign_count = len(os.listdir(benign_path)) if os.path.exists(benign_path) else 0
            malignant_count = len(os.listdir(malignant_path)) if os.path.exists(malignant_path) else 0
            total_split = benign_count + malignant_count
            
            print(f"📊 {split.upper():>12}: {benign_count:>3} benign, {malignant_count:>3} malignant, {total_split:>3} total")
            
            if split == 'train':
                total_train = total_split
            elif split == 'validation':
                total_val = total_split
            elif split == 'test':
                total_test = total_split
    
    print("="*50)
    print(f"📈 GRAND TOTAL: {total_train + total_val + total_test} images")
    print(f"   → Train: {total_train} ({total_train/(total_train+total_val+total_test)*100:.1f}%)")
    print(f"   → Validation: {total_val} ({total_val/(total_train+total_val+total_test)*100:.1f}%)")
    print(f"   → Test: {total_test} ({total_test/(total_train+total_val+total_test)*100:.1f}%)")

def check_original_data():
    """Check what's in the original data folder"""
    print("\n" + "="*50)
    print("ORIGINAL DATA CHECK")
    print("="*50)
    
    original_path = "data/training_set"
    if os.path.exists(original_path):
        benign_path = os.path.join(original_path, 'benign')
        malignant_path = os.path.join(original_path, 'malignant')
        
        benign_count = len(os.listdir(benign_path)) if os.path.exists(benign_path) else 0
        malignant_count = len(os.listdir(malignant_path)) if os.path.exists(malignant_path) else 0
        
        print(f"📁 Original training_set:")
        print(f"   👶 Benign: {benign_count} images")
        print(f"   🦠 Malignant: {malignant_count} images")
        print(f"   📦 Total: {benign_count + malignant_count} images")
    else:
        print("❌ Original training_set not found!")

if __name__ == "__main__":
    check_original_data()
    create_organized_splits()
    analyze_final_structure()
    
    print("\n🎉 READY FOR NEXT STEP!")
    print("👉 Now run: python train_traditional_ml.py")