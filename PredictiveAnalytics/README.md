# Breast Cancer Resource Allocation - Predictive Analytics

## 📋 Project Overview
A machine learning system that predicts breast cancer case priorities (High/Low) for efficient medical resource allocation using medical image analysis.

**🎯 Live Demo:** [https://charllote122-ai-assignment-week4-predictiveanalyticsapp-q2okrr.streamlit.app/](https://charllote122-ai-assignment-week4-predictiveanalyticsapp-q2okrr.streamlit.app/)

**🎯 Goal:** Preprocess medical images, train Random Forest model, predict priority levels, and deploy as a web application for healthcare resource optimization.

## 📊 Performance Results
- **Accuracy:** 80.95%
- **F1-Score:** 62.79%
- **Dataset:** 1,112 medical images (791 benign, 321 malignant)
- **High Priority Recall:** 55.1%
- **Low Priority Recall:** 91.6%

## 🚀 Quick Start

### 1. Use the Live Web Application
🌐 **Live Demo:** [Access the deployed app here](https://charllote122-ai-assignment-week4-predictiveanalyticsapp-q2okrr.streamlit.app/)

### 2. Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Launch the web app
streamlit run app.py

# Prepare data splits
python prepare_data.py

# Train Random Forest model
python train_traditional_ml.py

# Evaluate model performance
python prepare_data
python train_traditional_ml.py
python evaluate_model.py

PredictiveAnalytics/
├── data/
│   ├── training_set/              # Original medical images
│   └── processed/train_val_test/  # Processed data splits
├── models/
│   └── random_forest_model.pkl    # Trained model (80.95% accuracy)
├── results/                       # Performance metrics & plots
├── app.py                         # Main web application
├── train_traditional_ml.py        # Model training script
├── evaluate_model.py              # Model evaluation & analysis
├── prepare_data.py                # Data preprocessing & splitting
├── requirements.txt               # Python dependencies
└── packages.txt                   # System dependencies (OpenCV)