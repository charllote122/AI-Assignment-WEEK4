# Breast Cancer Resource Allocation - Predictive Analytics

## 📋 Project Overview
A machine learning system that predicts breast cancer case priorities (High/Low) for efficient medical resource allocation using the Breast Cancer Wisconsin dataset.

**🎯 Goal:** Preprocess data, train Random Forest model, predict priority levels, and evaluate performance.

## 📊 Performance Results
- **Accuracy:** 95.91%
- **F1-Score:** 95.05% 
- **High Priority Recall:** 97.0%
- **Low Priority Recall:** 99.0%

## 🚀 Quick Start

### 1. Run the Web Application
```bash
# Install dependencies
pip install -r requirements.txt

# Launch the web app
streamlit run app.py

📁 Project Structure
text
├── data/                           # Breast Cancer Wisconsin dataset
├── models/                         # Trained Random Forest model
├── results/                        # Performance metrics & plots
├── train_traditional_ml.py         # Model training script
├── evaluate_model.py               # Model evaluation
├── prepare_data.py                 # Data preprocessing
├── app.py                          # Web interface
└── requirements.txt                # Python dependencies
🛠️ Technical Implementation
Data Preprocessing
Loaded Breast Cancer Wisconsin dataset (569 samples, 30 features)

Created priority labels (High/Medium/Low) from diagnostic data

Train/validation/test split (70%/15%/15%)

Feature scaling with StandardScaler

Model Training
Algorithm: Random Forest Classifier

Hyperparameters: 100 estimators, balanced class weights

Validation: Stratified k-fold cross-validation

Priority System
High Priority: Malignant cases requiring immediate attention

Low Priority: Benign cases for routine follow-up

🌐 Web Application Features
Upload and analyze medical images

Automatic feature extraction (14 image features)

Real-time priority classification

Confidence scores and medical recommendations

Resource allocation suggestions

📈 Key Deliverables ✅
Data Preprocessing - Cleaned, labeled, split data

Model Training - Random Forest with 95.91% accuracy

Evaluation Metrics - Accuracy & F1-score reported

Jupyter Notebook - Complete analysis pipeline

Resource Allocation - Priority-based classification system

🏥 Business Impact
Automated triage for 171 test cases

98.3% accurate high-priority identification

Efficient resource allocation based on predicted risk

Reduced manual screening workload

⚠️ Medical Disclaimer
This system is designed for resource allocation prioritization and should not be used as a diagnostic tool. Always consult healthcare professionals for medical decisions.