# Detecting Insider Threats in Cybersecurity Using Machine Learning and Deep Learning Techniques

## 📌 Project Overview
This project focuses on detecting insider threats in cybersecurity using machine learning (ML) and deep learning (DL) techniques. The objective is to analyze user behavior, identify anomalous activities, and enhance security by detecting potential insider threats in an organizational network.

## 📂 Dataset
- **Dataset Name:**  UEBA (User and Entity Behavior Analytics)
- **Source:** Kaggle
- **Records:** 2000 user activity records
- **Keywords Used for Dataset Search:**
  - Insider threat detection
  - Cybersecurity insider threat
  - User behavior analytics
  - Employee activity monitoring
  - Anomalous activity detection
  - Login activity dataset

## 🛠 Technologies Used
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, TensorFlow, PyTorch, Matplotlib, Seaborn
- **Machine Learning Algorithms:** Random Forest, SVM, Decision Trees
- **Deep Learning Models:** LSTM, Autoencoders, ANN
- **Frameworks:** TensorFlow, Keras, PyTorch
- **Visualization Tools:** Matplotlib, Seaborn

## Features Extracted
- User login timestamps
- IP address tracking
- Unusual file access patterns
- Privileged account behavior
- Abnormal working hours
- Suspicious network activity

## Implementation Steps
1. **Data Preprocessing**
   - Handling missing values
   - Feature selection and scaling
2. **Exploratory Data Analysis (EDA)**
   - Visualizing user behavior trends
   - Detecting anomalies using statistical methods
3. **Model Training**
   - ML models: Decision Tree, Random Forest, SVM
   - DL models: LSTM, Autoencoder
4. **Evaluation Metrics**
   - Accuracy, Precision, Recall, F1-score
   - ROC-AUC for anomaly detection
5. **Deployment**
   - Flask/Django API for real-time threat detection

## Results & Performance
| Model | Accuracy | Precision | Recall | F1-Score |
|--------|----------|------------|--------|-----------|
| Random Forest | 92.3% | 91.7% | 90.2% | 90.9% |
| LSTM Autoencoder | 94.5% | 93.8% | 92.1% | 92.9% |
| SVM | 88.7% | 87.2% | 86.5% | 86.8% |

## How to Run
### Repository
```bash
git https://github.com/Aniketkr63/InsiderThreatDetectionCybersecurity.git
cd InsiderThreatDetectionCybersecurity
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
### Run the Model
```bash
python main.py
```
### (Optional) Run API Server for Deployment
```bash
python app.py
```

## Future Improvements
- Improve dataset quality with real-time monitoring data
- Enhance deep learning model performance
- Integrate with SIEM tools for better threat intelligence
- Develop a dashboard for real-time visualization

## Contact
**Author:** Aniket Kr Singh  
**Email:** aniketsingh7248@gmail.com  
**GitHub:** [Aniketkr63](https://github.com/Aniketkr63)  
**LinkedIn:** [Aniket Singh](https://www.linkedin.com/in/aniket-singh-416229261)
