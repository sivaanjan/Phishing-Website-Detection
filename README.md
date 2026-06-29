# 🛡️ Phishing Website Detection Using Machine Learning

A machine learning-based phishing website detection system that identifies malicious URLs using lexical, structural, and heuristic features. The project employs a Random Forest classifier trained on 34 engineered features and provides real-time URL analysis through an interactive Streamlit web application.

---

## 📌 Project Overview

Phishing attacks are one of the most common cyber threats, targeting users to steal sensitive information such as login credentials, banking details, and personal data. Traditional blacklist-based approaches struggle to detect newly created phishing websites. This project addresses this challenge by using machine learning to analyze URL characteristics and classify websites as **Legitimate** or **Phishing** in real time.

---

## 🚀 Features

- Real-time phishing URL detection
- Random Forest machine learning model
- 34 engineered URL-based features
- URL lexical and structural analysis
- Trust Score and Threat Probability
- Feature Importance Analysis
- Enterprise Security Checks
- Interactive Streamlit Dashboard
- Explainable predictions

---

## 🛠️ Technologies Used

- Python
- Scikit-learn
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Pickle

---

## 📂 Repository Structure

```
Phishing-Website-Detection/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── src/
│   ├── app.py
│   ├── model_train.py
│   ├── eda.py
│   ├── dataset.csv
│   ├── phishing_model.pkl
│   └── project_metrics.pkl
│
└── report/
    └── pythonreport.pdf
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Phishing-Website-Detection.git
```

Navigate to the project directory

```bash
cd Phishing-Website-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
cd src
streamlit run app.py
```

---

## 🔄 Workflow

1. Load the phishing URL dataset.
2. Perform exploratory data analysis.
3. Extract and engineer URL features.
4. Train the Random Forest classifier.
5. Save the trained model and evaluation metrics.
6. Launch the Streamlit application.
7. Analyze URLs in real time.
8. Display prediction, trust score, and detailed feature analysis.

---

## 📊 Applications

- Phishing Website Detection
- Enterprise Cybersecurity
- URL Threat Intelligence
- Browser Security Tools
- Security Operations Centers (SOC)
- Email Security Systems
- Web Security Research

---

## 🔮 Future Enhancements

- Deep Learning-based URL Classification
- Browser Extension Integration
- Dynamic Link Monitoring
- Deep Packet Inspection
- Cloud Deployment
- Real-time Threat Intelligence Integration

---

**Thota Siva Anjan Kumar**

M.Tech Data Science

SRM University-AP

---
