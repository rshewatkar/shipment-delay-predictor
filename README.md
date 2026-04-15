---
title: Shipment Delay Predictor
emoji: 🚚
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.32.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# 🚚 Shipment Delay Predictor — Logistics ML System

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red.svg)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-ML-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-green.svg)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen.svg)
![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Deployed-yellow.svg)

---

## 🚀 Live Demo

👉 **[Click Here to Try the App](https://rshewatkar-shipment-delay-predictor.hf.space)**

> Deployed on Hugging Face Spaces with CI/CD (GitHub Actions)

---

## 📌 Project Overview

An end-to-end Machine Learning system that predicts **shipment delay risk before dispatch**.

Built for logistics and courier companies to:
- Reduce late deliveries
- Improve customer satisfaction
- Enable proactive decision-making

> ⚠️ **Key Insight:** ~60% of shipments are delayed — this model flags high-risk orders early.

---

### 🏠 Dashboard Overview
![Dashboard Main](reports/screenshots/dashboard-main.png)

### 🔮 Prediction Flow
![Dashboard Result](reports/screenshots/dashboard-result.png)

### ⚠️ Risk Analysis
![Dashboard Risk](reports/screenshots/dashboard-risk.png)

---

## 🎯 Problem Statement

Late deliveries lead to:
- Refunds & penalties
- Customer dissatisfaction
- Operational inefficiencies

This system predicts delay probability at booking time — enabling early intervention.

---

## 📊 Dataset Overview

| Property | Details |
|---|---|
| Source | Kaggle (E-Commerce Shipping Dataset) |
| Records | 10,999 |
| Features | 12 |
| Target | `Reached.on.Time_Y.N` |
| Delay Rate | 59.7% |

---

## 🔍 Key Insights (EDA)

- 🎯 **Discount > 10% → High delay probability**
- ⚖️ **2–4 kg weight → Highest delay rate**
- 🚢 **Ship mode → Most delays**
- 📞 **Customer calls ≥ 4 → Strong delay signal**

---

## ⚙️ Feature Engineering

| Feature | Description |
|---|---|
| `high_discount` | Flag for discount > 10% |
| `weight_bucket` | Categorized weight ranges |
| `high_call_risk` | Calls ≥ 4 |

---

## 🤖 Model Performance

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Decision Tree | 64.18% | 69.57% | 71.06% | 70.31% | 62.53% |
| Random Forest | 66.32% | 77.82% | 60.93% | 68.35% | 73.49% |
| **XGBoost ✅** | **67.00%** | **73.77%** | **69.38%** | **71.51%** | **75.29%** |

---

## 💡 Business Impact

- 📈 69% of delayed shipments correctly identified
- 🎯 73% precision in delay prediction
- 📉 Potential **15–25% reduction in late deliveries**

---

## 🖥️ App Features

| Feature | Description |
|---|---|
| 📥 Input Form | Enter shipment details |
| 📊 Prediction | Delay probability output |
| ⚠️ Risk Analysis | Key contributing factors |
| 📢 Recommendations | Actionable insights |

---

## 📁 Project Structure

```bash
shipment-delay-predictor/
│
├── app.py                          # Streamlit app (entry point)
├── requirements.txt
├── README.md
│
├── .github/
│   └── workflows/
│       └── sync-to-hub.yml         # CI/CD pipeline (GitHub → Hugging Face)
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── preprocessing.ipynb
│   └── modeling.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── models/                        # Trained ML models
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── label_encoders.pkl
│
├── reports/                       
│   ├── figures/
│   │   ├──01_target_distribution.png
│   │   ├──02_delay_by_warehouse.png
│   │   ├──03_delay_by_mode.png
│   │   ├──04_discount_effect.png
│   │   ├──05_weight_effect.png
│   │   ├──06_customer_calls.png
│   │   ├──07_correlation_heatmap.png
│   │   ├──08_model_comparison.png
│   │   ├──09_confusion_matrix.png
│   │   ├──10_roc_curve.png
│   │   └──11_feature_importance.png
│   │ 
│   └── screenshots/
│        ├──dashboard-main.png
│        ├──dashboard-result.png
│        └──dashboard-risk.png

```

---

## ⚙️ How to Run Locally


**1. Clone the repository**

```bash
git clone https://github.com/rshewatkar/shipment-delay-predictor.git
cd shipment-delay-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

---

## 🔁 ML Pipeline

1. Data Collection
2. Data Preprocessing
3. Feature Engineering
4. Model Training (XGBoost)
5. Model Evaluation
6. Deployment (Streamlit + Hugging Face)

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML | Scikit-learn, XGBoost |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| App | Streamlit |
| Deployment | Hugging Face Spaces |
| CI/CD | GitHub Actions |

---

## 👤 Author

**Rahul Shewatkar**

- 💼 [LinkedIn](https://www.linkedin.com/in/rahul-shewatkar-ml-engineer/)
- 🐙 [GitHub](https://github.com/rshewatkar)
- 🤗 [Hugging Face](https://huggingface.co/rshewatkar)

---

## 📄 License

MIT License