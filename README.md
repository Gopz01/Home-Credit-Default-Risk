# Home Credit Default Risk: A Strategic Financial Pipeline

## 📊 Executive Summary
Developed a machine learning pipeline to predict loan default risk for 300,000+ applicants.
The final **Stacking Ensemble model (AUC 0.788)** outperforms the baseline by **4.2%** and is projected to generate **$1.55M in additional net profit** per portfolio cycle.

## 🛠️ Tech Stack
* **Core:** Python, Pandas, NumPy
* **ML Engines:** LightGBM, XGBoost, CatBoost, Scikit-Learn
* **Techniques:** Stacking Ensembles, Bayesian Optimization (Optuna), Behavioral Feature Engineering
* **Business:** Financial Stress Testing, Profit-Curve Optimization

## 📈 Key Results
| Model | AUC Score | Est. Portfolio Profit |
| :--- | :--- | :--- |
| Random Forest (Baseline) | 0.746 | $367.2M |
| **Stacking Ensemble** | **0.788** | **$368.8M** |
| **Net Impact** | **+4.2%** | **+$1.55M** |

## 🧠 Engineering Highlights
1.  **Bureau Intelligence:** Aggregated 5M+ rows of external credit history to create behavioral features (e.g., `BUREAU_ACTIVE_LOANS_PCT`).
2.  **Robust Handling:** Managed class imbalance (8% default rate) without destructive synthetic oversampling.
3.  **Resilience:** Passed stress tests simulating a 20% rise in global default rates.

## 📂 Notebook Guide
* `01-03`: Data Cleaning & Anomaly Detection
* `04-07`: Advanced Feature Engineering (Domain + Aggregations)
* `08-09`: Optimization & Stacking (LGBM + XGB + CatBoost)
* `10`: **Business Impact Analysis (ROI & Strategy)**
