# Home Credit Default Risk: End-to-End ML Pipeline

## 📖 Project Overview
This project builds a credit risk scoring model to predict the probability of loan default for clients with little to no credit history. By integrating alternative data sources (telecom, transactional, and bureau data), the final model achieves an **AUC of 0.788**, outperforming the standard baseline by **4.2%**.

Beyond the technical implementation, this repository includes a **Strategic Business Impact Analysis** (Notebook 10), translating model performance into a projected **$1.55M net profit increase** per 300,000 applicants.

* **Data Source:** [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/c/home-credit-default-risk/data)
* **Status:** Completed

---

## 📊 Executive Summary & Results

| Metric | Baseline (Random Forest) | Champion (Stacking Ensemble) | Lift |
| :--- | :--- | :--- | :--- |
| **ROC AUC** | 0.746 | **0.788** | +4.2% |
| **Max Profit** | $367.2M | **$368.8M** | +$1.55M |
| **Recession Resilience** | Negative ROI | **Positive ROI** | Robust |

**Key Insight:** The inclusion of behavioral features from external credit bureaus (Notebook 06) and previous loan history (Notebook 07) provided the single largest performance jump, validating the importance of historical behavioral data over static application demographics.

---

## 🛠️ Technical Architecture

### **Feature Engineering Strategy**
* **Domain Knowledge:** Constructed financial ratios (e.g., `CREDIT_TO_ANNUITY_RATIO`) to capture leverage.
* **Relational Aggregations:** Aggregated 5+ million rows from auxiliary tables (`bureau`, `installments_payments`) using statistical grouping (Mean, Max, Sum) to create a 360-degree customer view.
* **Target Encoding:** Implemented smooth target encoding for high-cardinality categorical variables inside Cross-Validation loops to prevent leakage.

### **Modeling Pipeline**
1.  **Baseline:** Random Forest & LightGBM (Single Models).
2.  **Optimization:** Bayesian Hyperparameter Tuning via **Optuna**.
3.  **Ensemble:** A Level-2 Stacking Classifier combining **LightGBM** (Leaf-wise), **XGBoost** (Level-wise), and **CatBoost** (Categorical).

---

## 📂 Repository Structure

### **Phase 1: Data Foundations**
* **[01_Exploratory_Data_Analysis.ipynb](notebooks/01_Exploratory_Data_Analysis.ipynb):** Initial inspection, distribution analysis, and correlation heatmaps.
* **[02_Data_Cleaning.ipynb](notebooks/02_Data_Cleaning.ipynb):** Handling missing values (IterativeImputer) and data casting.
* **[03_Anomaly_Detection.ipynb](notebooks/03_Anomaly_Detection.ipynb):** Identifying and capping outliers (e.g., `DAYS_EMPLOYED` artifacts).

### **Phase 2: Feature Engineering (The "Planets")**
* **[04_Domain_Features.ipynb](notebooks/04_Domain_Features.ipynb):** Engineering financial ratios based on domain expertise.
* **[05_Baseline_Model.ipynb](notebooks/05_Baseline_Model.ipynb):** establishing the LightGBM baseline (AUC 0.769).
* **[06_Bureau_Integration.ipynb](notebooks/06_Bureau_Integration.ipynb):** Merging external debt history from Credit Bureaus.
* **[07_Previous_History.ipynb](notebooks/07_Previous_History.ipynb):** Aggregating internal payment history and previous application rejections.

### **Phase 3: Production & Strategy**
* **[08_Optimization.ipynb](notebooks/08_Optimization.ipynb):** Tuning hyperparameters using Optuna (Bayesian Optimization).
* **[09_Ensemble_Stacking.ipynb](notebooks/09_Ensemble_Stacking.ipynb):** Building the final Stacking Ensemble (LGBM + XGB + CatBoost).
* **[10_Business_Impact.ipynb](notebooks/10_Business_Impact.ipynb):** Financial simulation, stress testing, and profit-curve optimization.

---
