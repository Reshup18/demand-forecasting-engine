# 📦 OptiSupply: End-to-End Demand Forecasting & Prescriptive Inventory Optimization Engine

An enterprise-grade supply chain demand forecasting engine and statistical safety stock optimization system trained on 1M+ retail sales records across 1,115 stores.

## 🎯 Executive Business Impact
* **Forecast Precision**: Trained XGBoost GBDT regressor achieving a **10.92% MAPE**, outperforming baseline moving average rules (25.38% MAPE) by **14.46% net accuracy gain**.
* **Working Capital Reduction**: Reduced daily tied-up inventory capital from $924.9K to $593.6K per store/day, achieving a **35.82% working capital optimization**.
* **Holding Cost Savings**: Generated **\(3,736,207.83 in net annual holding cost savings** while maintaining a 95% Cycle Service Level (CSL).

## 🛠 Tech Stack
* **Time-Series ML**: Python 3.12, XGBoost, Facebook Prophet, Scikit-Learn, Pandas, NumPy.
* **Inventory Optimization**: Statistical Normal Distributions, Combined Lead-Time Variance Propagation (\)\sigma_L$).
* **UI & Deployment**: Streamlit Web Framework, Streamlit Community Cloud.

## 🚀 Execution Guide
```bash
pip install -r requirements.txt
streamlit run app.py
```
