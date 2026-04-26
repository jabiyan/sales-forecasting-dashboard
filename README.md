# 📈 Sales Forecasting Dashboard

An interactive, end-to-end machine learning dashboard that analyzes historical retail sales data and forecasts future revenue using Facebook Prophet — built with Python and Streamlit.

🔗 **[Live Demo →](https://sales-forecasting-dashboard-f5kcd2ntgeg2y4d6pi5zgs.streamlit.app/)**

---

## 📌 What This App Does

Select a product category and region from the sidebar, choose how many months to forecast, and the dashboard instantly delivers:

- **Live KPI cards** — Total Sales, Profit, Orders, and Average Order Value
- **Historical trend chart** — interactive monthly sales trend with hover tooltips
- **12-month sales forecast** — predicted sales with confidence interval bands
- **Seasonality analysis** — which months are naturally high and low for sales
- **Category & Region breakdown** — pie and bar charts for sales distribution
- **Forecast data table** — clean exportable table of predicted values per month

---

## 🗂️ Project Workflow

```
Raw Data → Cleaning → EDA → Prophet Forecasting → Streamlit Dashboard → Deployment
```

### 1. Data Cleaning & EDA
- Loaded and cleaned the Superstore retail dataset (9,994 orders)
- Extracted date features (year, month, week)
- Analyzed sales trends, seasonality, and category distributions

### 2. Time-Series Forecasting with Prophet
- Aggregated sales to monthly level per category
- Trained separate Prophet models for Furniture, Office Supplies, and Technology
- Evaluated model accuracy using MAE and RMSE
- Generated forecasts with upper and lower confidence bounds

### 3. Interactive Streamlit Dashboard
- Built a fully interactive dashboard with sidebar filters
- Integrated Plotly charts for rich, hoverable visualizations
- Deployed live on Streamlit Cloud

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| Pandas | Data manipulation and cleaning |
| Prophet | Time-series forecasting model |
| Plotly | Interactive charts and visualizations |
| Streamlit | Web dashboard framework |
| Scikit-learn | Model evaluation (MAE, RMSE) |

---

## 📊 Key Results

- Built and deployed 3 category-level Prophet forecasting models
- Identified strong yearly seasonality — sales consistently peak in Q4
- Technology category shows the highest growth trend year-over-year
- Dashboard updates forecasts in real time based on user-selected filters

---

## 💼 Business Value

This dashboard gives any retail or e-commerce business the ability to:

- **Plan inventory** ahead of high-demand months
- **Set revenue targets** backed by data-driven forecasts
- **Identify seasonal patterns** to time promotions effectively
- **Compare performance** across product categories and regions
- **Reduce overstock and understock** by anticipating demand shifts

---

## 📁 Repository Structure

```
sales-forecasting-dashboard/
│
├── app.py                    # Streamlit dashboard
├── superstore_clean.csv      # Cleaned dataset
├── forecast_results.csv      # Pre-computed forecast data
├── requirements.txt          # Dependencies
└── README.md
```

---

## 🚀 Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/jabiyan/sales-forecasting-dashboard.git
   cd sales-forecasting-dashboard
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## 👤 Author

**Mustajab Hussain** — Data Scientist & ML Engineer

- 🔗 [LinkedIn](https://www.linkedin.com/in/mustajab-hussain-312475283/)
- 💻 [GitHub](https://github.com/jabiyan)
- 📧 [mustajabh015@gmail.com](mailto:mustajabh015@gmail.com)
- 💼 [Hire me on Upwork](https://www.upwork.com/freelancers/~011cf146030b8908fd)
