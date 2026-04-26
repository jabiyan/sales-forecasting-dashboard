import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('superstore_clean.csv', encoding='latin-1')
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    return df

@st.cache_data
def load_forecast():
    fc = pd.read_csv('forecast_results.csv')
    fc['ds'] = pd.to_datetime(fc['ds'])
    return fc

df = load_data()
forecast_df = load_forecast()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("📊 Dashboard Controls")
st.sidebar.markdown("---")

category = st.sidebar.selectbox(
    "Select Category",
    ["All", "Furniture", "Office Supplies", "Technology"]
)

region = st.sidebar.selectbox(
    "Select Region",
    ["All"] + sorted(df['Region'].unique().tolist())
)

forecast_months = st.sidebar.slider(
    "Forecast Months Ahead", 3, 24, 12
)

st.sidebar.markdown("---")
st.sidebar.markdown("Built by **Mustajab Hussain**")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/mustajab-hussain-312475283/) | [GitHub](https://github.com/jabiyan)")

# ── Filter data ───────────────────────────────────────────────
filtered = df.copy()
if category != "All":
    filtered = filtered[filtered['Category'] == category]
if region != "All":
    filtered = filtered[filtered['Region'] == region]

# ── Header ────────────────────────────────────────────────────
st.title("📈 Sales Forecasting Dashboard")
st.markdown("Analyze historical sales trends and forecast future revenue using Facebook Prophet.")
st.markdown("---")

# ── KPI Metric Cards ──────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

total_sales    = filtered['Sales'].sum()
total_profit   = filtered['Profit'].sum()
total_orders   = filtered['Order ID'].nunique()
avg_order_val  = total_sales / total_orders if total_orders > 0 else 0

col1.metric("Total Sales",   f"${total_sales:,.0f}")
col2.metric("Total Profit",  f"${total_profit:,.0f}")
col3.metric("Total Orders",  f"{total_orders:,}")
col4.metric("Avg Order Value",f"${avg_order_val:,.0f}")

st.markdown("---")

# ── Section 1: Historical Sales Trend ────────────────────────
st.subheader("📅 Historical Sales Trend")

monthly_sales = (
    filtered
    .groupby(pd.Grouper(key='Order Date', freq='M'))['Sales']
    .sum()
    .reset_index()
)

fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(
    x=monthly_sales['Order Date'],
    y=monthly_sales['Sales'],
    mode='lines+markers',
    name='Monthly Sales',
    line=dict(color='#7F77DD', width=2),
    marker=dict(size=5)
))
fig_hist.update_layout(
    xaxis_title="Date",
    yaxis_title="Sales ($)",
    hovermode="x unified",
    height=350,
    margin=dict(t=20, b=20)
)
st.plotly_chart(fig_hist, use_container_width=True)

# ── Section 2: Sales Forecast ─────────────────────────────────
st.subheader(f"🔮 Sales Forecast — Next {forecast_months} Months")

# Train Prophet on filtered data
prophet_input = monthly_sales.rename(
    columns={'Order Date': 'ds', 'Sales': 'y'}
)

with st.spinner("Training forecast model..."):
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.1
    )
    m.fit(prophet_input)
    future   = m.make_future_dataframe(periods=forecast_months, freq='M')
    forecast = m.predict(future)

# Plot forecast
fig_fc = go.Figure()

fig_fc.add_trace(go.Scatter(
    x=prophet_input['ds'], y=prophet_input['y'],
    mode='markers', name='Actual Sales',
    marker=dict(color='#7F77DD', size=6)
))
fig_fc.add_trace(go.Scatter(
    x=forecast['ds'], y=forecast['yhat'],
    mode='lines', name='Forecast',
    line=dict(color='#D85A30', width=2)
))
fig_fc.add_trace(go.Scatter(
    x=pd.concat([forecast['ds'], forecast['ds'].iloc[::-1]]),
    y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'].iloc[::-1]]),
    fill='toself', fillcolor='rgba(216,90,48,0.12)',
    line=dict(color='rgba(0,0,0,0)'),
    name='Confidence Interval'
))
fig_fc.update_layout(
    xaxis_title="Date",
    yaxis_title="Sales ($)",
    hovermode="x unified",
    height=400,
    margin=dict(t=20, b=20)
)
st.plotly_chart(fig_fc, use_container_width=True)

# ── Section 3: Seasonality Insight ───────────────────────────
st.subheader("🗓️ Monthly Seasonality Pattern")

season = forecast[forecast['ds'] <= prophet_input['ds'].max()]
month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
season_avg = season.copy()
season_avg['month'] = season_avg['ds'].dt.month
monthly_pattern = season_avg.groupby('month')['yearly'].mean()

fig_season = px.bar(
    x=month_names,
    y=monthly_pattern.values,
    color=monthly_pattern.values,
    color_continuous_scale=['#AFA9EC', '#7F77DD', '#534AB7'],
    labels={'x': 'Month', 'y': 'Seasonal Effect ($)'}
)
fig_season.update_layout(
    height=300,
    coloraxis_showscale=False,
    margin=dict(t=20, b=20)
)
st.plotly_chart(fig_season, use_container_width=True)

# ── Section 4: Sales by Category & Region ────────────────────
st.subheader("🗂️ Sales Breakdown")
col_a, col_b = st.columns(2)

with col_a:
    cat_sales = df.groupby('Category')['Sales'].sum().reset_index()
    fig_cat = px.pie(cat_sales, values='Sales', names='Category',
                      color_discrete_sequence=['#7F77DD','#1D9E75','#D85A30'],
                      title="Sales by Category")
    fig_cat.update_layout(height=320, margin=dict(t=40,b=10))
    st.plotly_chart(fig_cat, use_container_width=True)

with col_b:
    reg_sales = df.groupby('Region')['Sales'].sum().reset_index()
    fig_reg = px.bar(reg_sales.sort_values('Sales', ascending=True),
                      x='Sales', y='Region', orientation='h',
                      color='Sales',
                      color_continuous_scale=['#AFA9EC','#534AB7'],
                      title="Sales by Region")
    fig_reg.update_layout(height=320, coloraxis_showscale=False,
                            margin=dict(t=40,b=10))
    st.plotly_chart(fig_reg, use_container_width=True)

# ── Section 5: Raw Forecast Table ────────────────────────────
st.subheader("📋 Forecast Data Table")
future_only = forecast[forecast['ds'] > prophet_input['ds'].max()][[
    'ds', 'yhat', 'yhat_lower', 'yhat_upper'
]].copy()

future_only.columns = ['Month', 'Predicted Sales', 'Lower Bound', 'Upper Bound']
future_only['Month'] = future_only['Month'].dt.strftime('%b %Y')
for col in ['Predicted Sales', 'Lower Bound', 'Upper Bound']:
    future_only[col] = future_only[col].apply(lambda x: f"${x:,.0f}")

st.dataframe(future_only, use_container_width=True, hide_index=True)