import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="上证指数风险可视化与蒙特卡洛模拟", layout="wide")

st.title("上证指数风险可视化分析 + 蒙特卡洛上涨/下跌概率模拟")

# =========================
# Sidebar
# =========================
st.sidebar.header("参数设置")
start_date = st.sidebar.text_input("开始日期", "20180101")
end_date = st.sidebar.text_input("结束日期", "20260611")
future_days = st.sidebar.slider("预测天数", 5, 252, 60)
simulations = st.sidebar.slider("模拟路径数量", 500, 20000, 5000, step=500)
var_confidence = st.sidebar.selectbox("VaR 置信度", [0.90, 0.95, 0.99], index=1)
rolling_window = st.sidebar.slider("滚动窗口", 10, 120, 20)

# =========================
# Data
# =========================
@st.cache_data
def load_data(start_date, end_date):
    df = ak.stock_zh_index_daily(symbol="sh000001")
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]
    df = df.sort_values("date")
    df = df.rename(columns={
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    })
    df["Return"] = df["Close"].pct_change()
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    df["RollingVol"] = df["Return"].rolling(rolling_window).std() * np.sqrt(252)
    df["CumMax"] = df["Close"].cummax()
    df["Drawdown"] = df["Close"] / df["CumMax"] - 1
    return df.dropna()

df = load_data(start_date, end_date)

if df.empty:
    st.error("没有获取到数据，请检查日期或网络。")
    st.stop()

latest_close = df["Close"].iloc[-1]
daily_mu = df["LogReturn"].mean()
daily_sigma = df["LogReturn"].std()
annual_return = daily_mu * 252
annual_vol = daily_sigma * np.sqrt(252)
max_drawdown = df["Drawdown"].min()

# Historical VaR / CVaR
var_level = np.percentile(df["Return"], (1 - var_confidence) * 100)
cvar_level = df[df["Return"] <= var_level]["Return"].mean()

# Parametric VaR
parametric_var = norm.ppf(1 - var_confidence, df["Return"].mean(), df["Return"].std())

# =========================
# Monte Carlo
# =========================
np.random.seed(42)

simulated_paths = np.zeros((future_days, simulations))
simulated_paths[0] = latest_close

for t in range(1, future_days):
    random_returns = np.random.normal(daily_mu, daily_sigma, simulations)
    simulated_paths[t] = simulated_paths[t - 1] * np.exp(random_returns)

final_prices = simulated_paths[-1]
up_prob = np.mean(final_prices > latest_close)
down_prob = np.mean(final_prices < latest_close)
expected_price = np.mean(final_prices)
median_price = np.median(final_prices)
p5 = np.percentile(final_prices, 5)
p95 = np.percentile(final_prices, 95)

expected_return = expected_price / latest_close - 1
downside_5_return = p5 / latest_close - 1
upside_95_return = p95 / latest_close - 1

# =========================
# KPI Cards
# =========================
c1, c2, c3, c4 = st.columns(4)

c1.metric("最新收盘点位", f"{latest_close:,.2f}")
c2.metric("年化收益率", f"{annual_return:.2%}")
c3.metric("年化波动率", f"{annual_vol:.2%}")
c4.metric("最大回撤", f"{max_drawdown:.2%}")

c5, c6, c7, c8 = st.columns(4)

c5.metric(f"历史 VaR {int(var_confidence*100)}%", f"{var_level:.2%}")
c6.metric(f"CVaR {int(var_confidence*100)}%", f"{cvar_level:.2%}")
c7.metric("蒙特卡洛上涨概率", f"{up_prob:.2%}")
c8.metric("蒙特卡洛下跌概率", f"{down_prob:.2%}")

# =========================
# Charts
# =========================
st.subheader("1. 上证指数走势")
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="上证指数"))
fig_price.update_layout(height=420, template="plotly_dark")
st.plotly_chart(fig_price, use_container_width=True)

st.subheader("2. 滚动年化波动率")
fig_vol = go.Figure()
fig_vol.add_trace(go.Scatter(x=df["Date"], y=df["RollingVol"], mode="lines", name="滚动波动率"))
fig_vol.update_layout(height=380, template="plotly_dark", yaxis_tickformat=".0%")
st.plotly_chart(fig_vol, use_container_width=True)

st.subheader("3. 最大回撤分析")
fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(x=df["Date"], y=df["Drawdown"], mode="lines", name="回撤"))
fig_dd.update_layout(height=380, template="plotly_dark", yaxis_tickformat=".0%")
st.plotly_chart(fig_dd, use_container_width=True)

st.subheader("4. 日收益率分布")
fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(x=df["Return"], nbinsx=80, name="日收益率"))
fig_hist.add_vline(x=var_level, line_dash="dash", annotation_text="Historical VaR")
fig_hist.add_vline(x=parametric_var, line_dash="dot", annotation_text="Parametric VaR")
fig_hist.update_layout(height=380, template="plotly_dark", xaxis_tickformat=".1%")
st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("5. 蒙特卡洛未来路径模拟")
fig_mc = go.Figure()

sample_paths = min(200, simulations)
for i in range(sample_paths):
    fig_mc.add_trace(go.Scatter(
        y=simulated_paths[:, i],
        mode="lines",
        line=dict(width=0.5),
        opacity=0.15,
        showlegend=False
    ))

fig_mc.add_trace(go.Scatter(
    y=np.mean(simulated_paths, axis=1),
    mode="lines",
    name="平均路径",
    line=dict(width=3)
))

fig_mc.add_hline(y=latest_close, line_dash="dash", annotation_text="当前点位")
fig_mc.update_layout(height=500, template="plotly_dark")
st.plotly_chart(fig_mc, use_container_width=True)

# =========================
# Probability Table
# =========================
st.subheader("6. 蒙特卡洛结果汇总")

summary = pd.DataFrame({
    "指标": [
        "当前点位",
        "模拟期望点位",
        "模拟中位数点位",
        "5%悲观分位点",
        "95%乐观分位点",
        "期望收益率",
        "5%悲观收益率",
        "95%乐观收益率",
        "上涨概率",
        "下跌概率"
    ],
    "结果": [
        f"{latest_close:,.2f}",
        f"{expected_price:,.2f}",
        f"{median_price:,.2f}",
        f"{p5:,.2f}",
        f"{p95:,.2f}",
        f"{expected_return:.2%}",
        f"{downside_5_return:.2%}",
        f"{upside_95_return:.2%}",
        f"{up_prob:.2%}",
        f"{down_prob:.2%}"
    ]
})

st.dataframe(summary, use_container_width=True)

# =========================
# Macro Support
# =========================
st.subheader("7. 风险解释框架")

st.markdown(f"""
### 当前模型结论

在过去样本区间内：

- 上证指数年化波动率约为 **{annual_vol:.2%}**
- 最大历史回撤约为 **{max_drawdown:.2%}**
- 单日 Historical VaR {int(var_confidence*100)}% 为 **{var_level:.2%}**
- 单日 CVaR {int(var_confidence*100)}% 为 **{cvar_level:.2%}**

在未来 **{future_days} 个交易日** 的蒙特卡洛模拟中：

- 上涨概率：**{up_prob:.2%}**
- 下跌概率：**{down_prob:.2%}**
- 期望点位：**{expected_price:,.2f}**
- 5% 悲观情景：**{p5:,.2f}**
- 95% 乐观情景：**{p95:,.2f}**

### 可引入的支撑变量

你后续可以加入这些外部因子：

1. **利率因子**：LPR、10年期国债收益率、MLF、逆回购利率  
2. **宏观因子**：CPI、PPI、PMI、社融、M2、工业增加值  
3. **市场因子**：成交额、北向资金、融资融券余额、沪深300、创业板指  
4. **海外因子**：美元指数、美国10年期国债收益率、恒生指数、纳斯达克  
5. **情绪因子**：VIX、人民币汇率、商品价格、政策新闻强度  

### 注意

这个模型不是预测确定点位，而是用历史波动率生成未来概率分布。
如果市场出现政策刺激、流动性变化、地缘风险或极端行情，真实结果会偏离正态蒙特卡洛模型。
""")
