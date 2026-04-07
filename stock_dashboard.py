"""
Stock Market Dashboard — Streamlit
====================================
Course: Python for Data Analysis and Visualization

Features:
  • Live data via yfinance (falls back to synthetic data if offline)
  • Candlestick chart with Plotly
  • Moving averages (MA20, MA50, MA200)
  • Bollinger Bands
  • RSI (Relative Strength Index)
  • MACD indicator
  • Volume analysis
  • Key statistics sidebar

Run:
    pip install streamlit yfinance plotly pandas numpy
    streamlit run stock_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ── Try yfinance, fall back gracefully ────────────────────────────────────────
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; }
    .metric-container { background: #f8f9fa; border-radius: 8px; padding: 12px 16px; margin: 4px 0; }
    .stMetric label { font-size: 12px !important; color: #6c757d !important; }
    .stMetric [data-testid="metric-container"] { background: #f8f9fa; border-radius: 8px; padding: 12px; }
    div[data-testid="stMetricValue"] { font-size: 20px !important; font-family: monospace; }
    .section-header { font-size: 13px; font-weight: 600; color: #495057; 
                      text-transform: uppercase; letter-spacing: .06em; 
                      margin: 16px 0 6px; }
    .badge-up   { background: #d4edda; color: #155724; padding: 2px 8px; 
                  border-radius: 12px; font-size: 12px; font-weight: 600; }
    .badge-down { background: #f8d7da; color: #721c24; padding: 2px 8px; 
                  border-radius: 12px; font-size: 12px; font-weight: 600; }
    h1 { font-size: 1.6rem !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data_yfinance(ticker: str, period: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.index = pd.to_datetime(df.index)
    return df.dropna()


def generate_synthetic_data(ticker: str, days: int = 365) -> pd.DataFrame:
    """Reproducible synthetic OHLCV data when yfinance is unavailable."""
    seed = sum(ord(c) for c in ticker)
    rng = np.random.default_rng(seed)
    base_prices = {"AAPL": 178, "MSFT": 415, "GOOGL": 172,
                   "AMZN": 185, "TSLA": 175, "NVDA": 875, "META": 505}
    start = base_prices.get(ticker, 150) * 0.80

    dates = pd.bdate_range(end=datetime.today(), periods=days)
    log_returns = rng.normal(0.0003, 0.018, days)
    close = start * np.exp(np.cumsum(log_returns))
    high  = close * (1 + rng.uniform(0.002, 0.015, days))
    low   = close * (1 - rng.uniform(0.002, 0.015, days))
    open_ = low + rng.random(days) * (high - low)
    vol   = rng.integers(int(2e7), int(1.2e8), days)

    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": vol}, index=dates)


def load_data(ticker: str, period: str) -> pd.DataFrame:
    if YFINANCE_AVAILABLE:
        try:
            df = fetch_data_yfinance(ticker, period)
            if not df.empty:
                return df
        except Exception:
            pass
    # Synthetic fallback
    days_map = {"1mo": 21, "3mo": 63, "6mo": 126, "1y": 252, "2y": 504, "5y": 1260}
    return generate_synthetic_data(ticker, days_map.get(period, 252))


# ══════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"].squeeze()

    # Moving Averages
    for w in [20, 50, 200]:
        df[f"MA{w}"] = c.rolling(w).mean()

    # Bollinger Bands (20-day, 2σ)
    df["BB_mid"]   = c.rolling(20).mean()
    df["BB_std"]   = c.rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * df["BB_std"]
    df["BB_lower"] = df["BB_mid"] - 2 * df["BB_std"]

    # RSI-14
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12       = c.ewm(span=12, adjust=False).mean()
    ema26       = c.ewm(span=26, adjust=False).mean()
    df["MACD"]  = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Hist"]   = df["MACD"] - df["Signal"]

    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv

    return df


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY CHART BUILDER
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "price":   "#2563eb",
    "ma20":    "#f59e0b",
    "ma50":    "#ec4899",
    "ma200":   "#8b5cf6",
    "bb":      "#10b981",
    "up_vol":  "rgba(16,185,129,0.55)",
    "dn_vol":  "rgba(239,68,68,0.55)",
    "macd":    "#2563eb",
    "signal":  "#f59e0b",
    "hist_up": "rgba(16,185,129,0.7)",
    "hist_dn": "rgba(239,68,68,0.7)",
    "rsi":     "#8b5cf6",
}


def build_chart(df: pd.DataFrame, ticker: str, chart_type: str,
                show_ma: list, show_bb: bool,
                show_volume: bool, show_rsi: bool, show_macd: bool) -> go.Figure:

    # Determine subplot rows
    rows, row_heights = [1], [0.55]
    if show_volume: rows.append(len(rows) + 1); row_heights.append(0.15)
    if show_rsi:    rows.append(len(rows) + 1); row_heights.append(0.15)
    if show_macd:   rows.append(len(rows) + 1); row_heights.append(0.15)
    total_rows = len(rows)

    specs = [[{"secondary_y": False}]] * total_rows
    fig = make_subplots(
        rows=total_rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=row_heights, specs=specs,
    )

    x = df.index

    # ── Price ────────────────────────────────────────────────────────────────
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=x, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name=ticker,
            increasing_line_color="#10b981", decreasing_line_color="#ef4444",
            increasing_fillcolor="#10b981", decreasing_fillcolor="#ef4444",
            line_width=1,
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=x, y=df["Close"], name=ticker,
            line=dict(color=COLORS["price"], width=2),
            fill="tozeroy" if chart_type == "Area" else None,
            fillcolor="rgba(37,99,235,0.08)",
        ), row=1, col=1)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    if show_bb:
        fig.add_trace(go.Scatter(
            x=x, y=df["BB_upper"], name="BB Upper",
            line=dict(color=COLORS["bb"], width=1, dash="dot"),
            showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=df["BB_lower"], name="BB Lower",
            line=dict(color=COLORS["bb"], width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(16,185,129,0.06)",
            showlegend=False,
        ), row=1, col=1)

    # ── Moving Averages ───────────────────────────────────────────────────────
    ma_cfg = {"MA20": (COLORS["ma20"], []), "MA50": (COLORS["ma50"], [4, 3]), "MA200": (COLORS["ma200"], [2, 2])}
    for ma_name, (color, dash) in ma_cfg.items():
        if ma_name in show_ma:
            fig.add_trace(go.Scatter(
                x=x, y=df[ma_name], name=ma_name,
                line=dict(color=color, width=1.5, dash="dash" if dash else "solid"),
            ), row=1, col=1)

    cur_row = 2

    # ── Volume ────────────────────────────────────────────────────────────────
    if show_volume:
        colors = [COLORS["up_vol"] if df["Close"].iloc[i] >= df["Open"].iloc[i]
                  else COLORS["dn_vol"] for i in range(len(df))]
        fig.add_trace(go.Bar(
            x=x, y=df["Volume"], name="Volume",
            marker_color=colors, showlegend=False,
        ), row=cur_row, col=1)
        fig.update_yaxes(title_text="Volume", row=cur_row, col=1,
                         tickformat=".2s", title_font_size=10)
        cur_row += 1

    # ── RSI ───────────────────────────────────────────────────────────────────
    if show_rsi:
        fig.add_trace(go.Scatter(
            x=x, y=df["RSI"], name="RSI",
            line=dict(color=COLORS["rsi"], width=1.5), showlegend=False,
        ), row=cur_row, col=1)
        for level, color in [(70, "rgba(239,68,68,0.3)"), (30, "rgba(16,185,129,0.3)")]:
            fig.add_hline(y=level, line_color=color, line_dash="dash",
                          line_width=1, row=cur_row, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100],
                         row=cur_row, col=1, title_font_size=10)
        cur_row += 1

    # ── MACD ──────────────────────────────────────────────────────────────────
    if show_macd:
        hist_colors = [COLORS["hist_up"] if v >= 0 else COLORS["hist_dn"]
                       for v in df["Hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=x, y=df["Hist"], name="MACD Hist",
            marker_color=hist_colors, showlegend=False,
        ), row=cur_row, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=df["MACD"], name="MACD",
            line=dict(color=COLORS["macd"], width=1.5), showlegend=False,
        ), row=cur_row, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=df["Signal"], name="Signal",
            line=dict(color=COLORS["signal"], width=1.5), showlegend=False,
        ), row=cur_row, col=1)
        fig.update_yaxes(title_text="MACD", row=cur_row, col=1, title_font_size=10)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font_size=11),
        margin=dict(l=10, r=10, t=30, b=10),
        height=520 + (show_volume + show_rsi + show_macd) * 110,
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_family="monospace",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)",
                     showspikes=True, spikethickness=1, spikecolor="#aaa")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📈 Stock Dashboard")
    st.markdown("---")

    st.markdown('<div class="section-header">Ticker</div>', unsafe_allow_html=True)
    popular = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "BRK-B"]
    ticker = st.selectbox("Select stock", popular, index=0)
    custom = st.text_input("Or enter custom ticker", placeholder="e.g. NFLX")
    if custom.strip():
        ticker = custom.strip().upper()

    st.markdown('<div class="section-header">Time Period</div>', unsafe_allow_html=True)
    period_map = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo",
                  "1 Year": "1y", "2 Years": "2y", "5 Years": "5y"}
    period_label = st.select_slider("Period", options=list(period_map.keys()), value="1 Year")
    period = period_map[period_label]

    st.markdown('<div class="section-header">Chart Type</div>', unsafe_allow_html=True)
    chart_type = st.radio("", ["Candlestick", "Line", "Area"], horizontal=True)

    st.markdown('<div class="section-header">Overlays</div>', unsafe_allow_html=True)
    show_ma = st.multiselect("Moving Averages", ["MA20", "MA50", "MA200"],
                             default=["MA20", "MA50"])
    show_bb = st.checkbox("Bollinger Bands", value=True)

    st.markdown('<div class="section-header">Sub-charts</div>', unsafe_allow_html=True)
    show_volume = st.checkbox("Volume", value=True)
    show_rsi    = st.checkbox("RSI (14)", value=True)
    show_macd   = st.checkbox("MACD (12,26,9)", value=False)

    st.markdown("---")
    if not YFINANCE_AVAILABLE:
        st.warning("⚠ yfinance not installed — using synthetic data.\n\n"
                   "`pip install yfinance`")
    else:
        st.success("✅ Live data via yfinance")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# Load & process
with st.spinner(f"Loading {ticker}..."):
    df_raw = load_data(ticker, period)
    df = add_indicators(df_raw.copy())

# ── Header row ────────────────────────────────────────────────────────────────
last   = float(df["Close"].iloc[-1])
prev   = float(df["Close"].iloc[-2]) if len(df) > 1 else last
change = last - prev
pct    = (change / prev) * 100 if prev else 0
sign   = "+" if change >= 0 else ""
badge_cls = "badge-up" if change >= 0 else "badge-down"

col_title, col_badge = st.columns([6, 1])
with col_title:
    st.markdown(f"## {ticker} &nbsp; `${last:,.2f}`", unsafe_allow_html=True)
with col_badge:
    st.markdown(
        f'<div class="{badge_cls}" style="margin-top:18px;">'
        f'{sign}{change:.2f} ({sign}{pct:.2f}%)</div>',
        unsafe_allow_html=True,
    )

# ── Key metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
period_high = float(df["High"].max())
period_low  = float(df["Low"].min())
avg_vol     = float(df["Volume"].mean())
rsi_val     = float(df["RSI"].iloc[-1]) if not pd.isna(df["RSI"].iloc[-1]) else 0.0
ma20_val    = float(df["MA20"].iloc[-1]) if not pd.isna(df["MA20"].iloc[-1]) else 0.0
bb_width    = float((df["BB_upper"].iloc[-1] - df["BB_lower"].iloc[-1]) / df["BB_mid"].iloc[-1] * 100) \
              if not pd.isna(df["BB_upper"].iloc[-1]) else 0.0

c1.metric("Last Price",    f"${last:,.2f}",       f"{sign}${abs(change):.2f}")
c2.metric("Period High",   f"${period_high:,.2f}")
c3.metric("Period Low",    f"${period_low:,.2f}")
c4.metric("Avg Volume",    f"{avg_vol/1e6:.1f}M")
c5.metric("RSI (14)",      f"{rsi_val:.1f}",
          "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral"))
c6.metric("BB Width",      f"{bb_width:.1f}%",    "Volatility proxy")

st.markdown("---")

# ── Main chart ────────────────────────────────────────────────────────────────
fig = build_chart(df, ticker, chart_type, show_ma, show_bb,
                  show_volume, show_rsi, show_macd)
st.plotly_chart(fig, use_container_width=True)

# ── Analysis tabs ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Statistics", "📉 Returns Analysis", "📋 Raw Data"])

with tab1:
    st.markdown("#### Descriptive Statistics")
    cols = ["Open", "High", "Low", "Close", "Volume"]
    st.dataframe(
        df[cols].describe().round(2).style.format("{:.2f}"),
        use_container_width=True,
    )

    st.markdown("#### Moving Average Signals")
    ma_sig = []
    for ma in ["MA20", "MA50", "MA200"]:
        val = df[ma].iloc[-1]
        if not pd.isna(val):
            sig = "🟢 Bullish" if last > val else "🔴 Bearish"
            ma_sig.append({"Indicator": ma, "Value": f"${val:.2f}",
                           "Price vs MA": f"${last - val:+.2f}", "Signal": sig})
    if ma_sig:
        st.dataframe(pd.DataFrame(ma_sig).set_index("Indicator"), use_container_width=True)

with tab2:
    st.markdown("#### Daily Returns Distribution")
    df["Return"] = df["Close"].pct_change() * 100

    fig_ret = go.Figure()
    fig_ret.add_trace(go.Histogram(
        x=df["Return"].dropna(), nbinsx=50,
        marker_color="#2563eb", opacity=0.75, name="Daily Return %",
    ))
    fig_ret.update_layout(
        xaxis_title="Daily Return (%)", yaxis_title="Frequency",
        template="plotly_white", height=280, margin=dict(l=10, r=10, t=20, b=30),
    )
    st.plotly_chart(fig_ret, use_container_width=True)

    c1r, c2r, c3r, c4r = st.columns(4)
    ret = df["Return"].dropna()
    c1r.metric("Mean Daily Return", f"{ret.mean():.3f}%")
    c2r.metric("Std Dev",           f"{ret.std():.3f}%")
    c3r.metric("Best Day",          f"{ret.max():.2f}%")
    c4r.metric("Worst Day",         f"{ret.min():.2f}%")

    # Cumulative return
    st.markdown("#### Cumulative Return")
    df["Cum_Return"] = (1 + df["Return"] / 100).cumprod() - 1
    fig_cum = go.Figure(go.Scatter(
        x=df.index, y=df["Cum_Return"] * 100,
        fill="tozeroy",
        line=dict(color="#2563eb", width=2),
        fillcolor="rgba(37,99,235,0.08)",
    ))
    fig_cum.update_layout(
        yaxis_title="Cumulative Return (%)", template="plotly_white",
        height=250, margin=dict(l=10, r=10, t=10, b=30),
    )
    st.plotly_chart(fig_cum, use_container_width=True)

with tab3:
    st.markdown("#### Historical OHLCV Data")
    display_cols = ["Open", "High", "Low", "Close", "Volume"]
    display_df = df[display_cols].copy()
    display_df.index = display_df.index.strftime("%Y-%m-%d")
    st.dataframe(
        display_df.sort_index(ascending=False).round(2)
        .style.format({"Open": "${:.2f}", "High": "${:.2f}",
                       "Low": "${:.2f}", "Close": "${:.2f}",
                       "Volume": "{:,.0f}"}),
        use_container_width=True, height=400,
    )

    csv = display_df.to_csv().encode("utf-8")
    st.download_button("⬇ Download CSV", csv,
                       f"{ticker}_data.csv", "text/csv")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="font-size:11px; color:#9ca3af; text-align:center;">'
    'Python for Data Analysis & Visualization · '
    'Built with Streamlit + Plotly + yfinance · '
    'For educational purposes only — not financial advice.'
    '</p>',
    unsafe_allow_html=True,
)
