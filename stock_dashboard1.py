"""
Stock Market Dashboard — Streamlit (Static Data)
=================================================
Course: Python for Data Analysis and Visualization

Features:
  • 100% static synthetic data (no external dependencies)
  • Candlestick chart with Plotly
  • Moving averages (MA20, MA50, MA200)
  • Bollinger Bands
  • RSI (Relative Strength Index)
  • MACD indicator
  • Volume analysis
  • Key statistics sidebar
  • Sector Performance Dashboard
  • Portfolio Overview Dashboard
  • Market Heatmap Dashboard
  • Correlation Matrix Dashboard

Run:
    pip install streamlit plotly pandas numpy
    streamlit run stock_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

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
# STATIC DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

BASE_PRICES = {
    "AAPL": 178, "MSFT": 415, "GOOGL": 172,
    "AMZN": 185, "TSLA": 175, "NVDA": 875,
    "META": 505, "JPM":  198, "BRK-B": 368,
}

SECTOR_DATA = {
    "Technology":       {"return": 24.3,  "market_cap": 12.8, "pe": 28.4, "color": "#2563eb"},
    "Healthcare":       {"return": 8.7,   "market_cap": 5.2,  "pe": 22.1, "color": "#10b981"},
    "Financials":       {"return": 15.2,  "market_cap": 7.1,  "pe": 14.3, "color": "#f59e0b"},
    "Consumer Disc.":   {"return": 11.4,  "market_cap": 4.3,  "pe": 24.8, "color": "#8b5cf6"},
    "Industrials":      {"return": 13.8,  "market_cap": 3.9,  "pe": 19.6, "color": "#ec4899"},
    "Communication":    {"return": 19.6,  "market_cap": 4.7,  "pe": 21.5, "color": "#06b6d4"},
    "Consumer Staples": {"return": 4.2,   "market_cap": 3.1,  "pe": 18.7, "color": "#84cc16"},
    "Energy":           {"return": -2.1,  "market_cap": 2.8,  "pe": 11.2, "color": "#f97316"},
    "Utilities":        {"return": -5.3,  "market_cap": 1.4,  "pe": 15.9, "color": "#64748b"},
    "Real Estate":      {"return": -8.1,  "market_cap": 1.2,  "pe": 35.2, "color": "#a78bfa"},
    "Materials":        {"return": 6.5,   "market_cap": 1.9,  "pe": 16.4, "color": "#34d399"},
}

PORTFOLIO_HOLDINGS = {
    "AAPL":  {"shares": 50,  "cost_basis": 145.20, "sector": "Technology"},
    "MSFT":  {"shares": 30,  "cost_basis": 310.50, "sector": "Technology"},
    "NVDA":  {"shares": 20,  "cost_basis": 450.00, "sector": "Technology"},
    "JPM":   {"shares": 40,  "cost_basis": 155.80, "sector": "Financials"},
    "GOOGL": {"shares": 25,  "cost_basis": 130.00, "sector": "Communication"},
    "AMZN":  {"shares": 35,  "cost_basis": 140.00, "sector": "Consumer Disc."},
    "TSLA":  {"shares": 15,  "cost_basis": 220.00, "sector": "Consumer Disc."},
    "META":  {"shares": 18,  "cost_basis": 320.00, "sector": "Communication"},
}


@st.cache_data(show_spinner=False)
def generate_ohlcv(ticker: str, days: int = 252) -> pd.DataFrame:
    """Reproducible synthetic OHLCV data."""
    seed = sum(ord(c) for c in ticker)
    rng = np.random.default_rng(seed)
    start = BASE_PRICES.get(ticker, 150) * 0.80

    dates = pd.bdate_range(end=datetime.today(), periods=days)
    log_returns = rng.normal(0.0003, 0.018, days)
    close = start * np.exp(np.cumsum(log_returns))
    high  = close * (1 + rng.uniform(0.002, 0.015, days))
    low   = close * (1 - rng.uniform(0.002, 0.015, days))
    open_ = low + rng.random(days) * (high - low)
    vol   = rng.integers(int(2e7), int(1.2e8), days)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


@st.cache_data(show_spinner=False)
def get_current_price(ticker: str) -> float:
    df = generate_ohlcv(ticker)
    return float(df["Close"].iloc[-1])


@st.cache_data(show_spinner=False)
def generate_portfolio_history(days: int = 252) -> pd.DataFrame:
    """Combined portfolio value over time."""
    dates = pd.bdate_range(end=datetime.today(), periods=days)
    total = np.zeros(days)
    for ticker, info in PORTFOLIO_HOLDINGS.items():
        df = generate_ohlcv(ticker, days)
        total += df["Close"].values * info["shares"]
    return pd.DataFrame({"Portfolio": total}, index=dates)


# ══════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"].squeeze()

    for w in [20, 50, 200]:
        df[f"MA{w}"] = c.rolling(w).mean()

    df["BB_mid"]   = c.rolling(20).mean()
    df["BB_std"]   = c.rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * df["BB_std"]
    df["BB_lower"] = df["BB_mid"] - 2 * df["BB_std"]

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12        = c.ewm(span=12, adjust=False).mean()
    ema26        = c.ewm(span=26, adjust=False).mean()
    df["MACD"]   = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Hist"]   = df["MACD"] - df["Signal"]

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
# COLOR PALETTE
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


# ══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_price_chart(df, ticker, chart_type, show_ma, show_bb,
                      show_volume, show_rsi, show_macd):
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

    if show_bb:
        fig.add_trace(go.Scatter(
            x=x, y=df["BB_upper"], name="BB Upper",
            line=dict(color=COLORS["bb"], width=1, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=df["BB_lower"], name="BB Lower",
            line=dict(color=COLORS["bb"], width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(16,185,129,0.06)",
            showlegend=False,
        ), row=1, col=1)

    ma_cfg = {
        "MA20":  (COLORS["ma20"],  "solid"),
        "MA50":  (COLORS["ma50"],  "dash"),
        "MA200": (COLORS["ma200"], "dash"),
    }
    for ma_name, (color, dash) in ma_cfg.items():
        if ma_name in show_ma:
            fig.add_trace(go.Scatter(
                x=x, y=df[ma_name], name=ma_name,
                line=dict(color=color, width=1.5, dash=dash),
            ), row=1, col=1)

    cur_row = 2

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


def build_sector_chart():
    sectors = list(SECTOR_DATA.keys())
    returns = [SECTOR_DATA[s]["return"] for s in sectors]
    colors  = ["#10b981" if r >= 0 else "#ef4444" for r in returns]

    order = np.argsort(returns)
    sorted_sectors = [sectors[i] for i in order]
    sorted_returns = [returns[i] for i in order]
    sorted_colors  = [colors[i]  for i in order]

    fig = go.Figure(go.Bar(
        x=sorted_returns, y=sorted_sectors,
        orientation="h",
        marker_color=sorted_colors,
        text=[f"{r:+.1f}%" for r in sorted_returns],
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_white",
        xaxis_title="YTD Return (%)",
        margin=dict(l=10, r=60, t=20, b=10),
        height=380,
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="monospace",
    )
    fig.add_vline(x=0, line_width=1, line_color="#aaa")
    return fig


def build_sector_bubble():
    sectors = list(SECTOR_DATA.keys())
    fig = go.Figure()
    for s in sectors:
        d = SECTOR_DATA[s]
        fig.add_trace(go.Scatter(
            x=[d["pe"]],
            y=[d["return"]],
            mode="markers+text",
            text=[s],
            textposition="top center",
            marker=dict(
                size=d["market_cap"] * 6,
                color=d["color"],
                opacity=0.75,
                line=dict(width=1, color="white"),
            ),
            name=s,
            showlegend=False,
        ))
    fig.update_layout(
        template="plotly_white",
        xaxis_title="P/E Ratio",
        yaxis_title="YTD Return (%)",
        margin=dict(l=10, r=10, t=20, b=10),
        height=380,
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="monospace",
        hovermode="closest",
    )
    fig.add_hline(y=0, line_width=1, line_color="#ddd", line_dash="dash")
    return fig


def build_portfolio_charts():
    hist = generate_portfolio_history()
    current_value = float(hist["Portfolio"].iloc[-1])
    cost_total = sum(
        info["shares"] * info["cost_basis"]
        for info in PORTFOLIO_HOLDINGS.values()
    )

    # ── Donut: sector allocation ──────────────────────────────────────────────
    sector_alloc: dict = {}
    for ticker, info in PORTFOLIO_HOLDINGS.items():
        price = get_current_price(ticker)
        val   = price * info["shares"]
        s     = info["sector"]
        sector_alloc[s] = sector_alloc.get(s, 0) + val

    fig_donut = go.Figure(go.Pie(
        labels=list(sector_alloc.keys()),
        values=[round(v, 2) for v in sector_alloc.values()],
        hole=0.55,
        textinfo="label+percent",
        marker=dict(colors=["#2563eb", "#10b981", "#f59e0b", "#8b5cf6",
                             "#ec4899", "#06b6d4", "#84cc16"]),
    ))
    fig_donut.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=20, b=10),
        height=300,
        showlegend=False,
        paper_bgcolor="white",
        font_family="monospace",
    )

    # ── Line: portfolio value history ─────────────────────────────────────────
    pct_return = (hist["Portfolio"] / hist["Portfolio"].iloc[0] - 1) * 100
    fig_line = go.Figure(go.Scatter(
        x=hist.index, y=hist["Portfolio"],
        fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
        line=dict(color="#2563eb", width=2),
    ))
    fig_line.update_layout(
        template="plotly_white",
        yaxis_title="Portfolio Value ($)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=260,
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="monospace",
        hovermode="x unified",
    )
    fig_line.update_yaxes(tickformat="$,.0f")

    # ── Bar: individual stock P&L ─────────────────────────────────────────────
    tickers_list, pnl_vals, pnl_pcts = [], [], []
    for ticker, info in PORTFOLIO_HOLDINGS.items():
        price  = get_current_price(ticker)
        cost   = info["cost_basis"]
        gain   = (price - cost) * info["shares"]
        pct    = (price - cost) / cost * 100
        tickers_list.append(ticker)
        pnl_vals.append(round(gain, 2))
        pnl_pcts.append(round(pct, 2))

    bar_colors = ["#10b981" if v >= 0 else "#ef4444" for v in pnl_vals]
    fig_pnl = go.Figure(go.Bar(
        x=tickers_list,
        y=pnl_pcts,
        marker_color=bar_colors,
        text=[f"{p:+.1f}%" for p in pnl_pcts],
        textposition="outside",
    ))
    fig_pnl.update_layout(
        template="plotly_white",
        yaxis_title="Return (%)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=250,
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="monospace",
    )
    fig_pnl.add_hline(y=0, line_width=1, line_color="#aaa")

    return fig_donut, fig_line, fig_pnl, current_value, cost_total, pnl_vals


def build_heatmap():
    tickers = list(BASE_PRICES.keys())
    rng = np.random.default_rng(42)

    # Compute 1-day, 5-day, 1-month, 3-month, YTD returns for each ticker
    periods = {"1D": 1, "5D": 5, "1M": 21, "3M": 63, "YTD": 126}
    data = {}
    for ticker in tickers:
        df = generate_ohlcv(ticker)
        row = {}
        for label, days in periods.items():
            if len(df) > days:
                r = (df["Close"].iloc[-1] / df["Close"].iloc[-days - 1] - 1) * 100
                row[label] = round(float(r), 2)
            else:
                row[label] = 0.0
        data[ticker] = row

    heat_df = pd.DataFrame(data).T

    fig = go.Figure(go.Heatmap(
        z=heat_df.values,
        x=heat_df.columns.tolist(),
        y=heat_df.index.tolist(),
        colorscale=[
            [0.0, "#ef4444"],
            [0.4, "#fca5a5"],
            [0.5, "#f9fafb"],
            [0.6, "#86efac"],
            [1.0, "#10b981"],
        ],
        zmid=0,
        text=[[f"{v:+.1f}%" for v in row] for row in heat_df.values],
        texttemplate="%{text}",
        textfont_size=12,
        hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
        colorbar=dict(title="Return %", thickness=12, len=0.8),
    ))
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=20, b=10),
        height=340,
        paper_bgcolor="white",
        font_family="monospace",
        xaxis=dict(side="top"),
    )
    return fig


def build_correlation_matrix():
    tickers = list(BASE_PRICES.keys())
    returns = {}
    for ticker in tickers:
        df = generate_ohlcv(ticker)
        returns[ticker] = df["Close"].pct_change().dropna()

    ret_df = pd.DataFrame(returns).dropna()
    corr   = ret_df.corr().round(2)

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        zmin=-1, zmax=1,
        text=corr.values,
        texttemplate="%{text}",
        textfont_size=11,
        hovertemplate="<b>%{y} / %{x}</b><br>Corr: %{z:.2f}<extra></extra>",
        colorbar=dict(title="ρ", thickness=12, len=0.8),
    ))
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=20, b=10),
        height=400,
        paper_bgcolor="white",
        font_family="monospace",
        xaxis=dict(side="top"),
    )
    return fig


def build_volatility_chart():
    tickers = list(BASE_PRICES.keys())
    vols, means = [], []
    for ticker in tickers:
        df  = generate_ohlcv(ticker)
        ret = df["Close"].pct_change().dropna() * 100
        vols.append(round(float(ret.std() * np.sqrt(252)), 2))   # annualised
        means.append(round(float(ret.mean() * 252), 2))           # annualised mean

    fig = go.Figure(go.Scatter(
        x=vols, y=means,
        mode="markers+text",
        text=tickers,
        textposition="top center",
        marker=dict(
            size=14,
            color=means,
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(title="Ann. Return %", thickness=12),
            line=dict(width=1, color="white"),
        ),
    ))
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Annualised Volatility (%)",
        yaxis_title="Annualised Return (%)",
        margin=dict(l=10, r=10, t=20, b=10),
        height=360,
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="monospace",
        hovermode="closest",
    )
    fig.add_hline(y=0, line_width=1, line_color="#ddd", line_dash="dash")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📈 Stock Dashboard")
    st.info("⚙ All data is synthetic & static — no internet required.")
    st.markdown("---")

    dashboard_page = st.radio(
        "Select Dashboard",
        ["📊 Stock Analysis", "🏭 Sector Performance",
         "💼 Portfolio Overview", "🔥 Market Heatmap",
         "🔗 Correlation & Risk"],
        index=0,
    )

    st.markdown("---")

    if dashboard_page == "📊 Stock Analysis":
        st.markdown('<div class="section-header">Ticker</div>', unsafe_allow_html=True)
        popular = list(BASE_PRICES.keys())
        ticker  = st.selectbox("Select stock", popular, index=0)

        st.markdown('<div class="section-header">Time Period</div>', unsafe_allow_html=True)
        period_map = {"1 Month": 21, "3 Months": 63, "6 Months": 126,
                      "1 Year": 252, "2 Years": 504}
        period_label = st.select_slider("Period", options=list(period_map.keys()), value="1 Year")
        period_days  = period_map[period_label]

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


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD: STOCK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

if dashboard_page == "📊 Stock Analysis":
    df_raw = generate_ohlcv(ticker, period_days)
    df     = add_indicators(df_raw.copy())

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

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    period_high = float(df["High"].max())
    period_low  = float(df["Low"].min())
    avg_vol     = float(df["Volume"].mean())
    rsi_val     = float(df["RSI"].iloc[-1]) if not pd.isna(df["RSI"].iloc[-1]) else 0.0
    bb_width    = float(
        (df["BB_upper"].iloc[-1] - df["BB_lower"].iloc[-1])
        / df["BB_mid"].iloc[-1] * 100
    ) if not pd.isna(df["BB_upper"].iloc[-1]) else 0.0

    c1.metric("Last Price",  f"${last:,.2f}",    f"{sign}${abs(change):.2f}")
    c2.metric("Period High", f"${period_high:,.2f}")
    c3.metric("Period Low",  f"${period_low:,.2f}")
    c4.metric("Avg Volume",  f"{avg_vol/1e6:.1f}M")
    c5.metric("RSI (14)",    f"{rsi_val:.1f}",
              "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral"))
    c6.metric("BB Width",    f"{bb_width:.1f}%", "Volatility proxy")

    st.markdown("---")
    fig = build_price_chart(df, ticker, chart_type, show_ma, show_bb,
                            show_volume, show_rsi, show_macd)
    st.plotly_chart(fig, use_container_width=True)

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
        fig_ret = go.Figure(go.Histogram(
            x=df["Return"].dropna(), nbinsx=50,
            marker_color="#2563eb", opacity=0.75,
        ))
        fig_ret.update_layout(
            xaxis_title="Daily Return (%)", yaxis_title="Frequency",
            template="plotly_white", height=280,
            margin=dict(l=10, r=10, t=20, b=30),
        )
        st.plotly_chart(fig_ret, use_container_width=True)

        c1r, c2r, c3r, c4r = st.columns(4)
        ret = df["Return"].dropna()
        c1r.metric("Mean Daily Return", f"{ret.mean():.3f}%")
        c2r.metric("Std Dev",           f"{ret.std():.3f}%")
        c3r.metric("Best Day",          f"{ret.max():.2f}%")
        c4r.metric("Worst Day",         f"{ret.min():.2f}%")

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
        display_df   = df[display_cols].copy()
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


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD: SECTOR PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

elif dashboard_page == "🏭 Sector Performance":
    st.markdown("## 🏭 Sector Performance Dashboard")
    st.caption("YTD performance across S&P 500 sectors — static illustrative data")
    st.markdown("---")

    # Top metrics
    best_sector  = max(SECTOR_DATA, key=lambda s: SECTOR_DATA[s]["return"])
    worst_sector = min(SECTOR_DATA, key=lambda s: SECTOR_DATA[s]["return"])
    avg_return   = np.mean([d["return"] for d in SECTOR_DATA.values()])
    pos_sectors  = sum(1 for d in SECTOR_DATA.values() if d["return"] > 0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best Sector",   best_sector,  f"{SECTOR_DATA[best_sector]['return']:+.1f}%")
    m2.metric("Worst Sector",  worst_sector, f"{SECTOR_DATA[worst_sector]['return']:+.1f}%")
    m3.metric("Average Return", f"{avg_return:.1f}%")
    m4.metric("Sectors in Green", f"{pos_sectors} / {len(SECTOR_DATA)}")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("#### YTD Returns by Sector")
        st.plotly_chart(build_sector_chart(), use_container_width=True)

    with col_right:
        st.markdown("#### Valuation vs Return (bubble = market cap)")
        st.plotly_chart(build_sector_bubble(), use_container_width=True)

    st.markdown("#### Sector Summary Table")
    sector_rows = []
    for sector, data in SECTOR_DATA.items():
        sector_rows.append({
            "Sector":         sector,
            "YTD Return (%)": f"{data['return']:+.1f}%",
            "Market Cap ($T)": f"${data['market_cap']:.1f}T",
            "P/E Ratio":       f"{data['pe']:.1f}x",
            "Signal":          "🟢 Outperform" if data["return"] > avg_return
                               else ("🔴 Underperform" if data["return"] < 0
                                     else "🟡 Neutral"),
        })
    st.dataframe(pd.DataFrame(sector_rows).set_index("Sector"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD: PORTFOLIO OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

elif dashboard_page == "💼 Portfolio Overview":
    st.markdown("## 💼 Portfolio Overview Dashboard")
    st.caption("Synthetic 8-stock portfolio — static illustrative data")
    st.markdown("---")

    fig_donut, fig_line, fig_pnl, current_value, cost_total, pnl_vals = build_portfolio_charts()

    total_gain   = current_value - cost_total
    total_return = total_gain / cost_total * 100
    best_ticker  = list(PORTFOLIO_HOLDINGS.keys())[np.argmax(pnl_vals)]
    worst_ticker = list(PORTFOLIO_HOLDINGS.keys())[np.argmin(pnl_vals)]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Portfolio Value",  f"${current_value:,.0f}")
    m2.metric("Total Cost",       f"${cost_total:,.0f}")
    m3.metric("Total Gain/Loss",  f"${total_gain:+,.0f}", f"{total_return:+.1f}%")
    m4.metric("Best Holding",     best_ticker)
    m5.metric("Worst Holding",    worst_ticker)

    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### Sector Allocation")
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        st.markdown("#### Portfolio Value Over Time")
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("#### Individual Stock P&L (%)")
    st.plotly_chart(fig_pnl, use_container_width=True)

    # Detailed holdings table
    st.markdown("#### Holdings Detail")
    rows = []
    for ticker_sym, info in PORTFOLIO_HOLDINGS.items():
        price     = get_current_price(ticker_sym)
        mkt_val   = price * info["shares"]
        cost_val  = info["cost_basis"] * info["shares"]
        gain      = mkt_val - cost_val
        ret       = (price - info["cost_basis"]) / info["cost_basis"] * 100
        weight    = mkt_val / current_value * 100
        rows.append({
            "Ticker":       ticker_sym,
            "Sector":       info["sector"],
            "Shares":       info["shares"],
            "Cost Basis":   f"${info['cost_basis']:.2f}",
            "Current Price":f"${price:.2f}",
            "Mkt Value":    f"${mkt_val:,.0f}",
            "Gain/Loss":    f"${gain:+,.0f}",
            "Return":       f"{ret:+.1f}%",
            "Weight":       f"{weight:.1f}%",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Ticker"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD: MARKET HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

elif dashboard_page == "🔥 Market Heatmap":
    st.markdown("## 🔥 Market Heatmap Dashboard")
    st.caption("Multi-timeframe return heatmap — static illustrative data")
    st.markdown("---")

    # Quick stats
    tickers_all = list(BASE_PRICES.keys())
    one_day_rets = []
    for t in tickers_all:
        df_t = generate_ohlcv(t)
        one_day_rets.append(
            (df_t["Close"].iloc[-1] / df_t["Close"].iloc[-2] - 1) * 100
        )

    gainers  = sum(1 for r in one_day_rets if r > 0)
    avg_1d   = np.mean(one_day_rets)
    best_1d  = tickers_all[np.argmax(one_day_rets)]
    worst_1d = tickers_all[np.argmin(one_day_rets)]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Advancing Today",  f"{gainers} / {len(tickers_all)}")
    m2.metric("Avg 1-Day Return", f"{avg_1d:+.2f}%")
    m3.metric("Best Today",       best_1d,  f"{max(one_day_rets):+.2f}%")
    m4.metric("Worst Today",      worst_1d, f"{min(one_day_rets):+.2f}%")

    st.markdown("---")
    st.markdown("#### Multi-Timeframe Return Heatmap")
    st.plotly_chart(build_heatmap(), use_container_width=True)

    # Sparklines table
    st.markdown("#### 30-Day Price Trend (Sparkline)")
    spark_cols = st.columns(len(tickers_all))
    for i, t in enumerate(tickers_all):
        df_t  = generate_ohlcv(t, 30)
        ret30 = (df_t["Close"].iloc[-1] / df_t["Close"].iloc[0] - 1) * 100
        fig_spark = go.Figure(go.Scatter(
            y=df_t["Close"],
            mode="lines",
            line=dict(
                color="#10b981" if ret30 >= 0 else "#ef4444",
                width=1.5,
            ),
        ))
        fig_spark.update_layout(
            height=80, margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        with spark_cols[i]:
            st.caption(f"**{t}**  {ret30:+.1f}%")
            st.plotly_chart(fig_spark, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD: CORRELATION & RISK
# ══════════════════════════════════════════════════════════════════════════════

elif dashboard_page == "🔗 Correlation & Risk":
    st.markdown("## 🔗 Correlation & Risk Dashboard")
    st.caption("Return correlations and risk/return scatter — static illustrative data")
    st.markdown("---")

    # Risk metrics
    tickers_all = list(BASE_PRICES.keys())
    sharpe_list, vol_list = [], []
    rf = 0.05 / 252  # daily risk-free rate
    for t in tickers_all:
        df_t = generate_ohlcv(t)
        ret  = df_t["Close"].pct_change().dropna()
        ann_vol = ret.std() * np.sqrt(252) * 100
        ann_ret = ret.mean() * 252 * 100
        sharpe  = (ret.mean() - rf) / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
        vol_list.append(round(ann_vol, 2))
        sharpe_list.append(round(sharpe, 2))

    best_sharpe  = tickers_all[np.argmax(sharpe_list)]
    low_vol      = tickers_all[np.argmin(vol_list)]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best Sharpe Ratio", best_sharpe, f"{max(sharpe_list):.2f}")
    m2.metric("Lowest Volatility", low_vol,     f"{min(vol_list):.1f}%")
    m3.metric("Avg Annualised Vol", f"{np.mean(vol_list):.1f}%")
    m4.metric("Avg Sharpe Ratio",   f"{np.mean(sharpe_list):.2f}")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("#### Return Correlation Matrix")
        st.plotly_chart(build_correlation_matrix(), use_container_width=True)

    with col_right:
        st.markdown("#### Risk / Return Scatter")
        st.plotly_chart(build_volatility_chart(), use_container_width=True)

    # Sharpe table
    st.markdown("#### Risk Metrics Summary")
    risk_rows = []
    for i, t in enumerate(tickers_all):
        df_t    = generate_ohlcv(t)
        ret     = df_t["Close"].pct_change().dropna()
        ann_ret = ret.mean() * 252 * 100
        max_dd  = ((df_t["Close"] / df_t["Close"].cummax()) - 1).min() * 100
        risk_rows.append({
            "Ticker":          t,
            "Ann. Return (%)": f"{ann_ret:+.1f}%",
            "Ann. Vol (%)":    f"{vol_list[i]:.1f}%",
            "Sharpe Ratio":    f"{sharpe_list[i]:.2f}",
            "Max Drawdown (%)":f"{max_dd:.1f}%",
            "Risk Grade":      ("🟢 Low"    if vol_list[i] < 25
                                else "🟡 Medium" if vol_list[i] < 35
                                else "🔴 High"),
        })
    st.dataframe(pd.DataFrame(risk_rows).set_index("Ticker"), use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="font-size:11px; color:#9ca3af; text-align:center;">'
    'Python for Data Analysis &amp; Visualization · '
    'Built with Streamlit + Plotly · '
    'All data is synthetic — for educational purposes only, not financial advice.'
    '</p>',
    unsafe_allow_html=True,
)
