import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import re

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp {
    background: #0d0f14;
    color: #e8eaf0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #13161e !important;
    border-right: 1px solid #1f2330;
}
[data-testid="stSidebar"] * {
    color: #c8cad6 !important;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #191d28 0%, #1a1f2e 100%);
    border: 1px solid #252a3a;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
}
div[data-testid="metric-container"] label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7b82a0 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: #e8eaf0 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.78rem;
}

/* Section headers */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #e8eaf0;
    margin: 1.8rem 0 0.8rem;
    letter-spacing: -0.01em;
    border-left: 3px solid #4f6ef7;
    padding-left: 0.75rem;
}

/* Page title */
.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #4f6ef7, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.main-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    color: #6b7280;
    margin-top: 0.2rem;
    margin-bottom: 1.5rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #13161e;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #1f2330;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    color: #7b82a0;
    border-radius: 7px;
}
.stTabs [aria-selected="true"] {
    background: #1f2a4a !important;
    color: #4f6ef7 !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #1f2330;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4f6ef7, #7c3aed);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    padding: 0.5rem 1.4rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Info/warning/success boxes */
.stAlert { border-radius: 10px; }

/* Select boxes */
[data-testid="stSelectbox"] > div > div {
    background: #191d28;
    border: 1px solid #252a3a;
    border-radius: 8px;
    color: #e8eaf0;
}

/* Multiselect */
[data-testid="stMultiSelect"] > div {
    background: #191d28;
    border: 1px solid #252a3a;
    border-radius: 8px;
}

/* Expander */
details {
    background: #13161e;
    border: 1px solid #1f2330;
    border-radius: 10px;
    padding: 0.5rem 1rem;
}

/* Clean tag */
.clean-badge {
    display: inline-block;
    background: #0f2a1a;
    color: #34d399;
    border: 1px solid #064e3b;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-left: 8px;
}
.dirty-badge {
    display: inline-block;
    background: #2a1a0f;
    color: #f87171;
    border: 1px solid #7f1d1d;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-left: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly dark template ────────────────────────────────────────────────────────
PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#c8cad6"),
    colorway=["#4f6ef7", "#a78bfa", "#34d399", "#f59e0b", "#f87171", "#38bdf8", "#fb7185"],
)

# ── Data cleaning ───────────────────────────────────────────────────────────────
@st.cache_data
def clean_data(df: pd.DataFrame):
    report = []
    orig_shape = df.shape

    # 1. Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # 2. Remove exact duplicate rows
    dupes = df.duplicated().sum()
    if dupes:
        df = df.drop_duplicates()
        report.append(f"✅ Removed **{dupes}** exact duplicate rows")
    else:
        report.append("✅ No duplicate rows found")

    # 3. Parse dates (handles DD/MM/YYYY and MM/DD/YYYY gracefully)
    for col in ["Order Date", "Ship Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
            nulls = df[col].isna().sum()
            if nulls:
                report.append(f"⚠️ **{nulls}** unparseable dates in `{col}` set to NaT")
            else:
                report.append(f"✅ `{col}` parsed to datetime")

    # 4. Numeric cleaning for Sales
    if "Sales" in df.columns:
        df["Sales"] = pd.to_numeric(df["Sales"].astype(str).str.replace(",", ""), errors="coerce")
        nulls = df["Sales"].isna().sum()
        if nulls:
            report.append(f"⚠️ **{nulls}** non-numeric Sales values set to NaN")
            df["Sales"].fillna(0, inplace=True)
        report.append("✅ `Sales` converted to float")

    # 5. Postal Code → string (preserve leading zeros)
    if "Postal Code" in df.columns:
        df["Postal Code"] = df["Postal Code"].astype(str).str.zfill(5)
        report.append("✅ `Postal Code` preserved as zero-padded string")

    # 6. Strip string columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip() if s.dtype == "object" else s)
    report.append(f"✅ Stripped whitespace from {len(str_cols)} text columns")

    # 7. Standardise category casing
    for col in ["Category", "Sub-Category", "Segment", "Ship Mode", "Region"]:
        if col in df.columns:
            df[col] = df[col].str.title()

    # 8. Derive helper columns
    if "Order Date" in df.columns and df["Order Date"].dtype == "datetime64[ns]":
        df["Order Year"]  = df["Order Date"].dt.year
        df["Order Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
        df["Order Quarter"] = df["Order Date"].dt.to_period("Q").astype(str)
        report.append("✅ Derived `Order Year`, `Order Month`, `Order Quarter`")

    if "Ship Date" in df.columns and "Order Date" in df.columns:
        df["Ship Lag (days)"] = (df["Ship Date"] - df["Order Date"]).dt.days
        report.append("✅ Derived `Ship Lag (days)`")

    # 9. Handle remaining nulls
    null_total = df.isnull().sum().sum()
    if null_total:
        report.append(f"⚠️ **{null_total}** remaining null values in dataset")
    else:
        report.append("✅ No remaining null values")

    report.append(f"\n📐 Shape: **{orig_shape[0]} rows × {orig_shape[1]} cols** → **{df.shape[0]} rows × {df.shape[1]} cols**")
    return df, report

# ── Load data ───────────────────────────────────────────────────────────────────
def load_data(source) -> pd.DataFrame:
    if isinstance(source, str):
        return pd.read_csv(io.StringIO(source))
    return pd.read_csv(source)

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Data Source")
    mode = st.radio("", ["Use sample data", "Upload CSV"], label_visibility="collapsed")

    raw_df = None
    if mode == "Upload CSV":
        uploaded = st.file_uploader("Upload your sales CSV", type=["csv"])
        if uploaded:
            raw_df = load_data(uploaded)
    else:
        SAMPLE = """Row ID,Order ID,Order Date,Ship Date,Ship Mode,Customer ID,Customer Name,Segment,Country,City,State,Postal Code,Region,Product ID,Category,Sub-Category,Product Name,Sales
1,CA-2017-152156,08/11/2017,11/11/2017,Second Class,CG-12520,Claire Gute,Consumer,United States,Henderson,Kentucky,42420,South,FUR-BO-10001798,Furniture,Bookcases,Bush Somerset Collection Bookcase,261.96
2,CA-2017-152156,08/11/2017,11/11/2017,Second Class,CG-12520,Claire Gute,Consumer,United States,Henderson,Kentucky,42420,South,FUR-CH-10000454,Furniture,Chairs,"Hon Deluxe Fabric Upholstered Stacking Chairs, Rounded Back",731.94
3,CA-2017-138688,12/06/2017,16/06/2017,Second Class,DV-13045,Darrin Van Huff,Corporate,United States,Los Angeles,California,90036,West,OFF-LA-10000240,Office Supplies,Labels,Self-Adhesive Address Labels for Typewriters by Universal,14.62
4,US-2016-108966,11/10/2016,18/10/2016,Standard Class,SO-20335,Sean O'Donnell,Consumer,United States,Fort Lauderdale,Florida,33311,South,FUR-TA-10000577,Furniture,Tables,Bretford CR4500 Series Slim Rectangular Table,957.5775
5,US-2016-108966,11/10/2016,18/10/2016,Standard Class,SO-20335,Sean O'Donnell,Consumer,United States,Fort Lauderdale,Florida,33311,South,OFF-ST-10000760,Office Supplies,Storage,Eldon Fold 'N Roll Cart System,22.368
6,CA-2016-115812,09/06/2016,14/06/2016,Standard Class,BH-11710,Brosina Hoffman,Consumer,United States,Los Angeles,California,90032,West,FUR-FU-10001487,Furniture,Furnishings,Eldon Expressions Wood and Plastic Frames,48.86
7,CA-2016-115812,09/06/2016,14/06/2016,Standard Class,BH-11710,Brosina Hoffman,Consumer,United States,Los Angeles,California,90032,West,OFF-AR-10002833,Office Supplies,Art,Newell 322,7.28
8,CA-2016-115812,09/06/2016,14/06/2016,Standard Class,BH-11710,Brosina Hoffman,Consumer,United States,Los Angeles,California,90032,West,TEC-PH-10002275,Technology,Phones,Mitel 5320 IP Phone VoIP phone,907.152
9,CA-2016-115812,09/06/2016,14/06/2016,Standard Class,BH-11710,Brosina Hoffman,Consumer,United States,Los Angeles,California,90032,West,OFF-BI-10003910,Office Supplies,Binders,DXL Angle-View Binders with Locking Rings by Samsill,18.504
10,CA-2016-115812,09/06/2016,14/06/2016,Standard Class,BH-11710,Brosina Hoffman,Consumer,United States,Los Angeles,California,90032,West,OFF-AP-10002892,Office Supplies,Appliances,Belkin F5C206VTEL 6 Outlet,114.9
11,CA-2016-115812,09/06/2016,14/06/2016,Standard Class,BH-11710,Brosina Hoffman,Consumer,United States,Los Angeles,California,90032,West,FUR-TA-10001539,Furniture,Tables,Chromcraft Rectangular Conference Tables,1706.184
12,CA-2016-115812,09/06/2016,14/06/2016,Standard Class,BH-11710,Brosina Hoffman,Consumer,United States,Los Angeles,California,90032,West,TEC-PH-10002033,Technology,Phones,Konftel 250 Conference phone,911.424
13,CA-2015-114412,15/04/2015,20/04/2015,Standard Class,AA-10480,Andrew Allen,Consumer,United States,Concord,North Carolina,28027,South,OFF-PA-10002365,Office Supplies,Paper,Xerox 1967,15.552
14,CA-2017-161389,05/12/2017,10/12/2017,Second Class,IM-15070,Irene Maddox,Consumer,United States,Seattle,Washington,98103,West,OFF-BI-10003656,Office Supplies,Binders,Fellowes PB200 Electric Punch Plastic Comb Binding Machine,407.976
15,US-2015-118983,22/11/2015,26/11/2015,Standard Class,HP-14815,Harold Pawlan,Home Office,United States,Fort Worth,Texas,76106,Central,OFF-AP-10002311,Office Supplies,Appliances,Holmes Replacement Filter for HEPA Air Cleaner,68.81
16,US-2015-118983,22/11/2015,26/11/2015,Standard Class,HP-14815,Harold Pawlan,Home Office,United States,Fort Worth,Texas,76106,Central,OFF-BI-10000756,Office Supplies,Binders,Storex DuraTech Readers Rack,2.544
17,CA-2016-105893,11/11/2016,18/11/2016,Standard Class,PK-19075,Pete Kriz,Consumer,United States,Madison,Wisconsin,53711,Central,OFF-ST-10004186,Office Supplies,Storage,Snap-A-Way Fluorescent Orange Color Shipping Labels,665.88
18,CA-2016-167164,13/05/2016,15/05/2016,First Class,AG-10270,Alejandro Grove,Consumer,United States,West Jordan,Utah,84084,West,OFF-ST-10000107,Office Supplies,Storage,Fellowes Super Stor/Drawer,55.5
19,CA-2016-143567,27/08/2016,01/09/2016,Second Class,ZD-21925,Zuschuss Donatelli,Consumer,United States,San Francisco,California,94109,West,OFF-ST-10000304,Office Supplies,Storage,Safco Commercial Wire Shelving,213.48
20,CA-2016-143567,27/08/2016,01/09/2016,Second Class,ZD-21925,Zuschuss Donatelli,Consumer,United States,San Francisco,California,94109,West,TEC-PH-10003645,Technology,Phones,Mitel 5212 IP Phone VoIP phone,448.17
21,CA-2015-137330,16/11/2015,21/11/2015,Standard Class,KB-16585,Ken Black,Corporate,United States,Fremont,Nebraska,68025,Central,OFF-AP-10001492,Office Supplies,Appliances,Hoover Replacement Belts for Hoover Soft Guard Vacuums,19.46
22,CA-2015-137330,16/11/2015,21/11/2015,Standard Class,KB-16585,Ken Black,Corporate,United States,Fremont,Nebraska,68025,Central,TEC-PH-10004977,Technology,Phones,GE 30524EE4 Basic Phone,12.78"""
        raw_df = load_data(SAMPLE)

    st.markdown("---")
    st.markdown("### 🎛️ Filters")
    if raw_df is not None:
        df_clean, _ = clean_data(raw_df.copy())
        years = sorted(df_clean["Order Year"].dropna().unique().astype(int).tolist()) if "Order Year" in df_clean.columns else []
        sel_years = st.multiselect("Year", years, default=years)
        categories = sorted(df_clean["Category"].dropna().unique().tolist()) if "Category" in df_clean.columns else []
        sel_cats = st.multiselect("Category", categories, default=categories)
        segments = sorted(df_clean["Segment"].dropna().unique().tolist()) if "Segment" in df_clean.columns else []
        sel_segs = st.multiselect("Segment", segments, default=segments)
        regions = sorted(df_clean["Region"].dropna().unique().tolist()) if "Region" in df_clean.columns else []
        sel_regions = st.multiselect("Region", regions, default=regions)

# ── Main content ────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Sales Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="main-sub">End-to-end cleaning, quality report & interactive analytics</div>', unsafe_allow_html=True)

if raw_df is None:
    st.info("👈 Use the sidebar to load data or upload your own CSV.")
    st.stop()

df_clean, clean_report = clean_data(raw_df.copy())

# Apply filters
filtered = df_clean.copy()
if sel_years:    filtered = filtered[filtered["Order Year"].isin(sel_years)]
if sel_cats:     filtered = filtered[filtered["Category"].isin(sel_cats)]
if sel_segs:     filtered = filtered[filtered["Segment"].isin(sel_segs)]
if sel_regions:  filtered = filtered[filtered["Region"].isin(sel_regions)]

# ── Tabs ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🧹 Data Cleaning", "📊 Overview", "🔍 Deep Dive", "📋 Raw Data"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Cleaning Report</div>', unsafe_allow_html=True)
    for line in clean_report:
        st.markdown(line)

    st.markdown('<div class="section-title">Missing Values</div>', unsafe_allow_html=True)
    null_counts = df_clean.isnull().sum()
    null_df = pd.DataFrame({"Column": null_counts.index, "Missing": null_counts.values, "% Missing": (null_counts.values / len(df_clean) * 100).round(2)})
    null_df = null_df[null_df["Missing"] > 0]
    if null_df.empty:
        st.success("🎉 Dataset is fully clean — zero missing values after cleaning!")
    else:
        st.dataframe(null_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Column Data Types</div>', unsafe_allow_html=True)
    dtype_df = pd.DataFrame({"Column": df_clean.dtypes.index, "Type": df_clean.dtypes.astype(str).values})
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Statistical Summary</div>', unsafe_allow_html=True)
    num_cols = df_clean.select_dtypes(include="number").columns.tolist()
    st.dataframe(df_clean[num_cols].describe().T.round(2), use_container_width=True)

    # Download
    csv_bytes = df_clean.to_csv(index=False).encode()
    st.download_button("⬇️ Download Cleaned CSV", csv_bytes, "cleaned_sales.csv", "text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    total_sales    = filtered["Sales"].sum()
    total_orders   = filtered["Order ID"].nunique()
    avg_order_val  = filtered.groupby("Order ID")["Sales"].sum().mean()
    top_category   = filtered.groupby("Category")["Sales"].sum().idxmax() if not filtered.empty else "—"
    avg_ship_lag   = filtered["Ship Lag (days)"].mean() if "Ship Lag (days)" in filtered.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Sales",      f"${total_sales:,.0f}")
    c2.metric("Unique Orders",    f"{total_orders:,}")
    c3.metric("Avg Order Value",  f"${avg_order_val:,.0f}")
    c4.metric("Top Category",     top_category)
    c5.metric("Avg Ship Lag",     f"{avg_ship_lag:.1f} days")

    # Monthly trend
    if "Order Month" in filtered.columns:
        st.markdown('<div class="section-title">Monthly Sales Trend</div>', unsafe_allow_html=True)
        monthly = filtered.groupby("Order Month")["Sales"].sum().reset_index().sort_values("Order Month")
        fig = px.area(monthly, x="Order Month", y="Sales",
                      labels={"Sales": "Revenue ($)", "Order Month": ""},
                      **PLOTLY_THEME)
        fig.update_traces(fill="tozeroy", line=dict(color="#4f6ef7", width=2),
                          fillcolor="rgba(79,110,247,0.15)")
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Sales by Category</div>', unsafe_allow_html=True)
        cat_sales = filtered.groupby("Category")["Sales"].sum().reset_index()
        fig2 = px.pie(cat_sales, names="Category", values="Sales", hole=0.55,
                      color_discrete_sequence=["#4f6ef7","#a78bfa","#34d399"], **PLOTLY_THEME)
        fig2.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0), showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Sales by Region</div>', unsafe_allow_html=True)
        reg_sales = filtered.groupby("Region")["Sales"].sum().sort_values().reset_index()
        fig3 = px.bar(reg_sales, x="Sales", y="Region", orientation="h",
                      color="Sales", color_continuous_scale=["#1e254a","#4f6ef7"],
                      labels={"Sales": "Revenue ($)"}, **PLOTLY_THEME)
        fig3.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0), coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — DEEP DIVE
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    col_x, col_y = st.columns(2)

    with col_x:
        st.markdown('<div class="section-title">Sub-Category Breakdown</div>', unsafe_allow_html=True)
        sub_sales = filtered.groupby(["Category","Sub-Category"])["Sales"].sum().reset_index().sort_values("Sales", ascending=False).head(15)
        fig4 = px.bar(sub_sales, x="Sales", y="Sub-Category", color="Category",
                      orientation="h", labels={"Sales": "Revenue ($)"},
                      color_discrete_sequence=["#4f6ef7","#a78bfa","#34d399"],
                      **PLOTLY_THEME)
        fig4.update_layout(height=420, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig4, use_container_width=True)

    with col_y:
        st.markdown('<div class="section-title">Sales by Segment</div>', unsafe_allow_html=True)
        seg_sales = filtered.groupby("Segment")["Sales"].sum().reset_index()
        fig5 = px.funnel(seg_sales, x="Sales", y="Segment",
                         color_discrete_sequence=["#4f6ef7"],
                         **PLOTLY_THEME)
        fig5.update_layout(height=420, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown('<div class="section-title">Ship Mode Performance</div>', unsafe_allow_html=True)
    if "Ship Lag (days)" in filtered.columns:
        ship = filtered.groupby("Ship Mode").agg(
            Orders=("Order ID","nunique"),
            Avg_Lag=("Ship Lag (days)","mean"),
            Total_Sales=("Sales","sum")
        ).reset_index()
        fig6 = px.scatter(ship, x="Avg_Lag", y="Total_Sales", size="Orders",
                          color="Ship Mode", text="Ship Mode",
                          labels={"Avg_Lag":"Avg Ship Lag (days)","Total_Sales":"Total Revenue ($)"},
                          color_discrete_sequence=["#4f6ef7","#a78bfa","#34d399","#f59e0b"],
                          **PLOTLY_THEME)
        fig6.update_traces(textposition="top center")
        fig6.update_layout(height=340, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown('<div class="section-title">Top 10 Products by Revenue</div>', unsafe_allow_html=True)
    top_prod = filtered.groupby("Product Name")["Sales"].sum().nlargest(10).reset_index()
    top_prod["Product Name"] = top_prod["Product Name"].str[:40] + "…"
    fig7 = px.bar(top_prod, x="Sales", y="Product Name", orientation="h",
                  color="Sales", color_continuous_scale=["#1e254a","#a78bfa"],
                  labels={"Sales": "Revenue ($)"}, **PLOTLY_THEME)
    fig7.update_layout(height=380, margin=dict(l=0,r=0,t=20,b=0), coloraxis_showscale=False)
    st.plotly_chart(fig7, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — RAW DATA
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">Cleaned Dataset Preview</div>', unsafe_allow_html=True)
    cols_to_show = [c for c in ["Order ID","Order Date","Ship Date","Ship Lag (days)","Customer Name",
                                 "Segment","Region","Category","Sub-Category","Product Name","Sales"]
                    if c in filtered.columns]
    st.dataframe(filtered[cols_to_show].reset_index(drop=True), use_container_width=True, height=460)
    st.caption(f"{len(filtered):,} rows × {len(cols_to_show)} columns (filtered view)")
