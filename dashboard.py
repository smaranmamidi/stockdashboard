"""
Interactive Sales Dashboard using Streamlit
This dashboard provides comprehensive sales analytics with filters and visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .title-style {
            color: #1f77b4;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and prepare data"""
    df = pd.read_csv('train_cleaned.csv')
    
    # Convert date columns to datetime
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    
    # Extract useful time features
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Month_Name'] = df['Order Date'].dt.strftime('%B')
    df['Year_Month'] = df['Order Date'].dt.to_period('M')
    df['Quarter'] = df['Order Date'].dt.quarter
    df['Day of Week'] = df['Order Date'].dt.day_name()
    
    # Calculate Profit (estimated as 20% of Sales for demonstration)
    # In real scenario, this would come from actual profit column
    df['Profit'] = df['Sales'] * 0.20
    df['Profit Margin'] = 20
    
    # Calculate shipping days
    df['Shipping Days'] = (df['Ship Date'] - df['Order Date']).dt.days
    
    return df


def display_kpi_cards(filtered_df):
    """Display key performance indicators"""
    st.markdown("### 📈 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_sales = filtered_df['Sales'].sum()
        st.metric(label="Total Sales", value=f"${total_sales:,.2f}")
    
    with col2:
        total_profit = filtered_df['Profit'].sum()
        st.metric(label="Total Profit", value=f"${total_profit:,.2f}")
    
    with col3:
        total_orders = filtered_df.shape[0]
        st.metric(label="Total Orders", value=f"{total_orders:,}")
    
    with col4:
        avg_order_value = filtered_df['Sales'].mean()
        st.metric(label="Avg Order Value", value=f"${avg_order_value:,.2f}")
    
    with col5:
        profit_margin = (filtered_df['Profit'].sum() / filtered_df['Sales'].sum() * 100)
        st.metric(label="Profit Margin", value=f"{profit_margin:.1f}%")


def create_sales_trend_chart(filtered_df):
    """Sales trend line chart"""
    df_trend = filtered_df.groupby('Year_Month')['Sales'].sum().reset_index()
    df_trend['Year_Month'] = df_trend['Year_Month'].astype(str)
    
    fig = px.line(
        df_trend,
        x='Year_Month',
        y='Sales',
        markers=True,
        title='Sales Trend Over Time',
        labels={'Year_Month': 'Month', 'Sales': 'Sales ($)'},
        template='plotly_white'
    )
    fig.update_traces(line=dict(color='#1f77b4', width=3), marker=dict(size=8))
    return fig


def create_category_sales_chart(filtered_df):
    """Category sales bar chart"""
    df_category = filtered_df.groupby('Category')['Sales'].sum().sort_values(ascending=False).reset_index()
    
    fig = px.bar(
        df_category,
        x='Category',
        y='Sales',
        title='Sales by Category',
        labels={'Category': 'Category', 'Sales': 'Sales ($)'},
        template='plotly_white',
        color='Sales',
        color_continuous_scale='Blues'
    )
    fig.update_layout(showlegend=False)
    return fig


def create_profit_vs_sales_scatter(filtered_df):
    """Profit vs Sales scatter plot"""
    fig = px.scatter(
        filtered_df,
        x='Sales',
        y='Profit',
        color='Category',
        size='Profit Margin',
        hover_name='Product Name',
        title='Profit vs Sales Analysis',
        labels={'Sales': 'Sales ($)', 'Profit': 'Profit ($)'},
        template='plotly_white'
    )
    return fig


def create_correlation_heatmap(filtered_df):
    """Correlation heatmap"""
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['Year', 'Month', 'Quarter']]
    if len(numeric_cols) > 1:
        corr_matrix = filtered_df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text:.2f}',
            textfont={"size": 10}
        ))
        fig.update_layout(title='Correlation Heatmap', width=600, height=600)
        return fig
    return None


def create_region_pie_chart(filtered_df):
    """Region pie chart"""
    df_region = filtered_df.groupby('Region')['Sales'].sum()
    
    fig = px.pie(
        values=df_region.values,
        names=df_region.index,
        title='Sales Distribution by Region',
        template='plotly_white'
    )
    return fig


def create_top_products_chart(filtered_df, top_n=10):
    """Top products bar chart"""
    df_products = filtered_df.groupby('Product Name')['Sales'].sum().nlargest(top_n).reset_index()
    df_products.columns = ['Product Name', 'Sales']
    
    fig = px.barh(
        df_products,
        x='Sales',
        y='Product Name',
        title=f'Top {top_n} Products by Sales',
        labels={'Sales': 'Sales ($)', 'Product Name': 'Product'},
        template='plotly_white',
        color='Sales',
        color_continuous_scale='Greens'
    )
    fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
    return fig


def create_segment_sales_chart(filtered_df):
    """Sales by customer segment"""
    df_segment = filtered_df.groupby('Segment')['Sales'].sum().sort_values(ascending=False).reset_index()
    
    fig = px.bar(
        df_segment,
        x='Segment',
        y='Sales',
        title='Sales by Customer Segment',
        labels={'Segment': 'Segment', 'Sales': 'Sales ($)'},
        template='plotly_white',
        color='Sales',
        color_continuous_scale='Oranges'
    )
    fig.update_layout(showlegend=False)
    return fig


def create_monthly_profit_chart(filtered_df):
    """Monthly profit trend"""
    df_monthly = filtered_df.groupby('Month_Name')[['Sales', 'Profit']].sum().reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_monthly['Month_Name'],
        y=df_monthly['Sales'],
        name='Sales',
        marker_color='lightblue'
    ))
    fig.add_trace(go.Bar(
        x=df_monthly['Month_Name'],
        y=df_monthly['Profit'],
        name='Profit',
        marker_color='orange'
    ))
    fig.update_layout(
        title='Monthly Sales vs Profit',
        barmode='group',
        xaxis_title='Month',
        yaxis_title='Amount ($)',
        template='plotly_white'
    )
    return fig


def main():
    """Main application"""
    st.markdown("<h1 class='title-style'>📊 Sales Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    # Sidebar filters
    st.sidebar.markdown("## 🔍 Filters")
    
    # Region filter
    regions = st.sidebar.multiselect(
        "Select Region(s)",
        options=df['Region'].unique(),
        default=df['Region'].unique(),
        key='region_filter'
    )
    
    # Category filter
    categories = st.sidebar.multiselect(
        "Select Category/Categories",
        options=df['Category'].unique(),
        default=df['Category'].unique(),
        key='category_filter'
    )
    
    # Segment filter
    segments = st.sidebar.multiselect(
        "Select Customer Segment(s)",
        options=df['Segment'].unique(),
        default=df['Segment'].unique(),
        key='segment_filter'
    )
    
    # Ship Mode filter
    ship_modes = st.sidebar.multiselect(
        "Select Ship Mode(s)",
        options=df['Ship Mode'].unique(),
        default=df['Ship Mode'].unique(),
        key='ship_mode_filter'
    )
    
    # Date range filter
    st.sidebar.markdown("### 📅 Date Range")
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['Order Date'].min(), df['Order Date'].max()),
        min_value=df['Order Date'].min(),
        max_value=df['Order Date'].max()
    )
    
    # Apply filters
    filtered_df = df[
        (df['Region'].isin(regions)) &
        (df['Category'].isin(categories)) &
        (df['Segment'].isin(segments)) &
        (df['Ship Mode'].isin(ship_modes)) &
        (df['Order Date'].dt.date >= date_range[0]) &
        (df['Order Date'].dt.date <= date_range[1])
    ]
    
    # Display filter info
    st.sidebar.markdown("---")
    st.sidebar.info(f"📊 Showing {filtered_df.shape[0]} orders from {filtered_df.shape[0]:,} total")
    
    # Display KPI cards
    display_kpi_cards(filtered_df)
    st.markdown("---")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "📊 Analysis", "🔗 Correlations", "🏆 Rankings"])
    
    with tab1:
        st.markdown("### Sales Performance Trends")
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_sales_trend_chart(filtered_df), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_monthly_profit_chart(filtered_df), use_container_width=True)
    
    with tab2:
        st.markdown("### Sales Breakdown Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_category_sales_chart(filtered_df), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_segment_sales_chart(filtered_df), use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.plotly_chart(create_region_pie_chart(filtered_df), use_container_width=True)
        
        with col4:
            st.plotly_chart(create_profit_vs_sales_scatter(filtered_df), use_container_width=True)
    
    with tab3:
        st.markdown("### Data Correlations")
        corr_chart = create_correlation_heatmap(filtered_df)
        if corr_chart:
            st.plotly_chart(corr_chart, use_container_width=True)
        else:
            st.warning("Insufficient data for correlation analysis")
    
    with tab4:
        st.markdown("### Top Performers")
        
        # Add slider for top products count
        top_n = st.slider("Number of top products to display", min_value=5, max_value=20, value=10)
        
        st.plotly_chart(create_top_products_chart(filtered_df, top_n), use_container_width=True)
    
    # Detailed data view
    st.markdown("---")
    with st.expander("📋 View Detailed Data"):
        st.dataframe(
            filtered_df[[
                'Order Date', 'Order ID', 'Customer Name', 'Category', 
                'Sub-Category', 'Sales', 'Profit', 'Region', 'Segment'
            ]].sort_values('Order Date', ascending=False),
            use_container_width=True,
            height=400
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888; font-size: 12px;'>"
        "Sales Analytics Dashboard | Data-driven insights for better decision making"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
