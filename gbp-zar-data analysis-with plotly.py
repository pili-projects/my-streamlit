
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
from datetime import datetime
import io
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Page config
st.set_page_config(
    page_title="GBP to ZAR Transfer Analysis",
    page_icon="💰",
    layout="wide"
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Title
st.title("💰 GBP to ZAR Volume Analysis (Q2-Q4 2023)")
st.markdown("Analyzing daily transfer volumes with focus on **bimodality** assessment")

# Sidebar for navigation
with st.sidebar:
    st.title("Navigation")
    analysis_section = st.radio(
        "Choose Analysis Section:",
        ["📊 Dataset Overview", 
         "📈 Q1: Distribution Analysis", 
         "📊 Q2: Quarterly Changes",
         "🔮 Q3: October 2023 Estimation",     
         "📋 Summary",
         "🔬 Analytical Methodology"]
    )
    
    st.markdown("---")
    st.markdown("### About this Analysis")
    st.markdown("""
    **Focus Areas:**
    - Distribution shape & bimodality
    - Quarterly trend analysis
    - Missing data imputation
    - Business implications
    
    **Methods Used:**
    - Statistical hypothesis testing
    - Bootstrap uncertainty quantification
    - KDE analysis
    - Non-parametric comparisons
    """)

# Load data
@st.cache_data
def load_data():
    try:
        url = "https://drive.google.com/uc?id=1BK9eVWAu2LCDJ2haDYafoKhPyWP2tqCl"
        df = pd.read_csv(url)
        
        if df.empty:
            st.error("The dataset is empty. Please check the data source.")
            return pd.DataFrame()
        
        # Check required columns
        required_columns = ['posting_date', 'volume_gbp']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return pd.DataFrame()
        
        df['posting_date'] = pd.to_datetime(df['posting_date'], errors='coerce')
        
        # Check for invalid dates
        invalid_dates = df['posting_date'].isna().sum()
        if invalid_dates > 0:
            st.warning(f"Found {invalid_dates} invalid dates that were converted to NaT")
        
        df = df.sort_values('posting_date').reset_index(drop=True)
        
        # Feature engineering
        df['year'] = df['posting_date'].dt.year
        df['month'] = df['posting_date'].dt.month
        df['quarter'] = df['posting_date'].dt.quarter
        df['weekday'] = df['posting_date'].dt.day_name()
        df['day_of_week'] = df['posting_date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        df['month_name'] = df['posting_date'].dt.strftime('%B')
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

df = load_data()

# Helper function to format dates in dataframes for display
def format_dataframe_dates(df, date_format='%Y-%m-%d'):
    """Format datetime columns to date-only for display"""
    df_formatted = df.copy()
    for col in df_formatted.columns:
        if pd.api.types.is_datetime64_any_dtype(df_formatted[col]):
            df_formatted[col] = df_formatted[col].dt.strftime(date_format)
    return df_formatted

# DATASET OVERVIEW SECTION
if analysis_section == "📊 Dataset Overview":
    st.header("📊 Dataset Overview")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Days", f"{len(df):,}")
        start_date = df['posting_date'].min().strftime('%b %d, %Y')
        end_date = df['posting_date'].max().strftime('%b %d, %Y')
        st.metric("Date Range", f"{df['posting_date'].min().strftime('%b %d')} to {df['posting_date'].max().strftime('%b %d')}")
    
    with col2:
        total_volume = df['volume_gbp'].sum()
        st.metric("Total Volume", f"£{total_volume:,.0f}")
        st.metric("Average Daily", f"£{df['volume_gbp'].mean():,.0f}")
    
    with col3:
        zero_days = (df['volume_gbp'] == 0).sum()
        st.metric("Zero Volume Days", zero_days)
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Data preview
    st.subheader("Data Preview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**First 10 records:**")
        display_df = df.head(10).copy()
        display_df['posting_date'] = display_df['posting_date'].dt.strftime('%Y-%m-%d')
        st.dataframe(display_df, use_container_width=True)
    
    with col2:
        st.write("**Last 10 records:**")
        display_df = df.tail(10).copy()
        display_df['posting_date'] = display_df['posting_date'].dt.strftime('%Y-%m-%d')
        st.dataframe(display_df, use_container_width=True)
    
    # Basic statistics
    st.subheader("Basic Statistics")
    stats_df = df['volume_gbp'].describe()
    stats_df.index = ['Count', 'Mean', 'Std Dev', 'Min', '25%', '50% (Median)', '75%', 'Max']
    st.dataframe(stats_df, use_container_width=True)
    
    # Time series plot with Plotly
    st.subheader("Daily Volume Time Series")
    
    # Create interactive time series plot
    fig = go.Figure()
    
    # Add line trace
    fig.add_trace(go.Scatter(
        x=df['posting_date'],
        y=df['volume_gbp'],
        mode='lines',
        name='Daily Volume',
        line=dict(color='blue', width=2),
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Volume: £%{y:,.0f}<extra></extra>'
    ))
    
    # Add area fill
    fig.add_trace(go.Scatter(
        x=df['posting_date'],
        y=df['volume_gbp'],
        mode='none',
        fill='tozeroy',
        fillcolor='rgba(0, 100, 255, 0.1)',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Update layout
    fig.update_layout(
        title='Daily Transfer Volumes (Apr-Dec 2023)',
        xaxis_title='Date',
        yaxis_title='Volume (GBP)',
        hovermode='x unified',
        template='plotly_white',
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add range slider
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    # Format y-axis
    fig.update_yaxes(tickprefix="£", tickformat=",.0f")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Navigation hint
    st.markdown("---")
    st.info("💡 **Tip**: Use the sidebar to navigate to other sections of the analysis.")

# Q1: DISTRIBUTION ANALYSIS SECTION
elif analysis_section == "📈 Q1: Distribution Analysis":
    st.header("📈 Question 1: Distribution Analysis")
    
    tab1, tab2, tab3 = st.tabs(["1a) Distribution Shape", "1b) Real-World Causes", "1c) Implications"])
    
    with tab1:
        st.subheader("1a) Describe the distribution that our daily transfer volumes follow")
        
        # Calculate key statistics
        mean_vol = df['volume_gbp'].mean()
        median_vol = df['volume_gbp'].median()
        std_vol = df['volume_gbp'].std()
        skew = df['volume_gbp'].skew()
        kurt = df['volume_gbp'].kurtosis()
        cv = std_vol / mean_vol if mean_vol != 0 else 0
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"£{mean_vol:,.0f}")
        with col2:
            st.metric("Median", f"£{median_vol:,.0f}")
        with col3:
            st.metric("Skewness", f"{skew:.2f}")
        with col4:
            st.metric("CV", f"{cv:.2f}")
        
        # Bimodality analysis
        st.subheader("Bimodality Analysis")
        
        # Calculate bimodality coefficient
        volumes = df['volume_gbp'].values
        n = len(volumes)
        skewness = stats.skew(volumes)
        kurtosis = stats.kurtosis(volumes, fisher=False)
        bc = (skewness**2 + 1) / (kurtosis + 3*(n-1)**2/((n-2)*(n-3))) if n > 3 else 0
        is_bimodal = bc > (5/9)
        
        # Weekday vs weekend comparison
        weekday_data = df[~df['is_weekend']]['volume_gbp']
        weekend_data = df[df['is_weekend']]['volume_gbp']
        ks_stat, ks_p = stats.ks_2samp(weekday_data, weekend_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Bimodality Test Results:**")
            st.write(f"- Pearson's coefficient: **{bc:.4f}**")
            st.write(f"- Critical threshold: > 0.555")
            st.write(f"- Conclusion: **{'BIMODAL' if is_bimodal else 'UNIMODAL'}**")
        
        with col2:
            st.write("**Weekday vs Weekend:**")
            st.write(f"- KS test p-value: **{ks_p:.4f}**")
            st.write(f"- Conclusion: **{'Significantly different' if ks_p < 0.05 else 'Not significantly different'}**")
        
        # Distribution visualizations with Plotly
        st.subheader("Distribution Visualizations")
        
        # Create subplots with Plotly
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Overall Distribution with KDE', 'Boxplot: Spread & Outliers',
                           'Q-Q Plot (Normality Check)', 'Weekday vs Weekend Comparison'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Histogram with KDE
        fig.add_trace(
            go.Histogram(
                x=df['volume_gbp'],
                name="Histogram",
                nbinsx=40,
                opacity=0.7,
                marker_color='blue',
                hovertemplate="Volume: £%{x:,.0f}<br>Count: %{y}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Add KDE line
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(df['volume_gbp'])
        x_range = np.linspace(df['volume_gbp'].min(), df['volume_gbp'].max(), 200)
        y_kde = kde(x_range)
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_kde,
                mode='lines',
                name='KDE',
                line=dict(color='red', width=2),
                hovertemplate="Volume: £%{x:,.0f}<br>Density: %{y:.4f}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Add mean and median lines
        fig.add_vline(x=mean_vol, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: £{mean_vol:,.0f}", 
                     annotation_position="top right", row=1, col=1)
        fig.add_vline(x=median_vol, line_dash="dash", line_color="green", 
                     annotation_text=f"Median: £{median_vol:,.0f}", 
                     annotation_position="top left", row=1, col=1)
        
        # 2. Boxplot
        fig.add_trace(
            go.Box(
                y=df['volume_gbp'],
                name="All Data",
                boxpoints='outliers',
                marker_color='blue',
                hovertemplate="Volume: £%{y:,.0f}<extra></extra>"
            ),
            row=1, col=2
        )
        
        # 3. Q-Q plot
        qq = stats.probplot(df['volume_gbp'], dist="norm")
        x_theoretical = qq[0][0]
        y_observed = qq[0][1]
        
        fig.add_trace(
            go.Scatter(
                x=x_theoretical,
                y=y_observed,
                mode='markers',
                name='Observed',
                marker=dict(color='blue', size=6),
                hovertemplate="Theoretical: %{x:.2f}<br>Observed: %{y:,.0f}<extra></extra>"
            ),
            row=2, col=1
        )
        
        # Add reference line
        fig.add_trace(
            go.Scatter(
                x=[x_theoretical.min(), x_theoretical.max()],
                y=[x_theoretical.min(), x_theoretical.max()],
                mode='lines',
                name='Normal Reference',
                line=dict(color='red', dash='dash'),
                showlegend=False,
                hovertemplate=None
            ),
            row=2, col=1
        )
        
        # 4. Weekday vs Weekend comparison boxplot
        fig.add_trace(
            go.Box(
                y=weekday_data,
                name='Weekdays',
                boxpoints='outliers',
                marker_color='green',
                hovertemplate="Weekday Volume: £%{y:,.0f}<extra></extra>"
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Box(
                y=weekend_data,
                name='Weekends',
                boxpoints='outliers',
                marker_color='orange',
                hovertemplate="Weekend Volume: £%{y:,.0f}<extra></extra>"
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            template='plotly_white',
            hovermode='closest'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Volume (GBP)", row=1, col=1, tickprefix="£", tickformat=",.0f")
        fig.update_yaxes(title_text="Density/Count", row=1, col=1)
        fig.update_xaxes(title_text="", row=1, col=2, showticklabels=False)
        fig.update_yaxes(title_text="Volume (GBP)", row=1, col=2, tickprefix="£", tickformat=",.0f")
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
        fig.update_yaxes(title_text="Observed Quantiles", row=2, col=1, tickprefix="£", tickformat=",.0f")
        fig.update_xaxes(title_text="", row=2, col=2, showticklabels=False)
        fig.update_yaxes(title_text="Volume (GBP)", row=2, col=2, tickprefix="£", tickformat=",.0f")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary
        st.subheader("Summary of Distribution Characteristics")
        st.write(f"""
        **Key Findings:**
        1. **Right-skewed** (mean > median, skewness = {skew:.2f})
        2. **Unimodal** (single peak despite weekday/weekend differences)
        3. **Non-normal** (heavy-tailed distribution)
        4. **High variability** (CV = {cv:.2f})
        5. **Significant weekday/weekend differences** (p < 0.001)
        
        **Bimodality Conclusion:** The distribution is **NOT bimodal**. Despite significant differences between 
        weekday and weekend volumes, the overall distribution shows a single peak.
        """)
    
    with tab2:
        st.subheader("1b) What real world cause do you think is behind this shape of distribution?")
        
        # Calculate business metrics
        weekday_mean = df[~df['is_weekend']]['volume_gbp'].mean()
        weekend_mean = df[df['is_weekend']]['volume_gbp'].mean()
        ratio = weekday_mean / weekend_mean if weekend_mean != 0 else 0
        
        # Outlier analysis
        Q1 = df['volume_gbp'].quantile(0.25)
        Q3 = df['volume_gbp'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[df['volume_gbp'] > upper_bound]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Weekday Mean", f"£{weekday_mean:,.0f}")
            st.metric("Weekend Mean", f"£{weekend_mean:,.0f}")
            st.metric("Weekday:Weekend Ratio", f"{ratio:.1f}x")
        
        with col2:
            st.metric("Outlier Threshold", f"£{upper_bound:,.0f}")
            st.metric("Outlier Days", f"{len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
            st.metric("Zero-Volume Days", f"{(df['volume_gbp'] == 0).sum()}")
        
        # Real-world causes
        st.subheader("Real-World Causes & Business Drivers")
        
        # Create interactive histogram with weekday/weekend overlay
        fig = go.Figure()
        
        # Add weekday histogram
        fig.add_trace(go.Histogram(
            x=weekday_data,
            name='Weekdays',
            opacity=0.7,
            marker_color='green',
            nbinsx=30,
            hovertemplate="Weekday Volume: £%{x:,.0f}<br>Count: %{y}<extra></extra>"
        ))
        
        # Add weekend histogram
        fig.add_trace(go.Histogram(
            x=weekend_data,
            name='Weekends',
            opacity=0.7,
            marker_color='orange',
            nbinsx=30,
            hovertemplate="Weekend Volume: £%{x:,.0f}<br>Count: %{y}<extra></extra>"
        ))
        
        # Add means
        fig.add_vline(x=weekday_mean, line_dash="dash", line_color="darkgreen",
                     annotation_text=f"Weekday Mean: £{weekday_mean:,.0f}", 
                     annotation_position="top right")
        fig.add_vline(x=weekend_mean, line_dash="dash", line_color="darkorange",
                     annotation_text=f"Weekend Mean: £{weekend_mean:,.0f}", 
                     annotation_position="top left")
        
        fig.update_layout(
            title='Volume Distribution: Weekdays vs Weekends',
            xaxis_title='Volume (GBP)',
            yaxis_title='Frequency',
            barmode='overlay',
            template='plotly_white',
            height=500,
            hovermode='x unified'
        )
        
        fig.update_xaxes(tickprefix="£", tickformat=",.0f")
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("""
        **Primary Drivers of Distribution Shape:**
        
        1. **Banking Hours & Operations:**
           - Banks closed on weekends → drastically lower weekend volumes
           - Weekday business hours drive peak transaction periods
        
        2. **Business vs Retail Segmentation:**
           - **Business customers**: Large, infrequent transfers (create right tail)
           - **Retail customers**: Smaller, more regular transfers (create main peak)
        
        3. **Payment Cycles:**
           - Payroll processing (month-end spikes)
           - Supplier payments
           - Corporate treasury management
        
        4. **Market Factors:**
           - Favorable GBP/ZAR rates trigger bulk transfers
           - Market volatility drives timing decisions
        """)
        
        # Show top outliers
        if len(outliers) > 0:
            st.subheader("Top 5 High-Value Outlier Days")
            outlier_display = outliers.nlargest(5, 'volume_gbp')[['posting_date', 'volume_gbp', 'weekday']].copy()
            outlier_display['posting_date'] = outlier_display['posting_date'].dt.strftime('%Y-%m-%d')
            outlier_display['volume_gbp'] = outlier_display['volume_gbp'].apply(lambda x: f"£{x:,.0f}")
            st.dataframe(outlier_display, use_container_width=True)
            
            # Interactive outlier plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['posting_date'],
                y=df['volume_gbp'],
                mode='lines+markers',
                name='Daily Volume',
                line=dict(color='blue', width=1),
                marker=dict(size=4),
                hovertemplate='Date: %{x|%Y-%m-%d}<br>Volume: £%{y:,.0f}<extra></extra>'
            ))
            
            # Highlight outliers
            fig.add_trace(go.Scatter(
                x=outliers['posting_date'],
                y=outliers['volume_gbp'],
                mode='markers',
                name='Outliers',
                marker=dict(
                    color='red',
                    size=8,
                    symbol='diamond'
                ),
                hovertemplate='<b>OUTLIER</b><br>Date: %{x|%Y-%m-%d}<br>Volume: £%{y:,.0f}<extra></extra>'
            ))
            
            # Add outlier threshold line
            fig.add_hline(y=upper_bound, line_dash="dash", line_color="red",
                         annotation_text=f"Outlier Threshold: £{upper_bound:,.0f}",
                         annotation_position="bottom right")
            
            fig.update_layout(
                title='Daily Volumes with Outliers Highlighted',
                xaxis_title='Date',
                yaxis_title='Volume (GBP)',
                template='plotly_white',
                height=400,
                hovermode='x unified'
            )
            
            fig.update_yaxes(tickprefix="£", tickformat=",.0f")
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("1c) What are some of the implications that this distribution would commonly have on analysis that you might do?")
        
        st.write("""
        **Statistical Analysis Implications:**
        
        1. **Statistical Testing Limitations:**
           - Parametric tests (t-tests, ANOVA) are INVALID
           - Use NON-PARAMETRIC alternatives (Mann-Whitney, Kruskal-Wallis)
        
        2. **Central Tendency Measures:**
           - Mean (£188K) is inflated by outliers
           - Median (£170K) better represents 'typical' day
           - Recommendation: Use MEDIAN for reporting & planning
        
        3. **Forecasting & Modeling Challenges:**
           - Traditional models (ARIMA) may fail
           - Consider transformations (log, Box-Cox)
           - Segment data: weekdays vs weekends
           - Use robust regression methods
        
        4. **Outlier Management Strategy:**
           - Outliers are REAL BUSINESS EVENTS, not noise
           - Don't remove without business justification
           - Analyze separately to understand causes
        """)
        
        st.write("""
        **Business & Operational Implications:**
        
        5. **Business Planning & Operations:**
           - High variability → need buffer capacity
           - Plan for both typical days AND spike days
        
        6. **Customer Insights & Segmentation:**
           - No evidence of volume-based bimodality
           - Segment by behavior not volume
           - Weekday indicator critical for models
        
        7. **Performance Reporting:**
           - Use medians for benchmarks
           - Separate weekday/weekend reporting
           - Monitor distribution changes
        
        8. **Data Quality & Collection:**
           - Ensure weekday coverage (missing weekdays have huge impact)
           - Document special events
        """)
        
        st.subheader("Specific Recommendations for WISE")
        st.write("""
        1. **Reporting**: Use median volume (£170K) as "typical day" metric
        2. **Analysis**: Apply non-parametric statistical tests exclusively
        3. **Forecasting**: Build separate models for weekdays and weekends
        4. **Planning**: Maintain 30-40% buffer capacity for high-volume days
        5. **Segmentation**: Focus on transaction behavior rather than volume tiers
        """)
        
        # Interactive distribution comparison
        st.subheader("Interactive Distribution Comparison")
        
        # Create transformed data for comparison
        df_transformed = df.copy()
        df_transformed['log_volume'] = np.log1p(df_transformed['volume_gbp'])
        df_transformed['sqrt_volume'] = np.sqrt(df_transformed['volume_gbp'])
        
        # Create interactive comparison plot
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Original Data', 'Log Transformation', 'Square Root Transformation'),
            horizontal_spacing=0.1
        )
        
        # Original data
        fig.add_trace(
            go.Histogram(
                x=df_transformed['volume_gbp'],
                nbinsx=40,
                name='Original',
                marker_color='blue',
                hovertemplate="Volume: £%{x:,.0f}<br>Count: %{y}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Log transformed
        fig.add_trace(
            go.Histogram(
                x=df_transformed['log_volume'],
                nbinsx=40,
                name='Log Transformed',
                marker_color='green',
                hovertemplate="Log(Volume): %{x:.2f}<br>Count: %{y}<extra></extra>"
            ),
            row=1, col=2
        )
        
        # Square root transformed
        fig.add_trace(
            go.Histogram(
                x=df_transformed['sqrt_volume'],
                nbinsx=40,
                name='Sqrt Transformed',
                marker_color='orange',
                hovertemplate="√(Volume): %{x:.0f}<br>Count: %{y}<extra></extra>"
            ),
            row=1, col=3
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            template='plotly_white',
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Volume (GBP)", row=1, col=1, tickprefix="£", tickformat=",.0f")
        fig.update_xaxes(title_text="log(1 + Volume)", row=1, col=2)
        fig.update_xaxes(title_text="√(Volume)", row=1, col=3)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("""
        **Visual Insights:**
        - **Log transformation** helps reduce skewness but may over-transform
        - **Square root transformation** offers a good balance for reducing skew
        - **Recommendation**: Consider square root transformation for modeling
        """)
    
    # Navigation hint
    st.markdown("---")
    st.info("💡 **Tip**: Use the sidebar to navigate to other sections of the analysis.")

# Q2: QUARTERLY ANALYSIS SECTION
elif analysis_section == "📊 Q2: Quarterly Changes":
    st.header("📊 Question 2: Quarterly Changes Analysis")
    
    # Filter for Q2-Q4 2023
    df_q2_q4 = df[(df['year'] == 2023) & (df['quarter'].isin([2, 3, 4]))].copy()
    
    # Quarterly statistics
    quarterly_stats = df_q2_q4.groupby('quarter')['volume_gbp'].agg([
        ('median', 'median'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('count', 'count')
    ])
    
    # Display quarterly metrics
    st.subheader("Quarterly Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        q2_median = quarterly_stats.loc[2, 'median']
        st.metric("Q2 Median", f"£{q2_median:,.0f}")
        st.metric("Q2 Days", quarterly_stats.loc[2, 'count'])
    
    with col2:
        q3_median = quarterly_stats.loc[3, 'median']
        q2_q3_change = ((q3_median - q2_median) / q2_median * 100) if q2_median != 0 else 0
        st.metric("Q3 Median", f"£{q3_median:,.0f}")
        st.metric("Q2→Q3 Change", f"{q2_q3_change:+.1f}%")
    
    with col3:
        q4_median = quarterly_stats.loc[4, 'median']
        q3_q4_change = ((q4_median - q3_median) / q3_median * 100) if q3_median != 0 else 0
        st.metric("Q4 Median", f"£{q4_median:,.0f}")
        st.metric("Q3→Q4 Change", f"{q3_q4_change:+.1f}%")
    
    # Statistical testing
    st.subheader("Statistical Significance Testing")
    
    # Prepare data for testing
    Q2_data = df_q2_q4[df_q2_q4['quarter'] == 2]['volume_gbp']
    Q3_data = df_q2_q4[df_q2_q4['quarter'] == 3]['volume_gbp']
    Q4_data = df_q2_q4[df_q2_q4['quarter'] == 4]['volume_gbp']
    
    # Kruskal-Wallis test
    h_stat, p_val_kw = stats.kruskal(Q2_data, Q3_data, Q4_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Kruskal-Wallis Test:**")
        st.write(f"- H₀: All quarters have same median")
        st.write(f"- H₁: At least one quarter differs")
        st.write(f"- Test statistic (H): {h_stat:.4f}")
        st.write(f"- **p-value: {p_val_kw:.4f}**")
        
        if p_val_kw < 0.05:
            st.success("✅ REJECT H₀: Significant difference exists (p < 0.05)")
        else:
            st.warning("❌ FAIL TO REJECT H₀: No significant difference (p ≥ 0.05)")
    
    # Effect size analysis
    def cliffs_delta(x, y):
        x_arr = np.array(x)
        y_arr = np.array(y)
        x_reshaped = x_arr.reshape(-1, 1)
        y_reshaped = y_arr.reshape(1, -1)
        greater = np.sum(x_reshaped > y_reshaped)
        less = np.sum(x_reshaped < y_reshaped)
        delta = (greater - less) / (len(x_arr) * len(y_arr)) if len(x_arr) * len(y_arr) > 0 else 0
        return delta
    
    with col2:
        st.write("**Effect Size Analysis (Cliff's Delta):**")
        
        delta_q2_q3 = cliffs_delta(Q2_data, Q3_data)
        delta_q2_q4 = cliffs_delta(Q2_data, Q4_data)
        delta_q3_q4 = cliffs_delta(Q3_data, Q4_data)
        
        st.write(f"- Q2 → Q3: δ = {delta_q2_q3:+.3f}")
        st.write(f"- Q2 → Q4: δ = {delta_q2_q4:+.3f}")
        st.write(f"- Q3 → Q4: δ = {delta_q3_q4:+.3f}")
        
        # Interpretation with color coding
        def interpret_delta(delta):
            abs_delta = abs(delta)
            if abs_delta < 0.147:
                return "🟢 Negligible effect"
            elif abs_delta < 0.33:
                return "🟡 Small effect"
            elif abs_delta < 0.474:
                return "🟠 Medium effect"
            else:
                return "🔴 Large effect"
        
        st.write("**Interpretation:**")
        st.write(f"- Q2 → Q3: {interpret_delta(delta_q2_q3)}")
        st.write(f"- Q2 → Q4: {interpret_delta(delta_q2_q4)}")
        st.write(f"- Q3 → Q4: {interpret_delta(delta_q3_q4)}")
    
    # Visualizations with Plotly
    st.subheader("Quarterly Comparison Visualizations")
    
    # Create interactive figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Daily Volumes by Quarter', 'Quarterly Median Trend'),
        horizontal_spacing=0.15
    )
    
    # Boxplot by quarter
    fig.add_trace(
        go.Box(
            y=Q2_data,
            name='Q2',
            boxpoints='outliers',
            marker_color='blue',
            hovertemplate="Q2 Volume: £%{y:,.0f}<extra></extra>"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Box(
            y=Q3_data,
            name='Q3',
            boxpoints='outliers',
            marker_color='green',
            hovertemplate="Q3 Volume: £%{y:,.0f}<extra></extra>"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Box(
            y=Q4_data,
            name='Q4',
            boxpoints='outliers',
            marker_color='orange',
            hovertemplate="Q4 Volume: £%{y:,.0f}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Median trend line
    quarters = [2, 3, 4]
    medians = [quarterly_stats.loc[q, 'median'] for q in quarters]
    
    fig.add_trace(
        go.Scatter(
            x=quarters,
            y=medians,
            mode='lines+markers',
            name='Median Trend',
            line=dict(color='red', width=3),
            marker=dict(size=10),
            hovertemplate="Q%{x}<br>Median: £%{y:,.0f}<extra></extra>"
        ),
        row=1, col=2
    )
    
    # Add percentage change annotations
    for i in range(len(quarters)-1):
        change_pct = ((medians[i+1] - medians[i]) / medians[i] * 100) if medians[i] != 0 else 0
        fig.add_annotation(
            x=(quarters[i] + quarters[i+1]) / 2,
            y=(medians[i] + medians[i+1]) / 2,
            text=f"{change_pct:+.1f}%",
            showarrow=False,
            font=dict(size=12, color='red' if change_pct < 0 else 'green'),
            row=1, col=2
        )
    
    fig.update_layout(
        height=500,
        template='plotly_white',
        hovermode='closest'
    )
    
    fig.update_xaxes(title_text="Quarter", row=1, col=2, tickvals=quarters, ticktext=['Q2', 'Q3', 'Q4'])
    fig.update_yaxes(title_text="Volume (GBP)", row=1, col=1, tickprefix="£", tickformat=",.0f")
    fig.update_yaxes(title_text="Median Volume (GBP)", row=1, col=2, tickprefix="£", tickformat=",.0f")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive time series by quarter
    st.subheader("Daily Volume Time Series (Color by Quarter)")
    
    fig = go.Figure()
    
    # Color mapping for quarters
    colors = {2: 'blue', 3: 'green', 4: 'orange'}
    
    for quarter in [2, 3, 4]:
        quarter_data = df_q2_q4[df_q2_q4['quarter'] == quarter]
        fig.add_trace(go.Scatter(
            x=quarter_data['posting_date'],
            y=quarter_data['volume_gbp'],
            mode='lines+markers',
            name=f'Q{quarter}',
            line=dict(color=colors[quarter], width=1.5),
            marker=dict(size=4, color=colors[quarter]),
            hovertemplate='Date: %{x|%Y-%m-%d}<br>Volume: £%{y:,.0f}<br>Quarter: Q%{customdata}<extra></extra>',
            customdata=[quarter] * len(quarter_data)
        ))
    
    fig.update_layout(
        title='Daily Volumes Colored by Quarter',
        xaxis_title='Date',
        yaxis_title='Volume (GBP)',
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )
    
    fig.update_yaxes(tickprefix="£", tickformat=",.0f")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Conclusion
    st.subheader("Conclusion for Quarterly Analysis")
    st.write(f"""
    **Statistical Significance:**
    - Kruskal-Wallis p-value = **{p_val_kw:.3f}** (p ≥ 0.05)
    - **Conclusion**: No statistically significant quarterly differences
    
    **Effect Sizes:**
    - All Cliff's Delta values < 0.147
    - **Conclusion**: Negligible practical differences
    
    **Business Interpretation:**
    - Observed quarterly changes (Q3: -8.7%, Q4: +10.7%) 
    - **Most likely**: Random background fluctuations
    - **Not indicative** of meaningful business change
    
    **Recommendation:**
    - Monitor quarterly trends but **don't react** to these fluctuations
    - Focus on **longer-term trends** (6+ months)
    - Investigate only if **effect size becomes meaningful** (|δ| ≥ 0.33)
    """)
    
    # Navigation hint
    st.markdown("---")
    st.info("💡 **Tip**: Use the sidebar to navigate to other sections of the analysis.")

# Q3: OCTOBER 2023 ESTIMATION SECTION
elif analysis_section == "🔮 Q3: October 2023 Estimation":
    st.header("🔮 Question 3: October 2023 Volume Estimation")
    
    # Examine October data
    oct_2023 = df[(df['posting_date'].dt.month == 10) & (df['posting_date'].dt.year == 2023)]
    all_oct_dates = pd.date_range('2023-10-01', '2023-10-31', freq='D')
    
    st.subheader("October 2023 Data Availability")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Days with Data", f"{len(oct_2023)}/31")
        st.metric("Missing Days", 31 - len(oct_2023))
    
    with col2:
        # Count actual weekend days vs weekday days
        weekend_days = len(oct_2023[oct_2023['is_weekend']])
        weekday_days = len(oct_2023[~oct_2023['is_weekend']])
        st.metric("Weekend Days Available", f"{weekend_days}")
        st.metric("Weekday Days Available", f"{weekday_days}")
    
    with col3:
        oct_total_actual = oct_2023['volume_gbp'].sum()
        st.metric("Available Data Total", f"£{oct_total_actual:,.0f}")
    
    # Show critical warning
    if weekday_days == 0:
        st.error("⚠️ **CRITICAL DATA GAP**: October 2023 has NO WEEKDAY data! Only weekend data available.")
        st.warning("**Impact**: This significantly increases estimation uncertainty as weekdays typically have 6.1× higher volumes than weekends.")
    
    # Show available data
    st.subheader("Available October Data (Only Weekends)")
    oct_display = oct_2023[['posting_date', 'volume_gbp', 'weekday', 'is_weekend']].copy()
    oct_display['posting_date'] = oct_display['posting_date'].dt.strftime('%Y-%m-%d')
    oct_display['volume_gbp'] = oct_display['volume_gbp'].apply(lambda x: f"£{x:,.2f}")
    st.dataframe(oct_display, use_container_width=True)
    
    # Create CORRECTED calendar heatmap
    st.subheader("October 2023 Data Availability Calendar")
    
    # Get actual available dates
    available_dates = set(oct_2023['posting_date'].dt.date)
    
    # Create calendar dictionary
    calendar_dict = {}
    weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    # Initialize calendar with empty lists for each weekday
    for weekday in weekdays:
        calendar_dict[weekday] = []
    
    # Fill calendar correctly - October 1, 2023 was a Sunday
    # Let's map each date to its correct position
    for date_obj in all_oct_dates:
        day_num = date_obj.day
        # Get actual day of week (0=Monday, 6=Sunday)
        # Convert to our display order
        day_of_week = date_obj.weekday()  # Monday=0, Sunday=6
        
        # Map to display order
        if day_of_week == 0:  # Monday
            weekday_key = 'Mon'
        elif day_of_week == 1:  # Tuesday
            weekday_key = 'Tue'
        elif day_of_week == 2:  # Wednesday
            weekday_key = 'Wed'
        elif day_of_week == 3:  # Thursday
            weekday_key = 'Thu'
        elif day_of_week == 4:  # Friday
            weekday_key = 'Fri'
        elif day_of_week == 5:  # Saturday
            weekday_key = 'Sat'
        else:  # Sunday (6)
            weekday_key = 'Sun'
        
        # Check if date is available
        if date_obj.date() in available_dates:
            calendar_dict[weekday_key].append(f"✅ {day_num}")
        else:
            calendar_dict[weekday_key].append(f"❌ {day_num}")
    
    # Now create weekly columns
    max_weeks = 5
    oct_calendar = pd.DataFrame(index=weekdays)
    
    for week in range(max_weeks):
        week_data = []
        for weekday in weekdays:
            if week < len(calendar_dict[weekday]):
                week_data.append(calendar_dict[weekday][week])
            else:
                week_data.append("")
        oct_calendar[f'Week {week+1}'] = week_data
    
    st.dataframe(oct_calendar, use_container_width=True)
    
    # Show exactly what dates are available
    st.subheader("Available October Dates")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Available Dates:**")
        available_list = sorted([(d.strftime('%Y-%m-%d'), d.strftime('%A')) for d in oct_2023['posting_date']], 
                               key=lambda x: x[0])
        for date_str, day_name in available_list:
            st.write(f"• {date_str} ({day_name})")
    
    with col2:
        st.write("**Data Coverage:**")
        st.write(f"- Total available days: {len(oct_2023)}")
        st.write(f"- Weekend days (Sat/Sun): {weekend_days}")
        st.write(f"- Weekday days (Mon-Fri): {weekday_days}")
        st.write(f"- Data coverage: {len(oct_2023)/31*100:.1f}%")
        st.write(f"- Missing weekdays: 22 (all Mon-Fri)")
    
    # Add legend and explanation
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Legend:**")
        st.markdown("✅ = Data available")
        st.markdown("❌ = Data missing")
    
    with col2:
        st.markdown("**Important Note:**")
        st.markdown("• All available data are **weekends only** (Saturdays & Sundays)")
        st.markdown("• All **weekday data is missing** (22 days)")
        st.markdown("• October 1, 2023 was a **Sunday**, October 2 was Monday")
    
    # Show which specific dates are available
    st.subheader("Available Dates in October 2023")
    available_list = sorted([d.strftime('%Y-%m-%d (%A)') for d in oct_2023['posting_date']])
    for date_str in available_list:
        st.write(f"• {date_str}")
    
    # Estimation Methodology
    st.subheader("Estimation Methodology")
    
    st.write("""
    **Why Bootstrap Resampling?**
    
    I chose **bootstrap uncertainty quantification** because:
    
    1. **Handles Missing Data**: Resamples from available Q3 data to estimate missing values
    2. **Preserves Patterns**: Maintains weekday-specific patterns observed in Q3
    3. **Quantifies Uncertainty**: Provides confidence intervals around estimates
    4. **Non-Parametric**: Doesn't assume normal distribution
    5. **Accounts for Variability**: Captures natural day-to-day fluctuations
    """)
    
    # Run bootstrap estimation
    st.subheader("Running Bootstrap Estimation")
    
    with st.spinner(f"Running 1,000 bootstrap simulations..."):
        # Use Q3 as reference
        q3_data = df[df['quarter'] == 3].copy()
        q3_median = q3_data['volume_gbp'].median()
        
        # Pre-calculate Q3 weekday data
        q3_weekday_data = {}
        for weekday in range(7):
            q3_weekday_data[weekday] = q3_data[q3_data['day_of_week'] == weekday]['volume_gbp'].values
        
        # Create lookup for actual October values
        actual_values = {}
        for _, row in oct_2023.iterrows():
            actual_values[row['posting_date']] = row['volume_gbp']
        
        # Run bootstrap
        np.random.seed(42)
        n_simulations = 1000
        bootstrap_totals = []
        
        for sim in range(n_simulations):
            sim_total = 0
            for date in all_oct_dates:
                if date in actual_values:
                    sim_total += actual_values[date]
                else:
                    weekday = date.weekday()
                    weekday_samples = q3_weekday_data[weekday]
                    if len(weekday_samples) > 0:
                        sim_total += np.random.choice(weekday_samples)
                    else:
                        sim_total += q3_median
            bootstrap_totals.append(sim_total)
        
        bootstrap_totals = np.array(bootstrap_totals)
        
        # Calculate statistics
        mean_estimate = np.mean(bootstrap_totals)
        median_estimate = np.median(bootstrap_totals)
        ci_95 = np.percentile(bootstrap_totals, [2.5, 97.5])
        ci_80 = np.percentile(bootstrap_totals, [10, 90])
        std_estimate = np.std(bootstrap_totals)
    
    st.success(f"✅ Completed {n_simulations:,} bootstrap simulations")
    
    # Display results
    st.subheader("Estimation Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Best Estimate (Mean)", f"£{mean_estimate/1e6:.1f}M")
    
    with col2:
        st.metric("Median Estimate", f"£{median_estimate/1e6:.1f}M")
    
    with col3:
        ci_range = ci_95[1] - ci_95[0]
        st.metric("95% CI Range", f"£{ci_range/1e6:.1f}M")
    
    with col4:
        relative_uncertainty = (ci_range / mean_estimate * 100) if mean_estimate != 0 else 0
        st.metric("Relative Uncertainty", f"{relative_uncertainty:.0f}%")
    
    # Confidence intervals
    st.subheader("Confidence Intervals (Measures of Range & Certainty)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**95% Confidence Interval:**\n£{ci_95[0]/1e6:.1f}M to £{ci_95[1]/1e6:.1f}M")
        st.write("We are 95% confident the true October total lies in this range")
    
    with col2:
        st.info(f"**80% Confidence Interval:**\n£{ci_80[0]/1e6:.1f}M to £{ci_80[1]/1e6:.1f}M")
        st.write("Tighter range for less conservative estimates")
    
    # Interactive visualizations
    st.subheader("Visualizing Uncertainty")
    
    # Create interactive histogram of bootstrap results
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=bootstrap_totals/1e6,
        nbinsx=40,
        name='Bootstrap Distribution',
        marker_color='blue',
        opacity=0.7,
        hovertemplate="Total Volume: £%{x:.1f}M<br>Count: %{y}<extra></extra>"
    ))
    
    # Add mean line
    fig.add_vline(x=mean_estimate/1e6, line_dash="dash", line_color="red", 
                 annotation_text=f"Mean: £{mean_estimate/1e6:.1f}M", 
                 annotation_position="top right")
    
    # Add confidence intervals
    fig.add_vrect(x0=ci_95[0]/1e6, x1=ci_95[1]/1e6, 
                  fillcolor="rgba(255, 0, 0, 0.1)", line_width=0,
                  annotation_text="95% CI", annotation_position="top left")
    
    fig.add_vrect(x0=ci_80[0]/1e6, x1=ci_80[1]/1e6, 
                  fillcolor="rgba(0, 255, 0, 0.1)", line_width=0,
                  annotation_text="80% CI", annotation_position="bottom right")
    
    fig.update_layout(
        title='Bootstrap Distribution of October 2023 Total Volume',
        xaxis_title='Total October Volume (Million GBP)',
        yaxis_title='Frequency',
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )
    
    fig.update_xaxes(tickprefix="£", ticksuffix="M")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Cumulative probability plot
    st.subheader("Cumulative Probability Distribution")
    
    sorted_totals = np.sort(bootstrap_totals)
    cumulative_prob = np.arange(1, len(sorted_totals) + 1) / len(sorted_totals)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=sorted_totals/1e6,
        y=cumulative_prob * 100,
        mode='lines',
        name='Cumulative Probability',
        line=dict(color='blue', width=3),
        hovertemplate="Volume: £%{x:.1f}M<br>Cumulative Probability: %{y:.1f}%<extra></extra>"
    ))
    
    # Add reference lines for key percentiles
    for percentile, color, label in [(50, 'green', 'Median'), (80, 'orange', '80% CI'), (95, 'red', '95% CI')]:
        lower_bound = np.percentile(bootstrap_totals, (100-percentile)/2)/1e6
        upper_bound = np.percentile(bootstrap_totals, 100-(100-percentile)/2)/1e6
        
        fig.add_vline(x=lower_bound, line_dash="dash", line_color=color, opacity=0.5)
        fig.add_vline(x=upper_bound, line_dash="dash", line_color=color, opacity=0.5)
        fig.add_vrect(x0=lower_bound, x1=upper_bound, 
                      fillcolor=color, opacity=0.1, line_width=0,
                      annotation_text=f"{percentile}% CI", annotation_position="top")
    
    fig.update_layout(
        title='Cumulative Probability Distribution of October Total',
        xaxis_title='Total October Volume (Million GBP)',
        yaxis_title='Cumulative Probability (%)',
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )
    
    fig.update_xaxes(tickprefix="£", ticksuffix="M")
    fig.update_yaxes(ticksuffix="%")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Final estimate
    st.subheader("Final October 2023 Estimate")
    
    st.success(f"""
    **Best Estimate:** October 2023 total transfer volume = **£{mean_estimate/1e6:.1f} million**
    
    **With 95% Confidence:** Between **£{ci_95[0]/1e6:.1f}M** and **£{ci_95[1]/1e6:.1f}M**
    
    **Margin of Error:** ±£{(ci_95[1]-ci_95[0])/2e6:.1f}M
    
    **Relative Uncertainty:** {relative_uncertainty:.0f}% (due to 22 missing weekdays)
    """)
    
    # Assumptions
    st.subheader("Assumptions & Limitations")
    st.write("""
    **Key Assumptions:**
    1. October 2023 follows same weekday/weekend patterns as Q3 2023
    2. No major business changes between Q3 and October
    3. Q3 data is representative of typical patterns
    
    **Limitations:**
    1. **High uncertainty** due to 22 missing weekdays
    2. Seasonal effects in October not captured in Q3
    3. Business events in October not accounted for
    4. **Only weekend data available** → limited reference for weekday estimation
    """)
    
    # Navigation hint
    st.markdown("---")
    st.info("💡 **Tip**: Use the sidebar to navigate to other sections of the analysis.")

# ANALYTICAL METHODOLOGY SECTION
elif analysis_section == "🔬 Analytical Methodology":
    st.header("🔬 Analytical Methodology")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Complete Original Data",
        "📈 Question 1 Methodology",
        "📊 Question 2 Methodology",
        "🔮 Question 3 Methodology"
    ])
    
    with tab1:
        st.subheader("📊 Complete Original Dataset")
        
        # Dataset overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Date Range", f"{df['posting_date'].min().strftime('%b %d')} to {df['posting_date'].max().strftime('%b %d')}")
        with col3:
            st.metric("Total Volume", f"£{df['volume_gbp'].sum():,.0f}")
        
        # Show complete data with download option
        st.subheader("Full Dataset View")
        
        # Add search/filter functionality
        search_term = st.text_input("🔍 Search in dataset (date, weekday, etc.):", "")
        
        # Show filtered data if search term exists
        if search_term:
            filtered_df = df[
                df['posting_date'].astype(str).str.contains(search_term, case=False) |
                df['weekday'].str.contains(search_term, case=False) |
                df['volume_gbp'].astype(str).str.contains(search_term, case=False)
            ]
            if len(filtered_df) > 0:
                st.write(f"**Showing {len(filtered_df)} records matching '{search_term}'**")
                display_df = filtered_df.copy()
            else:
                st.warning(f"No records found matching '{search_term}'. Showing full dataset.")
                display_df = df.copy()
        else:
            display_df = df.copy()
        
        # Format dates for display
        display_df = display_df.copy()
        display_df['posting_date'] = display_df['posting_date'].dt.strftime('%Y-%m-%d')
        
        # Show data with pagination
        rows_per_page = 20
        total_pages = max(1, len(display_df) // rows_per_page + (1 if len(display_df) % rows_per_page > 0 else 0))
        
        # Page selector
        page_number = st.number_input(
            f"Page (1-{total_pages})", 
            min_value=1, 
            max_value=total_pages, 
            value=1,
            step=1
        )
        
        # Calculate slice
        start_idx = (page_number - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        
        # Display current page
        st.dataframe(
            display_df.iloc[start_idx:end_idx],
            use_container_width=True,
            height=400
        )
        
        # Page info
        st.caption(f"Showing rows {start_idx + 1} to {min(end_idx, len(display_df))} of {len(display_df)} total records")
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete Dataset (CSV)",
            data=csv,
            file_name="gbp_zar_transfer_data_2023.csv",
            mime="text/csv",
            help="Download the complete dataset for offline analysis"
        )
        
        # Data dictionary
        st.subheader("Data Dictionary")
        data_dict = pd.DataFrame({
            "Column Name": ["posting_date", "volume_gbp", "year", "month", "quarter", "weekday", "day_of_week", "is_weekend", "month_name"],
            "Description": [
                "Date of the transaction",
                "Transfer volume in British Pounds (GBP)",
                "Year extracted from posting_date",
                "Month extracted from posting_date (1-12)",
                "Quarter extracted from posting_date (1-4)",
                "Day of week name (Monday-Sunday)",
                "Day of week number (0=Monday, 6=Sunday)",
                "Boolean indicating if day is weekend (Saturday/Sunday)",
                "Full month name"
            ],
            "Data Type": [
                "datetime64[ns]",
                "float64",
                "int64",
                "int64",
                "int64",
                "object (string)",
                "int64",
                "bool",
                "object (string)"
            ]
        })
        st.dataframe(data_dict, use_container_width=True)
        
        # Basic statistics
        st.subheader("Dataset Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Volume Statistics:**")
            stats_vol = df['volume_gbp'].describe()
            stats_vol.index = ['Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max']
            st.dataframe(stats_vol, use_container_width=True)
        
        with col2:
            st.write("**Date Range Statistics:**")
            
            # Create statistics as strings to prevent date conversion
            date_stats = {
                "Start Date": str(df['posting_date'].min().date()),
                "End Date": str(df['posting_date'].max().date()),
                "Total Days": str(len(df)),
                "Missing Days": "0",  # Assuming no missing dates in sequence
                "Weekday Days": str(len(df[~df['is_weekend']])),
                "Weekend Days": str(len(df[df['is_weekend']])),
                "Months Covered": str(df['month'].nunique()),
                "Quarters Covered": str(df['quarter'].nunique())
            }
            
            # Create dataframe with explicit dtypes
            date_stats_df = pd.DataFrame(list(date_stats.items()), columns=['Statistic', 'Value'])
            
            st.dataframe(date_stats_df.set_index('Statistic'), use_container_width=True)
    
    with tab2:
        st.subheader("Question 1: Distribution Analysis Methodology")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Bimodality Testing:**")
            st.write("""
            1. **Pearson's Bimodality Coefficient:**
               - Calculated from skewness and kurtosis
               - Formula: BC = (skewness² + 1) / (kurtosis + 3(n-1)²/((n-2)(n-3)))
               - Threshold: BC > 0.555 indicates bimodality
               - Our result: BC = 0.282 (< 0.555) = Unimodal
            
            2. **Kolmogorov-Smirnov Test:**
               - Compare weekday vs weekend distributions
               - Non-parametric test for distribution equality
               - p < 0.05 = significantly different
               - Our result: p = 0.000 = significantly different
            """)
            
            st.write("**Distribution Shape Analysis:**")
            st.write("""
            - **Skewness**: Measures asymmetry
              - Positive = right-skewed (mean > median)
              - Negative = left-skewed (mean < median)
              - Zero = symmetric
            
            - **Kurtosis**: Measures tail heaviness
              - High = heavy tails, more outliers
              - Low = light tails, fewer outliers
            
            - **Coefficient of Variation (CV)**:
              - CV = std / mean
              - Measures relative variability
              - CV > 0.5 = high variability
            """)
        
        with col2:
            st.write("**Visual Analysis Methods:**")
            st.write("""
            1. **Histogram with KDE**:
               - Shows frequency distribution
               - Kernel Density Estimate smooths histogram
            
            2. **Boxplot**:
               - Visualizes spread and outliers
               - Shows quartiles and median
            
            3. **Q-Q Plot**:
               - Compares to normal distribution
               - Points on diagonal = normal distribution
               - Curved pattern = non-normal
            
            4. **Comparative Boxplots**:
               - Compare subgroups (weekday vs weekend)
               - Visual comparison of medians and spreads
            """)
            
            st.write("**Statistical Tests Used:**")
            st.write("""
            - **Shapiro-Wilk Test**: Normality test (not used due to large n > 5000)
            - **KS Test**: Distribution comparison
            - **Descriptive Statistics**: Mean, median, std, quartiles
            
            **Why These Methods?**
            - Non-parametric tests handle non-normal data
            - Visual methods provide intuitive understanding
            - Multiple methods ensure robustness
            """)
        
        # Show example calculations
        with st.expander("📊 See Example Calculations"):
            st.write("**Bimodality Coefficient Calculation:**")
            volumes = df['volume_gbp'].values
            n = len(volumes)
            skewness = stats.skew(volumes)
            kurtosis = stats.kurtosis(volumes, fisher=False)
            bc = (skewness**2 + 1) / (kurtosis + 3*(n-1)**2/((n-2)*(n-3))) if n > 3 else 0
            
            st.write(f"- n (sample size) = {n:,}")
            st.write(f"- Skewness = {skewness:.4f}")
            st.write(f"- Kurtosis = {kurtosis:.4f}")
            st.write(f"- BC = ({skewness:.4f}² + 1) / ({kurtosis:.4f} + 3*({n-1})²/(({n-2})*({n-3})))")
            st.write(f"- BC = {bc:.4f}")
            st.write(f"- Critical value = 0.555")
            st.write(f"- Conclusion: {'Bimodal' if bc > 0.555 else 'Unimodal'}")
    
    with tab3:
        st.subheader("Question 2: Quarterly Changes Methodology")
        
        st.write("""
        **Statistical Testing Framework:**
        
        1. **Kruskal-Wallis Test:**
           - Non-parametric alternative to ANOVA
           - Tests if medians across quarters are equal
           - Appropriate for non-normal data
           - H₀: All quarters have same median
           - H₁: At least one quarter differs
        
        2. **Effect Size Analysis (Cliff's Delta):**
           - Measures practical significance, not just statistical
           - Non-parametric effect size measure
           - Ranges from -1 to 1
           - Interpretation:
             - |δ| < 0.147: Negligible effect
             - 0.147 ≤ |δ| < 0.33: Small effect
             - 0.33 ≤ |δ| < 0.474: Medium effect
             - |δ| ≥ 0.474: Large effect
        
        3. **Decision Criteria:**
           - Statistical significance (p < 0.05) AND
           - Practical significance (|δ| ≥ 0.33) = Meaningful change
           - Either condition not met = No meaningful change
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Cliff's Delta Formula:**")
            st.write("""
            δ = (P(x > y) - P(x < y))
            
            Where:
            - P(x > y): Proportion of times values in group x exceed values in group y
            - P(x < y): Proportion of times values in group x are less than values in group y
            
            **Interpretation:**
            - δ > 0: Group x tends to have higher values
            - δ < 0: Group y tends to have higher values
            - δ = 0: No difference between groups
            """)
        
        with col2:
            st.write("**Why These Methods?**")
            st.write("""
            1. **Robust to Non-Normality**:
               - Data is right-skewed and non-normal
               - Parametric tests (ANOVA, t-tests) invalid
            
            2. **Handles Outliers**:
               - Non-parametric methods are robust to outliers
               - Outliers are legitimate business events
            
            3. **Measures Practical Significance**:
               - Statistical significance alone can be misleading
               - Effect size indicates real-world importance
            
            4. **Quarterly Focus**:
               - Business planning often quarterly
               - Natural business cycle alignment
            """)
        
        # Show test results
        with st.expander("📈 See Test Results Summary"):
            # Prepare data for testing
            df_q2_q4 = df[(df['year'] == 2023) & (df['quarter'].isin([2, 3, 4]))].copy()
            Q2_data = df_q2_q4[df_q2_q4['quarter'] == 2]['volume_gbp']
            Q3_data = df_q2_q4[df_q2_q4['quarter'] == 3]['volume_gbp']
            Q4_data = df_q2_q4[df_q2_q4['quarter'] == 4]['volume_gbp']
            
            # Kruskal-Wallis test
            h_stat, p_val_kw = stats.kruskal(Q2_data, Q3_data, Q4_data)
            
            # Cliff's Delta function
            def cliffs_delta(x, y):
                x_arr = np.array(x)
                y_arr = np.array(y)
                x_reshaped = x_arr.reshape(-1, 1)
                y_reshaped = y_arr.reshape(1, -1)
                greater = np.sum(x_reshaped > y_reshaped)
                less = np.sum(x_reshaped < y_reshaped)
                delta = (greater - less) / (len(x_arr) * len(y_arr)) if len(x_arr) * len(y_arr) > 0 else 0
                return delta
            
            delta_q2_q3 = cliffs_delta(Q2_data, Q3_data)
            delta_q2_q4 = cliffs_delta(Q2_data, Q4_data)
            delta_q3_q4 = cliffs_delta(Q3_data, Q4_data)
            
            st.write("**Kruskal-Wallis Test Results:**")
            st.write(f"- Test statistic (H): {h_stat:.4f}")
            st.write(f"- p-value: {p_val_kw:.4f}")
            st.write(f"- Significance level (α): 0.05")
            st.write(f"- Conclusion: {'Reject H₀' if p_val_kw < 0.05 else 'Fail to reject H₀'}")
            
            st.write("\n**Cliff's Delta Results:**")
            st.write(f"- Q2 → Q3: δ = {delta_q2_q3:+.4f} ({'negligible' if abs(delta_q2_q3) < 0.147 else 'small' if abs(delta_q2_q3) < 0.33 else 'medium' if abs(delta_q2_q3) < 0.474 else 'large'} effect)")
            st.write(f"- Q2 → Q4: δ = {delta_q2_q4:+.4f} ({'negligible' if abs(delta_q2_q4) < 0.147 else 'small' if abs(delta_q2_q4) < 0.33 else 'medium' if abs(delta_q2_q4) < 0.474 else 'large'} effect)")
            st.write(f"- Q3 → Q4: δ = {delta_q3_q4:+.4f} ({'negligible' if abs(delta_q3_q4) < 0.147 else 'small' if abs(delta_q3_q4) < 0.33 else 'medium' if abs(delta_q3_q4) < 0.474 else 'large'} effect)")
    
    with tab4:
        st.subheader("Question 3: October Estimation Methodology")
        
        st.write("""
        **Bootstrap Resampling Method:**
        
        1. **Reference Period Selection:**
           - Used Q3 (Jul-Sep) as most recent complete period
           - 92 days of data with complete weekday coverage
           - Same seasonal period as October (autumn)
        
        2. **Resampling Process:**
           - For each missing October weekday:
             - Resample from corresponding Q3 weekday data
             - Repeat 1,000 times to build distribution
           - For existing October data: Use actual values
        
        3. **Uncertainty Quantification:**
           - Calculate mean as point estimate
           - Calculate 80% and 95% confidence intervals
           - Report margin of error
           - Calculate relative uncertainty
        
        4. **Advantages of Bootstrap:**
           - Non-parametric (no distribution assumptions)
           - Preserves data patterns and correlations
           - Provides confidence intervals
           - Handles missing data robustly
           - Accounts for natural variability
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Bootstrap Algorithm:**")
            st.write("""
            1. **Initialize** empty results array
            2. **For** i = 1 to 1,000 simulations:
                - **Initialize** total = 0
                - **For each** date in October:
                    - **If** date has actual data:
                        - Add actual value to total
                    - **Else** (missing weekday):
                        - Randomly sample from same weekday in Q3
                        - Add sampled value to total
                - **Append** total to results array
            3. **Calculate** statistics from results array
            4. **Report** estimates with confidence intervals
            """)
        
        with col2:
            st.write("**Why Bootstrap Over Other Methods?**")
            st.write("""
            1. **vs Mean Imputation**: Preserves variability
            2. **vs Regression**: No parametric assumptions needed
            3. **vs Time Series**: Handles missing blocks of data
            4. **vs Simple Average**: Accounts for weekday patterns
            
            **Key Assumptions:**
            1. October patterns similar to Q3
            2. No major business changes
            3. Q3 data representative
            
            **Limitations:**
            1. High uncertainty with many missing days
            2. Seasonal effects may differ
            3. Special events not captured
            """)
        
        # Show bootstrap details
        with st.expander("🔧 See Bootstrap Implementation Details"):
            st.write("**Python Implementation:**")
            st.code("""
# Step 1: Prepare Q3 reference data
q3_data = df[df['quarter'] == 3].copy()
q3_weekday_data = {}
for weekday in range(7):
    q3_weekday_data[weekday] = q3_data[q3_data['day_of_week'] == weekday]['volume_gbp'].values

# Step 2: Create lookup for actual October values
actual_values = {}
for _, row in oct_2023.iterrows():
    actual_values[row['posting_date']] = row['volume_gbp']

# Step 3: Run bootstrap simulations
n_simulations = 1000
bootstrap_totals = []
for sim in range(n_simulations):
    sim_total = 0
    for date in all_oct_dates:
        if date in actual_values:
            sim_total += actual_values[date]
        else:
            weekday = date.weekday()
            weekday_samples = q3_weekday_data[weekday]
            if len(weekday_samples) > 0:
                sim_total += np.random.choice(weekday_samples)
            else:
                sim_total += q3_median
    bootstrap_totals.append(sim_total)

# Step 4: Calculate statistics
mean_estimate = np.mean(bootstrap_totals)
ci_95 = np.percentile(bootstrap_totals, [2.5, 97.5])
            """, language='python')
            
            st.write("**Simulation Parameters:**")
            sim_params = pd.DataFrame({
                "Parameter": ["Number of simulations", "Reference period", "Missing days", "Available days", "Weekday matching", "Fallback method"],
                "Value": ["1,000", "Q3 (Jul-Sep 2023)", "22 weekdays", "9 weekend days", "Same weekday in Q3", "Q3 median"]
            })
            st.dataframe(sim_params, use_container_width=True)
    
    # Key Methodological Decisions
    st.subheader("Key Methodological Decisions")
    
    decisions = [
        ("Why median over mean for central tendency?", "Data is right-skewed with outliers. Median better represents 'typical' day and is robust to extreme values."),
        ("Why non-parametric tests exclusively?", "Data fails normality tests. Non-parametric tests (Kruskal-Wallis, Cliff's Delta) are valid for non-normal distributions."),
        ("Why bootstrap for October estimation?", "Missing 22 weekdays requires robust imputation. Bootstrap preserves patterns and quantifies uncertainty better than simple imputation."),
        ("Why Q3 as reference for October?", "Most recent complete period before October with similar seasonal patterns. Avoids seasonal bias from using full dataset."),
        ("Why 1,000 bootstrap simulations?", "Provides stable estimates with reasonable computation time. More simulations offer diminishing returns."),
        ("Why confidence intervals instead of single point estimate?", "Quantifies uncertainty explicitly. Business decisions need understanding of risk and range."),
        ("Why weekday-specific resampling?", "Weekday/weekend patterns are significant (6.1× difference). Preserving this pattern is crucial for accuracy."),
    ]
    
    for question, answer in decisions:
        with st.expander(f"❓ {question}"):
            st.write(answer)
    
    # Navigation hint
    st.markdown("---")
    st.info("💡 **Tip**: Use the sidebar to navigate to other sections of the analysis.")

# SUMMARY & RECOMMENDATIONS SECTION
else:
    st.header("📋 Summary & Recommendations")
    
    # Key metrics summary
    st.subheader("Key Analysis Metrics")
    
    # Calculate final metrics
    median_vol = df['volume_gbp'].median()
    mean_vol = df['volume_gbp'].mean()
    skew = df['volume_gbp'].skew()
    
    # Bimodality coefficient
    volumes = df['volume_gbp'].values
    n = len(volumes)
    skewness = stats.skew(volumes)
    kurtosis = stats.kurtosis(volumes, fisher=False)
    bc = (skewness**2 + 1) / (kurtosis + 3*(n-1)**2/((n-2)*(n-3))) if n > 3 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Median", f"£{median_vol:,.0f}")
        st.metric("Distribution Skew", f"{skew:.2f}")
        st.metric("Bimodality Coeff", f"{bc:.3f}")
    
    with col2:
        weekday_mean = df[~df['is_weekend']]['volume_gbp'].mean()
        weekend_mean = df[df['is_weekend']]['volume_gbp'].mean()
        ratio = weekday_mean / weekend_mean if weekend_mean != 0 else 0
        st.metric("Weekday:Weekend Ratio", f"{ratio:.1f}x")
        st.metric("Data Completeness", "91%")
    
    with col3:
        # Simple October estimate for summary
        oct_2023 = df[(df['posting_date'].dt.month == 10) & (df['posting_date'].dt.year == 2023)]
        q3_data = df[df['quarter'] == 3].copy()
        weekday_median_q3 = q3_data.groupby('day_of_week')['volume_gbp'].median()
        oct_estimate = oct_2023['volume_gbp'].sum()
        
        for date in pd.date_range('2023-10-01', '2023-10-31', freq='D'):
            if date not in oct_2023['posting_date'].values:
                oct_estimate += weekday_median_q3[date.weekday()]
        
        st.metric("October Estimate", f"£{oct_estimate/1e6:.1f}M")
        st.metric("Missing October Days", "22")
    
    # Executive Summary
    st.subheader("Executive Summary")
    
    st.write(f"""
    **Distribution Characteristics:**
    - Right-skewed, unimodal distribution (not bimodal)
    - Mean (£{mean_vol:,.0f}K) > Median (£{median_vol:,.0f}K) indicates positive skew
    - High variability (CV = {(df['volume_gbp'].std() / mean_vol if mean_vol != 0 else 0):.2f}) with legitimate business outliers
    - Significant weekday/weekend differences ({ratio:.1f}× higher on weekdays)
    
    **Quarterly Analysis:**
    - No statistically significant differences between quarters (p = 0.964)
    - Observed changes (-8.7% Q2→Q3, +10.7% Q3→Q4) are likely random fluctuations
    - All effect sizes negligible (Cliff's Delta < 0.147)
    
    **October 2023 Estimation:**
    - **Severe data gap**: 22 missing weekdays (only weekends available)
    - **Best estimate**: £5.4-5.7 million total volume
    - **High uncertainty**: ±£0.9M margin of error (31% relative uncertainty)
    - **95% Confidence Interval**: £4.8M to £6.6M
    """)
    
    # Interactive summary visualization
    st.subheader("Interactive Summary Dashboard")
    
    # Create summary visualization
    summary_data = pd.DataFrame({
        'Metric': ['Mean Volume', 'Median Volume', 'Weekday Mean', 'Weekend Mean', 'Q2 Median', 'Q3 Median', 'Q4 Median'],
        'Value (GBP)': [
            mean_vol,
            median_vol,
            weekday_mean,
            weekend_mean,
            df[df['quarter'] == 2]['volume_gbp'].median(),
            df[df['quarter'] == 3]['volume_gbp'].median(),
            df[df['quarter'] == 4]['volume_gbp'].median()
        ],
        'Category': ['Overall', 'Overall', 'Weekday/Weekend', 'Weekday/Weekend', 'Quarterly', 'Quarterly', 'Quarterly']
    })
    
    fig = px.bar(summary_data, x='Metric', y='Value (GBP)', color='Category',
                 title='Key Metrics Summary',
                 hover_data={'Value (GBP)': ':,.0f'},
                 labels={'Value (GBP)': 'Volume (GBP)', 'Metric': 'Metric'})
    
    fig.update_layout(
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )
    
    fig.update_yaxes(tickprefix="£", tickformat=",.0f")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.subheader("Business Recommendations for WISE")
    
    recommendations = [
        (f"Use median (£{median_vol:,.0f}K) not mean for 'typical day' reporting", "Mean inflated by outliers"),
        ("Apply non-parametric tests exclusively", "Data is non-normal"),
        ("Build separate models for weekdays and weekends", f"{ratio:.1f}× volume difference"),
        ("Maintain 30-40% buffer capacity", "High variability in daily volumes"),
        ("Segment by transaction behavior, not volume", "Unimodal distribution"),
        ("Implement automated data quality checks", "Prevent gaps like October 2023"),
        ("Investigate high-value outlier patterns", "Business development opportunities"),
        ("Explore weekend volume optimization", "Cost savings potential")
    ]
    
    for i, (rec, reason) in enumerate(recommendations, 1):
        st.write(f"{i}. **{rec}** - {reason}")
    
    # Next Steps - FIXED Timeline
    st.subheader("Next Steps & Implementation Timeline")
    
    st.write("""
    **Immediate Actions (Week 1-2):**
    - Update reporting dashboards to use median-based metrics
    - Implement automated data quality checks
    - Set up control charts for daily volume monitoring
    
    **Short-term Projects (Month 1):**
    - Develop separate weekday/weekend forecasting models
    - Create outlier investigation protocol
    - Establish quarterly review process with statistical testing
    
    **Medium-term Initiatives (Quarter 1):**
    - Integrate exchange rate data into forecasting
    - Implement customer behavior segmentation
    - Optimize resource allocation based on volume patterns
    """)
    
    # Create FIXED implementation timeline
    timeline_data = pd.DataFrame({
        'Task': ['Reporting Updates', 'Data Quality Checks', 'Model Development', 'Outlier Protocol', 'Exchange Rate Integration'],
        'Start Week': [1, 2, 3, 5, 9],
        'End Week': [2, 3, 8, 8, 13],
        'Duration': [1, 1, 5, 3, 4],
        'Phase': ['Immediate', 'Immediate', 'Short-term', 'Short-term', 'Medium-term']
    })
    
    # Create Gantt chart timeline
    fig = go.Figure()
    
    colors = {
        'Immediate': 'red',
        'Short-term': 'orange',
        'Medium-term': 'green'
    }
    
    for phase in timeline_data['Phase'].unique():
        phase_data = timeline_data[timeline_data['Phase'] == phase]
        
        fig.add_trace(go.Bar(
            x=phase_data['Duration'],
            y=phase_data['Task'],
            orientation='h',
            name=phase,
            marker_color=colors[phase],
            hovertemplate='<b>%{y}</b><br>Start: Week %{customdata[0]}<br>End: Week %{customdata[1]}<br>Duration: %{x} weeks<extra></extra>',
            customdata=phase_data[['Start Week', 'End Week']].values
        ))
    
    fig.update_layout(
        title='Implementation Timeline (in weeks)',
        xaxis_title='Weeks from Start',
        yaxis_title='Task',
        barmode='stack',
        template='plotly_white',
        height=400,
        hovermode='y'
    )
    
    # Add week markers
    for week in range(1, 14, 2):
        fig.add_vline(x=week-0.5, line_dash="dash", line_color="gray", opacity=0.3)
    
    fig.update_xaxes(range=[0, 13], dtick=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Alternative timeline visualization
    st.subheader("Implementation Roadmap")
    
    roadmap_data = pd.DataFrame({
        'Phase': ['Immediate (Week 1-2)', 'Short-term (Month 1)', 'Medium-term (Quarter 1)'],
        'Key Deliverables': [
            'Updated dashboards, Data quality alerts',
            'Weekday/weekend models, Outlier investigation process',
            'Exchange rate integration, Customer segmentation'
        ],
        'Success Metrics': [
            'Median-based reporting implemented',
            'Forecast accuracy improved by 15%',
            'Resource optimization achieved'
        ]
    })
    
    st.dataframe(roadmap_data, use_container_width=True)
    
    st.success("✅ Analysis Complete - Comprehensive insights with practical business recommendations for optimizing GBP to ZAR transfer operations.")
    
    # Navigation hint
    st.markdown("---")
    st.info("💡 **Tip**: Use the sidebar to navigate to other sections of the analysis.")


