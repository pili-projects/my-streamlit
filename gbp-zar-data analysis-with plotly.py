import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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

# ============================================================
# PRE-COMPUTE ALL STATISTICS AND DATA AT THE TOP LEVEL
# ============================================================

@st.cache_data
def load_and_precompute_data():
    """Load data and pre-compute all statistics once"""
    try:
        url = "https://drive.google.com/uc?id=1BK9eVWAu2LCDJ2haDYafoKhPyWP2tqCl"
        df = pd.read_csv(url)
        
        # Basic processing
        df['posting_date'] = pd.to_datetime(df['posting_date'], errors='coerce')
        df = df.sort_values('posting_date').reset_index(drop=True)
        
        # Feature engineering
        df['year'] = df['posting_date'].dt.year
        df['month'] = df['posting_date'].dt.month
        df['quarter'] = df['posting_date'].dt.quarter
        df['weekday'] = df['posting_date'].dt.day_name()
        df['day_of_week'] = df['posting_date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        df['month_name'] = df['posting_date'].dt.strftime('%B')
        
        # Pre-compute key statistics
        stats_dict = {}
        
        # Basic stats
        stats_dict['total_days'] = len(df)
        stats_dict['total_volume'] = df['volume_gbp'].sum()
        stats_dict['mean_volume'] = df['volume_gbp'].mean()
        stats_dict['median_volume'] = df['volume_gbp'].median()
        stats_dict['std_volume'] = df['volume_gbp'].std()
        stats_dict['skew'] = df['volume_gbp'].skew()
        stats_dict['kurt'] = df['volume_gbp'].kurtosis()
        stats_dict['cv'] = stats_dict['std_volume'] / stats_dict['mean_volume'] if stats_dict['mean_volume'] != 0 else 0
        
        # Bimodality analysis
        volumes = df['volume_gbp'].values
        n = len(volumes)
        skewness = stats.skew(volumes)
        kurtosis = stats.kurtosis(volumes, fisher=False)
        bc = (skewness**2 + 1) / (kurtosis + 3*(n-1)**2/((n-2)*(n-3))) if n > 3 else 0
        stats_dict['bimodality_coefficient'] = bc
        stats_dict['is_bimodal'] = bc > (5/9)
        
        # Weekday vs weekend
        weekday_data = df[~df['is_weekend']]['volume_gbp']
        weekend_data = df[df['is_weekend']]['volume_gbp']
        stats_dict['weekday_mean'] = weekday_data.mean()
        stats_dict['weekend_mean'] = weekend_data.mean()
        stats_dict['weekday_median'] = weekday_data.median()
        stats_dict['weekend_median'] = weekend_data.median()
        stats_dict['weekday_weekend_ratio'] = stats_dict['weekday_mean'] / stats_dict['weekend_mean'] if stats_dict['weekend_mean'] != 0 else 0
        
        # KS test
        ks_stat, ks_p = stats.ks_2samp(weekday_data, weekend_data)
        stats_dict['ks_p'] = ks_p
        
        # Outlier analysis
        Q1 = df['volume_gbp'].quantile(0.25)
        Q3 = df['volume_gbp'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[df['volume_gbp'] > upper_bound]
        stats_dict['outlier_threshold'] = upper_bound
        stats_dict['outlier_count'] = len(outliers)
        stats_dict['zero_days'] = (df['volume_gbp'] == 0).sum()
        
        # Quarterly stats (Q2-Q4 2023)
        df_q2_q4 = df[(df['year'] == 2023) & (df['quarter'].isin([2, 3, 4]))].copy()
        quarterly_stats = df_q2_q4.groupby('quarter')['volume_gbp'].agg([
            ('median', 'median'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('count', 'count')
        ]).to_dict('index')
        
        # Quarterly statistical tests
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
        
        stats_dict['kruskal_p'] = p_val_kw
        stats_dict['kruskal_h'] = h_stat
        stats_dict['delta_q2_q3'] = cliffs_delta(Q2_data, Q3_data)
        stats_dict['delta_q2_q4'] = cliffs_delta(Q2_data, Q4_data)
        stats_dict['delta_q3_q4'] = cliffs_delta(Q3_data, Q4_data)
        
        # October 2023 analysis
        oct_2023 = df[(df['posting_date'].dt.month == 10) & (df['posting_date'].dt.year == 2023)]
        stats_dict['oct_available_days'] = len(oct_2023)
        stats_dict['oct_missing_days'] = 31 - len(oct_2023)
        stats_dict['oct_weekend_days'] = len(oct_2023[oct_2023['is_weekend']])
        stats_dict['oct_weekday_days'] = len(oct_2023[~oct_2023['is_weekend']])
        stats_dict['oct_total_actual'] = oct_2023['volume_gbp'].sum()
        
        # Prepare Q3 data for bootstrap
        q3_data = df[df['quarter'] == 3].copy()
        q3_weekday_data = {}
        for weekday in range(7):
            q3_weekday_data[weekday] = q3_data[q3_data['day_of_week'] == weekday]['volume_gbp'].values
        
        # Bootstrap estimation for October
        all_oct_dates = pd.date_range('2023-10-01', '2023-10-31', freq='D')
        actual_values = {}
        for _, row in oct_2023.iterrows():
            actual_values[row['posting_date']] = row['volume_gbp']
        
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
                        sim_total += q3_data['volume_gbp'].median()
            bootstrap_totals.append(sim_total)
        
        bootstrap_totals = np.array(bootstrap_totals)
        stats_dict['oct_mean_estimate'] = np.mean(bootstrap_totals)
        stats_dict['oct_median_estimate'] = np.median(bootstrap_totals)
        stats_dict['oct_ci_95'] = np.percentile(bootstrap_totals, [2.5, 97.5])
        stats_dict['oct_ci_80'] = np.percentile(bootstrap_totals, [10, 90])
        stats_dict['oct_std_estimate'] = np.std(bootstrap_totals)
        stats_dict['oct_ci_range'] = stats_dict['oct_ci_95'][1] - stats_dict['oct_ci_95'][0]
        stats_dict['oct_relative_uncertainty'] = (stats_dict['oct_ci_range'] / stats_dict['oct_mean_estimate'] * 100) if stats_dict['oct_mean_estimate'] != 0 else 0
        
        # Store prepared data
        stats_dict['df'] = df
        stats_dict['df_q2_q4'] = df_q2_q4
        stats_dict['oct_2023'] = oct_2023
        stats_dict['q3_data'] = q3_data
        stats_dict['weekday_data'] = weekday_data
        stats_dict['weekend_data'] = weekend_data
        stats_dict['Q2_data'] = Q2_data
        stats_dict['Q3_data'] = Q3_data
        stats_dict['Q4_data'] = Q4_data
        stats_dict['outliers'] = outliers
        stats_dict['quarterly_stats'] = quarterly_stats
        stats_dict['bootstrap_totals'] = bootstrap_totals
        
        return stats_dict
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load and precompute all data
data = load_and_precompute_data()

if data is None:
    st.stop()

# Extract precomputed data
df = data['df']
df_q2_q4 = data['df_q2_q4']
oct_2023 = data['oct_2023']
weekday_data = data['weekday_data']
weekend_data = data['weekend_data']
outliers = data['outliers']
Q2_data = data['Q2_data']
Q3_data = data['Q3_data']
Q4_data = data['Q4_data']
quarterly_stats = data['quarterly_stats']
bootstrap_totals = data['bootstrap_totals']

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
    """)

# ============================================================
# DATASET OVERVIEW SECTION
# ============================================================
if analysis_section == "📊 Dataset Overview":
    st.header("📊 Dataset Overview")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Days", f"{data['total_days']:,}")
        start_date = df['posting_date'].min().strftime('%b %d, %Y')
        end_date = df['posting_date'].max().strftime('%b %d, %Y')
        st.metric("Date Range", f"{start_date} to {end_date}")
    
    with col2:
        st.metric("Total Volume", f"£{data['total_volume']:,.0f}")
        st.metric("Average Daily", f"£{data['mean_volume']:,.0f}")
    
    with col3:
        st.metric("Zero Volume Days", data['zero_days'])
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
    
    # Time series plot
    st.subheader("Daily Volume Time Series")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['posting_date'],
        y=df['volume_gbp'],
        mode='lines',
        name='Daily Volume',
        line=dict(color='blue', width=2),
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Volume: £%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Daily Transfer Volumes (Apr-Dec 2023)',
        xaxis_title='Date',
        yaxis_title='Volume (GBP)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    fig.update_yaxes(tickprefix="£", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Q1: DISTRIBUTION ANALYSIS
# ============================================================
elif analysis_section == "📈 Q1: Distribution Analysis":
    st.header("📈 Question 1: Distribution Analysis")
    
    tab1, tab2, tab3 = st.tabs(["1a) Distribution Shape", "1b) Real-World Causes", "1c) Implications"])
    
    with tab1:
        st.subheader("1a) Describe the distribution that our daily transfer volumes follow")
        
        # Display precomputed metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"£{data['mean_volume']:,.0f}")
        with col2:
            st.metric("Median", f"£{data['median_volume']:,.0f}")
        with col3:
            st.metric("Skewness", f"{data['skew']:.2f}")
        with col4:
            st.metric("CV", f"{data['cv']:.2f}")
        
        # Bimodality analysis
        st.subheader("Bimodality Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Bimodality Test Results:**")
            st.write(f"- Pearson's coefficient: **{data['bimodality_coefficient']:.4f}**")
            st.write(f"- Critical threshold: > 0.555")
            st.write(f"- Conclusion: **{'BIMODAL' if data['is_bimodal'] else 'UNIMODAL'}**")
        
        with col2:
            st.write("**Weekday vs Weekend:**")
            st.write(f"- KS test p-value: **{data['ks_p']:.4f}**")
            st.write(f"- Conclusion: **{'Significantly different' if data['ks_p'] < 0.05 else 'Not significantly different'}**")
        
        # Distribution visualizations
        st.subheader("Distribution Visualizations")
        
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
        
        fig.update_layout(
            height=800,
            showlegend=True,
            template='plotly_white',
            hovermode='closest'
        )
        
        fig.update_xaxes(title_text="Volume (GBP)", row=1, col=1, tickprefix="£", tickformat=",.0f")
        fig.update_yaxes(title_text="Density/Count", row=1, col=1)
        fig.update_yaxes(title_text="Volume (GBP)", row=1, col=2, tickprefix="£", tickformat=",.0f")
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
        fig.update_yaxes(title_text="Observed Quantiles", row=2, col=1, tickprefix="£", tickformat=",.0f")
        fig.update_yaxes(title_text="Volume (GBP)", row=2, col=2, tickprefix="£", tickformat=",.0f")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("1b) What real world cause do you think is behind this shape of distribution?")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Weekday Mean", f"£{data['weekday_mean']:,.0f}")
            st.metric("Weekend Mean", f"£{data['weekend_mean']:,.0f}")
            st.metric("Weekday:Weekend Ratio", f"{data['weekday_weekend_ratio']:.1f}x")
        
        with col2:
            st.metric("Outlier Threshold", f"£{data['outlier_threshold']:,.0f}")
            st.metric("Outlier Days", f"{data['outlier_count']} ({data['outlier_count']/data['total_days']*100:.1f}%)")
            st.metric("Zero-Volume Days", f"{data['zero_days']}")
        
        # Weekday vs weekend histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=weekday_data,
            name='Weekdays',
            opacity=0.7,
            marker_color='green',
            nbinsx=30,
            hovertemplate="Weekday Volume: £%{x:,.0f}<br>Count: %{y}<extra></extra>"
        ))
        
        fig.add_trace(go.Histogram(
            x=weekend_data,
            name='Weekends',
            opacity=0.7,
            marker_color='orange',
            nbinsx=30,
            hovertemplate="Weekend Volume: £%{x:,.0f}<br>Count: %{y}<extra></extra>"
        ))
        
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
        
        # Show top outliers if they exist
        if len(outliers) > 0:
            st.subheader("Top 5 High-Value Outlier Days")
            outlier_display = outliers.nlargest(5, 'volume_gbp')[['posting_date', 'volume_gbp', 'weekday']].copy()
            outlier_display['posting_date'] = outlier_display['posting_date'].dt.strftime('%Y-%m-%d')
            outlier_display['volume_gbp'] = outlier_display['volume_gbp'].apply(lambda x: f"£{x:,.0f}")
            st.dataframe(outlier_display, use_container_width=True)
    
    with tab3:
        st.subheader("1c) What are some of the implications that this distribution would commonly have on analysis that you might do?")
        
        st.write(f"""
        **Statistical Analysis Implications:**
        
        1. **Statistical Testing Limitations:**
           - Parametric tests (t-tests, ANOVA) are INVALID
           - Use NON-PARAMETRIC alternatives (Mann-Whitney, Kruskal-Wallis)
        
        2. **Central Tendency Measures:**
           - Mean (£{data['mean_volume']:,.0f}K) is inflated by outliers
           - Median (£{data['median_volume']:,.0f}K) better represents 'typical' day
           - Recommendation: Use MEDIAN for reporting & planning
        
        3. **Forecasting & Modeling Challenges:**
           - Traditional models (ARIMA) may fail
           - Consider transformations (log, Box-Cox)
           - Segment data: weekdays vs weekends
           - Use robust regression methods
        """)

# ============================================================
# Q2: QUARTERLY ANALYSIS
# ============================================================
elif analysis_section == "📊 Q2: Quarterly Changes":
    st.header("📊 Question 2: Quarterly Changes Analysis")
    
    # Display quarterly metrics
    st.subheader("Quarterly Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        q2_median = quarterly_stats[2]['median']
        st.metric("Q2 Median", f"£{q2_median:,.0f}")
        st.metric("Q2 Days", quarterly_stats[2]['count'])
    
    with col2:
        q3_median = quarterly_stats[3]['median']
        q2_q3_change = ((q3_median - q2_median) / q2_median * 100) if q2_median != 0 else 0
        st.metric("Q3 Median", f"£{q3_median:,.0f}")
        st.metric("Q2→Q3 Change", f"{q2_q3_change:+.1f}%")
    
    with col3:
        q4_median = quarterly_stats[4]['median']
        q3_q4_change = ((q4_median - q3_median) / q3_median * 100) if q3_median != 0 else 0
        st.metric("Q4 Median", f"£{q4_median:,.0f}")
        st.metric("Q3→Q4 Change", f"{q3_q4_change:+.1f}%")
    
    # Statistical testing results
    st.subheader("Statistical Significance Testing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Kruskal-Wallis Test:**")
        st.write(f"- H₀: All quarters have same median")
        st.write(f"- H₁: At least one quarter differs")
        st.write(f"- Test statistic (H): {data['kruskal_h']:.4f}")
        st.write(f"- **p-value: {data['kruskal_p']:.4f}**")
        
        if data['kruskal_p'] < 0.05:
            st.success("✅ REJECT H₀: Significant difference exists (p < 0.05)")
        else:
            st.warning("❌ FAIL TO REJECT H₀: No significant difference (p ≥ 0.05)")
    
    with col2:
        st.write("**Effect Size Analysis (Cliff's Delta):**")
        st.write(f"- Q2 → Q3: δ = {data['delta_q2_q3']:+.3f}")
        st.write(f"- Q2 → Q4: δ = {data['delta_q2_q4']:+.3f}")
        st.write(f"- Q3 → Q4: δ = {data['delta_q3_q4']:+.3f}")
        
        # Interpretation helper
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
        st.write(f"- Q2 → Q3: {interpret_delta(data['delta_q2_q3'])}")
        st.write(f"- Q2 → Q4: {interpret_delta(data['delta_q2_q4'])}")
        st.write(f"- Q3 → Q4: {interpret_delta(data['delta_q3_q4'])}")
    
    # Quarterly visualization
    st.subheader("Quarterly Comparison Visualizations")
    
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
    medians = [quarterly_stats[q]['median'] for q in quarters]
    
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
    
    fig.update_layout(
        height=500,
        template='plotly_white',
        hovermode='closest'
    )
    
    fig.update_xaxes(title_text="Quarter", row=1, col=2, tickvals=quarters, ticktext=['Q2', 'Q3', 'Q4'])
    fig.update_yaxes(title_text="Volume (GBP)", row=1, col=1, tickprefix="£", tickformat=",.0f")
    fig.update_yaxes(title_text="Median Volume (GBP)", row=1, col=2, tickprefix="£", tickformat=",.0f")
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Q3: OCTOBER 2023 ESTIMATION
# ============================================================
elif analysis_section == "🔮 Q3: October 2023 Estimation":
    st.header("🔮 Question 3: October 2023 Volume Estimation")
    
    st.subheader("October 2023 Data Availability")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Days with Data", f"{data['oct_available_days']}/31")
        st.metric("Missing Days", data['oct_missing_days'])
    
    with col2:
        st.metric("Weekend Days Available", data['oct_weekend_days'])
        st.metric("Weekday Days Available", data['oct_weekday_days'])
    
    with col3:
        st.metric("Available Data Total", f"£{data['oct_total_actual']:,.0f}")
    
    # Warning about missing data
    if data['oct_weekday_days'] == 0:
        st.error("⚠️ **CRITICAL DATA GAP**: October 2023 has NO WEEKDAY data! Only weekend data available.")
    
    # Show available data
    st.subheader("Available October Data (Only Weekends)")
    oct_display = oct_2023[['posting_date', 'volume_gbp', 'weekday', 'is_weekend']].copy()
    oct_display['posting_date'] = oct_display['posting_date'].dt.strftime('%Y-%m-%d')
    oct_display['volume_gbp'] = oct_display['volume_gbp'].apply(lambda x: f"£{x:,.2f}")
    st.dataframe(oct_display, use_container_width=True)
    
    # Bootstrap results
    st.subheader("Estimation Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Best Estimate (Mean)", f"£{data['oct_mean_estimate']/1e6:.1f}M")
    
    with col2:
        st.metric("Median Estimate", f"£{data['oct_median_estimate']/1e6:.1f}M")
    
    with col3:
        st.metric("95% CI Range", f"£{data['oct_ci_range']/1e6:.1f}M")
    
    with col4:
        st.metric("Relative Uncertainty", f"{data['oct_relative_uncertainty']:.0f}%")
    
    # Confidence intervals
    st.subheader("Confidence Intervals")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**95% Confidence Interval:**\n£{data['oct_ci_95'][0]/1e6:.1f}M to £{data['oct_ci_95'][1]/1e6:.1f}M")
    
    with col2:
        st.info(f"**80% Confidence Interval:**\n£{data['oct_ci_80'][0]/1e6:.1f}M to £{data['oct_ci_80'][1]/1e6:.1f}M")
    
    # Bootstrap distribution visualization
    st.subheader("Bootstrap Distribution")
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=bootstrap_totals/1e6,
        nbinsx=40,
        name='Bootstrap Distribution',
        marker_color='blue',
        opacity=0.7,
        hovertemplate="Total Volume: £%{x:.1f}M<br>Count: %{y}<extra></extra>"
    ))
    
    fig.add_vline(x=data['oct_mean_estimate']/1e6, line_dash="dash", line_color="red", 
                 annotation_text=f"Mean: £{data['oct_mean_estimate']/1e6:.1f}M")
    
    fig.add_vrect(x0=data['oct_ci_95'][0]/1e6, x1=data['oct_ci_95'][1]/1e6, 
                  fillcolor="rgba(255, 0, 0, 0.1)", line_width=0,
                  annotation_text="95% CI")
    
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

# ============================================================
# SUMMARY SECTION
# ============================================================
elif analysis_section == "📋 Summary":
    st.header("📋 Summary & Recommendations")
    
    st.subheader("Executive Summary")
    
    st.write(f"""
    **Distribution Characteristics:**
    - Right-skewed, unimodal distribution (not bimodal)
    - Mean (£{data['mean_volume']:,.0f}K) > Median (£{data['median_volume']:,.0f}K) indicates positive skew
    - High variability (CV = {data['cv']:.2f}) with legitimate business outliers
    - Significant weekday/weekend differences ({data['weekday_weekend_ratio']:.1f}× higher on weekdays)
    
    **Quarterly Analysis:**
    - No statistically significant differences between quarters (p = {data['kruskal_p']:.3f})
    - All effect sizes negligible (Cliff's Delta < 0.147)
    
    **October 2023 Estimation:**
    - **Best estimate**: £{data['oct_mean_estimate']/1e6:.1f} million total volume
    - **95% Confidence Interval**: £{data['oct_ci_95'][0]/1e6:.1f}M to £{data['oct_ci_95'][1]/1e6:.1f}M
    - **High uncertainty**: {data['oct_relative_uncertainty']:.0f}% relative uncertainty
    """)
    
    st.subheader("Key Recommendations")
    
    recommendations = [
        (f"Use median (£{data['median_volume']:,.0f}K) not mean for 'typical day' reporting", "Mean inflated by outliers"),
        ("Apply non-parametric tests exclusively", "Data is non-normal"),
        (f"Build separate models for weekdays and weekends", f"{data['weekday_weekend_ratio']:.1f}× volume difference"),
        ("Maintain 30-40% buffer capacity", "High variability in daily volumes"),
        ("Implement automated data quality checks", "Prevent gaps like October 2023"),
        ("Explore weekend volume optimization", "Cost savings potential")
    ]
    
    for i, (rec, reason) in enumerate(recommendations, 1):
        st.write(f"{i}. **{rec}** - {reason}")

# ============================================================
# ANALYTICAL METHODOLOGY
# ============================================================
else:
    st.header("🔬 Analytical Methodology")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Complete Original Data",
        "📈 Question 1 Methodology",
        "📊 Question 2 Methodology",
        "🔮 Question 3 Methodology"
    ])
    
    with tab1:
        st.subheader("📊 Complete Original Dataset")
        
        # Show complete data
        display_df = df.copy()
        display_df['posting_date'] = display_df['posting_date'].dt.strftime('%Y-%m-%d')
        
        # Add pagination
        rows_per_page = 20
        total_pages = max(1, len(display_df) // rows_per_page + (1 if len(display_df) % rows_per_page > 0 else 0))
        
        page_number = st.number_input(
            f"Page (1-{total_pages})", 
            min_value=1, 
            max_value=total_pages, 
            value=1,
            step=1
        )
        
        start_idx = (page_number - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        
        st.dataframe(
            display_df.iloc[start_idx:end_idx],
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete Dataset (CSV)",
            data=csv,
            file_name="gbp_zar_transfer_data_2023.csv",
            mime="text/csv"
        )
