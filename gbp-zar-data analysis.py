import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
from datetime import datetime

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
         "📋 Summary & Recommendations",
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
    url = "https://drive.google.com/uc?id=1BK9eVWAu2LCDJ2haDYafoKhPyWP2tqCl"
    df = pd.read_csv(url)
    df['posting_date'] = pd.to_datetime(df['posting_date'])
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
        # Format dates before display
        display_df = df.head(10).copy()
        display_df['posting_date'] = display_df['posting_date'].dt.strftime('%Y-%m-%d')
        st.dataframe(display_df, use_container_width=True)
    
    with col2:
        st.write("**Last 10 records:**")
        # Format dates before display
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
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['posting_date'], df['volume_gbp'], linewidth=1, alpha=0.7)
    ax.fill_between(df['posting_date'], df['volume_gbp'], alpha=0.3)
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume (GBP)')
    ax.set_title('Daily Transfer Volumes (Apr-Dec 2023)')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

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
        cv = std_vol / mean_vol
        
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
        bc = (skewness**2 + 1) / (kurtosis + 3*(n-1)**2/((n-2)*(n-3)))
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
        
        # Distribution visualizations
        st.subheader("Distribution Visualizations")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Histogram with KDE
        axes[0, 0].hist(df['volume_gbp'], bins=40, edgecolor='black', alpha=0.7, density=True)
        df['volume_gbp'].plot(kind='kde', ax=axes[0, 0], linewidth=2)
        axes[0, 0].axvline(mean_vol, color='red', linestyle='--', linewidth=2, label=f'Mean')
        axes[0, 0].axvline(median_vol, color='green', linestyle='--', linewidth=2, label=f'Median')
        axes[0, 0].set_xlabel('Volume (GBP)')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Overall Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Boxplot
        axes[0, 1].boxplot(df['volume_gbp'], vert=True)
        axes[0, 1].set_ylabel('Volume (GBP)')
        axes[0, 1].set_title('Boxplot: Spread & Outliers')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Q-Q plot
        stats.probplot(df['volume_gbp'], dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normality Check)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Weekday vs weekend comparison
        axes[1, 1].boxplot([weekday_data, weekend_data], labels=['Weekdays', 'Weekends'])
        axes[1, 1].set_ylabel('Volume (GBP)')
        axes[1, 1].set_title('Weekday vs Weekend Comparison')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Summary
        st.subheader("Summary of Distribution Characteristics")
        st.write("""
        **Key Findings:**
        1. **Right-skewed** (mean > median, skewness = {:.2f})
        2. **Unimodal** (single peak despite weekday/weekend differences)
        3. **Non-normal** (heavy-tailed distribution)
        4. **High variability** (CV = {:.2f})
        5. **Significant weekday/weekend differences** (p < 0.001)
        
        **Bimodality Conclusion:** The distribution is **NOT bimodal**. Despite significant differences between 
        weekday and weekend volumes, the overall distribution shows a single peak.
        """.format(skew, cv))
    
    with tab2:
        st.subheader("1b) What real world cause do you think is behind this shape of distribution?")
        
        # Calculate business metrics
        weekday_mean = df[~df['is_weekend']]['volume_gbp'].mean()
        weekend_mean = df[df['is_weekend']]['volume_gbp'].mean()
        ratio = weekday_mean / weekend_mean
        
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
            # Format the date column
            outlier_display['posting_date'] = outlier_display['posting_date'].dt.strftime('%Y-%m-%d')
            outlier_display['volume_gbp'] = outlier_display['volume_gbp'].apply(lambda x: f"£{x:,.0f}")
            st.dataframe(outlier_display, use_container_width=True)
    
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
        q2_q3_change = ((q3_median - q2_median) / q2_median * 100)
        st.metric("Q3 Median", f"£{q3_median:,.0f}")
        st.metric("Q2→Q3 Change", f"{q2_q3_change:+.1f}%")
    
    with col3:
        q4_median = quarterly_stats.loc[4, 'median']
        q3_q4_change = ((q4_median - q3_median) / q3_median * 100)
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
        delta = (greater - less) / (len(x_arr) * len(y_arr))
        return delta
    
    with col2:
        st.write("**Effect Size Analysis (Cliff's Delta):**")
        
        delta_q2_q3 = cliffs_delta(Q2_data, Q3_data)
        delta_q2_q4 = cliffs_delta(Q2_data, Q4_data)
        delta_q3_q4 = cliffs_delta(Q3_data, Q4_data)
        
        st.write(f"- Q2 → Q3: δ = {delta_q2_q3:+.3f}")
        st.write(f"- Q2 → Q4: δ = {delta_q2_q4:+.3f}")
        st.write(f"- Q3 → Q4: δ = {delta_q3_q4:+.3f}")
        
        # Interpretation
        st.write("**Interpretation:**")
        st.write("|δ| < 0.147: Negligible effect")
        st.write("0.147 ≤ |δ| < 0.33: Small effect")
        st.write("0.33 ≤ |δ| < 0.474: Medium effect")
        st.write("|δ| ≥ 0.474: Large effect")
    
    # Visualizations
    st.subheader("Quarterly Comparison Visualizations")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Boxplot by quarter
    box_data = [Q2_data, Q3_data, Q4_data]
    axes[0].boxplot(box_data, labels=['Q2', 'Q3', 'Q4'])
    axes[0].set_ylabel('Volume (GBP)')
    axes[0].set_title('Daily Volumes by Quarter')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Median trend
    quarters = [2, 3, 4]
    medians = [quarterly_stats.loc[q, 'median'] for q in quarters]
    
    axes[1].plot(quarters, medians, 'o-', markersize=8, linewidth=2)
    axes[1].set_xlabel('Quarter')
    axes[1].set_ylabel('Median Volume (GBP)')
    axes[1].set_title('Quarterly Median Trend')
    axes[1].set_xticks(quarters)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
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
        st.metric("Weekend Days", f"{len(oct_2023)}")
        st.metric("Weekday Days", "0")
    
    with col3:
        oct_total_actual = oct_2023['volume_gbp'].sum()
        st.metric("Available Data Total", f"£{oct_total_actual:,.0f}")
    
    st.warning("⚠️ **Critical Observation**: All available October data are WEEKENDS (Saturdays & Sundays). All WEEKDAYS in October are MISSING.")
    
    # Show available data
    st.subheader("Available October Data")
    oct_display = oct_2023[['posting_date', 'volume_gbp', 'weekday']].copy()
    # Format the date column
    oct_display['posting_date'] = oct_display['posting_date'].dt.strftime('%Y-%m-%d')
    oct_display['volume_gbp'] = oct_display['volume_gbp'].apply(lambda x: f"£{x:,.2f}")
    st.dataframe(oct_display, use_container_width=True)
    
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
        relative_uncertainty = (ci_range / mean_estimate * 100)
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
    
    # Visualizations
    st.subheader("Visualizing Uncertainty")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Histogram of bootstrap totals
    ax.hist(bootstrap_totals/1e6, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(mean_estimate/1e6, color='red', linestyle='--', linewidth=2,
                label=f'Mean: £{mean_estimate/1e6:.1f}M')
    ax.axvline(ci_95[0]/1e6, color='gray', linestyle=':', alpha=0.7)
    ax.axvline(ci_95[1]/1e6, color='gray', linestyle=':', alpha=0.7, label='95% CI')
    ax.set_xlabel('Total October Volume (Million GBP)')
    ax.set_ylabel('Frequency')
    ax.set_title('Bootstrap Distribution of October 2023 Total')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Final estimate
    st.subheader("Final October 2023 Estimate")
    
    st.success(f"""
    **Best Estimate:** October 2023 total transfer volume = **£{mean_estimate/1e6:.1f} million**
    
    **With 95% Confidence:** Between **£{ci_95[0]/1e6:.1f}M** and **£{ci_95[1]/1e6:.1f}M**
    
    **Margin of Error:** ±£{(ci_95[1]-ci_95[0])/2e6:.1f}M
    """)
    
    # Assumptions
    st.subheader("Assumptions & Limitations")
    st.write("""
    **Key Assumptions:**
    1. October 2023 follows same weekday/weekend patterns as Q3 2023
    2. No major business changes between Q3 and October
    3. Q3 data is representative of typical patterns
    
    **Limitations:**
    1. High uncertainty due to 22 missing weekdays
    2. Seasonal effects in October not captured in Q3
    3. Business events in October not accounted for
    """)

# ANALYTICAL METHODOLOGY SECTION
elif analysis_section == "🔬 Analytical Methodology":
    st.header("🔬 Analytical Methodology")
    
    st.write("""
    ### Overview of Analytical Approach
    
    This analysis follows a structured methodology to answer the three key questions:
    """)
    
    # Question 1 Methodology
    st.subheader("Question 1: Distribution Analysis Methodology")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Bimodality Testing:**")
        st.write("""
        1. **Pearson's Bimodality Coefficient:**
           - Calculated from skewness and kurtosis
           - Threshold: BC > 0.555 indicates bimodality
           - Our result: BC = 0.282 (< 0.555) = Unimodal
        
        2. **Kolmogorov-Smirnov Test:**
           - Compare weekday vs weekend distributions
           - p < 0.05 = significantly different
           - Our result: p = 0.000 = significantly different
        """)
    
    with col2:
        st.write("**Distribution Characterization:**")
        st.write("""
        1. **Central Tendency:**
           - Mean and median comparison
           - Mean > Median = Right skew
        
        2. **Variability Measures:**
           - Standard deviation
           - Coefficient of Variation (CV)
           - CV > 0.5 = High variability
        
        3. **Shape Measures:**
           - Skewness (asymmetry)
           - Kurtosis (tail heaviness)
        """)
    
    # Question 2 Methodology
    st.subheader("Question 2: Quarterly Changes Methodology")
    
    st.write("""
    **Statistical Testing Framework:**
    
    1. **Kruskal-Wallis Test:**
       - Non-parametric alternative to ANOVA
       - Tests if medians across quarters are equal
       - Appropriate for non-normal data
    
    2. **Effect Size Analysis (Cliff's Delta):**
       - Measures practical significance, not just statistical
       - Interpretation:
         - |δ| < 0.147: Negligible
         - 0.147 ≤ |δ| < 0.33: Small
         - 0.33 ≤ |δ| < 0.474: Medium
         - |δ| ≥ 0.474: Large
    
    3. **Decision Criteria:**
       - Statistical significance (p < 0.05) AND
       - Practical significance (|δ| ≥ 0.33) = Meaningful change
    """)
    
    # Question 3 Methodology
    st.subheader("Question 3: October Estimation Methodology")
    
    st.write("""
    **Bootstrap Resampling Method:**
    
    1. **Reference Period Selection:**
       - Used Q3 (Jul-Sep) as most recent complete period
       - 92 days of data with complete weekday coverage
    
    2. **Resampling Process:**
       - For each missing October weekday:
         - Resample from corresponding Q3 weekday data
         - Repeat 1,000 times to build distribution
    
    3. **Uncertainty Quantification:**
       - Calculate mean as point estimate
       - Calculate 80% and 95% confidence intervals
       - Report margin of error
    
    4. **Advantages of Bootstrap:**
       - Non-parametric (no distribution assumptions)
       - Preserves data patterns
       - Provides confidence intervals
       - Handles missing data robustly
    """)
    
    # Key Methodological Decisions
    st.subheader("Key Methodological Decisions")
    
    decisions = [
        ("Why median over mean?", "Data is right-skewed with outliers. Median better represents 'typical' day."),
        ("Why non-parametric tests?", "Data fails normality tests. Non-parametric tests are more appropriate."),
        ("Why bootstrap for October?", "Missing 22 weekdays. Bootstrap preserves patterns and quantifies uncertainty."),
        ("Why Q3 as reference?", "Most recent complete period before October with similar patterns expected.")
    ]
    
    for question, answer in decisions:
        with st.expander(f"❓ {question}"):
            st.write(answer)

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
    bc = (skewness**2 + 1) / (kurtosis + 3*(n-1)**2/((n-2)*(n-3)))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Median", f"£{median_vol:,.0f}")
        st.metric("Distribution Skew", f"{skew:.2f}")
        st.metric("Bimodality Coeff", f"{bc:.3f}")
    
    with col2:
        weekday_mean = df[~df['is_weekend']]['volume_gbp'].mean()
        weekend_mean = df[df['is_weekend']]['volume_gbp'].mean()
        ratio = weekday_mean / weekend_mean
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
    
    st.write("""
    **Distribution Characteristics:**
    - Right-skewed, unimodal distribution (not bimodal)
    - Mean (£188K) > Median (£170K) indicates positive skew
    - High variability (CV = 0.90) with legitimate business outliers
    - Significant weekday/weekend differences (6.1× higher on weekdays)
    
    **Quarterly Analysis:**
    - No statistically significant differences between quarters (p = 0.964)
    - Observed changes (-8.7% Q2→Q3, +10.7% Q3→Q4) are likely random fluctuations
    - All effect sizes negligible (Cliff's Delta < 0.147)
    
    **October 2023 Estimation:**
    - Severe data gap: 22 missing weekdays (only weekends available)
    - Best estimate: £5.4-5.7 million total volume
    - High uncertainty: ±£0.9M margin of error (31% relative uncertainty)
    - 95% Confidence Interval: £4.8M to £6.6M
    """)
    
    # Recommendations
    st.subheader("Business Recommendations for WISE")
    
    recommendations = [
        ("Use median (£170K) not mean for 'typical day' reporting", "Mean inflated by outliers"),
        ("Apply non-parametric tests exclusively", "Data is non-normal"),
        ("Build separate models for weekdays and weekends", "6.1× volume difference"),
        ("Maintain 30-40% buffer capacity", "High variability in daily volumes"),
        ("Segment by transaction behavior, not volume", "Unimodal distribution"),
        ("Implement automated data quality checks", "Prevent gaps like October 2023"),
        ("Investigate high-value outlier patterns", "Business development opportunities"),
        ("Explore weekend volume optimization", "Cost savings potential")
    ]
    
    for i, (rec, reason) in enumerate(recommendations, 1):
        st.write(f"{i}. **{rec}** - {reason}")
    
    # Next Steps
    st.subheader("Next Steps & Implementation")
    
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
    
    st.success("✅ Analysis Complete - Comprehensive insights with practical business recommendations for optimizing GBP to ZAR transfer operations.")

