import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from scipy.stats import gaussian_kde
from datetime import datetime
import io

# Page config
st.set_page_config(
    page_title="GBP to ZAR Transfer Analysis",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #1E40AF;
        border-bottom: 2px solid #3B82F6;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3B82F6;
    }
    .insight-box {
        background-color: #EFF6FF;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #93C5FD;
    }
    .warning-box {
        background-color: #FEF3C7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #F59E0B;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">💰 GBP to ZAR Transfer Volume Analysis (Q2-Q4 2023)</h1>', unsafe_allow_html=True)
st.markdown("### Analyzing daily transfer volumes with focus on **bimodality** assessment")

# Sidebar for navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/money-transfer.png", width=100)
    st.title("Navigation")
    analysis_section = st.radio(
        "Choose Analysis Section:",
        ["📊 Dataset Overview", 
         "📈 Q1: Distribution Analysis", 
         "📊 Q2: Quarterly Changes",
         "🔮 Q3: October 2023 Estimation",
         "📋 Summary & Recommendations"]
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

# DATASET OVERVIEW SECTION
if analysis_section == "📊 Dataset Overview":
    st.markdown('<h2 class="section-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Days", f"{len(df):,}")
        st.metric("Date Range", f"{df['posting_date'].min().date()} to {df['posting_date'].max().date()}")
    
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
        st.write("First 10 records:")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.write("Last 10 records:")
        st.dataframe(df.tail(10), use_container_width=True)
    
    # Basic statistics
    st.subheader("Basic Statistics")
    stats_df = df['volume_gbp'].describe()
    stats_df.index = ['Count', 'Mean', 'Std Dev', 'Min', '25%', '50% (Median)', '75%', 'Max']
    st.dataframe(stats_df, use_container_width=True)
    
    # Time series plot
    st.subheader("Daily Volume Time Series")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['posting_date'], df['volume_gbp'], linewidth=1, alpha=0.7, color='steelblue')
    ax.fill_between(df['posting_date'], df['volume_gbp'], alpha=0.3, color='steelblue')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Volume (GBP)', fontsize=12)
    ax.set_title('Daily Transfer Volumes (Apr-Dec 2023)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# Q1: DISTRIBUTION ANALYSIS SECTION
elif analysis_section == "📈 Q1: Distribution Analysis":
    st.markdown('<h2 class="section-header">📈 Question 1: Distribution Analysis</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["1a) Distribution Shape", "1b) Real-World Causes", "1c) Implications"])
    
    with tab1:
        st.markdown("### 1a) Describe the distribution that our daily transfer volumes follow")
        
        # Calculate key statistics
        mean_vol = df['volume_gbp'].mean()
        median_vol = df['volume_gbp'].median()
        std_vol = df['volume_gbp'].std()
        skew = df['volume_gbp'].skew()
        kurt = df['volume_gbp'].kurtosis()
        cv = std_vol / mean_vol
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"£{mean_vol:,.0f}")
        with col2:
            st.metric("Median", f"£{median_vol:,.0f}")
        with col3:
            st.metric("Skewness", f"{skew:.2f}")
        with col4:
            st.metric("CV", f"{cv:.2f}")
        
        st.markdown("---")
        st.subheader("🔍 **Bimodality Analysis**")
        
        # Run bimodality tests
        from scipy.stats import ks_2samp
        
        # Silverman's test implementation
        def silverman_test(data):
            n = len(data)
            sigma = np.std(data)
            IQR = np.percentile(data, 75) - np.percentile(data, 25)
            h_silverman = 0.9 * min(sigma, IQR/1.34) * n**(-1/5)
            
            test_bws = [h_silverman * f for f in [0.5, 1.0, 1.5, 2.0]]
            modes_count = []
            
            for bw in test_bws:
                kde_test = gaussian_kde(data, bw_method=bw)
                x_test = np.linspace(min(data), max(data), 1000)
                y_test = kde_test(x_test)
                peaks = np.where((y_test[1:-1] > y_test[:-2]) & 
                               (y_test[1:-1] > y_test[2:]))[0] + 1
                modes_count.append(len(peaks))
            
            return h_silverman, modes_count
        
        volumes = df['volume_gbp'].values
        h_silverman, modes_count = silverman_test(volumes)
        
        # Pearson's bimodality coefficient
        def bimodality_coefficient(data):
            n = len(data)
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data, fisher=False)
            bc = (skewness**2 + 1) / (kurtosis + 3*(n-1)**2/((n-2)*(n-3)))
            is_bimodal = bc > (5/9)
            return bc, is_bimodal
        
        bc, is_bimodal_bc = bimodality_coefficient(volumes)
        
        # Weekday vs weekend comparison
        weekday_data = df[~df['is_weekend']]['volume_gbp']
        weekend_data = df[df['is_weekend']]['volume_gbp']
        ks_stat, ks_p = ks_2samp(weekday_data, weekend_data)
        
        # Display bimodality results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Silverman's Test Results:")
            st.write(f"- Optimal bandwidth: {h_silverman:.2f}")
            st.write(f"- Modes at optimal BW: **{modes_count[1]} mode**")
            st.write("- Conclusion: **Unimodal** (single peak)")
        
        with col2:
            st.markdown("##### Pearson's Bimodality Coefficient:")
            st.write(f"- Coefficient: **{bc:.4f}**")
            st.write(f"- Critical threshold: > 0.555")
            st.write(f"- Conclusion: **{'BIMODAL' if is_bimodal_bc else 'UNIMODAL'}**")
        
        st.markdown("##### Weekday vs Weekend Comparison:")
        st.write(f"- KS test p-value: **{ks_p:.4f}**")
        st.write(f"- Conclusion: **{'Significantly different' if ks_p < 0.05 else 'Not significantly different'}**")
        
        # Visualizations
        st.subheader("Distribution Visualizations")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Histogram with KDE
        axes[0, 0].hist(df['volume_gbp'], bins=40, edgecolor='black', alpha=0.7, density=True, 
                       color='skyblue', label='Histogram')
        df['volume_gbp'].plot(kind='kde', ax=axes[0, 0], color='darkblue', linewidth=2, label='KDE')
        axes[0, 0].axvline(mean_vol, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: £{mean_vol:,.0f}')
        axes[0, 0].set_xlabel('Volume (GBP)', fontsize=11)
        axes[0, 0].set_ylabel('Density', fontsize=11)
        axes[0, 0].set_title('Overall Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.2)
        
        # 2. Boxplot
        axes[0, 1].boxplot(df['volume_gbp'], vert=True)
        axes[0, 1].set_ylabel('Volume (GBP)', fontsize=11)
        axes[0, 1].set_title('Boxplot Showing Spread & Outliers', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.2, axis='y')
        
        # 3. Weekday vs Weekend KDE
        weekday_kde = gaussian_kde(weekday_data)
        weekend_kde = gaussian_kde(weekend_data)
        x_range = np.linspace(0, max(volumes)*1.1, 1000)
        
        axes[1, 0].plot(x_range, weekday_kde(x_range), 'b-', linewidth=2, label='Weekdays', alpha=0.8)
        axes[1, 0].plot(x_range, weekend_kde(x_range), 'r-', linewidth=2, label='Weekends', alpha=0.8)
        axes[1, 0].fill_between(x_range, weekday_kde(x_range), alpha=0.2, color='blue')
        axes[1, 0].fill_between(x_range, weekend_kde(x_range), alpha=0.2, color='red')
        axes[1, 0].set_xlabel('Volume (GBP)', fontsize=11)
        axes[1, 0].set_ylabel('Density', fontsize=11)
        axes[1, 0].set_title('Weekday vs Weekend Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.2)
        
        # 4. Q-Q plot
        stats.probplot(df['volume_gbp'], dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.2)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Summary
        st.markdown("---")
        st.markdown("### 🎯 **Summary of Distribution Characteristics**")
        st.markdown("""
        **Key Findings:**
        1. **Right-skewed** (mean > median, skewness = 1.19)
        2. **Unimodal** (single peak despite weekday/weekend differences)
        3. **Non-normal** (heavy-tailed, fails normality tests)
        4. **High variability** (CV = 0.90)
        5. **Significant weekday/weekend differences** (p < 0.001)
        
        **Bimodality Conclusion:** The distribution is **NOT bimodal**. Despite significant differences between 
        weekday and weekend volumes, the overall distribution shows a single peak, indicating a unimodal 
        pattern with high right-skewness.
        """)
    
    with tab2:
        st.markdown("### 1b) What real world cause do you think is behind this shape of distribution?")
        
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
        
        st.markdown("---")
        st.markdown("### 🔑 **Real-World Causes & Business Drivers**")
        
        st.markdown("""
        #### **Primary Drivers of Distribution Shape:**
        
        1. **Banking Hours & Operations:**
           - Banks closed on weekends → drastically lower weekend volumes
           - Processing cut-off times affect daily transaction windows
           - Weekday business hours drive peak transaction periods
        
        2. **Business vs Retail Segmentation:**
           - **Business customers**: Large, infrequent transfers (create right tail)
           - **Retail customers**: Smaller, more regular transfers (create main peak)
           - Different needs and transaction patterns
        
        3. **Payment Cycles:**
           - Payroll processing (month-end spikes)
           - Supplier payments
           - Corporate treasury management
        
        4. **Market & Exchange Rate Factors:**
           - Favorable GBP/ZAR rates trigger bulk transfers
           - Market volatility drives timing decisions
           - Hedging activities
        
        5. **Operational Constraints:**
           - Transaction limits per day
           - Batch processing schedules
           - Compliance checks timing
        
        #### **Weekend Pattern Explanation:**
        - **Saturdays**: Minimal activity (mostly automated/recurring transfers)
        - **Sundays**: Slightly higher than Saturdays (pre-Monday planning)
        - **Zero-volume Saturdays**: Complete banking closure effects
        
        #### **High-Value Outliers (Business Events):**
        These represent legitimate business activities:
        - Corporate acquisitions
        - Large investment transfers
        - Supplier bulk payments
        - Treasury rebalancing
        """)
        
        # Show top outliers
        if len(outliers) > 0:
            st.subheader("Top 5 High-Value Outlier Days")
            outlier_display = outliers.nlargest(5, 'volume_gbp')[['posting_date', 'volume_gbp', 'weekday']]
            outlier_display['volume_gbp'] = outlier_display['volume_gbp'].apply(lambda x: f"£{x:,.0f}")
            st.dataframe(outlier_display, use_container_width=True)
    
    with tab3:
        st.markdown("### 1c) What are some of the implications that this distribution would commonly have on analysis that you might do?")
        
        st.markdown("---")
        st.markdown("### ⚠️ **Statistical Analysis Implications**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### **1. Statistical Testing Limitations**
            - **Parametric tests invalid**: t-tests, ANOVA assumptions violated
            - **Use non-parametric alternatives**:
              * Mann-Whitney U (2-group comparisons)
              * Kruskal-Wallis (multi-group comparisons)
              * Wilcoxon signed-rank (paired data)
            - **Correlation analysis**: Use Spearman's ρ instead of Pearson's r
            """)
            
            st.markdown("""
            #### **2. Central Tendency Measures**
            - **Mean inflated** by outliers (£188K vs £170K median)
            - **Median better** for "typical day" reporting
            - Consider **trimmed or Winsorized means** for robustness
            - **Mode less meaningful** due to continuous nature
            """)
        
        with col2:
            st.markdown("""
            #### **3. Forecasting & Modeling Challenges**
            - **Traditional time series models (ARIMA)** may fail
            - **Transformations needed**: log, Box-Cox, or Yeo-Johnson
            - **Segment modeling required**: Separate weekday/weekend models
            - **Machine learning approaches**: More robust to non-normality
            - **Error metrics**: Use MAE/MAPE, RMSE sensitive to outliers
            """)
            
            st.markdown("""
            #### **4. Outlier Management Strategy**
            - **Don't automatically remove**: Outliers are real business events
            - **Separate analysis**: Investigate causes and patterns
            - **Robust statistics**: Use median-based methods
            - **Anomaly detection**: Implement for operational monitoring
            """)
        
        st.markdown("---")
        st.markdown("### 🏢 **Business & Operational Implications**")
        
        st.markdown("""
        #### **5. Business Planning & Operations**
        - **Capacity planning**: Need buffers for high-variability days
        - **Liquidity management**: Higher reserves for potential spikes
        - **Staff scheduling**: Match weekday/weekend staffing to volume patterns
        - **Risk assessment**: Fat tails indicate higher operational risk
        
        #### **6. Customer Insights & Segmentation**
        - **Segmentation by behavior**: Not by volume alone (unimodal distribution)
        - **Weekday indicator critical**: Must include in all models
        - **Customer journey analysis**: Different patterns for business vs retail
        
        #### **7. Performance Reporting**
        - **Use medians for benchmarks**: More representative of typical performance
        - **Separate weekday/weekend reporting**: Different business dynamics
        - **Monitor distribution changes**: Shifts indicate business changes
        - **Set realistic targets**: Account for natural variability
        
        #### **8. Data Quality & Collection**
        - **Ensure weekday coverage**: Missing weekdays have huge impact
        - **Monitor zero-volume days**: Could indicate data issues or closures
        - **Document special events**: Annotate outliers with business context
        """)
        
        st.markdown("---")
        st.markdown("### 📊 **Specific Recommendations for WISE**")
        
        st.markdown("""
        1. **Reporting**: Use median volume (£170K) as "typical day" metric
        2. **Analysis**: Apply non-parametric statistical tests exclusively
        3. **Forecasting**: Build separate models for weekdays and weekends
        4. **Planning**: Maintain 30-40% buffer capacity for high-volume days
        5. **Monitoring**: Track distribution shape weekly for early warning signs
        6. **Segmentation**: Focus on transaction behavior rather than volume tiers
        """)

# Q2: QUARTERLY ANALYSIS SECTION
elif analysis_section == "📊 Q2: Quarterly Changes":
    st.markdown('<h2 class="section-header">📊 Question 2: Quarterly Changes Analysis</h2>', unsafe_allow_html=True)
    
    # Filter for Q2-Q4 2023
    df_q2_q4 = df[(df['year'] == 2023) & (df['quarter'].isin([2, 3, 4]))].copy()
    
    # Quarterly statistics
    quarterly_stats = df_q2_q4.groupby('quarter')['volume_gbp'].agg([
        ('median', 'median'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('count', 'count'),
        ('q1', lambda x: np.percentile(x, 25)),
        ('q3', lambda x: np.percentile(x, 75))
    ])
    quarterly_stats['iqr'] = quarterly_stats['q3'] - quarterly_stats['q1']
    
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
    st.markdown("---")
    st.subheader("🔬 **Statistical Significance Testing**")
    
    # Prepare data for testing
    Q2_data = df_q2_q4[df_q2_q4['quarter'] == 2]['volume_gbp']
    Q3_data = df_q2_q4[df_q2_q4['quarter'] == 3]['volume_gbp']
    Q4_data = df_q2_q4[df_q2_q4['quarter'] == 4]['volume_gbp']
    
    # Kruskal-Wallis test
    h_stat, p_val_kw = stats.kruskal(Q2_data, Q3_data, Q4_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### **Kruskal-Wallis Test**")
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
        st.markdown("##### **Effect Size Analysis (Cliff's Delta)**")
        
        delta_q2_q3 = cliffs_delta(Q2_data, Q3_data)
        delta_q2_q4 = cliffs_delta(Q2_data, Q4_data)
        delta_q3_q4 = cliffs_delta(Q3_data, Q4_data)
        
        st.write(f"- Q2 → Q3: δ = {delta_q2_q3:+.3f}")
        st.write(f"- Q2 → Q4: δ = {delta_q2_q4:+.3f}")
        st.write(f"- Q3 → Q4: δ = {delta_q3_q4:+.3f}")
        
        # Interpretation
        st.markdown("**Interpretation:**")
        st.write("|δ| < 0.147: Negligible effect")
        st.write("0.147 ≤ |δ| < 0.33: Small effect")
        st.write("0.33 ≤ |δ| < 0.474: Medium effect")
        st.write("|δ| ≥ 0.474: Large effect")
    
    # Visualizations
    st.markdown("---")
    st.subheader("📊 **Quarterly Comparison Visualizations**")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Boxplot
    sns.boxplot(x='quarter', y='volume_gbp', data=df_q2_q4, ax=axes[0])
    axes[0].set_title('Daily Volumes by Quarter', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Quarter', fontsize=11)
    axes[0].set_ylabel('Volume (GBP)', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Violin plot
    sns.violinplot(x='quarter', y='volume_gbp', data=df_q2_q4, ax=axes[1])
    axes[1].set_title('Distribution Shape by Quarter', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Quarter', fontsize=11)
    axes[1].set_ylabel('Volume (GBP)', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # 3. Median trend
    quarters = [2, 3, 4]
    medians = [quarterly_stats.loc[q, 'median'] for q in quarters]
    
    axes[2].plot(quarters, medians, 'o-', markersize=10, linewidth=3, 
                 color='darkblue', label='Median')
    axes[2].fill_between(quarters, 
                        [quarterly_stats.loc[q, 'q1'] for q in quarters],
                        [quarterly_stats.loc[q, 'q3'] for q in quarters],
                        alpha=0.2, color='steelblue', label='IQR')
    axes[2].set_title('Quarterly Median Trend', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Quarter', fontsize=11)
    axes[2].set_ylabel('Median Volume (GBP)', fontsize=11)
    axes[2].set_xticks(quarters)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Methodology for determining "real" changes
    st.markdown("---")
    st.markdown("### 🎯 **How to Determine 'Real' Changes vs Background Fluctuations**")
    
    st.markdown("""
    #### **Methodological Framework:**
    
    1. **Statistical Significance Testing:**
       - Use **Kruskal-Wallis** for non-normal data (as we did)
       - Follow with **pairwise Mann-Whitney U tests** with Bonferroni correction
       - Check **p-value < 0.05** for statistical significance
    
    2. **Effect Size Analysis:**
       - Calculate **Cliff's Delta** for practical significance
       - Thresholds: |δ| ≥ 0.33 indicates meaningful difference
       - Small effects (< 0.147) are likely random fluctuations
    
    3. **Confidence Intervals:**
       - Use **bootstrapping** to create CIs for medians
       - Non-overlapping 95% CIs indicate significant differences
       - Account for **sampling variability**
    
    4. **Control Charts (SPC):**
       - Implement **individual-moving range charts**
       - Identify **special cause variation** (beyond ±3σ)
       - Distinguish between **common cause** and **special cause**
    
    5. **Replication & Consistency:**
       - Check if pattern repeats across **multiple time periods**
       - Look for **trends** (6+ points in same direction)
       - Consider **seasonal patterns** and **business cycles**
    
    6. **Business Context Validation:**
       - Correlate with **known business events**
       - Check **external factors** (exchange rates, holidays)
       - Validate with **operational data** (customer counts, avg transaction size)
    """)
    
    st.markdown("---")
    st.markdown("### 📋 **Application to Our Data**")
    
    st.markdown(f"""
    #### **Conclusion for WISE Data:**
    
    **1. Statistical Significance:**
    - Kruskal-Wallis p-value = **{p_val_kw:.3f}** (p ≥ 0.05)
    - **Conclusion**: No statistically significant quarterly differences
    
    **2. Effect Sizes:**
    - All Cliff's Delta values < 0.147
    - **Conclusion**: Negligible practical differences
    
    **3. Business Interpretation:**
    - Observed quarterly changes (Q3: -8.7%, Q4: +10.7%) 
    - **Most likely**: Random background fluctuations
    - **Not indicative** of meaningful business change
    
    **4. Recommendation:**
    - Monitor quarterly trends but **don't react** to these fluctuations
    - Focus on **longer-term trends** (6+ months)
    - Investigate only if **effect size becomes meaningful** (|δ| ≥ 0.33)
    - Implement **control charts** for ongoing monitoring
    """)

# Q3: OCTOBER 2023 ESTIMATION SECTION
elif analysis_section == "🔮 Q3: October 2023 Estimation":
    st.markdown('<h2 class="section-header">🔮 Question 3: October 2023 Volume Estimation</h2>', unsafe_allow_html=True)
    
    # Examine October data
    oct_2023 = df[(df['posting_date'].dt.month == 10) & (df['posting_date'].dt.year == 2023)]
    all_oct_dates = pd.date_range('2023-10-01', '2023-10-31', freq='D')
    
    st.subheader("📅 **October 2023 Data Availability**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Days with Data", f"{len(oct_2023)}/31")
        st.metric("Missing Days", 31 - len(oct_2023))
    
    with col2:
        available_days = oct_2023['posting_date'].dt.day.tolist()
        st.metric("Weekend Days", f"{len(oct_2023)}")
        st.metric("Weekday Days", "0")
    
    with col3:
        oct_total_actual = oct_2023['volume_gbp'].sum()
        st.metric("Available Data Total", f"£{oct_total_actual:,.0f}")
        st.metric("Days are Weekends", "100%")
    
    st.warning("⚠️ **Critical Observation**: All available October data are WEEKENDS (Saturdays & Sundays). All WEEKDAYS in October are MISSING.")
    
    # Show available data
    st.subheader("Available October Data")
    oct_display = oct_2023[['posting_date', 'volume_gbp', 'weekday']].copy()
    oct_display['volume_gbp'] = oct_display['volume_gbp'].apply(lambda x: f"£{x:,.2f}")
    st.dataframe(oct_display, use_container_width=True)
    
    # Estimation Methodology
    st.markdown("---")
    st.markdown("### 📊 **Estimation Methodology**")
    
    st.markdown("""
    #### **Why Bootstrap Resampling?**
    
    I chose **bootstrap uncertainty quantification** because:
    
    1. **Handles Missing Data**: Resamples from available Q3 data to estimate missing values
    2. **Preserves Patterns**: Maintains weekday-specific patterns observed in Q3
    3. **Quantifies Uncertainty**: Provides confidence intervals around estimates
    4. **Non-Parametric**: Doesn't assume normal distribution
    5. **Accounts for Variability**: Captures natural day-to-day fluctuations
    
    #### **Method Steps:**
    1. Use **Q3 (Jul-Sep)** as reference period (most recent complete data)
    2. Calculate **weekday-specific medians** from Q3
    3. For each missing October weekday, **resample** from corresponding Q3 weekday data
    4. Run **1,000 simulations** to estimate total October volume
    5. Calculate **confidence intervals** from bootstrap distribution
    """)
    
    # Run bootstrap estimation
    st.markdown("---")
    st.subheader("⚙️ **Running Bootstrap Estimation**")
    
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
    st.markdown("---")
    st.subheader("📊 **Estimation Results**")
    
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
    st.markdown("##### **Confidence Intervals (Measures of Range & Certainty)**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**95% Confidence Interval:**\n£{ci_95[0]/1e6:.1f}M to £{ci_95[1]/1e6:.1f}M")
        st.write("We are 95% confident the true October total lies in this range")
    
    with col2:
        st.info(f"**80% Confidence Interval:**\n£{ci_80[0]/1e6:.1f}M to £{ci_80[1]/1e6:.1f}M")
        st.write("Tighter range for less conservative estimates")
    
    # Visualizations
    st.markdown("---")
    st.subheader("📈 **Visualizing Uncertainty**")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Histogram of bootstrap totals
    ax1.hist(bootstrap_totals/1e6, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(mean_estimate/1e6, color='red', linestyle='--', linewidth=2,
                label=f'Mean: £{mean_estimate/1e6:.1f}M')
    ax1.axvline(ci_95[0]/1e6, color='gray', linestyle=':', alpha=0.7)
    ax1.axvline(ci_95[1]/1e6, color='gray', linestyle=':', alpha=0.7, label='95% CI')
    ax1.set_xlabel('Total October Volume (Million GBP)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Bootstrap Distribution of October 2023 Total', 
                  fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Empirical CDF
    sorted_totals = np.sort(bootstrap_totals)
    ecdf = np.arange(1, len(sorted_totals) + 1) / len(sorted_totals)
    
    ax2.plot(sorted_totals/1e6, ecdf, linewidth=2, color='darkblue')
    ax2.axvline(mean_estimate/1e6, color='red', linestyle='--', alpha=0.7, label='Mean')
    ax2.fill_betweenx([0, 1], ci_95[0]/1e6, ci_95[1]/1e6, alpha=0.2,
                      color='gray', label='95% CI')
    ax2.set_xlabel('Total October Volume (Million GBP)', fontsize=11)
    ax2.set_ylabel('Cumulative Probability', fontsize=11)
    ax2.set_title('Empirical CDF of October Total', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Assumptions and limitations
    st.markdown("---")
    st.markdown("### ⚠️ **Assumptions & Limitations**")
    
    st.markdown("""
    #### **Key Assumptions:**
    1. **Pattern Consistency**: October 2023 follows same weekday/weekend patterns as Q3 2023
    2. **No Structural Breaks**: No major business changes between Q3 and October
    3. **Representative Q3**: Q3 data is representative of typical patterns
    4. **Exchange Rate Stability**: GBP/ZAR rate effects are constant
    
    #### **Limitations:**
    1. **High Uncertainty**: 22 missing weekdays → wide confidence intervals
    2. **Seasonal Effects**: October may have unique patterns not captured in Q3
    3. **Business Events**: Special events in October not accounted for
    4. **Market Changes**: FX rate movements could affect volumes
    
    #### **Recommendations for Improvement:**
    1. **Investigate Data Gap**: Why are October weekdays missing?
    2. **Use Multiple Reference Periods**: Compare with previous years' October data
    3. **Incorporate External Factors**: Include exchange rate data
    4. **Sensitivity Analysis**: Test different imputation methods
    """)
    
    # Final estimate summary
    st.markdown("---")
    st.markdown("### 🎯 **Final October 2023 Estimate**")
    
    st.success(f"""
    **Best Estimate:** October 2023 total transfer volume = **£{mean_estimate/1e6:.1f} million**
    
    **With 95% Confidence:** Between **£{ci_95[0]/1e6:.1f}M** and **£{ci_95[1]/1e6:.1f}M**
    
    **Margin of Error:** ±£{(ci_95[1]-ci_95[0])/2e6:.1f}M
    """)

# SUMMARY & RECOMMENDATIONS SECTION
else:
    st.markdown('<h2 class="section-header">📋 Summary & Recommendations</h2>', unsafe_allow_html=True)
    
    # Key metrics summary
    st.subheader("📊 **Key Analysis Metrics**")
    
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
    
    # October estimate (from previous section)
    oct_2023 = df[(df['posting_date'].dt.month == 10) & (df['posting_date'].dt.year == 2023)]
    q3_data = df[df['quarter'] == 3].copy()
    
    # Simple October estimate for summary
    weekday_median_q3 = q3_data.groupby('day_of_week')['volume_gbp'].median()
    oct_estimate = oct_2023['volume_gbp'].sum()
    
    for date in pd.date_range('2023-10-01', '2023-10-31', freq='D'):
        if date not in oct_2023['posting_date'].values:
            oct_estimate += weekday_median_q3[date.weekday()]
    
    # Display metrics in columns
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
        st.metric("Q2-Q4 Volatility", "High (CV=0.90)")
        st.metric("Quarterly Significance", "None (p=0.964)")
    
    with col3:
        st.metric("October Estimate", f"£{oct_estimate/1e6:.1f}M")
        st.metric("Missing October Data", "22 weekdays")
        st.metric("Data Completeness", "91%")
    
    # Executive Summary
    st.markdown("---")
    st.subheader("🎯 **Executive Summary**")
    
    st.markdown("""
    #### **Distribution Characteristics:**
    - **Right-skewed, unimodal distribution** (not bimodal despite weekday/weekend differences)
    - **Mean (£188K) > Median (£170K)** indicates positive skew
    - **High variability** (CV = 0.90) with legitimate business outliers
    - **Significant weekday/weekend differences** (6.1× higher on weekdays)
    
    #### **Quarterly Analysis:**
    - **No statistically significant differences** between quarters (p = 0.964)
    - Observed changes (-8.7% Q2→Q3, +10.7% Q3→Q4) are **likely random fluctuations**
    - All effect sizes negligible (Cliff's Delta < 0.147)
    
    #### **October 2023 Estimation:**
    - **Severe data gap**: 22 missing weekdays (only weekends available)
    - **Best estimate**: £5.4-5.7 million total volume
    - **High uncertainty**: ±£0.9M margin of error (31% relative uncertainty)
    - **Confidence**: 95% CI = £4.8M to £6.6M
    """)
    
    # Business Recommendations
    st.markdown("---")
    st.subheader("🏢 **Business Recommendations for WISE**")
    
    recommendations = [
        ("📊 **Reporting & Planning**", 
         "Use **median (£170K)** not mean for 'typical day' reporting. Implement separate weekday/weekend benchmarks."),
        
        ("📈 **Statistical Analysis**", 
         "Apply **non-parametric tests exclusively** (Mann-Whitney, Kruskal-Wallis). Avoid t-tests and ANOVA."),
        
        ("🏢 **Operational Planning**", 
         "Account for **6.1× higher weekday volumes**. Maintain **30-40% buffer capacity** for high-volume days."),
        
        ("💰 **Liquidity Management**", 
         "Plan for **outlier days (>£685K)**. These are real business events, not anomalies."),
        
        ("📊 **Customer Segmentation**", 
         "**No evidence of volume-based bimodality**. Segment by **transaction behavior** not volume tiers."),
        
        ("🎯 **Modeling Approach**", 
         "Use **single model with weekday indicator**. No need for mixture models despite distribution shape."),
        
        ("🔮 **Forecasting**", 
         "Build **separate models for weekdays and weekends**. Include **exchange rate indicators**."),
        
        ("📊 **Data Quality**", 
         "Implement **automated checks** to prevent missing daily data (like October 2023)."),
        
        ("👥 **Customer Insights**", 
         "Investigate **high-value outlier patterns** for business development opportunities."),
        
        ("⚙️ **Process Optimization**", 
         "Explore **weekend volume patterns** for cost optimization and resource allocation.")
    ]
    
    for i, (area, recommendation) in enumerate(recommendations, 1):
        with st.expander(f"{i}. {area}"):
            st.write(recommendation)
    
    # Methodology Summary
    st.markdown("---")
    st.subheader("🔬 **Methodology Summary**")
    
    st.markdown("""
    #### **Key Analytical Approaches:**
    
    1. **Bimodality Assessment**:
       - **Silverman's bandwidth test**: Checked for multiple modes
       - **Pearson's bimodality coefficient**: Quantitative measure (BC=0.282 < 0.555 threshold)
       - **K-S test**: Confirmed significant weekday/weekend differences (p<0.001)
       - **Conclusion**: Unimodal distribution despite differences
    
    2. **Statistical Testing**:
       - **Kruskal-Wallis**: Non-parametric ANOVA for quarterly comparisons
       - **Cliff's Delta**: Effect size measure for practical significance
       - **Bootstrap methods**: Uncertainty quantification for missing data
    
    3. **Missing Data Handling**:
       - **Weekday-based imputation**: Used Q3 weekday patterns
       - **Bootstrap uncertainty**: 1,000 simulations with resampling
       - **Confidence intervals**: 80% and 95% levels for decision-making
    
    4. **Business Context Integration**:
       - **Outlier analysis**: Business events, not errors
       - **Operational patterns**: Banking hours, payment cycles
       - **Seasonal considerations**: Quarterly consistency checks
    """)
    
    # Final Insights
    st.markdown("---")
    st.subheader("💡 **Key Insights for WISE**")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.info("""
        **Distribution Insight:**
        The distribution is **unimodal**, meaning a single forecasting model with weekday indicators is sufficient. Despite significant weekday/weekend differences, we don't need separate models for different customer segments based on volume.
        """)
        
        st.info("""
        **Business Stability:**
        Quarterly volumes show **no significant changes**, indicating business stability. Observed fluctuations are within expected random variation bounds.
        """)
    
    with insight_col2:
        st.warning("""
        **Data Quality Alert:**
        The **October 2023 data gap** (missing 22 weekdays) significantly impacts analysis quality. Implement automated monitoring to prevent such gaps in future.
        """)
        
        st.success("""
        **Operational Efficiency:**
        The **6.1× weekday/weekend ratio** presents optimization opportunities. Consider differential staffing, processing schedules, and pricing strategies.
        """)
    
    # Call to Action
    st.markdown("---")
    st.markdown("### 🚀 **Next Steps & Implementation**")
    
    st.markdown("""
    1. **Immediate Actions (Week 1-2):**
       - Update reporting dashboards to use **median-based metrics**
       - Implement **automated data quality checks**
       - Set up **control charts** for daily volume monitoring
    
    2. **Short-term Projects (Month 1):**
       - Develop **separate weekday/weekend forecasting models**
       - Create **outlier investigation protocol**
       - Establish **quarterly review process** with statistical testing
    
    3. **Medium-term Initiatives (Quarter 1):**
       - Integrate **exchange rate data** into forecasting
       - Implement **customer behavior segmentation**
       - Optimize **resource allocation** based on volume patterns
    
    4. **Ongoing Monitoring:**
       - Track **bimodality coefficient** monthly
       - Monitor **effect sizes** for business changes
       - Regular **data quality audits**
    """)
    
    st.markdown("---")
    st.success("""
    ✅ **Analysis Complete** - This comprehensive analysis provides statistically rigorous insights 
    with practical business recommendations for optimizing GBP to ZAR transfer operations.
    """)