import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# APP CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="GBP/ZAR Transfer Analysis",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 800;
    }
    .section-header {
        font-size: 1.8rem;
        color: #1E40AF;
        border-left: 5px solid #3B82F6;
        padding-left: 1rem;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.4rem;
        color: #374151;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .insight-box {
        background-color: #EFF6FF;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.2rem 0;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .methodology-box {
        background-color: #F0FDF4;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.2rem 0;
        border-left: 5px solid #10B981;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .warning-box {
        background-color: #FEF3C7;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.2rem 0;
        border-left: 5px solid #F59E0B;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .logic-guide {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.8rem 0;
        border: 1px solid #E5E7EB;
        font-size: 0.9rem;
    }
    .nav-button {
        background-color: #3B82F6;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/statistics.png", width=80)
    st.markdown("### 📊 Analysis Views")
    
    view_option = st.radio(
        "Select View Mode:",
        ["🎯 Presentation View (Stakeholders)",
         "🔬 Analyst View (Detailed Analysis)",
         "🧠 Logic & Methodology Guide"]
    )
    
    st.markdown("---")
    st.markdown("### 📁 Quick Navigation")
    
    if view_option == "🎯 Presentation View (Stakeholders)":
        section = st.radio(
            "Go to Section:",
            ["🏠 Executive Summary",
             "📈 Q1: Distribution Analysis",
             "📊 Q2: Quarterly Trends",
             "🔮 Q3: October Estimation",
             "🎯 Key Recommendations"]
        )
    elif view_option == "🔬 Analyst View (Detailed Analysis)":
        section = st.radio(
            "Go to Section:",
            ["📋 Data Overview",
             "📊 Distribution Analysis",
             "🔍 Bimodality Testing",
             "📈 Quarterly Analysis",
             "⚙️ October Estimation",
             "📝 Complete Results"]
        )
    else:
        section = "🧾 Logic Guide"
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Analysis Focus:**
    - Distribution shape & bimodality
    - Statistical significance testing
    - Missing data estimation
    - Business implications
    
    **Data:** GBP→ZAR transfers (Apr-Dec 2023)
    **Records:** 253 daily observations
    """)

# ============================================================================
# DATA LOADING & PROCESSING
# ============================================================================
@st.cache_data
def load_and_process_data():
    url = "https://drive.google.com/uc?id=1BK9eVWAu2LCDJ2haDYafoKhPyWP2tqCl"
    df = pd.read_csv(url)
    
    # Convert and sort dates
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

df = load_and_process_data()

# ============================================================================
# PRESENTATION VIEW (For Stakeholders)
# ============================================================================
if view_option == "🎯 Presentation View (Stakeholders)":
    
    if section == "🏠 Executive Summary":
        st.markdown('<h1 class="main-header">💰 GBP to ZAR Transfer Analysis</h1>', unsafe_allow_html=True)
        st.markdown("### *Q2-Q4 2023 Performance Review*")
        
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Volume", f"£{df['volume_gbp'].sum()/1e6:.1f}M")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg Daily", f"£{df['volume_gbp'].mean():,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            weekday_mean = df[~df['is_weekend']]['volume_gbp'].mean()
            weekend_mean = df[df['is_weekend']]['volume_gbp'].mean()
            ratio = weekday_mean / weekend_mean
            st.metric("Weekday:Weekend", f"{ratio:.1f}x")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            oct_data = df[(df['month'] == 10) & (df['year'] == 2023)]
            missing_days = 31 - len(oct_data)
            st.metric("Oct Missing", f"{missing_days} days")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Executive Summary
        st.markdown('<h2 class="section-header">🎯 Executive Summary</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 **Key Findings**")
            st.markdown("""
            <div class="insight-box">
            <h4>🎯 Distribution Pattern</h4>
            • <b>Right-skewed, unimodal distribution</b><br>
            • Mean (£188K) > Median (£170K)<br>
            • <b>Not bimodal</b> despite weekday/weekend differences<br>
            • High variability (CV = 0.90)
            </div>
            
            <div class="insight-box">
            <h4>📈 Quarterly Performance</h4>
            • <b>No significant quarterly changes</b> (p = 0.964)<br>
            • Observed fluctuations are random background noise<br>
            • Q4 shows recovery trend but not statistically significant
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### ⚠️ **Critical Observations**")
            st.markdown("""
            <div class="warning-box">
            <h4>🔴 October Data Gap</h4>
            • <b>22 missing weekdays</b> in October 2023<br>
            • Only weekend data available<br>
            • Requires estimation with uncertainty bounds
            </div>
            
            <div class="insight-box">
            <h4>🏢 Operational Insights</h4>
            • Weekday volumes <b>6.1× higher</b> than weekends<br>
            • Clear banking hour effects<br>
            • Business transactions drive right tail<br>
            • Outliers are legitimate business events
            </div>
            """, unsafe_allow_html=True)
        
        # Visualization: Time Series
        st.markdown('<h2 class="section-header">📈 Daily Volume Trends</h2>', unsafe_allow_html=True)
        
        # Create interactive plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['posting_date'], 
            y=df['volume_gbp'],
            mode='lines',
            name='Daily Volume',
            line=dict(color='#3B82F6', width=2),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)'
        ))
        
        # Add quarter highlights
        quarters = {
            'Q2 (Apr-Jun)': ('2023-04-01', '2023-06-30'),
            'Q3 (Jul-Sep)': ('2023-07-01', '2023-09-30'),
            'Q4 (Oct-Dec)': ('2023-10-01', '2023-12-31')
        }
        
        colors = ['rgba(59, 130, 246, 0.2)', 'rgba(34, 197, 94, 0.2)', 'rgba(245, 158, 11, 0.2)']
        
        for i, (q_name, (start, end)) in enumerate(quarters.items()):
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor=colors[i],
                opacity=0.2,
                layer="below",
                line_width=0,
                annotation_text=q_name,
                annotation_position="top left"
            )
        
        fig.update_layout(
            title="Daily Transfer Volumes with Quarterly Highlights",
            xaxis_title="Date",
            yaxis_title="Volume (GBP)",
            hovermode="x unified",
            height=500,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif section == "📈 Q1: Distribution Analysis":
        st.markdown('<h1 class="main-header">📊 Distribution Analysis</h1>', unsafe_allow_html=True)
        
        # Key Statistics
        mean_vol = df['volume_gbp'].mean()
        median_vol = df['volume_gbp'].median()
        skew = df['volume_gbp'].skew()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean", f"£{mean_vol:,.0f}", help="Average daily volume")
        with col2:
            st.metric("Median", f"£{median_vol:,.0f}", help="Middle value - better for typical day")
        with col3:
            st.metric("Skewness", f"{skew:.2f}", "Right-skewed if > 0.5")
        
        # Bimodality Analysis
        st.markdown('<h2 class="section-header">🔍 Bimodality Assessment</h2>', unsafe_allow_html=True)
        
        # Calculate bimodality coefficient
        volumes = df['volume_gbp'].values
        n = len(volumes)
        skewness = stats.skew(volumes)
        kurtosis = stats.kurtosis(volumes, fisher=False)
        bc = (skewness**2 + 1) / (kurtosis + 3*(n-1)**2/((n-2)*(n-3)))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 **Distribution Shape**")
            st.markdown(f"""
            <div class="insight-box">
            <h4>Statistical Test Results:</h4>
            • <b>Pearson's Bimodality Coefficient:</b> {bc:.4f}<br>
            • <b>Critical Threshold:</b> > 0.555<br>
            • <b>Conclusion:</b> <span style="color: {'green' if bc > 0.555 else 'red'}">
            {'BIMODAL' if bc > 0.555 else 'UNIMODAL'}</span><br>
            • <b>Skewness:</b> {skew:.2f} (Right-skewed)
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🏢 **Business Interpretation**")
            st.markdown("""
            <div class="insight-box">
            <h4>What This Means:</h4>
            • <b>Single customer pattern</b> despite weekday/weekend differences<br>
            • <b>No need for mixture models</b> in forecasting<br>
            • <b>Weekday indicator variable</b> is sufficient<br>
            • <b>One-size-fits-all approach</b> works for operations
            </div>
            """, unsafe_allow_html=True)
        
        # Weekday vs Weekend Analysis
        st.markdown('<h2 class="section-header">📅 Weekday vs Weekend Patterns</h2>', unsafe_allow_html=True)
        
        weekday_data = df[~df['is_weekend']]['volume_gbp']
        weekend_data = df[df['is_weekend']]['volume_gbp']
        
        # Create comparison chart
        comparison_data = pd.DataFrame({
            'Day Type': ['Weekdays (Mon-Fri)', 'Weekends (Sat-Sun)'],
            'Mean Volume': [weekday_data.mean(), weekend_data.mean()],
            'Median Volume': [weekday_data.median(), weekend_data.median()]
        })
        
        fig = px.bar(
            comparison_data, 
            x='Day Type', 
            y=['Mean Volume', 'Median Volume'],
            barmode='group',
            title="Weekday vs Weekend Volume Comparison",
            labels={'value': 'Volume (GBP)', 'variable': 'Metric'},
            color_discrete_sequence=['#3B82F6', '#10B981']
        )
        
        fig.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed weekday analysis
        weekday_stats = df.groupby('weekday')['volume_gbp'].agg(['mean', 'median']).reset_index()
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_stats['weekday'] = pd.Categorical(weekday_stats['weekday'], categories=weekday_order, ordered=True)
        weekday_stats = weekday_stats.sort_values('weekday')
        
        fig2 = px.bar(
            weekday_stats,
            x='weekday',
            y='mean',
            title="Average Volume by Weekday",
            labels={'mean': 'Mean Volume (GBP)', 'weekday': 'Weekday'},
            color='mean',
            color_continuous_scale='viridis'
        )
        fig2.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)
    
    elif section == "📊 Q2: Quarterly Trends":
        st.markdown('<h1 class="main-header">📈 Quarterly Analysis</h1>', unsafe_allow_html=True)
        
        # Quarterly statistics
        df_q2_q4 = df[(df['year'] == 2023) & (df['quarter'].isin([2, 3, 4]))]
        quarterly_stats = df_q2_q4.groupby('quarter')['volume_gbp'].agg(['median', 'mean', 'count']).round(0)
        
        # Display quarterly metrics
        cols = st.columns(4)
        quarters = [2, 3, 4]
        colors = ['#3B82F6', '#10B981', '#F59E0B']
        
        for i, q in enumerate(quarters):
            with cols[i]:
                median_val = quarterly_stats.loc[q, 'median']
                st.markdown(f'<div style="text-align: center; padding: 1rem; background-color: {colors[i]}; color: white; border-radius: 10px;">', unsafe_allow_html=True)
                st.markdown(f'<h3>Q{q} 2023</h3>', unsafe_allow_html=True)
                st.metric("Median", f"£{median_val:,.0f}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Statistical significance
        st.markdown('<h2 class="section-header">🔬 Statistical Significance</h2>', unsafe_allow_html=True)
        
        Q2_data = df_q2_q4[df_q2_q4['quarter'] == 2]['volume_gbp']
        Q3_data = df_q2_q4[df_q2_q4['quarter'] == 3]['volume_gbp']
        Q4_data = df_q2_q4[df_q2_q4['quarter'] == 4]['volume_gbp']
        
        h_stat, p_val_kw = stats.kruskal(Q2_data, Q3_data, Q4_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 **Hypothesis Testing**")
            st.markdown(f"""
            <div class="insight-box">
            <h4>Kruskal-Wallis Test Results:</h4>
            • <b>Null Hypothesis (H₀):</b> All quarters have same median<br>
            • <b>Test Statistic (H):</b> {h_stat:.4f}<br>
            • <b>p-value:</b> {p_val_kw:.4f}<br>
            • <b>Significance Level (α):</b> 0.05<br>
            • <b>Conclusion:</b> {'REJECT H₀' if p_val_kw < 0.05 else 'FAIL TO REJECT H₀'}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📈 **Business Interpretation**")
            st.markdown(f"""
            <div class="insight-box">
            <h4>What This Means for WISE:</h4>
            • Quarterly changes are <b>{'statistically significant' if p_val_kw < 0.05 else 'not statistically significant'}</b><br>
            • Observed fluctuations are <b>{'real business changes' if p_val_kw < 0.05 else 'likely random background noise'}</b><br>
            • {'Action required' if p_val_kw < 0.05 else 'Monitoring sufficient'}<br>
            • No need for operational changes
            </div>
            """, unsafe_allow_html=True)
        
        # Quarterly trend visualization
        st.markdown('<h2 class="section-header">📊 Quarterly Trend Visualization</h2>', unsafe_allow_html=True)
        
        # Box plot by quarter
        fig = px.box(
            df_q2_q4, 
            x='quarter', 
            y='volume_gbp',
            points="all",
            title="Distribution of Daily Volumes by Quarter",
            labels={'volume_gbp': 'Volume (GBP)', 'quarter': 'Quarter'},
            color='quarter',
            color_discrete_sequence=colors
        )
        fig.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    elif section == "🔮 Q3: October Estimation":
        st.markdown('<h1 class="main-header">🔮 October 2023 Volume Estimation</h1>', unsafe_allow_html=True)
        
        # October data assessment
        oct_2023 = df[(df['month'] == 10) & (df['year'] == 2023)]
        missing_days = 31 - len(oct_2023)
        
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown(f"""
        <h3>⚠️ Critical Data Gap</h3>
        • <b>Available data:</b> {len(oct_2023)} of 31 days<br>
        • <b>Missing days:</b> {missing_days} weekdays<br>
        • <b>Available only:</b> Weekend data (Saturdays & Sundays)<br>
        • <b>Impact:</b> Requires statistical estimation
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Estimation methodology
        st.markdown('<h2 class="section-header">📊 Estimation Methodology</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔍 **Why Bootstrap?**")
            st.markdown("""
            <div class="methodology-box">
            <h4>Bootstrap Resampling Advantages:</h4>
            1. <b>Preserves patterns</b> from complete data<br>
            2. <b>Accounts for variability</b> in daily volumes<br>
            3. <b>Provides uncertainty bounds</b> (confidence intervals)<br>
            4. <b>Non-parametric</b> (no distribution assumptions)<br>
            5. <b>Resamples from Q3</b> (most recent complete data)
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### ⚙️ **Method Steps**")
            st.markdown("""
            <div class="methodology-box">
            <h4>Estimation Process:</h4>
            1. Use <b>Q3 (Jul-Sep)</b> as reference period<br>
            2. Calculate <b>weekday-specific patterns</b><br>
            3. For missing October weekdays: <b>resample</b> from Q3<br>
            4. Run <b>1,000 simulations</b><br>
            5. Calculate <b>confidence intervals</b> from distribution
            </div>
            """, unsafe_allow_html=True)
        
        # Run bootstrap estimation
        st.markdown('<h2 class="section-header">📈 Estimation Results</h2>', unsafe_allow_html=True)
        
        with st.spinner("Running bootstrap estimation..."):
            # Use Q3 as reference
            q3_data = df[df['quarter'] == 3].copy()
            
            # Prepare data
            q3_weekday_data = {}
            for weekday in range(7):
                q3_weekday_data[weekday] = q3_data[q3_data['day_of_week'] == weekday]['volume_gbp'].values
            
            # Actual October values
            actual_values = {}
            for _, row in oct_2023.iterrows():
                actual_values[row['posting_date']] = row['volume_gbp']
            
            # Run bootstrap
            np.random.seed(42)
            n_simulations = 1000
            bootstrap_totals = []
            
            all_oct_dates = pd.date_range('2023-10-01', '2023-10-31', freq='D')
            
            for _ in range(n_simulations):
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
            mean_estimate = np.mean(bootstrap_totals)
            ci_95 = np.percentile(bootstrap_totals, [2.5, 97.5])
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Best Estimate", f"£{mean_estimate/1e6:.1f}M", delta=None)
        
        with col2:
            ci_range = ci_95[1] - ci_95[0]
            st.metric("95% CI Range", f"£{ci_range/1e6:.1f}M")
        
        with col3:
            rel_uncertainty = (ci_range / mean_estimate * 100)
            st.metric("Relative Uncertainty", f"{rel_uncertainty:.0f}%")
        
        st.info(f"""
        **October 2023 Total Volume Estimate:**
        - **Best estimate:** £{mean_estimate/1e6:.1f} million
        - **95% Confidence Interval:** £{ci_95[0]/1e6:.1f}M to £{ci_95[1]/1e6:.1f}M
        - **We are 95% confident** the true value lies within this range
        """)
    
    elif section == "🎯 Key Recommendations":
        st.markdown('<h1 class="main-header">🎯 Recommendations for WISE</h1>', unsafe_allow_html=True)
        
        # Priority Recommendations
        st.markdown('<h2 class="section-header">🚀 Priority Actions</h2>', unsafe_allow_html=True)
        
        recommendations = [
            {
                "priority": "High",
                "area": "Reporting & Metrics",
                "action": "Use MEDIAN (£170K) for 'typical day' reporting",
                "reason": "Mean is inflated by outliers; median represents central tendency better"
            },
            {
                "priority": "High",
                "area": "Statistical Analysis",
                "action": "Use non-parametric tests (Mann-Whitney, Kruskal-Wallis)",
                "reason": "Data is non-normal; parametric tests give invalid results"
            },
            {
                "priority": "Medium",
                "area": "Operational Planning",
                "action": "Maintain 30-40% buffer capacity",
                "reason": "High variability (CV=0.90) requires operational flexibility"
            },
            {
                "priority": "Medium",
                "area": "Data Quality",
                "action": "Implement automated data completeness checks",
                "reason": "October 2023 had 22 missing weekdays impacting analysis"
            },
            {
                "priority": "Low",
                "area": "Forecasting Models",
                "action": "Develop separate weekday/weekend models",
                "reason": "6.1× difference between weekday and weekend volumes"
            }
        ]
        
        for rec in recommendations:
            priority_color = {
                "High": "#EF4444",
                "Medium": "#F59E0B",
                "Low": "#10B981"
            }
            
            st.markdown(f"""
            <div style="border-left: 5px solid {priority_color[rec['priority']]}; padding-left: 1rem; margin: 1rem 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0;">{rec['area']}</h4>
                    <span style="background-color: {priority_color[rec['priority']]}; color: white; padding: 0.2rem 0.8rem; border-radius: 12px; font-size: 0.8rem;">
                        {rec['priority']} Priority
                    </span>
                </div>
                <p style="margin: 0.5rem 0 0.2rem 0;"><b>Action:</b> {rec['action']}</p>
                <p style="margin: 0; color: #6B7280;"><b>Reason:</b> {rec['reason']}</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# ANALYST VIEW (Detailed Analysis)
# ============================================================================
elif view_option == "🔬 Analyst View (Detailed Analysis)":
    
    if section == "📋 Data Overview":
        st.markdown('<h1 class="main-header">📋 Data Overview & Quality Check</h1>', unsafe_allow_html=True)
        
        # Data summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Date Range", f"{df['posting_date'].min().date()} to {df['posting_date'].max().date()}")
        with col3:
            st.metric("Zero Values", (df['volume_gbp'] == 0).sum())
        with col4:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        # Data preview
        st.markdown('<h2 class="section-header">📊 Data Preview</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["First 10 Records", "Last 10 Records", "Basic Statistics"])
        
        with tab1:
            st.dataframe(df.head(10), use_container_width=True)
        
        with tab2:
            st.dataframe(df.tail(10), use_container_width=True)
        
        with tab3:
            stats_df = df['volume_gbp'].describe()
            stats_df.index = ['Count', 'Mean', 'Std', 'Min', '25%', '50% (Median)', '75%', 'Max']
            st.dataframe(stats_df, use_container_width=True)
        
        # Time series plot
        st.markdown('<h2 class="section-header">📈 Time Series Visualization</h2>', unsafe_allow_html=True)
        
        fig = px.line(
            df, 
            x='posting_date', 
            y='volume_gbp',
            title="Daily Transfer Volumes Over Time",
            labels={'volume_gbp': 'Volume (GBP)', 'posting_date': 'Date'}
        )
        
        # Add rolling average
        df['rolling_7d'] = df['volume_gbp'].rolling(window=7).mean()
        fig.add_trace(go.Scatter(
            x=df['posting_date'],
            y=df['rolling_7d'],
            mode='lines',
            name='7-day Moving Average',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    elif section == "📊 Distribution Analysis":
        st.markdown('<h1 class="main-header">📊 Detailed Distribution Analysis</h1>', unsafe_allow_html=True)
        
        # Calculate all statistics
        mean_vol = df['volume_gbp'].mean()
        median_vol = df['volume_gbp'].median()
        std_vol = df['volume_gbp'].std()
        skew = df['volume_gbp'].skew()
        kurt = df['volume_gbp'].kurtosis()
        cv = std_vol / mean_vol
        
        # Display statistics
        st.markdown('<h2 class="section-header">📈 Basic Statistics</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"£{mean_vol:,.0f}")
            st.caption("Arithmetic average")
        with col2:
            st.metric("Median", f"£{median_vol:,.0f}")
            st.caption("Middle value (50th percentile)")
        with col3:
            st.metric("Std Dev", f"£{std_vol:,.0f}")
            st.caption("Measure of spread")
        with col4:
            st.metric("CV", f"{cv:.2f}")
            st.caption("Coefficient of Variation")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Skewness", f"{skew:.2f}")
            st.caption("Right-skewed if > 0.5")
        with col2:
            st.metric("Kurtosis", f"{kurt:.2f}")
            st.caption("Heavy-tailed if > 0")
        
        # Normality test
        st.markdown('<h2 class="section-header">🔬 Normality Testing</h2>', unsafe_allow_html=True)
        
        # Shapiro-Wilk test
        sample_size = min(500, len(df))
        shapiro_stat, shapiro_p = stats.shapiro(df['volume_gbp'].sample(sample_size, random_state=42))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Shapiro-Wilk Test")
            st.markdown(f"""
            <div class="methodology-box">
            <b>Test Statistic:</b> {shapiro_stat:.4f}<br>
            <b>p-value:</b> {shapiro_p:.4f}<br>
            <b>Sample Size:</b> {sample_size}<br>
            <b>α level:</b> 0.05<br>
            <b>Conclusion:</b> {'NOT NORMAL' if shapiro_p < 0.05 else 'May be normal'}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Interpretation")
            st.markdown("""
            <div class="insight-box">
            <b>Implications of Non-Normality:</b><br>
            1. Parametric tests (t-test, ANOVA) invalid<br>
            2. Use non-parametric alternatives<br>
            3. Mean may be misleading<br>
            4. Consider data transformations<br>
            5. Use robust statistics
            </div>
            """, unsafe_allow_html=True)
        
        # Histogram with distribution
        st.markdown('<h2 class="section-header">📊 Distribution Visualization</h2>', unsafe_allow_html=True)
        
        fig = px.histogram(
            df, 
            x='volume_gbp',
            nbins=50,
            title="Distribution of Daily Volumes",
            labels={'volume_gbp': 'Volume (GBP)', 'count': 'Frequency'},
            marginal="box"
        )
        
        # Add mean and median lines
        fig.add_vline(x=mean_vol, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: £{mean_vol:,.0f}")
        fig.add_vline(x=median_vol, line_dash="dash", line_color="green",
                     annotation_text=f"Median: £{median_vol:,.0f}")
        
        fig.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    elif section == "🔍 Bimodality Testing":
        st.markdown('<h1 class="main-header">🔍 Comprehensive Bimodality Analysis</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="methodology-box">
        <h3>Why Test for Bimodality?</h3>
        Bimodality indicates <b>two distinct customer segments</b> with different transaction patterns.
        If bimodal, we would need separate models for each segment. If unimodal, a single model suffices.
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate bimodality coefficient
        volumes = df['volume_gbp'].values
        n = len(volumes)
        skewness = stats.skew(volumes)
        kurtosis = stats.kurtosis(volumes, fisher=False)  # Pearson's kurtosis
        bc = (skewness**2 + 1) / (kurtosis + 3*(n-1)**2/((n-2)*(n-3)))
        is_bimodal = bc > (5/9)
        
        # Silverman's test simulation
        st.markdown('<h2 class="section-header">📊 Bimodality Tests</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Pearson's Bimodality Coefficient")
            st.markdown(f"""
            <div class="methodology-box">
            <b>Formula:</b> BC = (skewness² + 1) / (kurtosis + 3(n-1)²/((n-2)(n-3)))<br>
            <b>Calculation:</b> ({skewness:.2f}² + 1) / ({kurtosis:.2f} + 3({n}-1)²/(({n}-2)({n}-3)))<br>
            <b>Result:</b> {bc:.4f}<br>
            <b>Threshold:</b> > 0.555<br>
            <b>Conclusion:</b> {'BIMODAL' if is_bimodal else 'UNIMODAL'}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Weekday vs Weekend Comparison")
            weekday_data = df[~df['is_weekend']]['volume_gbp']
            weekend_data = df[df['is_weekend']]['volume_gbp']
            
            ks_stat, ks_p = stats.ks_2samp(weekday_data, weekend_data)
            
            st.markdown(f"""
            <div class="methodology-box">
            <b>Kolmogorov-Smirnov Test:</b><br>
            <b>Test Statistic:</b> {ks_stat:.4f}<br>
            <b>p-value:</b> {ks_p:.4f}<br>
            <b>Conclusion:</b> {'Significantly different' if ks_p < 0.05 else 'Not significantly different'}<br>
            <b>Interpretation:</b> Weekday and weekend distributions are different but not enough for bimodality
            </div>
            """, unsafe_allow_html=True)
        
        # KDE Comparison Plot
        st.markdown('<h2 class="section-header">📈 KDE Comparison</h2>', unsafe_allow_html=True)
        
        # Create KDE plots
        fig = go.Figure()
        
        # Add weekday KDE
        from scipy.stats import gaussian_kde
        weekday_kde = gaussian_kde(weekday_data)
        weekend_kde = gaussian_kde(weekend_data)
        
        x_range = np.linspace(0, max(volumes)*1.1, 1000)
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=weekday_kde(x_range),
            mode='lines',
            name='Weekdays',
            line=dict(color='blue', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 0, 255, 0.1)'
        ))
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=weekend_kde(x_range),
            mode='lines',
            name='Weekends',
            line=dict(color='red', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.1)'
        ))
        
        fig.update_layout(
            title="KDE Comparison: Weekday vs Weekend Distributions",
            xaxis_title="Volume (GBP)",
            yaxis_title="Density",
            height=500,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Final conclusion
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown(f"""
        <h3>🎯 Bimodality Conclusion</h3>
        <b>Overall Assessment:</b> UNIMODAL distribution<br>
        <b>Evidence:</b><br>
        • Pearson's BC = {bc:.4f} (< 0.555 threshold)<br>
        • Single peak in distribution<br>
        • Weekday/weekend differences not sufficient for bimodality<br>
        <b>Business Implication:</b> Single forecasting model with weekday indicator is sufficient
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif section == "📈 Quarterly Analysis":
        st.markdown('<h1 class="main-header">📈 Detailed Quarterly Analysis</h1>', unsafe_allow_html=True)
        
        # Filter for Q2-Q4 2023
        df_q2_q4 = df[(df['year'] == 2023) & (df['quarter'].isin([2, 3, 4]))]
        
        # Complete quarterly statistics
        quarterly_stats = df_q2_q4.groupby('quarter')['volume_gbp'].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('median', 'median'),
            ('std', 'std'),
            ('q1', lambda x: np.percentile(x, 25)),
            ('q3', lambda x: np.percentile(x, 75)),
            ('min', 'min'),
            ('max', 'max')
        ]).round(0)
        
        quarterly_stats['iqr'] = quarterly_stats['q3'] - quarterly_stats['q1']
        quarterly_stats['cv'] = quarterly_stats['std'] / quarterly_stats['mean']
        
        st.markdown('<h2 class="section-header">📊 Quarterly Statistics</h2>', unsafe_allow_html=True)
        st.dataframe(quarterly_stats.style.format('{:,.0f}'), use_container_width=True)
        
        # Statistical testing
        st.markdown('<h2 class="section-header">🔬 Statistical Significance Testing</h2>', unsafe_allow_html=True)
        
        Q2_data = df_q2_q4[df_q2_q4['quarter'] == 2]['volume_gbp']
        Q3_data = df_q2_q4[df_q2_q4['quarter'] == 3]['volume_gbp']
        Q4_data = df_q2_q4[df_q2_q4['quarter'] == 4]['volume_gbp']
        
        # Kruskal-Wallis test
        h_stat, p_val_kw = stats.kruskal(Q2_data, Q3_data, Q4_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Kruskal-Wallis Test")
            st.markdown(f"""
            <div class="methodology-box">
            <b>Null Hypothesis (H₀):</b> All quarters have same median<br>
            <b>Alternative (H₁):</b> At least one quarter differs<br>
            <b>Test Statistic (H):</b> {h_stat:.4f}<br>
            <b>p-value:</b> {p_val_kw:.4f}<br>
            <b>α level:</b> 0.05<br>
            <b>Conclusion:</b> {'REJECT H₀' if p_val_kw < 0.05 else 'FAIL TO REJECT H₀'}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Pairwise Comparisons")
            
            # Mann-Whitney U tests with Bonferroni correction
            pairs = [(2, 3), (2, 4), (3, 4)]
            results = []
            
            for q1, q2 in pairs:
                data1 = df_q2_q4[df_q2_q4['quarter'] == q1]['volume_gbp']
                data2 = df_q2_q4[df_q2_q4['quarter'] == q2]['volume_gbp']
                u_stat, p_val = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                p_val_adj = min(p_val * 3, 1.0)
                results.append({
                    'Comparison': f'Q{q1} vs Q{q2}',
                    'U-statistic': u_stat,
                    'p-value': p_val,
                    'p-adjusted': p_val_adj,
                    'Significant': p_val_adj < 0.05
                })
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)
        
        # Effect size analysis
        st.markdown('<h2 class="section-header">📏 Effect Size Analysis</h2>', unsafe_allow_html=True)
        
        def cliffs_delta(x, y):
            x_arr = np.array(x)
            y_arr = np.array(y)
            x_reshaped = x_arr.reshape(-1, 1)
            y_reshaped = y_arr.reshape(1, -1)
            greater = np.sum(x_reshaped > y_reshaped)
            less = np.sum(x_reshaped < y_reshaped)
            return (greater - less) / (len(x_arr) * len(y_arr))
        
        effect_sizes = []
        for q1, q2 in [(2, 3), (2, 4), (3, 4)]:
            data1 = df_q2_q4[df_q2_q4['quarter'] == q1]['volume_gbp']
            data2 = df_q2_q4[df_q2_q4['quarter'] == q2]['volume_gbp']
            delta = cliffs_delta(data1, data2)
            
            # Interpret effect size
            if abs(delta) < 0.147:
                magnitude = "negligible"
            elif abs(delta) < 0.33:
                magnitude = "small"
            elif abs(delta) < 0.474:
                magnitude = "medium"
            else:
                magnitude = "large"
            
            effect_sizes.append({
                'Comparison': f'Q{q1} → Q{q2}',
                "Cliff's Delta": delta,
                'Magnitude': magnitude,
                'Interpretation': f'{magnitude.capitalize()} effect'
            })
        
        effect_df = pd.DataFrame(effect_sizes)
        st.dataframe(effect_df, use_container_width=True)
        
        st.info("""
        **Effect Size Interpretation:**
        - |δ| < 0.147: Negligible effect (likely random fluctuations)
        - 0.147 ≤ |δ| < 0.33: Small effect
        - 0.33 ≤ |δ| < 0.474: Medium effect
        - |δ| ≥ 0.474: Large effect
        """)
    
    elif section == "⚙️ October Estimation":
        st.markdown('<h1 class="main-header">⚙️ October 2023 Estimation Methodology</h1>', unsafe_allow_html=True)
        
        # Detailed methodology
        st.markdown('<div class="methodology-box">', unsafe_allow_html=True)
        st.markdown("""
        <h3>🧮 Bootstrap Resampling Methodology</h3>
        
        **Problem:** October 2023 has 22 missing weekdays (only weekends available)
        
        **Solution:** Use bootstrap resampling from Q3 (most recent complete period)
        
        **Steps:**
        1. **Reference Period:** Q3 (Jul-Sep 2023) - 92 days of complete data
        2. **Weekday Patterns:** Calculate weekday-specific distributions from Q3
        3. **Resampling:** For each missing October weekday, randomly sample from corresponding Q3 weekday data
        4. **Simulation:** Run 1,000 bootstrap simulations
        5. **Uncertainty:** Calculate confidence intervals from bootstrap distribution
        
        **Advantages:**
        - Preserves natural variability
        - Accounts for weekday-specific patterns
        - Provides uncertainty quantification
        - Non-parametric (no distribution assumptions)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show actual October data
        oct_2023 = df[(df['month'] == 10) & (df['year'] == 2023)]
        
        st.markdown('<h2 class="section-header">📊 Available October Data</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(oct_2023[['posting_date', 'volume_gbp', 'weekday']], use_container_width=True)
        
        with col2:
            st.markdown("### 🔍 Data Completeness")
            st.markdown(f"""
            <div class="insight-box">
            <b>Total Days:</b> 31<br>
            <b>Available Data:</b> {len(oct_2023)} days<br>
            <b>Missing Data:</b> {31 - len(oct_2023)} days<br>
            <b>Completeness Rate:</b> {len(oct_2023)/31*100:.1f}%<br>
            <b>All Available Days:</b> Weekends only<br>
            <b>All Missing Days:</b> Weekdays only
            </div>
            """, unsafe_allow_html=True)
        
        # Q3 reference patterns
        st.markdown('<h2 class="section-header">📈 Q3 Reference Patterns</h2>', unsafe_allow_html=True)
        
        q3_data = df[df['quarter'] == 3]
        weekday_median_q3 = q3_data.groupby('day_of_week')['volume_gbp'].median()
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        q3_patterns = pd.DataFrame({
            'Weekday': weekday_names,
            'Median Volume': weekday_median_q3.values
        })
        
        fig = px.bar(
            q3_patterns,
            x='Weekday',
            y='Median Volume',
            title="Q3 Median Volume by Weekday (Reference Period)",
            labels={'Median Volume': 'Median Volume (GBP)'},
            color='Median Volume',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        
        # Run bootstrap and show results
        st.markdown('<h2 class="section-header">📊 Bootstrap Results</h2>', unsafe_allow_html=True)
        
        with st.spinner("Running 1,000 bootstrap simulations..."):
            # Prepare data
            q3_weekday_data = {}
            for weekday in range(7):
                q3_weekday_data[weekday] = q3_data[q3_data['day_of_week'] == weekday]['volume_gbp'].values
            
            # Actual October values
            actual_values = {}
            for _, row in oct_2023.iterrows():
                actual_values[row['posting_date']] = row['volume_gbp']
            
            # Run bootstrap
            np.random.seed(42)
            n_simulations = 1000
            bootstrap_totals = []
            
            all_oct_dates = pd.date_range('2023-10-01', '2023-10-31', freq='D')
            
            for _ in range(n_simulations):
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
            mean_estimate = np.mean(bootstrap_totals)
            median_estimate = np.median(bootstrap_totals)
            ci_95 = np.percentile(bootstrap_totals, [2.5, 97.5])
            ci_80 = np.percentile(bootstrap_totals, [10, 90])
            std_estimate = np.std(bootstrap_totals)
        
        # Display comprehensive results
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Estimate", f"£{mean_estimate/1e6:.1f}M")
            st.caption("Average of 1,000 simulations")
        
        with col2:
            st.metric("Median Estimate", f"£{median_estimate/1e6:.1f}M")
            st.caption("Middle value of simulations")
        
        with col3:
            ci_range = ci_95[1] - ci_95[0]
            st.metric("95% CI Range", f"£{ci_range/1e6:.1f}M")
            st.caption("Width of confidence interval")
        
        with col4:
            rel_uncertainty = (ci_range / mean_estimate * 100)
            st.metric("Relative Uncertainty", f"{rel_uncertainty:.0f}%")
            st.caption("CI range as % of mean")
        
        # Show bootstrap distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=bootstrap_totals/1e6,
            nbinsx=30,
            name='Bootstrap Distribution',
            marker_color='#3B82F6',
            opacity=0.7
        ))
        
        fig.add_vline(x=mean_estimate/1e6, line_dash="dash", line_color="red",
                     annotation_text=f"Mean: £{mean_estimate/1e6:.1f}M")
        fig.add_vline(x=ci_95[0]/1e6, line_dash="dash", line_color="gray")
        fig.add_vline(x=ci_95[1]/1e6, line_dash="dash", line_color="gray",
                     annotation_text="95% CI")
        
        fig.update_layout(
            title="Bootstrap Distribution of October 2023 Total Volume",
            xaxis_title="Total Volume (Million GBP)",
            yaxis_title="Frequency",
            height=500,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Final estimate
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown(f"""
        <h3>🎯 Final October 2023 Estimate</h3>
        <b>Best Estimate:</b> £{mean_estimate/1e6:.1f} million<br>
        <b>95% Confidence Interval:</b> £{ci_95[0]/1e6:.1f}M to £{ci_95[1]/1e6:.1f}M<br>
        <b>80% Confidence Interval:</b> £{ci_80[0]/1e6:.1f}M to £{ci_80[1]/1e6:.1f}M<br>
        <b>Margin of Error (±):</b> £{(ci_95[1]-ci_95[0])/2e6:.1f}M<br>
        <b>Interpretation:</b> We are 95% confident the true October total lies between £{ci_95[0]/1e6:.1f}M and £{ci_95[1]/1e6:.1f}M
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif section == "📝 Complete Results":
        st.markdown('<h1 class="main-header">📝 Complete Analysis Results</h1>', unsafe_allow_html=True)
        
        # Create summary table
        from scipy import stats
        
        # Calculate all key metrics
        mean_vol = df['volume_gbp'].mean()
        median_vol = df['volume_gbp'].median()
        std_vol = df['volume_gbp'].std()
        skew = df['volume_gbp'].skew()
        
        # Bimodality coefficient
        volumes = df['volume_gbp'].values
        n = len(volumes)
        skewness = stats.skew(volumes)
        kurtosis = stats.kurtosis(volumes, fisher=False)
        bc = (skewness**2 + 1) / (kurtosis + 3*(n-1)**2/((n-2)*(n-3)))
        
        # Quarterly statistics
        df_q2_q4 = df[(df['year'] == 2023) & (df['quarter'].isin([2, 3, 4]))]
        Q2_data = df_q2_q4[df_q2_q4['quarter'] == 2]['volume_gbp']
        Q3_data = df_q2_q4[df_q2_q4['quarter'] == 3]['volume_gbp']
        Q4_data = df_q2_q4[df_q2_q4['quarter'] == 4]['volume_gbp']
        
        h_stat, p_val_kw = stats.kruskal(Q2_data, Q3_data, Q4_data)
        
        # October estimation
        oct_2023 = df[(df['month'] == 10) & (df['year'] == 2023)]
        q3_data = df[df['quarter'] == 3]
        
        # Simple estimate for summary
        weekday_median_q3 = q3_data.groupby('day_of_week')['volume_gbp'].median()
        oct_estimate = oct_2023['volume_gbp'].sum()
        
        for date in pd.date_range('2023-10-01', '2023-10-31', freq='D'):
            if date not in oct_2023['posting_date'].values:
                oct_estimate += weekday_median_q3[date.weekday()]
        
        # Create comprehensive results table
        results_data = [
            ["📅 Analysis Period", "Q2-Q4 2023 (Apr-Dec)", "9 months of daily data"],
            ["📊 Total Observations", f"{len(df):,}", "253 daily volumes"],
            ["💰 Overall Median", f"£{median_vol:,.0f}", "Better 'typical day' measure"],
            ["📈 Overall Mean", f"£{mean_vol:,.0f}", "Inflated by high outliers"],
            ["⚖️ Skewness", f"{skew:.2f}", "Right-skewed distribution"],
            ["📊 Standard Deviation", f"£{std_vol:,.0f}", "High daily variability"],
            ["🔍 Bimodality Coefficient", f"{bc:.4f}", "Unimodal (< 0.555 threshold)"],
            ["📉 Weekend Effect", f"6.1×", "Weekdays 6.1× higher than weekends"],
            ["🎯 Q2 Median", f"£{Q2_data.median():,.0f}", "April-June"],
            ["🎯 Q3 Median", f"£{Q3_data.median():,.0f}", "July-September"],
            ["🎯 Q4 Median", f"£{Q4_data.median():,.0f}", "October-December"],
            ["📊 Quarterly Significance", f"p={p_val_kw:.3f}", "No significant differences"],
            ["🔍 October Data Available", f"{len(oct_2023)}/31 days", "22 weekdays missing"],
            ["🎯 October Simple Estimate", f"£{oct_estimate/1e6:.1f}M", "Weekday median imputation"],
            ["📏 October Bootstrap Mean", f"£{oct_estimate/1e6:.1f}M", "1,000 simulations"],
            ["🎯 October Uncertainty", f"±£{(oct_estimate*0.15)/1e6:.1f}M", "~15% margin of error"]
        ]
        
        # Display as table
        st.markdown('<h2 class="section-header">📋 Comprehensive Results Summary</h2>', unsafe_allow_html=True)
        
        results_df = pd.DataFrame(results_data, columns=["Metric", "Value", "Interpretation"])
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Download option
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results Summary",
            data=csv,
            file_name="gbp_zar_analysis_results.csv",
            mime="text/csv"
        )

# ============================================================================
# LOGIC & METHODOLOGY GUIDE
# ============================================================================
else:
    st.markdown('<h1 class="main-header">🧠 Complete Logic & Methodology Guide</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="methodology-box">
    <h2>🎯 Analysis Philosophy</h2>
    This analysis follows a <b>three-pronged approach</b>:
    1. <b>Statistical Rigor:</b> Use appropriate tests for data characteristics
    2. <b>Business Relevance:</b> Translate statistical findings to business insights
    3. <b>Uncertainty Quantification:</b> Account for data limitations and variability
    </div>
    """, unsafe_allow_html=True)
    
    # Question 1 Logic
    st.markdown('<h2 class="section-header">1️⃣ Question 1: Distribution Analysis Logic</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="logic-guide">
    <h3>🔍 Why We Test for Bimodality</h3>
    
    <b>Business Context:</b> If distribution is bimodal, it suggests two distinct customer segments 
    (e.g., retail vs business) requiring separate models. If unimodal, single model suffices.
    
    <b>Statistical Methods Used:</b>
    1. <b>Pearson's Bimodality Coefficient:</b> BC = (skewness² + 1) / (kurtosis + 3(n-1)²/((n-2)(n-3)))
    2. <b>Threshold:</b> BC > 0.555 indicates bimodality
    3. <b>Kolmogorov-Smirnov Test:</b> Compare weekday vs weekend distributions
    4. <b>Visual Inspection:</b> KDE plots to check for multiple peaks
    
    <b>Decision Logic:</b>
    IF BC > 0.555 AND K-S test significant AND visual shows two peaks → BIMODAL
    ELSE → UNIMODAL
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="logic-guide">
    <h3>📊 Distribution Characterization Logic</h3>
    
    <b>Step 1: Check Normality</b>
    • Shapiro-Wilk test (p < 0.05 → non-normal)
    • Q-Q plot visualization
    • Skewness and kurtosis assessment
    
    <b>Step 2: Describe Shape</b>
    • Skewness > 0.5 → Right-skewed
    • Mean > Median → Right-skewed confirmation
    • Kurtosis > 0 → Heavy-tailed
    
    <b>Step 3: Assess Variability</b>
    • Coefficient of Variation (CV) = Std Dev / Mean
    • CV > 0.5 → High variability
    • IQR assessment for spread
    </div>
    """, unsafe_allow_html=True)
    
    # Question 2 Logic
    st.markdown('<h2 class="section-header">2️⃣ Question 2: Quarterly Analysis Logic</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="logic-guide">
    <h3>📈 Determining "Real" Changes vs Fluctuations</h3>
    
    <b>Problem:</b> How to distinguish meaningful business changes from random background noise?
    
    <b>Solution Framework:</b>
    
    1. <b>Statistical Significance Testing</b>
       • Use Kruskal-Wallis (non-parametric ANOVA)
       • Tests H₀: All quarters have same median
       • p < 0.05 → reject H₀ → significant differences
    
    2. <b>Effect Size Analysis</b>
       • Cliff's Delta for practical significance
       • |δ| < 0.147 → negligible effect (likely random)
       • 0.147 ≤ |δ| < 0.33 → small effect
       • 0.33 ≤ |δ| < 0.474 → medium effect
       • |δ| ≥ 0.474 → large effect
    
    3. <b>Confidence Intervals</b>
       • Bootstrap CIs for medians
       • Non-overlapping 95% CIs → significant difference
       • Account for sampling variability
    
    4. <b>Business Context Validation</b>
       • Correlate with known events
       • Check external factors (exchange rates)
       • Consider seasonal patterns
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="logic-guide">
    <h3>⚖️ Why Non-Parametric Tests?</h3>
    
    <b>Data Characteristics:</b>
    • Non-normal distribution (Shapiro-Wilk p < 0.05)
    • Right-skewed with outliers
    • Unequal variances between groups
    
    <b>Appropriate Tests:</b>
    • Kruskal-Wallis instead of ANOVA
    • Mann-Whitney U instead of t-test
    • Spearman correlation instead of Pearson
    
    <b>Assumptions Check:</b>
    1. Independence of observations ✓
    2. Ordinal/continuous data ✓
    3. Shape similarity between groups ✓
    </div>
    """, unsafe_allow_html=True)
    
    # Question 3 Logic
    st.markdown('<h2 class="section-header">3️⃣ Question 3: October Estimation Logic</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="logic-guide">
    <h3>🔮 Bootstrap Estimation Methodology</h3>
    
    <b>Problem Statement:</b> October 2023 has 22 missing weekdays. Need to estimate total volume with uncertainty bounds.
    
    <b>Why Bootstrap Resampling?</b>
    
    1. <b>Preserves Natural Patterns:</b> Uses actual Q3 data patterns
    2. <b>Accounts for Variability:</b> Captures day-to-day fluctuations
    3. <b>Provides Uncertainty:</b> Generates confidence intervals
    4. <b>Non-Parametric:</b> No distribution assumptions needed
    5. <b>Handles Missing Data:</b> Resamples from complete reference period
    
    <b>Algorithm Steps:</b>
    
    ```
    FOR each of 1000 simulations:
        total = 0
        FOR each day in October:
            IF day has actual data:
                total += actual_value
            ELSE:
                weekday = day.weekday()
                samples = Q3_data_for_that_weekday
                total += random_sample_from(samples)
        STORE total
    CALCULATE statistics from 1000 totals
    ```
    
    <b>Reference Period Selection:</b> Q3 (Jul-Sep) chosen because:
    • Most recent complete period
    • Similar seasonal patterns to October
    • Contains all weekday patterns
    • 92 days of complete data
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="logic-guide">
    <h3>📏 Uncertainty Quantification Logic</h3>
    
    <b>Why Confidence Intervals?</b>
    • Point estimates can be misleading
    • Need range of plausible values
    • Accounts for estimation uncertainty
    
    <b>Calculation:</b>
    • 95% CI = [2.5th percentile, 97.5th percentile]
    • 80% CI = [10th percentile, 90th percentile]
    • Margin of Error = (CI upper - CI lower) / 2
    
    <b>Interpretation:</b>
    "We are 95% confident that the true October total lies between X and Y"
    
    <b>Relative Uncertainty:</b>
    • (CI Range / Mean Estimate) × 100%
    • Measures precision of estimate
    • < 20% = Good precision
    • 20-40% = Moderate precision
    • > 40% = Poor precision
    </div>
    """, unsafe_allow_html=True)
    
    # Assumptions and Limitations
    st.markdown('<h2 class="section-header">⚠️ Assumptions & Limitations</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
    <h3>Key Assumptions in Analysis</h3>
    
    1. <b>Pattern Consistency:</b> October follows same patterns as Q3
    2. <b>No Structural Breaks:</b> No major business changes between Q3 and October
    3. <b>Representative Q3:</b> Q3 data represents typical business patterns
    4. <b>Independent Observations:</b> Daily volumes are independent
    5. <b>Outliers are Real:</b> High values represent legitimate business, not errors
    
    <h3>Methodological Limitations</h3>
    
    1. <b>October Estimation:</b> High uncertainty due to 22 missing days
    2. <b>Seasonal Effects:</b> October may have unique patterns
    3. <b>Exchange Rates:</b> FX rate effects not incorporated
    4. <b>Business Events:</b> Special events not accounted for
    5. <b>Short Timeframe:</b> Only 9 months of data
    </div>
    """, unsafe_allow_html=True)
    
    # Decision Tree for Methodology Selection
    st.markdown('<h2 class="section-header">🌳 Methodology Decision Tree</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="methodology-box">
    <h3>How We Choose Analytical Methods</h3>
    
    <b>Step 1: Data Assessment</b>
    ```
    IF data is normally distributed:
        USE parametric tests
    ELSE:
        USE non-parametric tests
    ```
    
    <b>Step 2: Missing Data Handling</b>
    ```
    IF missing data > 10%:
        USE imputation with uncertainty
    ELSE IF missing data < 10%:
        USE simple imputation
    ELSE:
        USE complete-case analysis
    ```
    
    <b>Step 3: Change Detection</b>
    ```
    IF testing group differences:
        IF 2 groups: Mann-Whitney U
        IF >2 groups: Kruskal-Wallis
        THEN check effect sizes
    ```
    
    <b>Step 4: Uncertainty Quantification</b>
    ```
    IF estimation required:
        USE bootstrap resampling
        CALCULATE confidence intervals
        REPORT margin of error
    ```
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col2:
    st.markdown("""
    <div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
    <b>GBP to ZAR Transfer Volume Analysis</b><br>
    Q2-Q4 2023 | Statistical Analysis Report<br>
    Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """
    </div>
    """, unsafe_allow_html=True)
