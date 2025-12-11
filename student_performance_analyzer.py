# -*- coding: utf-8 -*-
"""Student performance analyzer

Original file is located at
    https://colab.research.google.com/drive/1kTzDDUscB4TaQ5bfIikjAiyDbYOkFYTZ
"""



        # -*- coding: utf-8 -*-
"""Student performance analyzer 

"""
Student Academic Performance Analyzer
Author: Drina Musili - 250636DAI
Course: DAI011 - Programming for AI
Dataset: UCI Machine Learning Repository - Student Performance Dataset

This application analyzes factors affecting student academic performance in Mathematics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Student Performance Analyzer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR BETTER STYLING
# ============================================================================

st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA LOADING FUNCTION
# ============================================================================

@st.cache_data
def load_data():
    """
    Load the student performance dataset from UCI repository
    
    Returns:
        pd.DataFrame: Loaded and preprocessed dataset
    """
    # URL to the dataset (Math course data)
    url = "https://raw.githubusercontent.com/arunk13/MSDA-Assignments/master/IS607Fall2015/Assignment3/student-mat.csv"
    
    # Load the CSV file (semicolon-separated)
    df = pd.read_csv(url, sep=';')
    
    # Create calculated columns for better analysis
    # Average grade across all three periods
    df['average_grade'] = (df['G1'] + df['G2'] + df['G3']) / 3
    
    # Categorize performance based on final grade
    def categorize_performance(grade):
        """Categorize student performance into 4 levels"""
        if grade < 10:
            return 'Fail'
        elif grade < 14:
            return 'Pass'
        elif grade < 17:
            return 'Good'
        else:
            return 'Excellent'
    
    df['performance_category'] = df['G3'].apply(categorize_performance)
    
    return df

# ============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# ============================================================================

def calculate_overall_stats(df):
    """Calculate overall grade statistics"""
    return {
        'mean': df['G3'].mean(),
        'median': df['G3'].median(),
        'std': df['G3'].std(),
        'min': df['G3'].min(),
        'max': df['G3'].max(),
        'pass_rate': (df['G3'] >= 10).sum() / len(df) * 100,
        'fail_rate': (df['G3'] < 10).sum() / len(df) * 100
    }

def analyze_by_category(df, category_col, value_col='G3'):
    """Analyze performance by a specific category"""
    return df.groupby(category_col)[value_col].agg(['mean', 'median', 'count', 'std']).round(2)

def get_correlation_data(df):
    """Get correlation data for key numerical features"""
    numerical_cols = ['age', 'Medu', 'Fedu', 'studytime', 'failures', 
                      'absences', 'G1', 'G2']
    correlations = df[numerical_cols + ['G3']].corr()['G3'].sort_values(ascending=False)
    return correlations

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_grade_distribution(df):
    """Create histogram of final grade distribution"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df['G3'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(df['G3'].mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {df["G3"].mean():.2f}')
    ax.set_xlabel('Final Grade (G3)', fontsize=12)
    ax.set_ylabel('Number of Students', fontsize=12)
    ax.set_title('Distribution of Final Grades', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    return fig

def plot_performance_pie(df):
    """Create pie chart of performance categories"""
    fig, ax = plt.subplots(figsize=(8, 8))
    category_counts = df['performance_category'].value_counts()
    colors = ['#ff6b6b', '#ffd93d', '#6bcf7f', '#4d96ff']
    ax.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', 
           colors=colors, startangle=90, textprops={'fontsize': 11})
    ax.set_title('Performance Category Distribution', fontsize=14, fontweight='bold')
    return fig

def plot_study_time_impact(df):
    """Create bar chart showing study time impact on grades"""
    fig, ax = plt.subplots(figsize=(10, 6))
    study_labels = ['<2h', '2-5h', '5-10h', '>10h']
    study_means = df.groupby('studytime')['G3'].mean()
    bars = ax.bar(study_labels, study_means, 
                  color=['#ff6b6b', '#ffd93d', '#6bcf7f', '#4d96ff'], 
                  edgecolor='black', alpha=0.8)
    ax.set_xlabel('Weekly Study Time', fontsize=12)
    ax.set_ylabel('Average Final Grade', fontsize=12)
    ax.set_title('Study Time Impact on Performance', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 20)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    return fig

def plot_gender_comparison(df):
    """Create bar chart comparing gender performance"""
    fig, ax = plt.subplots(figsize=(8, 6))
    gender_data = df.groupby('sex')['G3'].mean()
    bars = ax.bar(['Female', 'Male'], gender_data, 
                  color=['#ff99cc', '#6699ff'], 
                  edgecolor='black', alpha=0.8)
    ax.set_ylabel('Average Final Grade', fontsize=12)
    ax.set_title('Gender-Based Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 20)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontsize=11)
    return fig

def plot_family_support(df):
    """Create bar chart showing family support impact"""
    fig, ax = plt.subplots(figsize=(8, 6))
    support_data = df.groupby('famsup')['G3'].mean()
    bars = ax.bar(['No Support', 'With Support'], support_data, 
                  color=['#ff6b6b', '#6bcf7f'], edgecolor='black', alpha=0.8)
    ax.set_ylabel('Average Final Grade', fontsize=12)
    ax.set_title('Family Support Impact on Performance', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 20)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontsize=11)
    return fig

def plot_correlation_heatmap(df):
    """Create correlation heatmap for key features"""
    fig, ax = plt.subplots(figsize=(10, 8))
    key_features = ['studytime', 'failures', 'absences', 'G1', 'G2', 'G3']
    corr_matrix = df[key_features].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    return fig

def plot_failures_impact(df):
    """Create bar chart showing impact of past failures"""
    fig, ax = plt.subplots(figsize=(10, 6))
    failures_mean = df.groupby('failures')['G3'].mean()
    bars = ax.bar(failures_mean.index, failures_mean.values, 
                  color='coral', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Number of Past Failures', fontsize=12)
    ax.set_ylabel('Average Final Grade', fontsize=12)
    ax.set_title('Impact of Past Failures on Performance', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 20)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.title("📚 Student Academic Performance Analyzer")
    st.markdown("""
    <div style='background-color: #282A36; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h4>📊 Project Information</h4>
        <p><strong>Dataset:</strong> UCI Machine Learning Repository - Student Performance Dataset</p>
        <p><strong>Course:</strong> DAI011 - Programming for AI</p>
        <p><strong>Author:</strong>Drina Musili- 250636DAI</p>
        <p><strong>Purpose:</strong> Analyze factors affecting student academic performance in Mathematics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📊 About This Application")
    st.sidebar.info("""
    **This analyzer explores:**
    - 📈 Grade distributions and statistics
    - ⏰ Study time impact on performance
    - 👥 Gender-based comparisons
    - 👨‍👩‍👧 Family support effects
    - 📉 Impact of past failures
    - 🔗 Key performance predictors
    
    **Dataset Details:**
    - 395 students from Portuguese schools
    - Math course performance data
    - 33 features including demographics, social factors, and grades
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.header("🛠️ Technical Details")
    st.sidebar.markdown("""
    **Libraries Used:**
    - 🐼 Pandas - Data manipulation
    - 🔢 NumPy - Numerical operations
    - 📊 Matplotlib - Data visualization
    - 🎨 Seaborn - Statistical graphics
    - 🚀 Streamlit - Web interface
    """)
    
    # Load data
    try:
        with st.spinner('Loading dataset...'):
            df = load_data()
        st.success(f"✅ Dataset loaded successfully! ({df.shape[0]} students, {df.shape[1]} features)")
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()
    
    # Calculate overall statistics
    overall_stats = calculate_overall_stats(df)
    
    # Key Metrics at the top
    st.markdown("### 📊 Key Metrics Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Students", df.shape[0])
    with col2:
        st.metric("Average Grade", f"{overall_stats['mean']:.2f}/20")
    with col3:
        st.metric("Pass Rate", f"{overall_stats['pass_rate']:.1f}%")
    with col4:
        st.metric("Median Grade", f"{overall_stats['median']:.1f}/20")
    with col5:
        st.metric("Std Deviation", f"{overall_stats['std']:.2f}")
    
    st.markdown("---")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Data Overview", 
        "📈 Statistical Analysis", 
        "📊 Visualizations", 
        "💡 Key Insights",
        "🔍 Data Explorer"
    ])
    
    # ========================================================================
    # TAB 1: DATA OVERVIEW
    # ========================================================================
    with tab1:
        st.header("📋 Dataset Overview")
        
        # Sample data
        st.subheader("Sample Data (First 10 Rows)")
        st.dataframe(df.head(10), use_container_width=True, height=400)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Dataset Structure")
            st.write(f"**Total Rows:** {df.shape[0]}")
            st.write(f"**Total Columns:** {df.shape[1]}")
            st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
            
            st.subheader("✅ Data Quality")
            missing = df.isnull().sum().sum()
            if missing == 0:
                st.success("✅ No missing values found - Clean dataset!")
            else:
                st.warning(f"⚠️ {missing} missing values detected")
            
            duplicates = df.duplicated().sum()
            if duplicates == 0:
                st.success("✅ No duplicate rows found")
            else:
                st.warning(f"⚠️ {duplicates} duplicate rows detected")
        
        with col2:
            st.subheader("📝 Column Information")
            info_df = pd.DataFrame({
                'Column': df.columns[:15],  # Show first 15 columns
                'Type': df.dtypes[:15].astype(str),
                'Non-Null': df.count()[:15].values
            })
            st.dataframe(info_df, use_container_width=True, height=400)
        
        # Feature descriptions
        st.subheader("📖 Key Features Description")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Demographic Features:**
            - `age`: Student's age (15-22)
            - `sex`: Student's gender (F/M)
            - `address`: Home address type (U=urban, R=rural)
            - `famsize`: Family size (LE3 or GT3)
            """)
        
        with col2:
            st.markdown("""
            **Academic Features:**
            - `studytime`: Weekly study time (1-4 scale)
            - `failures`: Number of past failures (0-4)
            - `absences`: Number of school absences
            - `G1, G2, G3`: Period grades (0-20)
            """)
        
        with col3:
            st.markdown("""
            **Social Features:**
            - `famsup`: Family educational support (yes/no)
            - `activities`: Extra-curricular activities (yes/no)
            - `freetime`: Free time after school (1-5)
            - `goout`: Going out with friends (1-5)
            """)
    
    # ========================================================================
    # TAB 2: STATISTICAL ANALYSIS
    # ========================================================================
    with tab2:
        st.header("📈 Statistical Analysis")
        
        # Overall Statistics
        st.subheader("1️⃣ Overall Grade Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean Grade", f"{overall_stats['mean']:.2f}/20", 
                     help="Average final grade across all students")
            st.metric("Median Grade", f"{overall_stats['median']:.2f}/20",
                     help="Middle value when grades are sorted")
        
        with col2:
            st.metric("Standard Deviation", f"{overall_stats['std']:.2f}",
                     help="Measure of grade variability")
            st.metric("Grade Range", f"{overall_stats['min']} - {overall_stats['max']}",
                     help="Minimum and maximum grades")
        
        with col3:
            st.metric("Pass Rate", f"{overall_stats['pass_rate']:.1f}%",
                     delta=f"{overall_stats['pass_rate'] - 50:.1f}% vs 50%",
                     help="Percentage of students with grade ≥ 10")
            st.metric("Fail Rate", f"{overall_stats['fail_rate']:.1f}%",
                     delta=f"{50 - overall_stats['fail_rate']:.1f}% vs 50%",
                     delta_color="inverse",
                     help="Percentage of students with grade < 10")
        
        st.markdown("---")
        
        # Performance Categories
        st.subheader("2️⃣ Performance Category Distribution")
        category_counts = df['performance_category'].value_counts()
        category_df = pd.DataFrame({
            'Category': category_counts.index,
            'Count': category_counts.values,
            'Percentage': (category_counts.values / len(df) * 100).round(1)
        })
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(category_df, use_container_width=True, hide_index=True)
        with col2:
            fig = plot_performance_pie(df)
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # Gender Analysis
        st.subheader("3️⃣ Gender-Based Performance Analysis")
        gender_stats = analyze_by_category(df, 'sex')
        gender_stats.index = ['Female', 'Male']
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(gender_stats, use_container_width=True)
            diff = abs(gender_stats.loc['Female', 'mean'] - gender_stats.loc['Male', 'mean'])
            if diff < 1:
                st.info(f"📊 Gender difference: {diff:.2f} points (minimal difference)")
            else:
                st.info(f"📊 Gender difference: {diff:.2f} points")
        
        with col2:
            fig = plot_gender_comparison(df)
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # Study Time Impact
        st.subheader("4️⃣ Study Time Impact Analysis")
        study_labels_map = {1: '<2 hours', 2: '2-5 hours', 3: '5-10 hours', 4: '>10 hours'}
        study_impact = analyze_by_category(df, 'studytime')
        study_impact.index = study_impact.index.map(study_labels_map)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(study_impact, use_container_width=True)
            min_study = study_impact['mean'].min()
            max_study = study_impact['mean'].max()
            st.success(f"✅ Students who study >10 hours score {max_study - min_study:.1f} points higher on average!")
        
        with col2:
            fig = plot_study_time_impact(df)
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # Family Support Analysis
        st.subheader("5️⃣ Family Support Impact")
        family_stats = analyze_by_category(df, 'famsup')
        family_stats.index = ['No Support', 'Yes - Support']
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(family_stats, use_container_width=True)
        with col2:
            fig = plot_family_support(df)
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # Correlation Analysis
        st.subheader("6️⃣ Correlation with Final Grade")
        correlations = get_correlation_data(df)
        corr_df = pd.DataFrame({
            'Feature': correlations.index,
            'Correlation': correlations.values
        }).round(3)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(corr_df, use_container_width=True, height=400)
            st.info("""
            **Interpretation:**
            - Values close to +1: Strong positive relationship
            - Values close to -1: Strong negative relationship
            - Values close to 0: Weak/no relationship
            """)
        
        with col2:
            fig = plot_correlation_heatmap(df)
            st.pyplot(fig)
            plt.close()
    
    # ========================================================================
    # TAB 3: VISUALIZATIONS
    # ========================================================================
    with tab3:
        st.header("📊 Data Visualizations")
        
        # Grade Distribution
        st.subheader("📈 Final Grade Distribution")
        fig = plot_grade_distribution(df)
        st.pyplot(fig)
        plt.close()
        st.caption("This histogram shows how final grades are distributed across all students. The red dashed line indicates the mean grade.")
        
        st.markdown("---")
        
        # Two column layout for comparisons
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⏰ Study Time vs Performance")
            fig = plot_study_time_impact(df)
            st.pyplot(fig)
            plt.close()
            st.caption("Clear positive correlation: more study time leads to better grades")
            
            st.subheader("👥 Gender Comparison")
            fig = plot_gender_comparison(df)
            st.pyplot(fig)
            plt.close()
            st.caption("Comparison of average performance between male and female students")
        
        with col2:
            st.subheader("👨‍👩‍👧 Family Support Impact")
            fig = plot_family_support(df)
            st.pyplot(fig)
            plt.close()
            st.caption("Students with family support tend to perform better")
            
            st.subheader("📉 Past Failures Impact")
            fig = plot_failures_impact(df)
            st.pyplot(fig)
            plt.close()
            st.caption("Past failures strongly correlate with lower current performance")
        
        st.markdown("---")
        
        # Correlation Heatmap (full width)
        st.subheader("🔥 Feature Correlation Matrix")
        fig = plot_correlation_heatmap(df)
        st.pyplot(fig)
        plt.close()
        st.caption("This heatmap shows relationships between different features. Red indicates positive correlation, blue indicates negative correlation.")
    
    # ========================================================================
    # TAB 4: KEY INSIGHTS
    # ========================================================================
    with tab4:
        st.header("💡 Key Insights & Conclusions")
        
        # Calculate key statistics for insights
        study_means = df.groupby('studytime')['G3'].mean()
        gender_means = df.groupby('sex')['G3'].mean()
        family_means = df.groupby('famsup')['G3'].mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Overall Performance")
            st.markdown(f"""
            - **Average Final Grade**: {overall_stats['mean']:.2f}/20
            - **Pass Rate**: {overall_stats['pass_rate']:.1f}% ({int(df[df['G3'] >= 10].shape[0])} students passed)
            - **Fail Rate**: {overall_stats['fail_rate']:.1f}% ({int(df[df['G3'] < 10].shape[0])} students failed)
            - **Grade Range**: {overall_stats['min']} to {overall_stats['max']}
            - **Standard Deviation**: {overall_stats['std']:.2f} (moderate variability)
            """)
            
            st.subheader("⏰ Study Time Findings")
            st.markdown(f"""
            - Students studying **>10 hours/week**: {study_means[4]:.1f} average
            - Students studying **<2 hours/week**: {study_means[1]:.1f} average
            - **Performance Gap**: {study_means[4] - study_means[1]:.1f} points
            - **Conclusion**: Strong positive correlation between study time and grades
            """)
            
            st.subheader("👨‍👩‍👧 Family Support Impact")
            st.markdown(f"""
            - **With family support**: {family_means['yes']:.2f} average grade
            - **Without support**: {family_means['no']:.2f} average grade
            - **Difference**: {abs(family_means['yes'] - family_means['no']):.2f} points
            - **Conclusion**: Family support positively impacts academic performance
            """)
        
        with col2:
            st.subheader("👥 Gender Analysis")
            st.markdown(f"""
            - **Female students**: {gender_means['F']:.2f} average
            - **Male students**: {gender_means['M']:.2f} average
            - **Difference**: {abs(gender_means['F'] - gender_means['M']):.2f} points
            - **Conclusion**: Minimal gender difference in performance
            """)
            
            st.subheader("🎯 Strongest Performance Predictors")
            top_correlations = get_correlation_data(df).head(5)
            st.markdown("**Based on correlation analysis:**")
            for i, (feature, corr) in enumerate(top_correlations.items(), 1):
                if feature != 'G3':
                    st.markdown(f"{i}. **{feature}**: {corr:.3f} correlation")
            
            st.subheader("📉 Past Failures Effect")
            failures_impact = df.groupby('failures')['G3'].mean()
            st.markdown(f"""
            - **No failures**: {failures_impact[0]:.1f} average
            - **1+ failures**: {failures_impact[failures_impact.index > 0].mean():.1f} average
            - **Conclusion**: Past failures significantly predict lower performance
            """)
        
        st.markdown("---")
        
        # Recommendations
        st.subheader("📝 Recommendations for Educational Improvement")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **For Students:**
            - 📚 Maintain 5+ hours of weekly study
            - 🎯 Seek help early if struggling
            - 👨‍👩‍👧 Engage family in your education
            - ⏰ Reduce absences
            - 📊 Track your progress regularly
            """)
        
        with col2:
            st.markdown("""
            **For Teachers:**
            - 🔍 Monitor first period grades closely
            - 📉 Provide extra support for students with past failures
            - 👥 Encourage peer study groups
            - 📱 Maintain regular parent communication
            - 🎓 Identify at-risk students early
            """)
        
        with col3:
            st.markdown("""
            **For Parents:**
            - 💬 Stay involved in your child's education
            - 📅 Create structured study schedules
            - 🏠 Provide a conducive learning environment
            - 🤝 Communicate with teachers regularly
            - 🎯 Set realistic academic goals
            """)
        
        st.markdown("---")
        
        # Summary
        st.success("""
        ### 🎯 Summary
        
        This analysis reveals that **study time, previous academic performance, and family support** are the most 
        significant factors affecting student success in mathematics. Students who study consistently, have no past 
        failures, and receive family support tend to perform significantly better. Early intervention and continuous 
        monitoring from the first period can help identify and support struggling students before it's too late.
        """)
    
    # ========================================================================
    # TAB 5: DATA EXPLORER
    # ========================================================================
    with tab5:
        st.header("🔍 Interactive Data Explorer")
        st.markdown("Filter and explore the dataset based on different criteria")
        
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            gender_filter = st.multiselect(
                "Gender",
                options=['F', 'M'],
                default=['F', 'M']
            )
        
        with col2:
            studytime_filter = st.multiselect(
                "Study Time",
                options=[1, 2, 3, 4],
                default=[1, 2, 3, 4],
                format_func=lambda x: {1: '<2h', 2: '2-5h', 3: '5-10h', 4: '>10h'}[x]
            )
        
        with col3:
            famsup_filter = st.multiselect(
                "Family Support",
                options=['yes', 'no'],
                default=['yes', 'no']
            )
        
        with col4:
            grade_range = st.slider(
                "Final Grade Range",
                min_value=0,
                max_value=20,
                value=(0, 20)
            )
        
        # Apply filters
        filtered_df = df[
            (df['sex'].isin(gender_filter)) &
            (df['studytime'].isin(studytime_filter)) &
            (df['famsup'].isin(famsup_filter)) &
            (df['G3'] >= grade_range[0]) &
            (df['G3'] <= grade_range[1])
        ]
        
        st.subheader(f"Filtered Results: {len(filtered_df)} students")
        
        # Show filtered statistics
        if len(filtered_df) > 0:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average Grade", f"{filtered_df['G3'].mean():.2f}")
            with col2:
                st.metric("Pass Rate", f"{(filtered_df['G3'] >= 10).sum() / len(filtered_df) * 100:.1f}%")
            with col3:
                st.metric("Median", f"{filtered_df['G3'].median():.1f}")
            with col4:
                st.metric("Std Dev", f"{filtered_df['G3'].std():.2f}")
            
            st.dataframe(filtered_df, use_container_width=True, height=400)
            
            # Download filtered data
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv,
                file_name="filtered_student_data.csv",
                mime="text/csv"
            )
        else:
            st.warning("No students match the selected filters. Try adjusting your criteria.")

# ============================================================================
# FOOTER
# ============================================================================

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Student Performance Analyzer</strong> | DAI011 - Programming for AI</p>
        <p>Dataset: UCI Machine Learning Repository</p>
        <p>Libraries: Pandas • NumPy • Matplotlib • Seaborn • Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    main()
