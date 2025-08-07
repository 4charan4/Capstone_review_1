import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
import random

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        padding: 2rem 0;
        border-bottom: 3px solid #2ecc71;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2ecc71;
        margin: 1.5rem 0 1rem 0;
        border-left: 4px solid #1f77b4;
        padding-left: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2ecc71;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and header
st.markdown('<div class="main-header">❤️ Heart Disease Prediction - Data Exploration & Feature Selection Demo</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## 📊 Navigation")
st.sidebar.markdown("Upload your heart disease dataset to begin analysis")

# File uploader
uploaded_file = st.file_uploader(
    "📁 Upload Heart Disease Dataset (CSV)", 
    type=['csv'],
    help="Upload a CSV file containing heart disease data"
)

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # Dataset Preview Section
    st.markdown('<div class="sub-header">📋 Dataset Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Records", df.shape[0])
    with col2:
        st.metric("📈 Features", df.shape[1])
    with col3:
        st.metric("❌ Missing Values", df.isnull().sum().sum())
    with col4:
        if 'HeartDisease' in df.columns:
            positive_cases = df['HeartDisease'].sum()
            st.metric("❤️ Positive Cases", f"{positive_cases} ({positive_cases/len(df)*100:.1f}%)")

    # Display dataset
    with st.expander("🔍 View Dataset Sample", expanded=False):
        st.dataframe(df.head(10))
        
    # Dataset Information
    st.markdown('<div class="sub-header">ℹ️ Dataset Information</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Data Types:**")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count()
        })
        st.dataframe(dtype_df)
    
    with col2:
        st.write("**Missing Values Analysis:**")
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum(),
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(missing_df[missing_df['Missing Count'] > 0])

    # Identify column types
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Target variable identification
    target_col = 'HeartDisease' if 'HeartDisease' in df.columns else df.columns[-1]
    
    # Data Distribution Section
    st.markdown('<div class="sub-header">📊 Data Distribution Analysis</div>', unsafe_allow_html=True)
    
    # Categorical features distribution
    if categorical_cols:
        st.write("### 🏷️ Categorical Features Distribution")
        n_cols = min(3, len(categorical_cols))
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        fig_cat, axes_cat = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes_cat = [axes_cat] if n_cols == 1 else axes_cat
        else:
            axes_cat = axes_cat.flatten()
        
        for i, col in enumerate(categorical_cols):
            if i < len(axes_cat):
                df[col].value_counts().plot(kind='bar', ax=axes_cat[i], color='#2ecc71')
                axes_cat[i].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
                axes_cat[i].tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        for j in range(len(categorical_cols), len(axes_cat)):
            axes_cat[j].set_visible(False)
            
        plt.tight_layout()
        st.pyplot(fig_cat)

    # Encode categorical variables for correlation analysis
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in categorical_cols:
        df_encoded[col] = le.fit_transform(df[col])

    # Correlation Analysis
    st.markdown('<div class="sub-header">🔥 Feature Correlation Analysis</div>', unsafe_allow_html=True)
    
    correlation_matrix = df_encoded.corr()
    
    # Correlation heatmap
    fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, ax=ax_corr, fmt='.2f')
    ax_corr.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    st.pyplot(fig_corr)

    # Top correlations with target
    if target_col in correlation_matrix.columns:
        target_corr = correlation_matrix[target_col].abs().sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("### 🎯 Top Features Correlated with Heart Disease")
            corr_df = pd.DataFrame({
                'Feature': target_corr[1:6].index,
                'Correlation': target_corr[1:6].values
            })
            st.dataframe(corr_df)
        
        with col2:
            fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
            target_corr[1:6].plot(kind='barh', ax=ax_bar, color='#1f77b4')
            ax_bar.set_title('Top 5 Feature Correlations with Heart Disease')
            ax_bar.set_xlabel('Absolute Correlation')
            plt.tight_layout()
            st.pyplot(fig_bar)

    # Feature Selection Section
    st.markdown('<div class="sub-header">🎯 Feature Selection Analysis</div>', unsafe_allow_html=True)
    
    # Prepare data
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Mutual Information Feature Selection
    st.write("### 🧠 Mutual Information Feature Ranking")
    
    with st.spinner('Calculating Mutual Information scores...'):
        mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
        mi_ranking = pd.DataFrame({
            'Feature': X.columns, 
            'MI_Score': mi_scores
        }).sort_values('MI_Score', ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(mi_ranking)
    
    with col2:
        fig_mi, ax_mi = plt.subplots(figsize=(8, 6))
        sns.barplot(data=mi_ranking.head(8), x='MI_Score', y='Feature', 
                   palette='viridis', ax=ax_mi)
        ax_mi.set_title('Top 8 Features by Mutual Information Score')
        ax_mi.set_xlabel('MI Score')
        plt.tight_layout()
        st.pyplot(fig_mi)

    top_features_mi = mi_ranking.head(8)['Feature'].tolist()
    X_train_mi = X_train[top_features_mi]
    X_test_mi = X_test[top_features_mi]

    # Simplified ACO Implementation
    st.write("### 🐜 Ant Colony Optimization Feature Selection")
    
    class SimpleACO:
        def __init__(self, n_ants=10, n_iterations=20, alpha=1.0, beta=2.0, evaporation=0.1):
            self.n_ants = n_ants
            self.n_iterations = n_iterations
            self.alpha = alpha
            self.beta = beta
            self.evaporation = evaporation
        
        def fit(self, X, y, classifier):
            n_features = X.shape[1]
            pheromone = np.ones(n_features)
            best_features = None
            best_score = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for iteration in range(self.n_iterations):
                for ant in range(self.n_ants):
                    probabilities = pheromone ** self.alpha
                    probabilities = probabilities / probabilities.sum()
                    
                    selected_size = min(8, n_features)
                    selected_features = np.random.choice(
                        n_features, size=selected_size, 
                        replace=False, p=probabilities
                    )
                    
                    X_subset = X.iloc[:, selected_features]
                    score = self._evaluate_features(X_subset, y, classifier)
                    
                    if score > best_score:
                        best_score = score
                        best_features = selected_features
                    
                    pheromone[selected_features] += score
                
                pheromone *= (1 - self.evaporation)
                
                # Update progress
                progress = (iteration + 1) / self.n_iterations
                progress_bar.progress(progress)
                status_text.text(f'ACO Progress: {iteration+1}/{self.n_iterations} iterations, Best Score: {best_score:.4f}')
            
            progress_bar.empty()
            status_text.empty()
            return best_features, best_score
        
        def _evaluate_features(self, X, y, classifier):
            scores = cross_val_score(classifier, X, y, cv=3, scoring='accuracy')
            return scores.mean()

    with st.spinner('Running ACO feature selection...'):
        aco = SimpleACO(n_iterations=15)  # Reduced for faster demo
        classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        aco_features_idx, aco_score = aco.fit(X_train, y_train, classifier)

    aco_features = X_train.columns[aco_features_idx].tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**ACO Selected Features** (CV Score: {aco_score:.4f})")
        for i, feature in enumerate(aco_features, 1):
            st.write(f"{i}. {feature}")
    
    with col2:
        # Comparison of selected features
        comparison_df = pd.DataFrame({
            'MI Selected': top_features_mi[:8],
            'ACO Selected': aco_features[:8]
        })
        st.write("**Feature Selection Comparison:**")
        st.dataframe(comparison_df)

    X_train_aco = X_train[aco_features]
    X_test_aco = X_test[aco_features]

    # Model Performance Comparison
    st.markdown('<div class="sub-header">🏆 Model Performance Comparison</div>', unsafe_allow_html=True)

    classifiers = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }

    results = {}
    
    with st.spinner('Training models and evaluating performance...'):
        # Original features
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[f'{name}_Original'] = accuracy

        # MI selected features
        for name, clf in classifiers.items():
            clf.fit(X_train_mi, y_train)
            y_pred = clf.predict(X_test_mi)
            accuracy = accuracy_score(y_test, y_pred)
            results[f'{name}_MI'] = accuracy

        # ACO selected features
        for name, clf in classifiers.items():
            clf.fit(X_train_aco, y_train)
            y_pred = clf.predict(X_test_aco)
            accuracy = accuracy_score(y_test, y_pred)
            results[f'{name}_ACO'] = accuracy

    results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
    
    # Display results table
    st.write("### 📈 Accuracy Results")
    results_pivot = results_df.copy()
    results_pivot[['Classifier', 'Method']] = results_pivot['Model'].str.rsplit('_', n=1, expand=True)
    results_pivot = results_pivot.pivot(index='Classifier', columns='Method', values='Accuracy')
    
    # Style the dataframe
    styled_results = results_pivot.style.format('{:.4f}').background_gradient(
        cmap='RdYlGn', subset=['Original', 'MI', 'ACO']
    )
    st.dataframe(styled_results)

    # Performance visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
        results_pivot.plot(kind='bar', ax=ax_acc, color=['#1f77b4', '#2ecc71', '#34495e'])
        ax_acc.set_title('Accuracy Comparison by Feature Selection Method', fontweight='bold')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.legend(title='Method')
        ax_acc.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig_acc)
    
    with col2:
        # Best model selection
        best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
        best_accuracy = results_df['Accuracy'].max()
        
        st.success(f"🏆 **Best Model:** {best_model_name}")
        st.success(f"🎯 **Best Accuracy:** {best_accuracy:.4f}")
        
        # Performance metrics
        method = best_model_name.split('_')[-1]
        classifier_name = best_model_name.replace(f'_{method}', '')
        
        if method == 'MI':
            X_train_best, X_test_best = X_train_mi, X_test_mi
        elif method == 'ACO':
            X_train_best, X_test_best = X_train_aco, X_test_aco
        else:
            X_train_best, X_test_best = X_train, X_test
        
        best_clf = classifiers[classifier_name]
        best_clf.fit(X_train_best, y_train)
        y_pred_best = best_clf.predict(X_test_best)

    # Detailed Analysis of Best Model
    st.markdown('<div class="sub-header">🔬 Detailed Analysis - Best Model</div>', unsafe_allow_html=True)
    
    # Feature importance (if Random Forest)
    if 'Random Forest' in best_model_name:
        st.write("### 🌳 Feature Importance Analysis")
        feature_names = X_train_best.columns
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': best_clf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(importance_df)
        
        with col2:
            fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
            sns.barplot(data=importance_df, x='Importance', y='Feature', 
                       palette='viridis', ax=ax_imp)
            ax_imp.set_title('Feature Importance (Random Forest)')
            plt.tight_layout()
            st.pyplot(fig_imp)

    # Model evaluation visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 📊 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred_best)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('Actual')
        ax_cm.set_title('Confusion Matrix')
        st.pyplot(fig_cm)
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        
        st.write("**Performance Metrics:**")
        st.write(f"- **Sensitivity (Recall):** {sensitivity:.4f}")
        st.write(f"- **Specificity:** {specificity:.4f}")
        st.write(f"- **Accuracy:** {best_accuracy:.4f}")
    
    with col2:
        st.write("### 📈 ROC Curve")
        if hasattr(best_clf, 'predict_proba'):
            y_pred_proba = best_clf.predict_proba(X_test_best)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
            ax_roc.plot(fpr, tpr, color='#2ecc71', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax_roc.plot([0, 1], [0, 1], color='#34495e', lw=2, linestyle='--')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('ROC Curve')
            ax_roc.legend()
            plt.tight_layout()
            st.pyplot(fig_roc)
            
            st.write(f"**AUC Score:** {roc_auc:.4f}")

    # Summary and Conclusions
    st.markdown('<div class="sub-header">📋 Summary & Key Insights</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Feature Selection Impact**")
        original_acc = results_pivot.loc[:, 'Original'].max()
        mi_acc = results_pivot.loc[:, 'MI'].max()
        aco_acc = results_pivot.loc[:, 'ACO'].max()
        
        st.write(f"- Original Features: {original_acc:.4f}")
        st.write(f"- MI Selection: {mi_acc:.4f}")
        st.write(f"- ACO Selection: {aco_acc:.4f}")
        
        improvement = max(mi_acc, aco_acc) - original_acc
        if improvement > 0:
            st.success(f"Improvement: +{improvement:.4f}")
        else:
            st.warning("No significant improvement")
    
    with col2:
        st.info("**Best Performing Model**")
        st.write(f"- **Algorithm:** {classifier_name}")
        st.write(f"- **Feature Selection:** {method}")
        st.write(f"- **Accuracy:** {best_accuracy:.4f}")
        st.write(f"- **Features Used:** {len(X_train_best.columns)}")
    
    with col3:
        st.info("**Recommendations**")
        if method == 'MI':
            st.write("- Mutual Information ranking shows best results")
            st.write("- Statistical approach preferred")
        elif method == 'ACO':
            st.write("- Ant Colony Optimization effective")
            st.write("- Bio-inspired approach successful")
        else:
            st.write("- All features provide best performance")
            st.write("- No dimensionality reduction needed")

else:
    # Instructions when no file is uploaded
    st.info("👆 Please upload a heart disease dataset CSV file to begin the analysis")
    
    st.markdown("### 📖 Expected Dataset Format:")
    st.markdown("""
    Your CSV file should contain the following columns:
    - **Age**: Patient age
    - **Sex**: M/F or 1/0
    - **ChestPainType**: Type of chest pain
    - **RestingBP**: Resting blood pressure
    - **Cholesterol**: Serum cholesterol level
    - **FastingBS**: Fasting blood sugar
    - **RestingECG**: Resting electrocardiogram
    - **MaxHR**: Maximum heart rate achieved
    - **ExerciseAngina**: Exercise induced angina
    - **Oldpeak**: ST depression
    - **ST_Slope**: Slope of peak exercise ST segment
    - **HeartDisease**: Target variable (0/1)
    """)
    
    st.markdown("### 🎯 What This App Does:")
    st.markdown("""
    1. **Data Exploration**: Comprehensive dataset analysis and visualization
    2. **Feature Selection**: Mutual Information ranking and Ant Colony Optimization
    3. **Model Training**: Decision Tree, Random Forest, and SVM classifiers
    4. **Performance Comparison**: Accuracy analysis across different feature selection methods
    5. **Detailed Results**: Confusion matrix, ROC curves, and feature importance analysis
    """)

# Footer
st.markdown("---")
st.markdown("**Heart Disease Prediction App** | Built for Review 1 Demo | Team: CHARAN B, DEBIKA N S, MABITHA V")
