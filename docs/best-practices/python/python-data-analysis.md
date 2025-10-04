# Python Data Analysis Best Practices

**Objective**: Master senior-level Python data analysis patterns for production systems. When you need to perform comprehensive data analysis, when you want to implement reproducible research workflows, when you need enterprise-grade data analysis strategies—these best practices become your weapon of choice.

## Core Principles

- **Reproducibility**: Ensure analysis can be reproduced and validated
- **Performance**: Optimize for large datasets and complex computations
- **Visualization**: Create clear, informative visualizations
- **Documentation**: Document analysis methodology and findings
- **Validation**: Verify results through multiple approaches

## Data Analysis Workflows

### Exploratory Data Analysis

```python
# python/01-exploratory-data-analysis.py

"""
Exploratory data analysis patterns and techniques
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

class DataAnalyzer:
    """Comprehensive data analyzer"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.analysis_results = {}
        self.visualizations = {}
        self.summary_stats = {}
    
    def basic_info(self) -> Dict[str, Any]:
        """Get basic dataset information"""
        info = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "memory_usage": self.data.memory_usage(deep=True).sum(),
            "null_counts": self.data.isnull().sum().to_dict(),
            "duplicate_rows": self.data.duplicated().sum()
        }
        
        self.summary_stats["basic_info"] = info
        return info
    
    def descriptive_stats(self) -> Dict[str, Any]:
        """Get descriptive statistics"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        
        stats = {
            "numeric_summary": self.data[numeric_cols].describe().to_dict(),
            "categorical_summary": {}
        }
        
        for col in categorical_cols:
            stats["categorical_summary"][col] = {
                "unique_count": self.data[col].nunique(),
                "most_frequent": self.data[col].mode().iloc[0] if not self.data[col].mode().empty else None,
                "frequency": self.data[col].value_counts().head().to_dict()
            }
        
        self.summary_stats["descriptive"] = stats
        return stats
    
    def correlation_analysis(self) -> Dict[str, Any]:
        """Perform correlation analysis"""
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return {"error": "No numeric columns found"}
        
        correlation_matrix = numeric_data.corr()
        
        # Find high correlations
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # High correlation threshold
                    high_correlations.append({
                        "var1": correlation_matrix.columns[i],
                        "var2": correlation_matrix.columns[j],
                        "correlation": corr_value
                    })
        
        analysis = {
            "correlation_matrix": correlation_matrix.to_dict(),
            "high_correlations": high_correlations,
            "strongest_correlation": max(high_correlations, key=lambda x: abs(x["correlation"])) if high_correlations else None
        }
        
        self.analysis_results["correlation"] = analysis
        return analysis
    
    def outlier_detection(self, method: str = "iqr") -> Dict[str, Any]:
        """Detect outliers in the data"""
        numeric_data = self.data.select_dtypes(include=[np.number])
        outliers = {}
        
        for col in numeric_data.columns:
            if method == "iqr":
                Q1 = numeric_data[col].quantile(0.25)
                Q3 = numeric_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (numeric_data[col] < lower_bound) | (numeric_data[col] > upper_bound)
            elif method == "zscore":
                z_scores = np.abs(stats.zscore(numeric_data[col]))
                outlier_mask = z_scores > 3
            else:
                continue
            
            outliers[col] = {
                "count": outlier_mask.sum(),
                "percentage": (outlier_mask.sum() / len(numeric_data)) * 100,
                "indices": numeric_data[outlier_mask].index.tolist()
            }
        
        self.analysis_results["outliers"] = outliers
        return outliers
    
    def missing_data_analysis(self) -> Dict[str, Any]:
        """Analyze missing data patterns"""
        missing_data = self.data.isnull().sum()
        missing_percentage = (missing_data / len(self.data)) * 100
        
        analysis = {
            "missing_counts": missing_data.to_dict(),
            "missing_percentages": missing_percentage.to_dict(),
            "columns_with_missing": missing_data[missing_data > 0].to_dict(),
            "complete_cases": len(self.data.dropna()),
            "complete_case_percentage": (len(self.data.dropna()) / len(self.data)) * 100
        }
        
        # Analyze missing data patterns
        if missing_data.sum() > 0:
            missing_patterns = self.data.isnull().sum(axis=1).value_counts().sort_index()
            analysis["missing_patterns"] = missing_patterns.to_dict()
        
        self.analysis_results["missing_data"] = analysis
        return analysis
    
    def distribution_analysis(self) -> Dict[str, Any]:
        """Analyze data distributions"""
        numeric_data = self.data.select_dtypes(include=[np.number])
        distributions = {}
        
        for col in numeric_data.columns:
            # Normality tests
            shapiro_stat, shapiro_p = stats.shapiro(numeric_data[col].dropna())
            ks_stat, ks_p = stats.kstest(numeric_data[col].dropna(), 'norm')
            
            distributions[col] = {
                "mean": numeric_data[col].mean(),
                "median": numeric_data[col].median(),
                "std": numeric_data[col].std(),
                "skewness": stats.skew(numeric_data[col].dropna()),
                "kurtosis": stats.kurtosis(numeric_data[col].dropna()),
                "shapiro_test": {"statistic": shapiro_stat, "p_value": shapiro_p},
                "ks_test": {"statistic": ks_stat, "p_value": ks_p},
                "is_normal": shapiro_p > 0.05
            }
        
        self.analysis_results["distributions"] = distributions
        return distributions

class DataVisualizer:
    """Data visualization utilities"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.figures = {}
    
    def plot_distributions(self, columns: List[str] = None) -> go.Figure:
        """Plot distributions of numeric columns"""
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        fig = make_subplots(
            rows=len(columns), cols=2,
            subplot_titles=[f"{col} - Histogram" for col in columns] + 
                          [f"{col} - Box Plot" for col in columns]
        )
        
        for i, col in enumerate(columns):
            # Histogram
            fig.add_trace(
                go.Histogram(x=self.data[col], name=f"{col}_hist"),
                row=i+1, col=1
            )
            
            # Box plot
            fig.add_trace(
                go.Box(y=self.data[col], name=f"{col}_box"),
                row=i+1, col=2
            )
        
        fig.update_layout(height=300*len(columns), showlegend=False)
        return fig
    
    def plot_correlation_heatmap(self) -> go.Figure:
        """Plot correlation heatmap"""
        numeric_data = self.data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title="Correlation Heatmap",
            xaxis_title="Variables",
            yaxis_title="Variables"
        )
        
        return fig
    
    def plot_missing_data(self) -> go.Figure:
        """Plot missing data patterns"""
        missing_data = self.data.isnull().sum()
        missing_percentage = (missing_data / len(self.data)) * 100
        
        fig = go.Figure(data=[
            go.Bar(x=missing_data.index, y=missing_percentage.values)
        ])
        
        fig.update_layout(
            title="Missing Data by Column",
            xaxis_title="Columns",
            yaxis_title="Missing Percentage (%)"
        )
        
        return fig
    
    def plot_time_series(self, date_col: str, value_col: str) -> go.Figure:
        """Plot time series data"""
        fig = go.Figure(data=[
            go.Scatter(x=self.data[date_col], y=self.data[value_col], mode='lines+markers')
        ])
        
        fig.update_layout(
            title=f"Time Series: {value_col}",
            xaxis_title=date_col,
            yaxis_title=value_col
        )
        
        return fig

class StatisticalAnalyzer:
    """Statistical analysis utilities"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.test_results = {}
    
    def t_test(self, group_col: str, value_col: str) -> Dict[str, Any]:
        """Perform t-test between groups"""
        groups = self.data[group_col].unique()
        if len(groups) != 2:
            return {"error": "T-test requires exactly 2 groups"}
        
        group1_data = self.data[self.data[group_col] == groups[0]][value_col].dropna()
        group2_data = self.data[self.data[group_col] == groups[1]][value_col].dropna()
        
        t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
        
        result = {
            "groups": groups.tolist(),
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "group1_mean": group1_data.mean(),
            "group2_mean": group2_data.mean(),
            "group1_std": group1_data.std(),
            "group2_std": group2_data.std()
        }
        
        self.test_results["t_test"] = result
        return result
    
    def chi_square_test(self, var1: str, var2: str) -> Dict[str, Any]:
        """Perform chi-square test of independence"""
        contingency_table = pd.crosstab(self.data[var1], self.data[var2])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        result = {
            "chi2_statistic": chi2,
            "p_value": p_value,
            "degrees_of_freedom": dof,
            "significant": p_value < 0.05,
            "contingency_table": contingency_table.to_dict(),
            "expected_frequencies": expected.tolist()
        }
        
        self.test_results["chi_square"] = result
        return result
    
    def anova_test(self, group_col: str, value_col: str) -> Dict[str, Any]:
        """Perform ANOVA test"""
        groups = self.data[group_col].unique()
        group_data = [self.data[self.data[group_col] == group][value_col].dropna() for group in groups]
        
        f_stat, p_value = stats.f_oneway(*group_data)
        
        result = {
            "groups": groups.tolist(),
            "f_statistic": f_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "group_means": {group: data.mean() for group, data in zip(groups, group_data)},
            "group_stds": {group: data.std() for group, data in zip(groups, group_data)}
        }
        
        self.test_results["anova"] = result
        return result
    
    def regression_analysis(self, x_col: str, y_col: str) -> Dict[str, Any]:
        """Perform simple linear regression"""
        x = self.data[x_col].dropna()
        y = self.data[y_col].dropna()
        
        # Align data
        common_index = x.index.intersection(y.index)
        x_aligned = x.loc[common_index]
        y_aligned = y.loc[common_index]
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_aligned, y_aligned)
        
        result = {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,
            "p_value": p_value,
            "std_error": std_err,
            "significant": p_value < 0.05,
            "correlation": r_value
        }
        
        self.test_results["regression"] = result
        return result

# Usage examples
def example_data_analysis():
    """Example data analysis usage"""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'satisfaction': np.random.normal(7, 2, n_samples)
    })
    
    # Create analyzer
    analyzer = DataAnalyzer(data)
    
    # Basic analysis
    basic_info = analyzer.basic_info()
    print(f"Dataset shape: {basic_info['shape']}")
    print(f"Memory usage: {basic_info['memory_usage']} bytes")
    
    # Descriptive statistics
    desc_stats = analyzer.descriptive_stats()
    print(f"Numeric summary: {desc_stats['numeric_summary']}")
    
    # Correlation analysis
    correlation = analyzer.correlation_analysis()
    print(f"High correlations: {correlation['high_correlations']}")
    
    # Outlier detection
    outliers = analyzer.outlier_detection(method="iqr")
    print(f"Outliers detected: {outliers}")
    
    # Missing data analysis
    missing = analyzer.missing_data_analysis()
    print(f"Missing data: {missing['missing_counts']}")
    
    # Distribution analysis
    distributions = analyzer.distribution_analysis()
    print(f"Distribution analysis: {distributions}")
    
    # Create visualizer
    visualizer = DataVisualizer(data)
    
    # Create plots
    dist_plot = visualizer.plot_distributions(['age', 'income'])
    corr_plot = visualizer.plot_correlation_heatmap()
    
    # Statistical analysis
    stat_analyzer = StatisticalAnalyzer(data)
    
    # T-test
    t_test_result = stat_analyzer.t_test('gender', 'income')
    print(f"T-test result: {t_test_result}")
    
    # Chi-square test
    chi2_result = stat_analyzer.chi_square_test('gender', 'education')
    print(f"Chi-square result: {chi2_result}")
    
    # Regression analysis
    regression_result = stat_analyzer.regression_analysis('age', 'income')
    print(f"Regression result: {regression_result}")
```

### Advanced Analytics

```python
# python/02-advanced-analytics.py

"""
Advanced analytics patterns including clustering, dimensionality reduction, and time series analysis
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, t-SNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

class ClusteringAnalyzer:
    """Advanced clustering analysis"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.numeric_data = data.select_dtypes(include=[np.number])
        self.scaler = StandardScaler()
        self.scaled_data = None
        self.cluster_results = {}
    
    def prepare_data(self) -> None:
        """Prepare data for clustering"""
        self.scaled_data = self.scaler.fit_transform(self.numeric_data)
    
    def kmeans_clustering(self, n_clusters: int = 3) -> Dict[str, Any]:
        """Perform K-means clustering"""
        if self.scaled_data is None:
            self.prepare_data()
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.scaled_data)
        
        # Calculate metrics
        silhouette_avg = silhouette_score(self.scaled_data, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(self.scaled_data, cluster_labels)
        
        result = {
            "n_clusters": n_clusters,
            "cluster_labels": cluster_labels.tolist(),
            "cluster_centers": kmeans.cluster_centers_.tolist(),
            "silhouette_score": silhouette_avg,
            "calinski_harabasz_score": calinski_harabasz,
            "inertia": kmeans.inertia_
        }
        
        self.cluster_results["kmeans"] = result
        return result
    
    def dbscan_clustering(self, eps: float = 0.5, min_samples: int = 5) -> Dict[str, Any]:
        """Perform DBSCAN clustering"""
        if self.scaled_data is None:
            self.prepare_data()
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(self.scaled_data)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        result = {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "cluster_labels": cluster_labels.tolist(),
            "eps": eps,
            "min_samples": min_samples
        }
        
        self.cluster_results["dbscan"] = result
        return result
    
    def hierarchical_clustering(self, n_clusters: int = 3) -> Dict[str, Any]:
        """Perform hierarchical clustering"""
        if self.scaled_data is None:
            self.prepare_data()
        
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = hierarchical.fit_predict(self.scaled_data)
        
        result = {
            "n_clusters": n_clusters,
            "cluster_labels": cluster_labels.tolist(),
            "linkage": "ward"
        }
        
        self.cluster_results["hierarchical"] = result
        return result
    
    def find_optimal_clusters(self, max_clusters: int = 10) -> Dict[str, Any]:
        """Find optimal number of clusters using elbow method and silhouette analysis"""
        if self.scaled_data is None:
            self.prepare_data()
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(self.scaled_data)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.scaled_data, cluster_labels))
        
        # Find optimal k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        result = {
            "k_range": list(k_range),
            "inertias": inertias,
            "silhouette_scores": silhouette_scores,
            "optimal_k": optimal_k,
            "max_silhouette_score": max(silhouette_scores)
        }
        
        self.cluster_results["optimal_clusters"] = result
        return result

class DimensionalityReducer:
    """Dimensionality reduction analysis"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.numeric_data = data.select_dtypes(include=[np.number])
        self.scaler = StandardScaler()
        self.scaled_data = None
        self.reduction_results = {}
    
    def prepare_data(self) -> None:
        """Prepare data for dimensionality reduction"""
        self.scaled_data = self.scaler.fit_transform(self.numeric_data)
    
    def pca_analysis(self, n_components: int = 2) -> Dict[str, Any]:
        """Perform PCA analysis"""
        if self.scaled_data is None:
            self.prepare_data()
        
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(self.scaled_data)
        
        result = {
            "n_components": n_components,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "components": pca.components_.tolist(),
            "transformed_data": pca_result.tolist()
        }
        
        self.reduction_results["pca"] = result
        return result
    
    def tsne_analysis(self, n_components: int = 2, perplexity: float = 30) -> Dict[str, Any]:
        """Perform t-SNE analysis"""
        if self.scaled_data is None:
            self.prepare_data()
        
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        tsne_result = tsne.fit_transform(self.scaled_data)
        
        result = {
            "n_components": n_components,
            "perplexity": perplexity,
            "transformed_data": tsne_result.tolist(),
            "kl_divergence": tsne.kl_divergence_
        }
        
        self.reduction_results["tsne"] = result
        return result
    
    def find_optimal_components(self, max_components: int = None) -> Dict[str, Any]:
        """Find optimal number of components for PCA"""
        if self.scaled_data is None:
            self.prepare_data()
        
        if max_components is None:
            max_components = min(self.scaled_data.shape[1], 10)
        
        pca_full = PCA()
        pca_full.fit(self.scaled_data)
        
        explained_variance_ratio = pca_full.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Find number of components that explain 95% of variance
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        result = {
            "explained_variance_ratio": explained_variance_ratio.tolist(),
            "cumulative_variance": cumulative_variance.tolist(),
            "n_components_95_variance": n_components_95,
            "total_variance_explained": cumulative_variance[-1]
        }
        
        self.reduction_results["optimal_components"] = result
        return result

class TimeSeriesAnalyzer:
    """Time series analysis utilities"""
    
    def __init__(self, data: pd.DataFrame, date_col: str, value_col: str):
        self.data = data.copy()
        self.date_col = date_col
        self.value_col = value_col
        self.time_series = None
        self.analysis_results = {}
    
    def prepare_time_series(self) -> None:
        """Prepare time series data"""
        self.data[self.date_col] = pd.to_datetime(self.data[self.date_col])
        self.data = self.data.sort_values(self.date_col)
        self.time_series = self.data.set_index(self.date_col)[self.value_col]
    
    def trend_analysis(self) -> Dict[str, Any]:
        """Analyze time series trend"""
        if self.time_series is None:
            self.prepare_time_series()
        
        # Linear trend
        x = np.arange(len(self.time_series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, self.time_series.values)
        
        # Moving averages
        ma_7 = self.time_series.rolling(window=7).mean()
        ma_30 = self.time_series.rolling(window=30).mean()
        
        result = {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,
            "p_value": p_value,
            "trend_direction": "increasing" if slope > 0 else "decreasing",
            "moving_average_7": ma_7.dropna().tolist(),
            "moving_average_30": ma_30.dropna().tolist()
        }
        
        self.analysis_results["trend"] = result
        return result
    
    def seasonality_analysis(self) -> Dict[str, Any]:
        """Analyze seasonality patterns"""
        if self.time_series is None:
            self.prepare_time_series()
        
        # Add time components
        self.data['year'] = self.data[self.date_col].dt.year
        self.data['month'] = self.data[self.date_col].dt.month
        self.data['day_of_week'] = self.data[self.date_col].dt.dayofweek
        
        # Monthly patterns
        monthly_avg = self.data.groupby('month')[self.value_col].mean()
        yearly_avg = self.data.groupby('year')[self.value_col].mean()
        weekday_avg = self.data.groupby('day_of_week')[self.value_col].mean()
        
        result = {
            "monthly_patterns": monthly_avg.to_dict(),
            "yearly_patterns": yearly_avg.to_dict(),
            "weekday_patterns": weekday_avg.to_dict(),
            "seasonal_strength": monthly_avg.std() / monthly_avg.mean() if monthly_avg.mean() != 0 else 0
        }
        
        self.analysis_results["seasonality"] = result
        return result
    
    def stationarity_test(self) -> Dict[str, Any]:
        """Test for stationarity using Augmented Dickey-Fuller test"""
        if self.time_series is None:
            self.prepare_time_series()
        
        from statsmodels.tsa.stattools import adfuller
        
        adf_result = adfuller(self.time_series.dropna())
        
        result = {
            "adf_statistic": adf_result[0],
            "p_value": adf_result[1],
            "critical_values": adf_result[4],
            "is_stationary": adf_result[1] < 0.05
        }
        
        self.analysis_results["stationarity"] = result
        return result

class AnomalyDetector:
    """Anomaly detection utilities"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.numeric_data = data.select_dtypes(include=[np.number])
        self.anomaly_results = {}
    
    def isolation_forest(self, contamination: float = 0.1) -> Dict[str, Any]:
        """Detect anomalies using Isolation Forest"""
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(self.numeric_data)
        
        anomalies = self.data[anomaly_labels == -1]
        anomaly_scores = iso_forest.decision_function(self.numeric_data)
        
        result = {
            "n_anomalies": len(anomalies),
            "anomaly_percentage": (len(anomalies) / len(self.data)) * 100,
            "anomaly_indices": anomalies.index.tolist(),
            "anomaly_scores": anomaly_scores.tolist(),
            "contamination": contamination
        }
        
        self.anomaly_results["isolation_forest"] = result
        return result
    
    def statistical_outliers(self, method: str = "iqr") -> Dict[str, Any]:
        """Detect statistical outliers"""
        outliers = {}
        
        for col in self.numeric_data.columns:
            if method == "iqr":
                Q1 = self.numeric_data[col].quantile(0.25)
                Q3 = self.numeric_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (self.numeric_data[col] < lower_bound) | (self.numeric_data[col] > upper_bound)
            elif method == "zscore":
                z_scores = np.abs(stats.zscore(self.numeric_data[col]))
                outlier_mask = z_scores > 3
            else:
                continue
            
            outliers[col] = {
                "count": outlier_mask.sum(),
                "percentage": (outlier_mask.sum() / len(self.numeric_data)) * 100,
                "indices": self.numeric_data[outlier_mask].index.tolist()
            }
        
        result = {
            "method": method,
            "outliers_by_column": outliers,
            "total_outliers": sum(col_data["count"] for col_data in outliers.values())
        }
        
        self.anomaly_results["statistical"] = result
        return result

# Usage examples
def example_advanced_analytics():
    """Example advanced analytics usage"""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'feature4': np.random.normal(0, 1, n_samples),
        'feature5': np.random.normal(0, 1, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples)
    })
    
    # Clustering analysis
    cluster_analyzer = ClusteringAnalyzer(data)
    kmeans_result = cluster_analyzer.kmeans_clustering(n_clusters=3)
    print(f"K-means clustering: {kmeans_result['silhouette_score']:.3f}")
    
    optimal_clusters = cluster_analyzer.find_optimal_clusters(max_clusters=10)
    print(f"Optimal clusters: {optimal_clusters['optimal_k']}")
    
    # Dimensionality reduction
    reducer = DimensionalityReducer(data)
    pca_result = reducer.pca_analysis(n_components=2)
    print(f"PCA explained variance: {pca_result['explained_variance_ratio']}")
    
    tsne_result = reducer.tsne_analysis(n_components=2)
    print(f"t-SNE KL divergence: {tsne_result['kl_divergence']:.3f}")
    
    # Time series analysis (with sample time series data)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    values = np.random.normal(100, 10, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 5
    
    ts_data = pd.DataFrame({
        'date': dates,
        'value': values
    })
    
    ts_analyzer = TimeSeriesAnalyzer(ts_data, 'date', 'value')
    trend_result = ts_analyzer.trend_analysis()
    print(f"Trend direction: {trend_result['trend_direction']}")
    
    seasonality_result = ts_analyzer.seasonality_analysis()
    print(f"Seasonal strength: {seasonality_result['seasonal_strength']:.3f}")
    
    # Anomaly detection
    anomaly_detector = AnomalyDetector(data)
    iso_forest_result = anomaly_detector.isolation_forest(contamination=0.1)
    print(f"Anomalies detected: {iso_forest_result['n_anomalies']}")
    
    statistical_outliers = anomaly_detector.statistical_outliers(method="iqr")
    print(f"Statistical outliers: {statistical_outliers['total_outliers']}")
```

## TL;DR Runbook

### Quick Start

```python
# 1. Basic data analysis
analyzer = DataAnalyzer(data)
basic_info = analyzer.basic_info()
desc_stats = analyzer.descriptive_stats()

# 2. Correlation analysis
correlation = analyzer.correlation_analysis()
outliers = analyzer.outlier_detection()

# 3. Visualization
visualizer = DataVisualizer(data)
dist_plot = visualizer.plot_distributions()
corr_plot = visualizer.plot_correlation_heatmap()

# 4. Statistical tests
stat_analyzer = StatisticalAnalyzer(data)
t_test_result = stat_analyzer.t_test('group_col', 'value_col')

# 5. Advanced analytics
cluster_analyzer = ClusteringAnalyzer(data)
kmeans_result = cluster_analyzer.kmeans_clustering(n_clusters=3)
```

### Essential Patterns

```python
# Complete data analysis setup
def setup_data_analysis():
    """Setup complete data analysis environment"""
    
    # Data analyzer
    analyzer = DataAnalyzer(data)
    
    # Data visualizer
    visualizer = DataVisualizer(data)
    
    # Statistical analyzer
    stat_analyzer = StatisticalAnalyzer(data)
    
    # Clustering analyzer
    cluster_analyzer = ClusteringAnalyzer(data)
    
    # Dimensionality reducer
    reducer = DimensionalityReducer(data)
    
    # Time series analyzer
    ts_analyzer = TimeSeriesAnalyzer(data, 'date_col', 'value_col')
    
    # Anomaly detector
    anomaly_detector = AnomalyDetector(data)
    
    print("Data analysis setup complete!")
```

---

*This guide provides the complete machinery for Python data analysis. Each pattern includes implementation examples, analysis strategies, and real-world usage patterns for enterprise data analysis management.*
