# Python Machine Learning Best Practices

**Objective**: Master senior-level Python machine learning patterns for production systems. When you need to build robust ML pipelines, when you want to implement comprehensive model evaluation, when you need enterprise-grade ML strategies—these best practices become your weapon of choice.

## Core Principles

- **Reproducibility**: Ensure experiments can be reproduced and validated
- **Performance**: Optimize for training speed and inference efficiency
- **Validation**: Implement robust model evaluation and testing
- **Monitoring**: Track model performance and data drift
- **Deployment**: Ensure models can be deployed and maintained

## ML Pipeline Design

### Model Training Pipeline

```python
# python/01-ml-pipeline-design.py

"""
Machine learning pipeline design patterns and model training workflows
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import json
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Model type enumeration"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"

class ModelStatus(Enum):
    """Model status enumeration"""
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATED = "evaluated"
    DEPLOYED = "deployed"
    RETIRED = "retired"

@dataclass
class ModelMetrics:
    """Model metrics definition"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mse: float = 0.0
    r2_score: float = 0.0
    cross_val_score: float = 0.0
    training_time: float = 0.0
    inference_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

class DataPreprocessor:
    """Data preprocessing utilities"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.preprocessing_pipeline = None
    
    def prepare_features(self, X: pd.DataFrame, y: pd.Series = None, 
                        fit: bool = True) -> Tuple[np.ndarray, List[str]]:
        """Prepare features for training"""
        # Separate numeric and categorical columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create preprocessing pipeline
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
        
        if fit:
            X_processed = preprocessor.fit_transform(X)
            self.preprocessing_pipeline = preprocessor
        else:
            X_processed = self.preprocessing_pipeline.transform(X)
        
        # Get feature names
        feature_names = numeric_cols.copy()
        if categorical_cols:
            cat_feature_names = self.preprocessing_pipeline.named_transformers_['cat'].get_feature_names_out(categorical_cols)
            feature_names.extend(cat_feature_names)
        
        self.feature_names = feature_names
        return X_processed, feature_names
    
    def save_preprocessor(self, filepath: str) -> None:
        """Save preprocessing pipeline"""
        if self.preprocessing_pipeline is not None:
            joblib.dump(self.preprocessing_pipeline, filepath)
    
    def load_preprocessor(self, filepath: str) -> None:
        """Load preprocessing pipeline"""
        self.preprocessing_pipeline = joblib.load(filepath)

class ModelTrainer(ABC):
    """Abstract model trainer"""
    
    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        self.model = None
        self.metrics = ModelMetrics()
        self.training_history = []
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Evaluate the model"""
        pass
    
    def save_model(self, filepath: str) -> None:
        """Save trained model"""
        if self.model is not None:
            joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load trained model"""
        self.model = joblib.load(filepath)

class ClassificationTrainer(ModelTrainer):
    """Classification model trainer"""
    
    def __init__(self, algorithm: str = "random_forest"):
        super().__init__(ModelType.CLASSIFICATION)
        self.algorithm = algorithm
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the model"""
        if self.algorithm == "random_forest":
            self.model = RandomForestClassifier(random_state=42)
        elif self.algorithm == "gradient_boosting":
            self.model = GradientBoostingClassifier(random_state=42)
        elif self.algorithm == "logistic_regression":
            self.model = LogisticRegression(random_state=42)
        elif self.algorithm == "svm":
            self.model = SVC(random_state=42)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train the classification model"""
        start_time = datetime.utcnow()
        
        # Hyperparameter tuning if specified
        if 'param_grid' in kwargs:
            param_grid = kwargs['param_grid']
            cv = kwargs.get('cv', 5)
            
            grid_search = GridSearchCV(
                self.model, param_grid, cv=cv, scoring='accuracy'
            )
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        else:
            self.model.fit(X, y)
        
        training_time = (datetime.utcnow() - start_time).total_seconds()
        self.metrics.training_time = training_time
        
        logger.info(f"Model trained in {training_time:.2f} seconds")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        start_time = datetime.utcnow()
        predictions = self.model.predict(X)
        inference_time = (datetime.utcnow() - start_time).total_seconds()
        
        self.metrics.inference_time = inference_time
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError(f"Model {self.algorithm} does not support probability predictions")
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Evaluate the classification model"""
        predictions = self.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, average='weighted')
        recall = recall_score(y, predictions, average='weighted')
        f1 = f1_score(y, predictions, average='weighted')
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        cv_score = cv_scores.mean()
        
        self.metrics.accuracy = accuracy
        self.metrics.precision = precision
        self.metrics.recall = recall
        self.metrics.f1_score = f1
        self.metrics.cross_val_score = cv_score
        
        logger.info(f"Model evaluation - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return self.metrics

class RegressionTrainer(ModelTrainer):
    """Regression model trainer"""
    
    def __init__(self, algorithm: str = "linear_regression"):
        super().__init__(ModelType.REGRESSION)
        self.algorithm = algorithm
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the model"""
        if self.algorithm == "linear_regression":
            self.model = LinearRegression()
        elif self.algorithm == "random_forest":
            self.model = RandomForestClassifier(random_state=42)
        elif self.algorithm == "gradient_boosting":
            self.model = GradientBoostingClassifier(random_state=42)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train the regression model"""
        start_time = datetime.utcnow()
        
        # Hyperparameter tuning if specified
        if 'param_grid' in kwargs:
            param_grid = kwargs['param_grid']
            cv = kwargs.get('cv', 5)
            
            grid_search = GridSearchCV(
                self.model, param_grid, cv=cv, scoring='neg_mean_squared_error'
            )
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        else:
            self.model.fit(X, y)
        
        training_time = (datetime.utcnow() - start_time).total_seconds()
        self.metrics.training_time = training_time
        
        logger.info(f"Model trained in {training_time:.2f} seconds")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        start_time = datetime.utcnow()
        predictions = self.model.predict(X)
        inference_time = (datetime.utcnow() - start_time).total_seconds()
        
        self.metrics.inference_time = inference_time
        return predictions
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Evaluate the regression model"""
        predictions = self.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='neg_mean_squared_error')
        cv_score = -cv_scores.mean()
        
        self.metrics.mse = mse
        self.metrics.r2_score = r2
        self.metrics.cross_val_score = cv_score
        
        logger.info(f"Model evaluation - MSE: {mse:.4f}, R²: {r2:.4f}")
        return self.metrics

class ModelEvaluator:
    """Model evaluation utilities"""
    
    def __init__(self):
        self.evaluation_results = {}
    
    def cross_validate(self, model, X: np.ndarray, y: np.ndarray, 
                      cv: int = 5, scoring: str = None) -> Dict[str, Any]:
        """Perform cross-validation"""
        if scoring is None:
            scoring = 'accuracy' if len(np.unique(y)) < 10 else 'neg_mean_squared_error'
        
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        result = {
            "cv_scores": cv_scores.tolist(),
            "mean_score": cv_scores.mean(),
            "std_score": cv_scores.std(),
            "cv_folds": cv,
            "scoring": scoring
        }
        
        return result
    
    def learning_curve(self, model, X: np.ndarray, y: np.ndarray, 
                      train_sizes: List[float] = None) -> Dict[str, Any]:
        """Generate learning curve"""
        from sklearn.model_selection import learning_curve
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y, train_sizes=train_sizes, cv=5
        )
        
        result = {
            "train_sizes": train_sizes_abs.tolist(),
            "train_scores": train_scores.tolist(),
            "val_scores": val_scores.tolist(),
            "train_scores_mean": train_scores.mean(axis=1).tolist(),
            "val_scores_mean": val_scores.mean(axis=1).tolist()
        }
        
        return result
    
    def feature_importance(self, model, feature_names: List[str]) -> Dict[str, Any]:
        """Get feature importance"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            return {"error": "Model does not support feature importance"}
        
        # Sort features by importance
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        result = {
            "feature_importance": feature_importance,
            "top_features": feature_importance[:10]
        }
        
        return result

class ModelPipeline:
    """Complete ML pipeline"""
    
    def __init__(self, model_type: ModelType, algorithm: str = None):
        self.model_type = model_type
        self.algorithm = algorithm
        self.preprocessor = DataPreprocessor()
        self.trainer = None
        self.evaluator = ModelEvaluator()
        self.status = ModelStatus.TRAINING
        self.pipeline_metrics = {}
    
    def setup_trainer(self, algorithm: str) -> None:
        """Setup model trainer"""
        if self.model_type == ModelType.CLASSIFICATION:
            self.trainer = ClassificationTrainer(algorithm)
        elif self.model_type == ModelType.REGRESSION:
            self.trainer = RegressionTrainer(algorithm)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train_pipeline(self, X: pd.DataFrame, y: pd.Series, 
                      test_size: float = 0.2, **kwargs) -> Dict[str, Any]:
        """Train complete pipeline"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Preprocess features
        X_train_processed, feature_names = self.preprocessor.prepare_features(
            X_train, y_train, fit=True
        )
        X_test_processed, _ = self.preprocessor.prepare_features(
            X_test, fit=False
        )
        
        # Train model
        self.trainer.train(X_train_processed, y_train.values, **kwargs)
        self.status = ModelStatus.TRAINED
        
        # Evaluate model
        train_metrics = self.trainer.evaluate(X_train_processed, y_train.values)
        test_metrics = self.trainer.evaluate(X_test_processed, y_test.values)
        
        # Cross-validation
        cv_results = self.evaluator.cross_validate(
            self.trainer.model, X_train_processed, y_train.values
        )
        
        # Feature importance
        feature_importance = self.evaluator.feature_importance(
            self.trainer.model, feature_names
        )
        
        self.pipeline_metrics = {
            "train_metrics": train_metrics.to_dict(),
            "test_metrics": test_metrics.to_dict(),
            "cv_results": cv_results,
            "feature_importance": feature_importance,
            "feature_names": feature_names
        }
        
        self.status = ModelStatus.EVALUATED
        return self.pipeline_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        if self.status not in [ModelStatus.TRAINED, ModelStatus.EVALUATED, ModelStatus.DEPLOYED]:
            raise ValueError("Model not trained yet")
        
        X_processed, _ = self.preprocessor.prepare_features(X, fit=False)
        return self.trainer.predict(X_processed)
    
    def save_pipeline(self, filepath: str) -> None:
        """Save complete pipeline"""
        pipeline_data = {
            "model_type": self.model_type.value,
            "algorithm": self.algorithm,
            "status": self.status.value,
            "metrics": self.pipeline_metrics,
            "feature_names": self.preprocessor.feature_names
        }
        
        # Save model
        self.trainer.save_model(f"{filepath}_model.pkl")
        
        # Save preprocessor
        self.preprocessor.save_preprocessor(f"{filepath}_preprocessor.pkl")
        
        # Save pipeline metadata
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(pipeline_data, f, indent=2)
    
    def load_pipeline(self, filepath: str) -> None:
        """Load complete pipeline"""
        # Load metadata
        with open(f"{filepath}_metadata.json", 'r') as f:
            pipeline_data = json.load(f)
        
        self.model_type = ModelType(pipeline_data["model_type"])
        self.algorithm = pipeline_data["algorithm"]
        self.status = ModelStatus(pipeline_data["status"])
        self.pipeline_metrics = pipeline_data["metrics"]
        
        # Load model
        self.setup_trainer(self.algorithm)
        self.trainer.load_model(f"{filepath}_model.pkl")
        
        # Load preprocessor
        self.preprocessor.load_preprocessor(f"{filepath}_preprocessor.pkl")
        self.preprocessor.feature_names = pipeline_data["feature_names"]

# Usage examples
def example_ml_pipeline():
    """Example ML pipeline usage"""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples)
    })
    
    y = (X['feature1'] + X['feature2'] + np.random.normal(0, 0.1, n_samples) > 0).astype(int)
    
    # Create classification pipeline
    pipeline = ModelPipeline(ModelType.CLASSIFICATION)
    pipeline.setup_trainer("random_forest")
    
    # Train pipeline
    metrics = pipeline.train_pipeline(X, y, test_size=0.2)
    print(f"Training metrics: {metrics['train_metrics']}")
    print(f"Test metrics: {metrics['test_metrics']}")
    
    # Make predictions
    new_data = pd.DataFrame({
        'feature1': [1, -1, 0],
        'feature2': [1, -1, 0],
        'feature3': [1, -1, 0],
        'category': ['A', 'B', 'C']
    })
    
    predictions = pipeline.predict(new_data)
    print(f"Predictions: {predictions}")
    
    # Save pipeline
    pipeline.save_pipeline("my_model")
    
    # Load pipeline
    new_pipeline = ModelPipeline(ModelType.CLASSIFICATION)
    new_pipeline.load_pipeline("my_model")
    
    # Make predictions with loaded pipeline
    loaded_predictions = new_pipeline.predict(new_data)
    print(f"Loaded pipeline predictions: {loaded_predictions}")
```

### Model Validation

```python
# python/02-model-validation.py

"""
Model validation patterns and evaluation strategies
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class ModelValidator:
    """Comprehensive model validation"""
    
    def __init__(self):
        self.validation_results = {}
        self.cv_results = {}
    
    def stratified_cv(self, model, X: np.ndarray, y: np.ndarray, 
                     cv: int = 5) -> Dict[str, Any]:
        """Perform stratified cross-validation"""
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        cv_scores = []
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            fold_metrics = {
                "fold": fold + 1,
                "accuracy": accuracy_score(y_val, y_pred),
                "precision": precision_score(y_val, y_pred, average='weighted'),
                "recall": recall_score(y_val, y_pred, average='weighted'),
                "f1": f1_score(y_val, y_pred, average='weighted')
            }
            
            if y_pred_proba is not None:
                fold_metrics["auc"] = roc_auc_score(y_val, y_pred_proba)
            
            cv_scores.append(fold_metrics["accuracy"])
            fold_results.append(fold_metrics)
        
        result = {
            "cv_scores": cv_scores,
            "mean_score": np.mean(cv_scores),
            "std_score": np.std(cv_scores),
            "fold_results": fold_results,
            "cv_folds": cv
        }
        
        self.cv_results["stratified_cv"] = result
        return result
    
    def time_series_cv(self, model, X: np.ndarray, y: np.ndarray, 
                      cv: int = 5) -> Dict[str, Any]:
        """Perform time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=cv)
        
        cv_scores = []
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            fold_metrics = {
                "fold": fold + 1,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "accuracy": accuracy_score(y_val, y_pred),
                "mse": mean_squared_error(y_val, y_pred),
                "r2": r2_score(y_val, y_pred)
            }
            
            cv_scores.append(fold_metrics["accuracy"])
            fold_results.append(fold_metrics)
        
        result = {
            "cv_scores": cv_scores,
            "mean_score": np.mean(cv_scores),
            "std_score": np.std(cv_scores),
            "fold_results": fold_results,
            "cv_folds": cv
        }
        
        self.cv_results["time_series_cv"] = result
        return result
    
    def confusion_matrix_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        result = {
            "confusion_matrix": cm.tolist(),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1_score": f1
        }
        
        self.validation_results["confusion_matrix"] = result
        return result
    
    def roc_analysis(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Analyze ROC curve and AUC"""
        auc_score = roc_auc_score(y_true, y_pred_proba)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        
        # Find optimal threshold
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        result = {
            "auc_score": auc_score,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "optimal_threshold": optimal_threshold,
            "optimal_tpr": tpr[optimal_idx],
            "optimal_fpr": fpr[optimal_idx]
        }
        
        self.validation_results["roc_analysis"] = result
        return result
    
    def precision_recall_analysis(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Analyze precision-recall curve"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        ap_score = average_precision_score(y_true, y_pred_proba)
        
        # Find optimal threshold
        f1_scores = 2 * (precision * recall) / (precision + recall)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
        
        result = {
            "ap_score": ap_score,
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
            "f1_scores": f1_scores.tolist(),
            "optimal_threshold": optimal_threshold,
            "optimal_precision": precision[optimal_idx],
            "optimal_recall": recall[optimal_idx],
            "optimal_f1": f1_scores[optimal_idx]
        }
        
        self.validation_results["precision_recall"] = result
        return result
    
    def bias_analysis(self, model, X: np.ndarray, y: np.ndarray, 
                     sensitive_features: List[str]) -> Dict[str, Any]:
        """Analyze model bias across sensitive features"""
        bias_results = {}
        
        for feature in sensitive_features:
            if feature in X.columns:
                feature_values = X[feature].unique()
                feature_bias = {}
                
                for value in feature_values:
                    mask = X[feature] == value
                    X_subset = X[mask]
                    y_subset = y[mask]
                    
                    if len(X_subset) > 0:
                        y_pred = model.predict(X_subset)
                        
                        # Calculate metrics for this subgroup
                        subgroup_metrics = {
                            "sample_size": len(X_subset),
                            "accuracy": accuracy_score(y_subset, y_pred),
                            "precision": precision_score(y_subset, y_pred, average='weighted'),
                            "recall": recall_score(y_subset, y_pred, average='weighted'),
                            "f1": f1_score(y_subset, y_pred, average='weighted')
                        }
                        
                        feature_bias[value] = subgroup_metrics
                
                bias_results[feature] = feature_bias
        
        self.validation_results["bias_analysis"] = bias_results
        return bias_results

class ModelComparator:
    """Model comparison utilities"""
    
    def __init__(self):
        self.comparison_results = {}
    
    def compare_models(self, models: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Compare multiple models"""
        comparison_results = {}
        
        for model_name, model in models.items():
            # Train model
            model.fit(X, y)
            
            # Make predictions
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y, y_pred),
                "precision": precision_score(y, y_pred, average='weighted'),
                "recall": recall_score(y, y_pred, average='weighted'),
                "f1": f1_score(y, y_pred, average='weighted')
            }
            
            if y_pred_proba is not None:
                metrics["auc"] = roc_auc_score(y, y_pred_proba)
            
            comparison_results[model_name] = metrics
        
        # Find best model
        best_model = max(comparison_results.items(), key=lambda x: x[1]["f1"])
        
        result = {
            "model_comparison": comparison_results,
            "best_model": best_model[0],
            "best_metrics": best_model[1]
        }
        
        self.comparison_results = result
        return result
    
    def statistical_significance_test(self, model1_metrics: List[float], 
                                     model2_metrics: List[float]) -> Dict[str, Any]:
        """Test statistical significance between models"""
        from scipy import stats
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(model1_metrics, model2_metrics)
        
        result = {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "model1_mean": np.mean(model1_metrics),
            "model2_mean": np.mean(model2_metrics),
            "difference": np.mean(model1_metrics) - np.mean(model2_metrics)
        }
        
        return result

# Usage examples
def example_model_validation():
    """Example model validation usage"""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'sensitive_feature': np.random.choice(['A', 'B'], n_samples)
    })
    
    y = (X['feature1'] + X['feature2'] + np.random.normal(0, 0.1, n_samples) > 0).astype(int)
    
    # Create models
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }
    
    # Model validation
    validator = ModelValidator()
    
    # Stratified CV
    rf_model = RandomForestClassifier(random_state=42)
    cv_results = validator.stratified_cv(rf_model, X.values, y.values)
    print(f"CV Results: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
    
    # Confusion matrix analysis
    rf_model.fit(X.values, y.values)
    y_pred = rf_model.predict(X.values)
    cm_results = validator.confusion_matrix_analysis(y.values, y_pred)
    print(f"Confusion Matrix Analysis: {cm_results['accuracy']:.4f}")
    
    # ROC analysis
    y_pred_proba = rf_model.predict_proba(X.values)[:, 1]
    roc_results = validator.roc_analysis(y.values, y_pred_proba)
    print(f"ROC AUC: {roc_results['auc_score']:.4f}")
    
    # Bias analysis
    bias_results = validator.bias_analysis(rf_model, X, y.values, ['sensitive_feature'])
    print(f"Bias Analysis: {bias_results}")
    
    # Model comparison
    comparator = ModelComparator()
    comparison_results = comparator.compare_models(models, X.values, y.values)
    print(f"Best Model: {comparison_results['best_model']}")
    print(f"Best Metrics: {comparison_results['best_metrics']}")
```

## TL;DR Runbook

### Quick Start

```python
# 1. Data preprocessing
preprocessor = DataPreprocessor()
X_processed, feature_names = preprocessor.prepare_features(X, y, fit=True)

# 2. Model training
trainer = ClassificationTrainer("random_forest")
trainer.train(X_processed, y.values)

# 3. Model evaluation
metrics = trainer.evaluate(X_test, y_test)
print(f"Model accuracy: {metrics.accuracy:.4f}")

# 4. Complete pipeline
pipeline = ModelPipeline(ModelType.CLASSIFICATION)
pipeline.setup_trainer("random_forest")
metrics = pipeline.train_pipeline(X, y)

# 5. Model validation
validator = ModelValidator()
cv_results = validator.stratified_cv(model, X, y)
```

### Essential Patterns

```python
# Complete ML setup
def setup_ml_pipeline():
    """Setup complete ML pipeline environment"""
    
    # Data preprocessor
    preprocessor = DataPreprocessor()
    
    # Model trainer
    trainer = ClassificationTrainer("random_forest")
    
    # Model evaluator
    evaluator = ModelEvaluator()
    
    # Model validator
    validator = ModelValidator()
    
    # Model comparator
    comparator = ModelComparator()
    
    # Complete pipeline
    pipeline = ModelPipeline(ModelType.CLASSIFICATION)
    
    print("ML pipeline setup complete!")
```

---

*This guide provides the complete machinery for Python machine learning. Each pattern includes implementation examples, training strategies, and real-world usage patterns for enterprise ML management.*
