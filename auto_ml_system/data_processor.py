"""
Data processing module for loading and preprocessing datasets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression


class DataProcessor:
    """
    Data processing class for loading and preprocessing datasets
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataProcessor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger('AutoML.DataProcessor')
        self.encoders = {}
        self.scalers = {}
        self.imputers = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from various file formats
        
        Args:
            file_path: Path to data file
            
        Returns:
            Loaded DataFrame
        """
        file_path = Path(file_path)
        
        self.logger.info(f"Loading data from {file_path}")
        
        try:
            if file_path.suffix.lower() == '.csv':
                # Try different encodings for CSV
                for encoding in ['utf-8', 'latin-1', 'cp949', 'euc-kr']:
                    try:
                        data = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("Could not read CSV file with any supported encoding")
                    
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                data = pd.read_excel(file_path)
                
            elif file_path.suffix.lower() == '.json':
                data = pd.read_json(file_path)
                
            elif file_path.suffix.lower() == '.parquet':
                data = pd.read_parquet(file_path)
                
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            self.logger.info(f"Successfully loaded data with shape {data.shape}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Preprocess the dataset
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary containing processed features and target
        """
        self.logger.info("Starting data preprocessing")
        
        # Basic data info
        self.logger.info(f"Data shape: {data.shape}")
        self.logger.info(f"Columns: {list(data.columns)}")
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Detect target column
        target_col = self._detect_target_column(data)
        
        # Separate features and target
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Encode categorical variables
        X_encoded = self._encode_categorical_features(X)
        
        # Handle target encoding if needed
        y_encoded, target_type = self._encode_target(y)
        
        # Feature scaling
        X_scaled = self._scale_features(X_encoded)
        
        # Feature selection for high-dimensional data
        if X_scaled.shape[1] > 50:
            X_selected = self._select_features(X_scaled, y_encoded, target_type)
        else:
            X_selected = X_scaled
        
        result = {
            'X': X_selected,
            'y': y_encoded,
            'feature_names': list(X.columns) if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X_selected.shape[1])],
            'target_name': target_col,
            'target_type': target_type,
            'original_shape': data.shape,
            'processed_shape': (X_selected.shape[0], X_selected.shape[1])
        }
        
        self.logger.info(f"Preprocessing completed. Final shape: {X_selected.shape}")
        return result
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        
        missing_ratio = data.isnull().sum() / len(data)
        
        # Drop columns with >50% missing values
        high_missing_cols = missing_ratio[missing_ratio > 0.5].index
        if len(high_missing_cols) > 0:
            self.logger.warning(f"Dropping columns with >50% missing values: {list(high_missing_cols)}")
            data = data.drop(columns=high_missing_cols)
        
        # Impute remaining missing values
        for col in data.columns:
            if data[col].isnull().sum() > 0:
                if data[col].dtype in ['object', 'category']:
                    # Mode imputation for categorical
                    data[col].fillna(data[col].mode()[0], inplace=True)
                else:
                    # Median imputation for numerical
                    data[col].fillna(data[col].median(), inplace=True)
        
        return data
    
    def _detect_target_column(self, data: pd.DataFrame) -> str:
        """Detect the target column automatically"""
        
        # Common target column names
        target_keywords = ['target', 'label', 'class', 'y', 'output', 'result', 'outcome']
        
        # Check for exact matches
        for col in data.columns:
            if col.lower() in target_keywords:
                self.logger.info(f"Detected target column by name: {col}")
                return col
        
        # Check for partial matches
        for col in data.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                self.logger.info(f"Detected target column by keyword: {col}")
                return col
        
        # If no clear target found, use the last column
        target_col = data.columns[-1]
        self.logger.warning(f"No clear target column found. Using last column: {target_col}")
        return target_col
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        
        X_encoded = X.copy()
        
        for col in X.columns:
            if X[col].dtype in ['object', 'category']:
                # Use label encoding for high cardinality, one-hot for low cardinality
                unique_values = X[col].nunique()
                
                if unique_values > 10:
                    # Label encoding
                    encoder = LabelEncoder()
                    X_encoded[col] = encoder.fit_transform(X[col].astype(str))
                    self.encoders[col] = encoder
                else:
                    # One-hot encoding
                    dummies = pd.get_dummies(X[col], prefix=col)
                    X_encoded = X_encoded.drop(columns=[col])
                    X_encoded = pd.concat([X_encoded, dummies], axis=1)
        
        return X_encoded
    
    def _encode_target(self, y: pd.Series) -> Tuple[pd.Series, str]:
        """Encode target variable and determine its type"""
        
        if y.dtype in ['object', 'category']:
            # Categorical target - classification
            encoder = LabelEncoder()
            y_encoded = pd.Series(encoder.fit_transform(y.astype(str)), index=y.index)
            self.encoders['target'] = encoder
            target_type = 'classification'
        else:
            # Numerical target - could be regression or classification
            unique_values = y.nunique()
            
            if unique_values <= 20 and all(y.dropna() == y.dropna().astype(int)):
                # Integer values with few unique values - classification
                target_type = 'classification'
                y_encoded = y.astype(int)
            else:
                # Continuous values - regression
                target_type = 'regression'
                y_encoded = y.astype(float)
        
        return y_encoded, target_type
    
    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features"""
        
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 0:
            scaler = StandardScaler()
            X_scaled = X.copy()
            X_scaled[numerical_cols] = scaler.fit_transform(X[numerical_cols])
            self.scalers['standard'] = scaler
            return X_scaled
        
        return X
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series, target_type: str) -> pd.DataFrame:
        """Select top features for high-dimensional data"""
        
        max_features = min(50, X.shape[1])
        
        if target_type == 'classification':
            selector = SelectKBest(f_classif, k=max_features)
        else:
            selector = SelectKBest(f_regression, k=max_features)
        
        X_selected = selector.fit_transform(X, y)
        
        self.logger.info(f"Selected {X_selected.shape[1]} features from {X.shape[1]}")
        
        return pd.DataFrame(X_selected)