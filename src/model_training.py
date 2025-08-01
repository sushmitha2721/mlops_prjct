import os
import pickle
import logging
import yaml
import numpy as np
from scipy.sparse import load_npz, hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from logger import setup_logging
from pathlib import Path
from time import time
from typing import Dict, Any, Union

logger = setup_logging(logger_name='model_training')

def load_config() -> Dict[str, Any]:
    """Load and validate training configuration with parameter type conversion"""
    try:
        config_path = Path('params.yaml')
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'model_training' not in config:
            raise KeyError("Missing 'model_training' section in config")
            
        # Convert string numbers to proper numeric types
        if 'svm_params' in config['model_training']:
            svm_params = config['model_training']['svm_params']
            if 'tol' in svm_params:
                if isinstance(svm_params['tol'], str):
                    try:
                        svm_params['tol'] = float(svm_params['tol'])
                    except ValueError as e:
                        logger.error(f"Could not convert tol value {svm_params['tol']} to float")
                        raise ValueError("Invalid tol value in config") from e
            
        logger.info("Configuration loaded successfully")
        return config['model_training']
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        raise

def load_training_data(data_dir: str) -> tuple:
    """Load training features and labels with validation"""
    try:
        logger.info(f"Loading training data from {data_dir}")
        
        if not Path(data_dir).exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        X_path = Path(data_dir) / "X_train.npz"
        y_path = Path(data_dir) / "y_train.npy"
        
        if not X_path.exists():
            raise FileNotFoundError(f"Features file not found: {X_path}")
        if not y_path.exists():
            raise FileNotFoundError(f"Labels file not found: {y_path}")
        
        X_train = load_npz(X_path)
        y_train = np.load(y_path)
        
        logger.info(f"Training data loaded - Features: {X_train.shape}, Labels: {y_train.shape}")
        return X_train, y_train
        
    except Exception as e:
        logger.error(f"Failed to load training data: {str(e)}")
        raise

def scale_features(X: csr_matrix, model_type: str) -> csr_matrix:
    """
    Scale features appropriately for different model types
    Args:
        X: Input sparse feature matrix
        model_type: One of 'logistic', 'svm', or 'random_forest'
    Returns:
        Scaled sparse feature matrix
    """
    try:
        logger.info(f"Scaling features for {model_type}")
        
        if model_type == 'svm':
            # MaxAbsScaler works best with sparse data for SVM
            scaler = MaxAbsScaler()
            X_scaled = scaler.fit_transform(X)
            logger.info("Applied MaxAbsScaler for SVM")
        elif model_type == 'logistic':
            # StandardScaler (with_mean=False for sparse) works well for Logistic Regression
            scaler = StandardScaler(with_mean=False)
            X_scaled = scaler.fit_transform(X)
            logger.info("Applied StandardScaler for Logistic Regression")
        else:
            # No scaling for tree-based models
            X_scaled = X
            logger.info("No scaling applied for Random Forest")
            
        return X_scaled
        
    except Exception as e:
        logger.error(f"Feature scaling failed for {model_type}: {str(e)}")
        raise

def initialize_model(model_type: str, params: Dict[str, Any]) -> tuple:
    """Initialize model with parameters and validation including type conversion"""
    model_map = {
        'logistic': {
            'class': LogisticRegression,
            'required_params': ['C', 'penalty', 'solver'],
            'default_params': {
                'max_iter': 1000,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            },
            'param_types': {
                'C': float,
                'max_iter': int
            }
        },
        'svm': {
            'class': SVC,
            'required_params': ['C', 'kernel'],
            'default_params': {
                'max_iter': -1,  # No limit
                'class_weight': 'balanced',
                'probability': True,
                'random_state': 42,
                'verbose': True,
                'tol': 1e-3  # Default float value
            },
            'param_types': {
                'C': float,
                'tol': float,
                'max_iter': int
            }
        },
        'random_forest': {
            'class': RandomForestClassifier,
            'required_params': ['n_estimators', 'max_depth'],
            'default_params': {
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'bootstrap': True,
                'class_weight': 'balanced_subsample',
                'random_state': 42,
                'n_jobs': -1,
                'verbose': 1
            },
            'param_types': {
                'n_estimators': int,
                'max_depth': int
            }
        }
    }
    
    try:
        if model_type not in model_map:
            raise ValueError(f"Unknown model type: {model_type}. Choose from {list(model_map.keys())}")
            
        model_info = model_map[model_type]
        model_class = model_info['class']
        model_params = {**model_info['default_params'], **params.get(f"{model_type}_params", {})}
        
        # Validate required parameters
        missing_params = [p for p in model_info['required_params'] if p not in model_params]
        if missing_params:
            raise ValueError(f"Missing required parameters for {model_type}: {missing_params}")
        
        # Convert parameters to correct types
        if 'param_types' in model_info:
            for param, param_type in model_info['param_types'].items():
                if param in model_params and not isinstance(model_params[param], param_type):
                    try:
                        # Handle scientific notation strings
                        if isinstance(model_params[param], str) and 'e' in model_params[param].lower():
                            model_params[param] = float(model_params[param])
                        else:
                            model_params[param] = param_type(model_params[param])
                    except (ValueError, TypeError) as e:
                        raise ValueError(
                            f"Invalid {param} value for {model_type}. "
                            f"Expected {param_type.__name__}, got {model_params[param]}"
                        ) from e
        
        logger.info(f"Initialized {model_type} model with parameters: {model_params}")
        return model_class(**model_params), model_type
        
    except Exception as e:
        logger.error(f"Model initialization failed for {model_type}: {str(e)}")
        raise

def save_model(model, model_type: str, output_dir: str = "models") -> str:
    """Save trained model with versioning"""
    try:
        model_dir = Path(output_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time())
        model_path = model_dir / f"{model_type}_model.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Saved {model_type} model to {model_path}")
        return str(model_path)
        
    except Exception as e:
        logger.error(f"Failed to save {model_type} model: {str(e)}")
        raise

def train_all_models():
    """Train all three models with proper feature scaling and parameter handling"""
    try:
        logger.info("===== Starting Multi-Model Training Pipeline =====")
        
        # 1. Load configuration and data
        config = load_config()
        data_dir = config.get('data_dir', 'data/feature')
        X_train, y_train = load_training_data(data_dir)
        
        # 2. Initialize all models
        models_to_train = ['logistic', 'svm', 'random_forest']
        trained_models = {}
        
        for model_type in models_to_train:
            try:
                logger.info(f"\n{'='*40}")
                logger.info(f"Training {model_type} model".upper())
                logger.info(f"{'='*40}")
                
                # Initialize model with proper parameter types
                model, _ = initialize_model(model_type, config)
                
                # Scale features appropriately for each model type
                X_train_processed = scale_features(X_train, model_type)
                
                # Train model
                logger.info(f"Starting training for {model_type}...")
                model.fit(X_train_processed, y_train)
                logger.info(f"{model_type} training completed successfully")
                
                # Save model
                model_path = save_model(model, model_type)
                trained_models[model_type] = model_path
                
                logger.info(f"{'='*40}")
                logger.info(f"{model_type.upper()} TRAINING COMPLETE")
                logger.info(f"{'='*40}\n")
                
            except Exception as e:
                logger.error(f"Failed to train {model_type}: {str(e)}")
                continue
        
        # Final summary
        logger.info("\n===== Training Summary =====")
        for model_type, path in trained_models.items():
            logger.info(f"{model_type.upper():<15}: {path}")
        
        if not trained_models:
            raise RuntimeError("No models were successfully trained")
            
        logger.info("===== All Model Training Completed =====")
        return trained_models
        
    except Exception as e:
        logger.critical(f"Training pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    train_all_models()
    # Test data loading
    X, y = load_training_data('data/feature')
    print(f"Features shape: {X.shape}, Labels shape: {y.shape}")

    # Test config loading
    config = load_config()
    print("Config valid:", bool(config))