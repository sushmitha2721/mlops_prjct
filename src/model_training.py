import os
import pickle
import logging
import yaml
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.model_selection import GridSearchCV
from logger import setup_logging
from pathlib import Path
from time import time
from typing import Dict, Any, Tuple, Union
from dvclive import Live

logger = setup_logging(logger_name='model_training')

def load_config() -> Dict[str, Any]:
    """Load and validate training configuration"""
    try:
        config_path = Path('params.yaml')
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'model_training' not in config:
            raise KeyError("Missing 'model_training' section in config")
            
        # Set default grid search parameters if not specified
        if 'grid_search' not in config['model_training']:
            config['model_training']['grid_search'] = {
                'cv': 5,
                'scoring': 'f1',
                'n_jobs': -1,
                'verbose': 2,
                'refit': True
            }
            
        logger.info("Configuration loaded successfully")
        return config['model_training']
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        raise

def load_training_data(data_dir: str) -> Tuple[csr_matrix, np.ndarray]:
    """Load training features and labels"""
    try:
        logger.info(f"Loading training data from {data_dir}")
        
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
    """Scale features based on model type"""
    try:
        logger.info(f"Scaling features for {model_type}")
        
        if model_type == 'svm':
            scaler = MaxAbsScaler()
            X_scaled = scaler.fit_transform(X)
            logger.info("Applied MaxAbsScaler for SVM")
        elif model_type == 'logistic':
            scaler = StandardScaler(with_mean=False)
            X_scaled = scaler.fit_transform(X)
            logger.info("Applied StandardScaler for Logistic Regression")
        else:
            X_scaled = X
            logger.info("No scaling applied for Random Forest")
            
        return X_scaled
        
    except Exception as e:
        logger.error(f"Feature scaling failed for {model_type}: {str(e)}")
        raise
def initialize_model(model_type: str, params: Dict[str, Any]) -> Any:
    """Initialize model with parameters - Updated version"""
    model_map = {
        'logistic': {
            'class': LogisticRegression,
            'required_params': ['C', 'penalty', 'solver'],
            'default_params': {
                'max_iter': 1000,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }
        },
        'svm': {
            'class': SVC,
            'required_params': ['C', 'kernel'],
            'default_params': {
                'probability': True,
                'class_weight': 'balanced',
                'random_state': 42
            }
        },
        'random_forest': {
            'class': RandomForestClassifier,
            'required_params': ['n_estimators', 'max_depth'],
            'default_params': {
                'class_weight': 'balanced_subsample',
                'random_state': 42,
                'n_jobs': -1
            }
        }
    }
    
    try:
        if model_type not in model_map:
            raise ValueError(f"Unknown model type: {model_type}")
            
        model_info = model_map[model_type]
        
        # Get both fixed params and param grid
        model_config = params.get(model_type, {})
        fixed_params = model_config.get('fixed_params', {})
        param_grid = model_config.get('param_grid', {})
        
        # Combine default and fixed params
        model_params = {
            **model_info['default_params'],
            **fixed_params
        }
        
        # Validate required parameters
        missing_params = [
            p for p in model_info['required_params'] 
            if p not in model_params
        ]
        if missing_params:
            raise ValueError(f"Missing required parameters for {model_type}: {missing_params}")
        
        logger.info(f"Initialized {model_type} model with parameters: {model_params}")
        return model_info['class'](**model_params), param_grid
        
    except Exception as e:
        logger.error(f"Model initialization failed for {model_type}: {str(e)}")
        raise

def save_gridsearch_results(grid, model_type: str) -> str:
    """Save complete GridSearchCV results and best estimator"""
    results_dir = Path("results/gridsearch")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    results_df = pd.DataFrame(grid.cv_results_)
    results_path = results_dir / f"{model_type}_results.csv"
    results_df.to_csv(results_path, index=False)
    
    # Save best parameters
    best_params_path = Path("results") / f"best_params_{model_type}.yaml"
    with open(best_params_path, 'w') as f:
        yaml.dump(grid.best_params_, f)
    
    # Save best estimator with metadata
    model_metadata = {
        'best_estimator': grid.best_estimator_,
        'best_params': grid.best_params_,
        'best_score': grid.best_score_,
        'model_type': model_type,
        'timestamp': time()
    }
    
    model_path = Path("models") / f"{model_type}_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_metadata, f)
    
    return str(model_path)

def run_grid_search(model, X_train, y_train, param_grid, grid_config, model_type: str) -> Tuple[Any, str]:
    """Execute grid search with enhanced logging and saving"""
    try:
        logger.info(f"Starting GridSearchCV for {model_type}")
        
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            **grid_config
        )
        
        grid.fit(X_train, y_train)
        
        # Save all results
        model_path = save_gridsearch_results(grid, model_type)
        
        logger.info(f"Best parameters for {model_type}: {grid.best_params_}")
        logger.info(f"Best cross-validation score: {grid.best_score_:.4f}")
        
        return grid, model_path  # Return both grid object and model path
        
    except Exception as e:
        logger.error(f"Grid search failed for {model_type}: {str(e)}")
        raise

def train_all_models() -> Dict[str, str]:
    """Main training function with GridSearchCV"""
    try:
        logger.info("===== Starting Training Pipeline with GridSearch =====")
        
        with Live(save_dvc_exp=True) as live:
            # Load configuration and data
            config = load_config()
            live.log_params(config)
            
            X_train, y_train = load_training_data('data/feature')
            grid_config = config.get('grid_search', {})
            
            # Train each model
            trained_models = {}
            models_to_train = ['logistic', 'svm', 'random_forest']
            
            for model_type in models_to_train:
                try:
                    logger.info(f"\n{'='*40}")
                    logger.info(f"TRAINING {model_type.upper()} WITH GRIDSEARCH")
                    logger.info(f"{'='*40}")
                    
                    # Initialize base model
                    model, param_grid = initialize_model(model_type, config)
                    
                    # Scale features
                    X_train_processed = scale_features(X_train, model_type)
                    
                    # Run grid search
                    grid, model_path = run_grid_search(  # Now receives both grid and path
                        model,
                        X_train_processed,
                        y_train,
                        param_grid,
                        grid_config,
                        model_type
                    )
                    
                    trained_models[model_type] = model_path
                    
                    # Log best parameters
                    live.log_params({f"{model_type}_best_params": grid.best_params_})
                    live.log_metric(f"{model_type}/best_cv_score", grid.best_score_)
                    
                    logger.info(f"{'='*40}")
                    logger.info(f"{model_type.upper()} TRAINING COMPLETE")
                    logger.info(f"{'='*40}\n")
                    
                except Exception as e:
                    logger.error(f"Failed to train {model_type}: {str(e)}")
                    continue
            
            # Final summary
            if not trained_models:
                raise RuntimeError("No models were successfully trained")
                
            logger.info("===== Training Completed =====")
            return trained_models
            
    except Exception as e:
        logger.critical(f"Training pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    train_all_models()