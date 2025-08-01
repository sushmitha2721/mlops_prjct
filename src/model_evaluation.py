import os
import pickle
import logging
import yaml
from time import time
import numpy as np
from scipy.sparse import load_npz, csr_matrix
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)
from logger import setup_logging
from pathlib import Path
from typing import Dict, Any, Tuple
from dvclive import Live
import matplotlib.pyplot as plt
import json

logger = setup_logging(logger_name='model_evaluation')

def load_params(params_path: str) -> dict:
    """Load parameters from YAML file with enhanced validation"""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        
        if 'model_training' not in params:
            raise ValueError("Missing 'model_training' section in params")
            
        logger.debug("Parameters retrieved from %s", params_path)
        return params
        
    except Exception as e:
        logger.error(f"Failed to load parameters: {str(e)}")
        raise

def load_test_data(data_dir: str) -> Tuple[csr_matrix, np.ndarray]:
    """Load test features and labels with enhanced validation"""
    try:
        logger.info(f"Loading test data from {data_dir}")
        
        data_path = Path(data_dir)
        required_files = {
            'features': data_path / "X_test.npz",
            'labels': data_path / "y_test.npy"
        }
        
        for file_type, file_path in required_files.items():
            if not file_path.exists():
                raise FileNotFoundError(f"Missing {file_type} file: {file_path}")
        
        X_test = load_npz(required_files['features'])
        y_test = np.load(required_files['labels'])
        
        logger.info(f"Test data loaded - Features: {X_test.shape}, Labels: {y_test.shape}")
        return X_test, y_test
        
    except Exception as e:
        logger.error(f"Failed to load test data: {str(e)}")
        raise

def load_models(models_dir: str = "models") -> Dict[str, Any]:
    """Load trained models with GridSearchCV metadata"""
    try:
        models_dir_path = Path(models_dir).absolute()
        logger.info(f"Loading models from {models_dir_path}")
        
        models = {}
        for model_file in models_dir_path.glob('*.pkl'):
            model_type = model_file.stem.split('_')[0]
            
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
                
                if not isinstance(model_data, dict) or 'best_estimator' not in model_data:
                    logger.warning(f"Model {model_file} is not a GridSearchCV result")
                    continue
                    
                models[model_type] = {
                    'model': model_data['best_estimator'],
                    'metadata': {
                        'best_params': model_data.get('best_params', {}),
                        'best_score': model_data.get('best_score', 0),
                        'model_type': model_data.get('model_type', 'unknown'),
                        'timestamp': model_data.get('timestamp', 0)
                    },
                    'path': str(model_file)
                }
                
            logger.info(f"Loaded {model_type} model with params: {models[model_type]['metadata']['best_params']}")
            
        return models
        
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise

def evaluate_model(model, model_metadata: dict, X_test: csr_matrix, y_test: np.ndarray) -> Dict[str, Any]:
    """Enhanced evaluation with GridSearchCV metadata"""
    try:
        model_type = model_metadata.get('model_type', 'unknown')
        logger.info(f"Evaluating {model_type} model...")
        
        # Log training metrics
        logger.info(f"Best CV Score: {model_metadata.get('best_score', 'N/A')}")
        logger.info(f"Best Params: {model_metadata.get('best_params', {})}")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'training': model_metadata,
            'testing': {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            }
        }
        
        if y_prob is not None:
            metrics['testing'].update({
                'roc_auc': roc_auc_score(y_test, y_prob),
                'pr_auc': average_precision_score(y_test, y_prob)
            })
        
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        logger.info(f"Confusion Matrix:\n{metrics['testing']['confusion_matrix']}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to evaluate {model_type} model: {str(e)}")
        raise

def save_evaluation_results(results: Dict[str, Any], output_dir: str = "results") -> str:
    """Save evaluation results with training metadata"""
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save full results as JSON
        results_path = output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # 2. Save simplified version as YAML
        summary_path = output_dir / "evaluation_summary.yaml"
        summary = {
            model: {
                'training': {
                    'best_cv_score': metrics['training']['best_score'],
                    'best_params': metrics['training']['best_params']
                },
                'testing': {k: v for k, v in metrics['testing'].items() 
                          if k not in ['confusion_matrix', 'classification_report']}
            }
            for model, metrics in results.items()
        }
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f)
        
        # 3. Save markdown summary table
        md_path = output_dir / "evaluation_summary.md"
        with open(md_path, 'w') as f:
            f.write("# Model Evaluation Summary\n\n")
            f.write("| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Best CV Score |\n")
            f.write("|-------|----------|-----------|--------|----|---------|---------------|\n")
            for model, metrics in results.items():
                f.write(
                    f"| {model} | "
                    f"{metrics['testing']['accuracy']:.4f} | "
                    f"{metrics['testing']['precision']:.4f} | "
                    f"{metrics['testing']['recall']:.4f} | "
                    f"{metrics['testing']['f1']:.4f} | "
                    f"{metrics['testing'].get('roc_auc', 'N/A'):.4f} | "
                    f"{metrics['training']['best_score']:.4f} |\n"
                )
        
        logger.info(f"Saved evaluation results to {results_path}")
        return str(results_path)
        
    except Exception as e:
        logger.error(f"Failed to save evaluation results: {str(e)}")
        raise

def ensure_plot_files(model_type: str):
    """Ensure all required plot files exist"""
    plot_dir = Path("dvclive") / "plots" / "sklearn" / model_type
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Create empty plot files with basic structure if they don't exist
    plots = {
        "confusion_matrix.json": {
            "actual": [],
            "predicted": [],
            "type": "confusion_matrix"
        },
        "roc_curve.json": {
            "fpr": [],
            "tpr": [],
            "type": "roc_curve"
        },
        "pr_curve.json": {
            "precision": [],
            "recall": [],
            "type": "pr_curve"
        }
    }
    
    for filename, content in plots.items():
        filepath = plot_dir / filename
        if not filepath.exists():
            with open(filepath, 'w') as f:
                json.dump(content, f)

def evaluate_all_models():
    """Enhanced evaluation pipeline with comprehensive tracking"""
    try:
        logger.info("===== Starting Model Evaluation Pipeline =====")
        params = load_params('params.yaml')
        
        with Live(dir="dvclive", save_dvc_exp=True) as live:
            # Log parameters and environment info
            live.log_params(params)
            live.log_param("eval_timestamp", int(time()))
            
            # Load data and models
            X_test, y_test = load_test_data('data/feature')
            models = load_models()
            
            evaluation_results = {}
            for model_type, model_data in models.items():
                try:
                    logger.info(f"\n{'='*40}")
                    logger.info(f"EVALUATING {model_type.upper()} MODEL")
                    logger.info(f"{'='*40}")
                    
                    model = model_data['model']
                    metrics = evaluate_model(model, model_data['metadata'], X_test, y_test)
                    evaluation_results[model_type] = metrics
                    
                    # Get predictions for plotting
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    # Log metrics
                    for metric_name, value in metrics['testing'].items():
                        if isinstance(value, (int, float)):
                            live.log_metric(f"{model_type}/{metric_name}", value)
                    
                    # Log training metrics
                    live.log_metric(f"{model_type}/best_cv_score", metrics['training']['best_score'])
                    
                  
                    
                    # Log model metadata
                    live.log_artifact(
                        model_data['path'],
                        type="model",
                        name=f"{model_type}_model"
                    )
                    
                    logger.info(f"{'='*40}")
                    logger.info(f"{model_type.upper()} EVALUATION COMPLETE")
                    logger.info(f"{'='*40}\n")
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate {model_type}: {str(e)}")
                    continue
            
            # Save and log results
            results_path = save_evaluation_results(evaluation_results)
            live.log_artifact(results_path, type="eval_results")
            
            # Generate summary
            logger.info("\n===== Evaluation Summary =====")
            for model_type, metrics in evaluation_results.items():
                logger.info(f"\n{model_type.upper()} Metrics:")
                logger.info(f"Training CV Score: {metrics['training']['best_score']:.4f}")
                logger.info(f"Test Accuracy: {metrics['testing']['accuracy']:.4f}")
                logger.info(f"Test F1: {metrics['testing']['f1']:.4f}")
                if 'roc_auc' in metrics['testing']:
                    logger.info(f"Test ROC AUC: {metrics['testing']['roc_auc']:.4f}")
            
            logger.info(f"\nDetailed results saved to: {results_path}")
            logger.info("===== All Model Evaluation Completed =====")
            
            return evaluation_results
            
    except Exception as e:
        logger.critical(f"Evaluation pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    evaluate_all_models()