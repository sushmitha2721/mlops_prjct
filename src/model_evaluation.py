import os
import pickle
import logging
import yaml
import numpy as np
from scipy.sparse import load_npz, csr_matrix
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from logger import setup_logging
from pathlib import Path
from typing import Dict, Any, Tuple

logger = setup_logging(logger_name='model_evaluation')

def load_test_data(data_dir: str) -> Tuple[csr_matrix, np.ndarray]:
    """Load test features and labels with validation"""
    try:
        logger.info(f"Loading test data from {data_dir}")
        
        X_path = Path(data_dir) / "X_test.npz"
        y_path = Path(data_dir) / "y_test.npy"
        
        if not X_path.exists():
            raise FileNotFoundError(f"Features file not found: {X_path}")
        if not y_path.exists():
            raise FileNotFoundError(f"Labels file not found: {y_path}")
        
        X_test = load_npz(X_path)
        y_test = np.load(y_path)
        
        logger.info(f"Test data loaded - Features: {X_test.shape}, Labels: {y_test.shape}")
        return X_test, y_test
        
    except Exception as e:
        logger.error(f"Failed to load test data: {str(e)}")
        raise
def load_models(models_dir: str = "models") -> Dict[str, Any]:
    """Load all trained models from directory"""
    try:
        models_dir_path = Path(models_dir).absolute()  # Get absolute path
        logger.info(f"Loading models from {models_dir_path}")
        
        # Verify directory exists
        if not models_dir_path.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir_path}")
        
        models = {}
        model_files = list(models_dir_path.glob('*.pkl'))
        
        if not model_files:
            raise FileNotFoundError(f"No .pkl files found in {models_dir_path}")
        
        for model_file in model_files:
            model_type = model_file.stem.split('_')[0]
            with open(model_file, 'rb') as f:
                models[model_type] = pickle.load(f)
            logger.info(f"Loaded {model_type} model from {model_file}")
            
        return models
        
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise

def evaluate_model(model, model_type: str, X_test: csr_matrix, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate a single model and return metrics"""
    try:
        logger.info(f"Evaluating {model_type} model...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Log detailed report
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        logger.info(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to evaluate {model_type} model: {str(e)}")
        raise

def save_evaluation_results(results: Dict[str, Any], output_dir: str = "results") -> str:
    """Save evaluation results to file"""
    try:
        output_path = Path(output_dir) / "evaluation_results.yaml"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(results, f)
        
        logger.info(f"Saved evaluation results to {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Failed to save evaluation results: {str(e)}")
        raise

def evaluate_all_models():
    """Evaluate all trained models on test data"""
    try:
        logger.info("===== Starting Model Evaluation Pipeline =====")
        
        # 1. Load test data
        X_test, y_test = load_test_data('data/feature')
        
        # 2. Load trained models
        models = load_models()
        
        # 3. Evaluate each model
        evaluation_results = {}
        for model_type, model in models.items():
            try:
                logger.info(f"\n{'='*40}")
                logger.info(f"EVALUATING {model_type.upper()} MODEL")
                logger.info(f"{'='*40}")
                
                metrics = evaluate_model(model, model_type, X_test, y_test)
                evaluation_results[model_type] = metrics
                
                logger.info(f"{'='*40}")
                logger.info(f"{model_type.upper()} EVALUATION COMPLETE")
                logger.info(f"{'='*40}\n")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_type}: {str(e)}")
                continue
        
        # 4. Save results
        results_path = save_evaluation_results(evaluation_results)
        
        # Final summary
        logger.info("\n===== Evaluation Summary =====")
        for model_type, metrics in evaluation_results.items():
            logger.info(f"\n{model_type.upper()} Metrics:")
            for metric, value in metrics.items():
                if metric != 'confusion_matrix':
                    logger.info(f"{metric:<12}: {value:.4f}")
        
        logger.info(f"\nDetailed results saved to: {results_path}")
        logger.info("===== All Model Evaluation Completed =====")
        
        return evaluation_results
        
    except Exception as e:
        logger.critical(f"Evaluation pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    evaluate_all_models()