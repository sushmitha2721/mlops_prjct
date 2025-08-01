import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from typing import Tuple, List
from pathlib import Path
from logger import setup_logging
import os
from scipy.sparse import save_npz, load_npz

logger = setup_logging(logger_name="feature_engineering")

class FeatureEngineer:
    def __init__(self, max_tfidf_features: int = 1000, ngram_range: Tuple[int, int] = (1, 2)):
        """Initialize feature engineering pipeline with detailed configuration"""
        self.max_tfidf_features = max_tfidf_features
        self.ngram_range = ngram_range
        self.tfidf = None
        self.scaler = None
        self.stemmer = PorterStemmer()
        self.stopwords = set(stopwords.words('english'))
        self.stemmed_stopwords = set(self.stemmer.stem(w) for w in stopwords.words('english'))
        
        logger.info(
            f"Initialized FeatureEngineer with: "
            f"max_tfidf_features={max_tfidf_features}, "
            f"ngram_range={ngram_range}"
        )

    def _validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean input data with detailed logging"""
        logger.info("Starting input validation")
        
        # Check required columns
        required_columns = {'cleaned_text', 'processed_text', 'target'}
        missing = set(required_columns) - set(df.columns)
        if missing:
            logger.error(f"Missing required columns: {missing}")
            raise ValueError(f"Missing required columns: {missing}")
        logger.debug("All required columns present")

        # Handle null values
        null_counts = df[['cleaned_text', 'processed_text']].isnull().sum()
        total_nulls = null_counts.sum()
        
        if total_nulls > 0:
            logger.warning(
                f"Found {total_nulls} null values:\n"
                f"- cleaned_text: {null_counts['cleaned_text']}\n"
                f"- processed_text: {null_counts['processed_text']}\n"
                "Filling with empty strings"
            )
            df = df.copy()  # Avoid SettingWithCopyWarning
            df['cleaned_text'] = df['cleaned_text'].fillna('')
            df['processed_text'] = df['processed_text'].fillna('')
            
            # Verify correction
            new_nulls = df[['cleaned_text', 'processed_text']].isnull().sum().sum()
            if new_nulls > 0:
                logger.error(f"Failed to fill all nulls. Remaining: {new_nulls}")
                raise ValueError("Null values persist after filling")
            logger.info("All null values handled successfully")
        
        return df

    def _get_tfidf_features(self, processed_text: pd.Series) -> csr_matrix:
        """Generate TF-IDF features with comprehensive logging"""
        logger.info("Initializing TF-IDF vectorizer")
        try:
            self.tfidf = TfidfVectorizer(
                tokenizer=lambda x: x.split(),
                token_pattern=None,
                preprocessor=None,
                lowercase=False,
                max_features=self.max_tfidf_features,
                ngram_range=self.ngram_range,
                stop_words=None
            )
            
            logger.info(
                f"Fitting TF-IDF on {len(processed_text)} documents "
                f"(max_features={self.max_tfidf_features})"
            )
            X_text = self.tfidf.fit_transform(processed_text)
            
            logger.info(
                f"TF-IDF transformation complete. "
                f"Shape: {X_text.shape}, "
                f"Vocab size: {len(self.tfidf.get_feature_names_out())}"
            )
            return X_text
            
        except Exception as e:
            logger.error(
                "TF-IDF failed on sample texts:\n"
                f"{processed_text.head().tolist()}\n"
                f"Error: {str(e)}"
            )
            raise

    def _get_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features with progress tracking"""
        logger.info("Creating engineered features")
        features = pd.DataFrame(index=df.index)
        
        # Length features
        logger.debug("Calculating length features")
        features['char_count'] = df['cleaned_text'].str.len().fillna(0)
        tokenized = df['processed_text'].str.split().fillna('')
        features['word_count'] = tokenized.apply(len).fillna(0)
        features['avg_word_length'] = np.where(
            features['word_count'] > 0,
            features['char_count'] / features['word_count'],
            0
        )

        # Special tokens
        logger.debug("Counting special tokens")
        special_tokens = ['PHONENUM', 'URL', 'EMAIL', 'CURRENCY', 'NUMBER']
        for token in special_tokens:
            features[f'count_{token}'] = (
                df['processed_text']
                .str.count(rf'\b{token}\b')
                .fillna(0)
                .astype(int))
            logger.debug(f"Added feature: count_{token}")

        # Linguistic features
        logger.debug("Calculating linguistic features")
        features['stopword_ratio'] = tokenized.apply(
            lambda words: sum(w in self.stemmed_stopwords for w in words)/max(len(words), 1)
        ).fillna(0)
        
        features['unique_word_ratio'] = tokenized.apply(
            lambda words: len(set(words))/max(len(words), 1)
        ).fillna(0)

        # Spam indicators
        logger.debug("Calculating spam indicators")
        features['has_exclamation'] = (
            df['cleaned_text']
            .str.contains(r'!+')
            .fillna(0)
            .astype(int))
        
        features['all_caps_ratio'] = df['cleaned_text'].apply(
            lambda x: sum(1 for c in x if c.isupper())/max(len(x), 1) 
            if isinstance(x, str) else 0
        )

        logger.info(
            f"Engineered {len(features.columns)} features: "
            f"{list(features.columns)}"
        )
        return features

    def transform(self, df: pd.DataFrame) -> csr_matrix:
        """Orchestrate full feature generation with error handling"""
        logger.info("Starting feature transformation pipeline")
        
        try:
            # Input validation
            df = self._validate_input(df)
            
            # TF-IDF features
            logger.info("Phase 1: Text feature extraction")
            X_text = self._get_tfidf_features(df['processed_text'])
            
            # Engineered features
            logger.info("Phase 2: Feature engineering")
            X_engineered = self._get_engineered_features(df)
            
            
            
            # Feature combination
            logger.info("Phase 3: Feature combination")
            combined = hstack([X_text, csr_matrix(X_engineered)])
            logger.info(
                f"Feature combination complete. Final shape: {combined.shape}\n"
                f"- Text features: {X_text.shape[1]}\n"
                f"- Engineered features: {X_engineered.shape[1]}"
            )
            
            return combined
            
        except Exception as e:
            logger.critical(
                "Feature generation failed\n"
                f"Error: {str(e)}\n"
                f"Data shape: {df.shape}\n"
                f"Columns: {df.columns.tolist()}"
            )
            raise

    def get_feature_names(self) -> List[str]:
        """Get feature names with validation"""
        if not self.tfidf:
            logger.error("Attempted to get feature names before fitting TF-IDF")
            raise RuntimeError("Fit the transformer first")
            
        feature_names = (
            list(self.tfidf.get_feature_names_out()) + 
            ['char_count', 'word_count', 'avg_word_length',
             'count_PHONENUM', 'count_URL', 'count_EMAIL',
             'count_CURRENCY', 'count_NUMBER',
             'stopword_ratio', 'unique_word_ratio',
             'has_exclamation', 'all_caps_ratio']
        )
        logger.debug(f"Retrieved {len(feature_names)} feature names")
        return feature_names

if __name__ == '__main__':
    try:
        logger.info("===== Starting Feature Engineering Pipeline =====")
        
        # Data loading
        logger.info("Loading processed data")
        train_path = 'data/processed/train_processed.csv'
        test_path = 'data/processed/test_processed.csv'
        
        logger.info(f"Loading train data from {train_path}")
        train = pd.read_csv(train_path)
        logger.info(f"Train data loaded. Shape: {train.shape}")
        
        logger.info(f"Loading test data from {test_path}")
        test = pd.read_csv(test_path)
        logger.info(f"Test data loaded. Shape: {test.shape}")

        # Extract target variables
        y_train = train['target'].values
        y_test = test['target'].values
        
        # Null check
        logger.info("Checking for null values")
        train_nulls = train[['cleaned_text', 'processed_text']].isnull().sum()
        test_nulls = test[['cleaned_text', 'processed_text']].isnull().sum()
        
        logger.info(
            "Null value summary:\n"
            f"Train:\n{train_nulls}\n"
            f"Test:\n{test_nulls}"
        )

        # Feature engineering
        logger.info("Initializing FeatureEngineer")
        fe = FeatureEngineer()
        
        logger.info("Transforming training data")
        X_train = fe.transform(train)
        
        logger.info("Transforming test data")
        X_test = fe.transform(test)
        
        # Success message
        logger.info(
            "Pipeline completed successfully\n"
            f"Final feature matrices:\n"
            f"- Train: {X_train.shape}\n"
            f"- Test: {X_test.shape}"
        )

        #save results
        os.makedirs('./data/feature',exist_ok=True)
        # Save sparse matrices
        save_npz('./data/feature/X_train.npz', X_train)  
        save_npz('./data/feature/X_test.npz', X_test)
        

        #Save target variables
        np.save('./data/feature/y_train.npy', y_train)
        np.save('./data/feature/y_test.npy', y_test)
        logger.info("Results saved in the folder")
        
    except Exception as e:
        logger.critical(
            "Pipeline failed catastrophically\n"
            f"Error type: {type(e).__name__}\n"
            f"Error message: {str(e)}",
            exc_info=True
        )
        raise