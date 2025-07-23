import pandas as pd
import re
import unicodedata
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk 
from logger import setup_logging
import os
import logging

logger = setup_logging(logger_name="data_preprocessing")

# Download NLTK resources with error handling
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    logger.debug("NLTK resources downloaded successfully")
except Exception as e:
    logger.error(f"Failed to download NLTK resources: {e}")
    raise

class TextPreprocessor:
    def __init__(self, use_stemming=True, debug_mode=False):
        """
        Enhanced text preprocessor that maintains your cleaning approach 
        while preventing empty results
        """
        self.use_stemming = use_stemming
        self.debug_mode = debug_mode
        self.stemmer = PorterStemmer() if use_stemming else None
        self.stopwords = set(stopwords.words('english'))
        self._compile_regex_patterns()
        logger.info(f"Initialized TextPreprocessor (stemming={use_stemming})")

    def _compile_regex_patterns(self):
        """Pre-compile regex patterns for efficiency"""
        self.replacements = [
            (re.compile(r'[\+\(]?\d[\d\-\(\) ]{7,}\d'), ' PHONENUM '),
            (re.compile(r'http[s]?://\S+|www\.\S+'), ' URL '),
            (re.compile(r'\S+@\S+'), ' EMAIL '),
            (re.compile(r'£|\$|€|¥|₹'), ' CURRENCY '),
            (re.compile(r'\b\d+\b'), ' NUMBER '),
            (re.compile(r'[^\w\s]'), ' '),  # Remove punctuation
        ]
        logger.debug("Compiled regex patterns")

    def clean_text(self, text):
        """Your original cleaning method with enhanced unicode handling"""
        try:
            if not isinstance(text, str):
                logger.debug(f"Non-string input received: {type(text)}")
                return ""
            
            # Enhanced unicode normalization
            text = unicodedata.normalize('NFKD', text)
            text = text.encode('ascii', 'ignore').decode('ascii').lower()
            
            # Apply all replacements
            for pattern, replacement in self.replacements:
                text = pattern.sub(replacement, text)
                
            # Clean whitespace and return
            result = re.sub(r'\s+', ' ', text).strip()
            
            if self.debug_mode and not result:
                logger.debug(f"Empty result after cleaning text: {text[:50]}...")
                
            return result
            
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return ""

    def tokenize_and_process(self, text):
        """
        tokenization that:
        1. Preserves cleaning approach
        2. Prevents empty results
        3. Maintains special tokens
        """
        try:
            # Skip empty inputs
            if not text.strip():
                logger.debug("Empty input text received in tokenizer")
                return ""
                
            # Tokenize
            tokens = word_tokenize(text)
            
            # Process tokens with safeguards
            processed_tokens = []
            for token in tokens:
                # Always keep special tokens
                if token in ['PHONENUM', 'URL', 'EMAIL', 'CURRENCY', 'NUMBER']:
                    processed_tokens.append(token)
                    continue
                    
                # Apply your standard filtering
                if token.isalnum() and token not in self.stopwords:
                    if self.use_stemming and self.stemmer:
                        processed_tokens.append(self.stemmer.stem(token))
                    else:
                        processed_tokens.append(token)
            
            # Fallback: if nothing remains, return cleaned text (truncated)
            if not processed_tokens:
                logger.debug(f"No valid tokens after processing: {text[:50]}...")
                return text[:200].strip()  # Return first 200 chars as fallback
                
            return " ".join(processed_tokens)
            
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            return text[:200].strip()  # Fallback to cleaned text

def preprocess_data(df, text_col='message', target_col='target', use_stemming=True, debug_mode=False):
    """Full preprocessing pipeline with error handling"""
    logger.info("Starting data preprocessing")
    
    try:
        # Validate input
        if df.empty:
            logger.error("Empty DataFrame received")
            raise ValueError("Empty DataFrame received")
        
        logger.info(f"Processing {len(df)} records")
        
        # Encode target
        logger.info("Encoding target variable")
        encoder = LabelEncoder()
        df[target_col] = encoder.fit_transform(df[target_col])
        
        # Clean and process text
        preprocessor = TextPreprocessor(use_stemming=use_stemming, debug_mode=debug_mode)
        
        logger.info("Cleaning raw texts")
        tqdm.pandas(desc="Cleaning texts")
        df['cleaned_text'] = df[text_col].progress_apply(preprocessor.clean_text)
        
        logger.info("Tokenizing and processing cleaned texts")
        tqdm.pandas(desc="Processing texts")
        df['processed_text'] = df['cleaned_text'].progress_apply(preprocessor.tokenize_and_process)
        
        logger.info("Preprocessing completed successfully")
        return df
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        raise

def analyze_dataset(df, name, original_df=None):
    """Analyze processed dataset with detailed logging"""
    logger.info(f"\n{'='*30} {name} Dataset Analysis {'='*30}")
    
    # Basic stats
    logger.info(f"Total records: {len(df)}")
    
    # Empty strings analysis
    empty_cleaned = (df['cleaned_text'].str.strip() == '').sum()
    empty_processed = (df['processed_text'].str.strip() == '').sum()
    
    logger.info(f"\nEmpty String Counts:")
    logger.info(f"- Cleaned text: {empty_cleaned} records ({empty_cleaned/len(df):.2%})")
    logger.info(f"- Processed text: {empty_processed} records ({empty_processed/len(df):.2%})")
    
    # Null values check
    null_counts = df[['cleaned_text', 'processed_text']].isnull().sum()
    logger.info("\nNull Value Verification:")
    if null_counts.sum() == 0:
        logger.info("[OK] No null values found")
    else:
        logger.warning(f"Null values detected:\n{null_counts.to_string()}")
    
    # Sample analysis of empty strings
    if empty_cleaned > 0 or empty_processed > 0:
        logger.info("\nEmpty String Samples:")
        
        empty_samples = df[
            (df['cleaned_text'].str.strip() == '') | 
            (df['processed_text'].str.strip() == '')
        ].head(2)
        
        for idx, row in empty_samples.iterrows():
            logger.info(f"\nSample ID: {idx}")
            logger.info(f"Original: {original_df.loc[idx, 'message'][:100]}")
            logger.info(f"Cleaned: {row['cleaned_text']}")
            logger.info(f"Processed: {row['processed_text']}")

if __name__ == '__main__':
    try:
        logger.info("Starting data preprocessing pipeline")
        
        # Load data
        logger.info("Loading raw data")
        train = pd.read_csv('./data/raw/train.csv')
        test = pd.read_csv('./data/raw/test.csv')
        logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")
        
        # Process data
        logger.info("Processing training data")
        train_processed = preprocess_data(train, use_stemming=True, debug_mode=False)
        
        logger.info("Processing test data")
        test_processed = preprocess_data(test, use_stemming=True, debug_mode=False)
        
        # Analysis
        logger.info("Analyzing results")
        analyze_dataset(train_processed, "Training", train)
        analyze_dataset(test_processed, "Test", test)
        
        # Save results
        logger.info("Saving processed data")
        os.makedirs('./data/processed', exist_ok=True)
        train_processed.to_csv('./data/processed/train_processed.csv', index=False)
        test_processed.to_csv('./data/processed/test_processed.csv', index=False)
        
        logger.info("Pipeline completed successfully. Processed data saved to /data/processed/")
        
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)
        raise