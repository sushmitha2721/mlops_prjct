import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml
from pathlib import Path
from typing import Tuple

# data_ingestion.py and data_preprocessing.py
from logger import setup_logging

logger = setup_logging(logger_name="data_ingestion")  

def load_params(params_path: str)->dict:
    """ Load parameters from a YAML file"""
    try:
        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
        logger.info(f'successfully loaded parameters from {params_path}')
        return params
    except FileNotFoundError:
        logger.error(f'parameters file not found:{params_path}')
        raise
    except yaml.YAMLError as e:
        logger.error(f'unexpected error loading parameters:{e}')
        raise
    except Exception as e:
        logger.error(f'unexpected error loading parameters:{e}')
        raise

def load_data(data_path: str)->pd.DataFrame:
    """load data from csv file"""
    try:
        #Check if the path exists locally
        if os.path.exists(data_path):
            df=pd.read_csv(data_path)
            logger.info(f'Successfully loaded data from local file: {data_path}')
        else:
            #Fallback to treating it as a URL
            df = pd.read_csv(data_path)
            logger.info(f'Successfully loaded data from URL: {data_path}')
        logger.debug(f'Data shape is {df.shape}')
        return df
    
    except pd.errors.ParserError as e:
        logger.error(f'failed to parse the csv file {data_path}:{e}')
        raise
    except Exception as e:
        logger.error(f'unexpected error loading data:{e}')
        raise

def preprocess_data(df: pd.DataFrame)->pd.DataFrame:
    """Preprocess the data with comprehensive column cleaning and validation.
    
    Args:
        df: Input DataFrame to be processed
        
    Returns:
        Processed DataFrame with cleaned columns
        
    Raises:
        KeyError: If required columns are missing
        ValueError: If data quality checks fail
        Exception: For unexpected errors
    """
    try:
        #1. High null-value  column handling
        high_null_cols = [col for col in df.columns 
                           if df[col].isnull().mean()*100 > 0.9]
        if high_null_cols:
            null_percentages = {col:f'{df[col].isnull().mean() * 100:.1f}%' for col in high_null_cols}
            logger.info(f'dropping high-null columns: {high_null_cols}')
            logger.debug(f'Null percentages : {null_percentages}')
            df.drop(columns = high_null_cols,inplace=True)

        #2. Column renaming with validation
        required_columns = {'v1': 'target', 'v2': 'message'}
        missing_columns = [orig for orig in required_columns
                           if orig not in df.columns ]
        
        if missing_columns:
            raise KeyError(f'Missing required columns :{required_columns}')
        
        df.rename(columns=required_columns, inplace = True)

        #3. Post-cleaning validation
        if df.empty:
            raise ValueError("DataFrame is empty after preprocessing")

        logger.info("Data preprocessing completed successfully")
        logger.debug(f'final columns: {df.columns.tolist()}')
        return df

    except KeyError as e:
        logger.error(f'column error: {str(e)}')
        raise
    except ValueError as e:
        logger.error(f"Data validation error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected preprocessing error: {str(e)}", exc_info=True)
        raise        

def split_data(df: pd.DataFrame, test_size: float, random_state: 42)-> Tuple[pd.DataFrame,pd.DataFrame]:
    """Split data into train and test sets"""
    try:
        train_data,test_data = train_test_split(df,test_size= test_size, random_state=random_state,
                                                stratify= df['target'] if 'target' in df.columns else None)
        logger.info(f'Split data into train ({1 - test_size:.1%}) and test ({test_size:.1%}) sets')
        logger.debug(f'Train shape: {train_data.shape}, Test shape: {test_data.shape}')
        return train_data,test_data
    except Exception as e:
        logging.error(f"Error spliting data: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, output_dir: str)->None:
    """Save the train and tesrt datasets"""
    try:
        output_path= Path(output_dir)/'raw'
        output_path.mkdir(parents=True, exist_ok= True)

        train_path = output_path/'train.csv'
        test_path= output_path/'test.csv'

        train_data.to_csv(train_path)
        test_data.to_csv(test_path)
        
        logger.info(f'Saved train data to {train_path}')
        logger.info(f'Saved test data to {test_path}')

    except Exception as e:
        logging.error("Error saving data {e}")
        raise

def main():
   
    
    try:
        logger.info("Starting data ingestion process")

        #Load parameters
        #params= load_params('params.yaml')
        #ingestion_params = params.get('data_ingestion,{}')
        '''
        test_size = ingestion_params.get('test_size',0.2)
        random_state = ingestion_params.get('random_state', 42)
        data_path = ingestion_params.get('data_path', 
            'D:\Sushmitha\project_ds\mlops_prjct\experiments\spam.csv')
        output_dir = ingestion_params.get('output_dir', './data')
        '''

        #Data pipeline
        df = load_data('D:\Sushmitha\project_ds\mlops_prjct\experiments\spam.csv')
        preprocess_df = preprocess_data(df)
        train_data,test_data = split_data(preprocess_df,0.2,42)
        save_data(train_data,test_data,'./data')

        logger.info('Data ingestion process completed successfully')

    except Exception as e:
        logger.error(f'Data ingestion process failed: {e}')
        raise

if __name__ == '__main__':
    main()