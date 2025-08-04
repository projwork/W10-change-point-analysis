"""
Data Loading and Preprocessing Module for Brent Oil Price Analysis

This module handles loading, cleaning, and preprocessing of Brent oil price data
for change point analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BrentOilDataLoader:
    """
    Data loader for Brent oil price data with preprocessing capabilities.
    """
    
    def __init__(self, data_path: str = "data/BrentOilPrices.csv"):
        """
        Initialize the data loader.
        
        Args:
            data_path (str): Path to the Brent oil prices CSV file
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load raw Brent oil price data from CSV file.
        
        Returns:
            pd.DataFrame: Raw data with Date and Price columns
        """
        try:
            self.raw_data = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(self.raw_data)} records from {self.data_path}")
            return self.raw_data.copy()
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the loaded data: parse dates, handle missing values, etc.
        
        Returns:
            pd.DataFrame: Processed data ready for analysis
        """
        if self.raw_data is None:
            self.load_data()
        
        # Create a copy for processing
        df = self.raw_data.copy()
        
        # Parse dates - handle multiple date formats
        try:
            # Try the primary format first
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
        except ValueError:
            # If that fails, use mixed format parsing
            df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
        
        # Handle potential year ambiguity (87 could be 1987 or 2087)
        # Adjust years that are interpreted as 20xx to 19xx for dates before 2000
        df.loc[df['Date'].dt.year > 2022, 'Date'] = df.loc[df['Date'].dt.year > 2022, 'Date'] - pd.DateOffset(years=100)
        
        # Set Date as index
        df.set_index('Date', inplace=True)
        
        # Sort by date
        df.sort_index(inplace=True)
        
        # Handle missing values in Price column
        if df['Price'].isnull().any():
            missing_count = df['Price'].isnull().sum()
            logger.warning(f"Found {missing_count} missing price values")
            
            # Forward fill missing values
            df['Price'].fillna(method='ffill', inplace=True)
            df['Price'].fillna(method='bfill', inplace=True)
        
        # Ensure Price is numeric
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        
        # Remove any remaining NaN values
        df.dropna(inplace=True)
        
        # Add derived features
        df['Returns'] = df['Price'].pct_change()
        df['Log_Price'] = np.log(df['Price'])
        df['Log_Returns'] = df['Log_Price'].diff()
        
        # Add moving averages
        df['MA_30'] = df['Price'].rolling(window=30).mean()
        df['MA_90'] = df['Price'].rolling(window=90).mean()
        df['MA_365'] = df['Price'].rolling(window=365).mean()
        
        self.processed_data = df
        logger.info(f"Preprocessed data shape: {df.shape}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        
        return df.copy()
    
    def get_data_summary(self) -> dict:
        """
        Get summary statistics of the loaded data.
        
        Returns:
            dict: Summary statistics and data information
        """
        if self.processed_data is None:
            self.preprocess_data()
        
        df = self.processed_data
        
        summary = {
            'total_records': len(df),
            'date_range': {
                'start': df.index.min(),
                'end': df.index.max(),
                'days': (df.index.max() - df.index.min()).days
            },
            'price_statistics': {
                'mean': df['Price'].mean(),
                'std': df['Price'].std(),
                'min': df['Price'].min(),
                'max': df['Price'].max(),
                'median': df['Price'].median()
            },
            'missing_values': df.isnull().sum().to_dict(),
            'returns_statistics': {
                'mean': df['Returns'].mean(),
                'std': df['Returns'].std(),
                'skewness': df['Returns'].skew(),
                'kurtosis': df['Returns'].kurtosis()
            }
        }
        
        return summary
    
    def get_data_for_period(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get data for a specific time period.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Data for the specified period
        """
        if self.processed_data is None:
            self.preprocess_data()
        
        return self.processed_data.loc[start_date:end_date].copy()
    
    def export_processed_data(self, output_path: str) -> None:
        """
        Export processed data to CSV file.
        
        Args:
            output_path (str): Path for the output CSV file
        """
        if self.processed_data is None:
            self.preprocess_data()
        
        self.processed_data.to_csv(output_path)
        logger.info(f"Processed data exported to {output_path}")


def load_brent_oil_data(data_path: str = "data/BrentOilPrices.csv") -> pd.DataFrame:
    """
    Convenience function to load and preprocess Brent oil data.
    
    Args:
        data_path (str): Path to the data file
        
    Returns:
        pd.DataFrame: Processed Brent oil price data
    """
    loader = BrentOilDataLoader(data_path)
    return loader.preprocess_data()


if __name__ == "__main__":
    # Example usage
    loader = BrentOilDataLoader()
    data = loader.preprocess_data()
    summary = loader.get_data_summary()
    
    print("Data Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")