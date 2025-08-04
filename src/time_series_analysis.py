"""
Time Series Analysis Module for Brent Oil Price Data

This module provides functions for analyzing time series properties including
stationarity tests, trend analysis, and seasonal decomposition.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesAnalyzer:
    """
    Comprehensive time series analysis for oil price data.
    """
    
    def __init__(self, data: pd.DataFrame, price_column: str = 'Price'):
        """
        Initialize the analyzer with time series data.
        
        Args:
            data (pd.DataFrame): Time series data with datetime index
            price_column (str): Name of the price column to analyze
        """
        self.data = data.copy()
        self.price_column = price_column
        self.results = {}
        
    def test_stationarity(self, series: pd.Series = None, 
                         significance_level: float = 0.05) -> dict:
        """
        Test for stationarity using Augmented Dickey-Fuller and KPSS tests.
        
        Args:
            series (pd.Series): Time series to test (default: price column)
            significance_level (float): Significance level for tests
            
        Returns:
            dict: Results of stationarity tests
        """
        if series is None:
            series = self.data[self.price_column]
        
        # Remove any NaN values
        series = series.dropna()
        
        results = {}
        
        # Augmented Dickey-Fuller Test
        adf_result = adfuller(series)
        results['ADF'] = {
            'statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < significance_level,
            'interpretation': 'Stationary' if adf_result[1] < significance_level else 'Non-stationary'
        }
        
        # KPSS Test
        kpss_result = kpss(series, regression='ct')  # trend stationary
        results['KPSS'] = {
            'statistic': kpss_result[0],
            'p_value': kpss_result[1],
            'critical_values': kpss_result[3],
            'is_stationary': kpss_result[1] > significance_level,
            'interpretation': 'Stationary' if kpss_result[1] > significance_level else 'Non-stationary'
        }
        
        # Overall conclusion
        adf_stationary = results['ADF']['is_stationary']
        kpss_stationary = results['KPSS']['is_stationary']
        
        if adf_stationary and kpss_stationary:
            conclusion = "Both tests indicate stationarity"
        elif adf_stationary and not kpss_stationary:
            conclusion = "Mixed results: ADF suggests stationary, KPSS suggests non-stationary"
        elif not adf_stationary and kpss_stationary:
            conclusion = "Mixed results: ADF suggests non-stationary, KPSS suggests stationary"
        else:
            conclusion = "Both tests indicate non-stationarity"
            
        results['conclusion'] = conclusion
        
        return results
    
    def analyze_trend(self, series: pd.Series = None) -> dict:
        """
        Analyze trend components in the time series.
        
        Args:
            series (pd.Series): Time series to analyze
            
        Returns:
            dict: Trend analysis results
        """
        if series is None:
            series = self.data[self.price_column]
        
        # Linear trend
        x = np.arange(len(series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, series.values)
        
        trend_results = {
            'linear_trend': {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'is_significant': p_value < 0.05,
                'direction': 'Increasing' if slope > 0 else 'Decreasing' if slope < 0 else 'No trend'
            }
        }
        
        # Moving average trends
        ma_30 = series.rolling(window=30).mean()
        ma_365 = series.rolling(window=365).mean()
        
        trend_results['moving_averages'] = {
            'ma_30_final': ma_30.iloc[-1] if not ma_30.empty else None,
            'ma_365_final': ma_365.iloc[-1] if not ma_365.empty else None,
            'ma_30_vs_365': ma_30.iloc[-1] > ma_365.iloc[-1] if not ma_30.empty and not ma_365.empty else None
        }
        
        return trend_results
    
    def seasonal_decomposition(self, series: pd.Series = None, 
                             period: int = 365) -> dict:
        """
        Perform seasonal decomposition of the time series.
        
        Args:
            series (pd.Series): Time series to decompose
            period (int): Period for seasonal decomposition
            
        Returns:
            dict: Decomposition components
        """
        if series is None:
            series = self.data[self.price_column]
        
        # Ensure we have enough data points
        if len(series) < 2 * period:
            period = len(series) // 2
        
        decomposition = seasonal_decompose(series, model='additive', period=period)
        
        return {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid,
            'original': decomposition.observed
        }
    
    def analyze_volatility(self, series: pd.Series = None, 
                          window: int = 30) -> dict:
        """
        Analyze volatility patterns in the time series.
        
        Args:
            series (pd.Series): Time series to analyze
            window (int): Rolling window for volatility calculation
            
        Returns:
            dict: Volatility analysis results
        """
        if series is None:
            series = self.data[self.price_column]
        
        # Calculate returns
        returns = series.pct_change().dropna()
        
        # Rolling volatility (standard deviation of returns)
        rolling_vol = returns.rolling(window=window).std()
        
        volatility_results = {
            'returns_stats': {
                'mean': returns.mean(),
                'std': returns.std(),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis(),
                'min': returns.min(),
                'max': returns.max()
            },
            'rolling_volatility': {
                'mean': rolling_vol.mean(),
                'std': rolling_vol.std(),
                'max': rolling_vol.max(),
                'max_date': rolling_vol.idxmax()
            }
        }
        
        return volatility_results
    
    def comprehensive_analysis(self) -> dict:
        """
        Perform comprehensive time series analysis.
        
        Returns:
            dict: Complete analysis results
        """
        series = self.data[self.price_column]
        
        results = {
            'data_info': {
                'start_date': series.index.min(),
                'end_date': series.index.max(),
                'total_observations': len(series),
                'missing_values': series.isnull().sum()
            },
            'stationarity': self.test_stationarity(series),
            'trend_analysis': self.analyze_trend(series),
            'volatility_analysis': self.analyze_volatility(series)
        }
        
        # Test stationarity of returns
        if 'Returns' in self.data.columns:
            returns_stationarity = self.test_stationarity(self.data['Returns'].dropna())
            results['returns_stationarity'] = returns_stationarity
        
        # Test stationarity of log prices
        if 'Log_Price' in self.data.columns:
            log_stationarity = self.test_stationarity(self.data['Log_Price'].dropna())
            results['log_price_stationarity'] = log_stationarity
        
        self.results = results
        return results
    
    def plot_analysis(self, figsize: tuple = (15, 12)) -> plt.Figure:
        """
        Create comprehensive plots for time series analysis.
        
        Args:
            figsize (tuple): Figure size
            
        Returns:
            plt.Figure: Figure object with all plots
        """
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle('Brent Oil Price Time Series Analysis', fontsize=16)
        
        series = self.data[self.price_column]
        
        # Original time series
        axes[0, 0].plot(series.index, series.values)
        axes[0, 0].set_title('Brent Oil Prices Over Time')
        axes[0, 0].set_ylabel('Price (USD)')
        axes[0, 0].grid(True)
        
        # Returns
        if 'Returns' in self.data.columns:
            returns = self.data['Returns'].dropna()
            axes[0, 1].plot(returns.index, returns.values)
            axes[0, 1].set_title('Daily Returns')
            axes[0, 1].set_ylabel('Returns')
            axes[0, 1].grid(True)
        
        # Moving averages
        axes[1, 0].plot(series.index, series.values, label='Price', alpha=0.7)
        if 'MA_30' in self.data.columns:
            axes[1, 0].plot(self.data.index, self.data['MA_30'], label='30-day MA')
        if 'MA_365' in self.data.columns:
            axes[1, 0].plot(self.data.index, self.data['MA_365'], label='365-day MA')
        axes[1, 0].set_title('Price with Moving Averages')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Volatility
        if 'Returns' in self.data.columns:
            rolling_vol = self.data['Returns'].rolling(window=30).std()
            axes[1, 1].plot(rolling_vol.index, rolling_vol.values)
            axes[1, 1].set_title('30-Day Rolling Volatility')
            axes[1, 1].set_ylabel('Volatility')
            axes[1, 1].grid(True)
        
        # Distribution of returns
        if 'Returns' in self.data.columns:
            returns = self.data['Returns'].dropna()
            axes[2, 0].hist(returns, bins=50, alpha=0.7, density=True)
            axes[2, 0].set_title('Distribution of Returns')
            axes[2, 0].set_xlabel('Returns')
            axes[2, 0].set_ylabel('Density')
            axes[2, 0].grid(True)
        
        # Q-Q plot
        if 'Returns' in self.data.columns:
            returns = self.data['Returns'].dropna()
            stats.probplot(returns, dist="norm", plot=axes[2, 1])
            axes[2, 1].set_title('Q-Q Plot (Returns vs Normal)')
            axes[2, 1].grid(True)
        
        plt.tight_layout()
        return fig


def analyze_oil_price_series(data: pd.DataFrame, price_column: str = 'Price') -> dict:
    """
    Convenience function for comprehensive time series analysis.
    
    Args:
        data (pd.DataFrame): Time series data
        price_column (str): Name of price column
        
    Returns:
        dict: Analysis results
    """
    analyzer = TimeSeriesAnalyzer(data, price_column)
    return analyzer.comprehensive_analysis()


if __name__ == "__main__":
    # Example usage
    from data_loader import load_brent_oil_data
    
    # Load data
    data = load_brent_oil_data()
    
    # Analyze
    analyzer = TimeSeriesAnalyzer(data)
    results = analyzer.comprehensive_analysis()
    
    print("Time Series Analysis Results:")
    for key, value in results.items():
        print(f"\n{key.upper()}:")
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")
        else:
            print(f"  {value}")