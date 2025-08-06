"""
Advanced Models for Oil Price Analysis

This module implements advanced econometric models including VAR (Vector Autoregression)
and Markov-Switching models for comprehensive oil market analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Statistical and econometric libraries
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.api import VAR as VARModel
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.diagnostic import acorr_ljungbox
import yfinance as yf


class VARAnalyzer:
    """
    Vector Autoregression (VAR) analysis for oil prices and macroeconomic variables.
    """
    
    def __init__(self, oil_data: pd.DataFrame):
        """
        Initialize VAR analyzer.
        
        Args:
            oil_data (pd.DataFrame): Oil price time series data
        """
        self.oil_data = oil_data.copy()
        self.var_data = None
        self.var_model = None
        self.var_results = None
        self.external_data = {}
        
        print(f"📊 Initialized VAR Analyzer")
        print(f"   Oil data: {len(oil_data)} observations")
        print(f"   Date range: {oil_data.index.min().strftime('%Y-%m-%d')} to {oil_data.index.max().strftime('%Y-%m-%d')}")
    
    def fetch_macroeconomic_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fetch macroeconomic data for VAR analysis.
        
        Args:
            start_date (str): Start date for data fetch
            end_date (str): End date for data fetch
            
        Returns:
            pd.DataFrame: Combined macroeconomic dataset
        """
        print("\n📥 Fetching macroeconomic data...")
        
        if start_date is None:
            start_date = self.oil_data.index.min().strftime('%Y-%m-%d')
        if end_date is None:
            end_date = self.oil_data.index.max().strftime('%Y-%m-%d')
        
        try:
            # Define tickers for macro data
            tickers = {
                'SPY': 'S&P 500 ETF',
                'GLD': 'Gold ETF', 
                'UUP': 'US Dollar ETF',
                'TLT': 'Treasury Bond ETF',
                'VIX': 'Volatility Index',
                'DJP': 'Commodity ETF'
            }
            
            macro_data = pd.DataFrame()
            
            for ticker, description in tickers.items():
                try:
                    print(f"   Fetching {ticker} ({description})...")
                    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    if not data.empty:
                        # Use adjusted close price
                        macro_data[ticker] = data['Adj Close']
                        print(f"   ✅ {ticker}: {len(data)} observations")
                    else:
                        print(f"   ⚠️ {ticker}: No data available")
                except Exception as e:
                    print(f"   ❌ {ticker}: Error - {str(e)}")
            
            if macro_data.empty:
                print("   ⚠️ No macroeconomic data fetched, using synthetic data")
                return self._create_synthetic_macro_data()
            
            # Forward fill missing values
            macro_data = macro_data.fillna(method='ffill')
            
            # Calculate returns
            for col in macro_data.columns:
                macro_data[f'{col}_Return'] = macro_data[col].pct_change()
            
            self.external_data = macro_data
            print(f"   ✅ Macroeconomic data: {macro_data.shape}")
            
            return macro_data
            
        except Exception as e:
            print(f"   ❌ Error fetching data: {str(e)}")
            print("   Using synthetic data instead")
            return self._create_synthetic_macro_data()
    
    def _create_synthetic_macro_data(self) -> pd.DataFrame:
        """Create synthetic macroeconomic data for testing."""
        print("   📊 Creating synthetic macroeconomic data...")
        
        dates = self.oil_data.index
        n_obs = len(dates)
        
        np.random.seed(42)
        
        # Create correlated synthetic data
        returns = np.random.multivariate_normal(
            mean=[0, 0, 0, 0], 
            cov=[[0.0004, 0.0001, -0.0001, 0.0002],
                 [0.0001, 0.0009, 0.0001, 0.0001],
                 [-0.0001, 0.0001, 0.0001, -0.0001],
                 [0.0002, 0.0001, -0.0001, 0.0004]], 
            size=n_obs
        )
        
        synthetic_data = pd.DataFrame({
            'Stock_Index_Return': returns[:, 0],
            'Gold_Return': returns[:, 1], 
            'USD_Index_Return': returns[:, 2],
            'Bond_Return': returns[:, 3]
        }, index=dates)
        
        return synthetic_data
    
    def prepare_var_data(self, include_oil_returns: bool = True,
                        max_lags: int = 5) -> pd.DataFrame:
        """
        Prepare data for VAR analysis.
        
        Args:
            include_oil_returns (bool): Whether to include oil returns
            max_lags (int): Maximum number of lags for stationarity testing
            
        Returns:
            pd.DataFrame: Prepared VAR dataset
        """
        print("\n🔧 Preparing data for VAR analysis...")
        
        # Start with oil data
        oil_returns = self.oil_data['Price'].pct_change().dropna()
        
        var_data = pd.DataFrame({
            'Oil_Return': oil_returns
        })
        
        # Add macroeconomic data if available
        if self.external_data is not None and not self.external_data.empty:
            # Align dates
            common_dates = var_data.index.intersection(self.external_data.index)
            
            if len(common_dates) > 0:
                var_data = var_data.loc[common_dates]
                external_aligned = self.external_data.loc[common_dates]
                
                # Add return series
                for col in external_aligned.columns:
                    if 'Return' in col:
                        var_data[col] = external_aligned[col]
        
        # Remove NaN values
        var_data = var_data.dropna()
        
        # Check stationarity
        print(f"   📊 Checking stationarity of {var_data.shape[1]} variables...")
        stationary_vars = []
        
        for col in var_data.columns:
            adf_stat, adf_pvalue, _, _, _, _ = adfuller(var_data[col], maxlag=max_lags)
            is_stationary = adf_pvalue < 0.05
            
            print(f"   {col}: {'✅ Stationary' if is_stationary else '⚠️ Non-stationary'} (p={adf_pvalue:.4f})")
            
            if is_stationary:
                stationary_vars.append(col)
        
        # Keep only stationary variables
        if len(stationary_vars) < len(var_data.columns):
            print(f"   📋 Keeping {len(stationary_vars)} stationary variables")
            var_data = var_data[stationary_vars]
        
        self.var_data = var_data
        print(f"   ✅ VAR data prepared: {var_data.shape}")
        
        return var_data
    
    def fit_var_model(self, max_lags: int = 10, ic: str = 'aic') -> Dict:
        """
        Fit VAR model with optimal lag selection.
        
        Args:
            max_lags (int): Maximum number of lags to consider
            ic (str): Information criterion for lag selection ('aic', 'bic', 'hqic')
            
        Returns:
            Dict: VAR model results
        """
        print(f"\n🔍 Fitting VAR model (max_lags={max_lags}, ic={ic})...")
        
        if self.var_data is None:
            raise ValueError("VAR data not prepared. Call prepare_var_data() first.")
        
        # Fit VAR model
        model = VAR(self.var_data)
        
        # Select optimal lag length
        lag_order_results = model.select_order(maxlags=max_lags)
        optimal_lags = getattr(lag_order_results, ic)
        
        print(f"   🎯 Optimal lags ({ic}): {optimal_lags}")
        
        # Fit model with optimal lags
        self.var_results = model.fit(optimal_lags)
        self.var_model = model
        
        # Model summary
        results_summary = {
            'optimal_lags': optimal_lags,
            'n_variables': len(self.var_data.columns),
            'n_observations': len(self.var_data),
            'variables': list(self.var_data.columns),
            'aic': self.var_results.aic,
            'bic': self.var_results.bic,
            'hqic': self.var_results.hqic,
            'log_likelihood': self.var_results.llf,
            'deterministic_terms': self.var_results.deterministic
        }
        
        print(f"   ✅ VAR({optimal_lags}) model fitted")
        print(f"   📊 AIC: {self.var_results.aic:.2f}, BIC: {self.var_results.bic:.2f}")
        
        return results_summary
    
    def granger_causality_analysis(self, target_variable: str = 'Oil_Return',
                                  max_lag: int = 5) -> Dict:
        """
        Perform Granger causality analysis.
        
        Args:
            target_variable (str): Target variable for causality testing
            max_lag (int): Maximum lag for testing
            
        Returns:
            Dict: Granger causality test results
        """
        print(f"\n🔍 Granger causality analysis for {target_variable}...")
        
        if self.var_data is None:
            raise ValueError("VAR data not prepared.")
        
        causality_results = {}
        
        for variable in self.var_data.columns:
            if variable != target_variable:
                try:
                    # Prepare data for Granger test (target, cause)
                    test_data = self.var_data[[target_variable, variable]].dropna()
                    
                    # Run Granger causality test
                    gc_result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
                    
                    # Extract p-values for each lag
                    p_values = []
                    f_stats = []
                    
                    for lag in range(1, max_lag + 1):
                        if lag in gc_result:
                            test_stat = gc_result[lag][0]['ssr_ftest']
                            p_values.append(test_stat[1])  # p-value
                            f_stats.append(test_stat[0])   # F-statistic
                    
                    # Overall conclusion (minimum p-value)
                    min_p_value = min(p_values) if p_values else 1.0
                    granger_causes = min_p_value < 0.05
                    
                    causality_results[variable] = {
                        'granger_causes': granger_causes,
                        'min_p_value': min_p_value,
                        'optimal_lag': p_values.index(min_p_value) + 1 if p_values else None,
                        'all_p_values': p_values,
                        'all_f_stats': f_stats
                    }
                    
                    status = "✅ Granger causes" if granger_causes else "❌ No causality"
                    print(f"   {variable} → {target_variable}: {status} (p={min_p_value:.4f})")
                    
                except Exception as e:
                    print(f"   ⚠️ Error testing {variable}: {str(e)}")
                    causality_results[variable] = {'error': str(e)}
        
        return causality_results
    
    def impulse_response_analysis(self, periods: int = 20) -> Dict:
        """
        Compute impulse response functions.
        
        Args:
            periods (int): Number of periods for IRF
            
        Returns:
            Dict: Impulse response analysis results
        """
        print(f"\n📊 Computing impulse response functions ({periods} periods)...")
        
        if self.var_results is None:
            raise ValueError("VAR model not fitted.")
        
        # Compute impulse response functions
        irf = self.var_results.irf(periods)
        
        # Extract IRF data
        irf_data = {}
        
        for i, response_var in enumerate(self.var_data.columns):
            irf_data[response_var] = {}
            for j, impulse_var in enumerate(self.var_data.columns):
                irf_data[response_var][impulse_var] = irf.irfs[:, i, j]
        
        results = {
            'irf_data': irf_data,
            'periods': periods,
            'variables': list(self.var_data.columns),
            'irf_object': irf
        }
        
        print(f"   ✅ IRF computed for {len(self.var_data.columns)} variables")
        
        return results
    
    def plot_var_results(self, figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
        """
        Plot VAR analysis results.
        
        Args:
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Figure object
        """
        if self.var_results is None:
            raise ValueError("VAR model not fitted.")
        
        print("\n📊 Creating VAR analysis plots...")
        
        n_vars = len(self.var_data.columns)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Residuals plot
        residuals = self.var_results.resid
        
        axes[0, 0].plot(residuals.index, residuals.iloc[:, 0], alpha=0.7)
        axes[0, 0].set_title('VAR Model Residuals (Oil Returns)')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Autocorrelation of residuals
        oil_resid = residuals.iloc[:, 0].dropna()
        if len(oil_resid) > 20:
            from statsmodels.graphics.tsaplots import plot_acf
            plot_acf(oil_resid, ax=axes[0, 1], lags=20, alpha=0.05)
            axes[0, 1].set_title('Residuals Autocorrelation')
        
        # 3. Fitted vs Actual (Oil returns)
        oil_data_aligned = self.var_data.iloc[:, 0]  # Assuming oil is first column
        fitted_values = oil_data_aligned - residuals.iloc[:, 0]
        
        axes[1, 0].scatter(fitted_values, oil_data_aligned, alpha=0.6)
        axes[1, 0].plot([oil_data_aligned.min(), oil_data_aligned.max()], 
                       [oil_data_aligned.min(), oil_data_aligned.max()], 'r--')
        axes[1, 0].set_xlabel('Fitted Values')
        axes[1, 0].set_ylabel('Actual Values')
        axes[1, 0].set_title('Fitted vs Actual (Oil Returns)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Model summary table
        axes[1, 1].axis('off')
        
        summary_text = f"""
VAR MODEL SUMMARY

Model: VAR({self.var_results.k_ar})
Variables: {n_vars}
Observations: {len(self.var_data)}

Information Criteria:
• AIC: {self.var_results.aic:.2f}
• BIC: {self.var_results.bic:.2f}
• HQIC: {self.var_results.hqic:.2f}

Log Likelihood: {self.var_results.llf:.2f}

Variables:
{chr(10).join([f"• {var}" for var in self.var_data.columns])}
        """
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        return fig


class MarkovSwitchingAnalyzer:
    """
    Markov-Switching model for oil price regime analysis.
    """
    
    def __init__(self, oil_data: pd.DataFrame, target_column: str = 'Price'):
        """
        Initialize Markov-Switching analyzer.
        
        Args:
            oil_data (pd.DataFrame): Oil price data
            target_column (str): Column to analyze
        """
        self.oil_data = oil_data.copy()
        self.target_column = target_column
        self.returns = oil_data[target_column].pct_change().dropna()
        self.ms_model = None
        self.ms_results = None
        
        print(f"📊 Initialized Markov-Switching Analyzer")
        print(f"   Data: {len(self.returns)} observations")
        print(f"   Target: {target_column} returns")
    
    def fit_markov_switching_model(self, k_regimes: int = 2, 
                                  switching_variance: bool = True) -> Dict:
        """
        Fit Markov-Switching model.
        
        Args:
            k_regimes (int): Number of regimes
            switching_variance (bool): Whether variance switches between regimes
            
        Returns:
            Dict: Model fitting results
        """
        print(f"\n🔍 Fitting Markov-Switching model ({k_regimes} regimes)...")
        
        try:
            # Prepare data
            y = self.returns.dropna().values
            
            # Fit Markov-Switching model
            if switching_variance:
                # Switching mean and variance
                model = MarkovRegression(
                    y, 
                    k_regimes=k_regimes, 
                    trend='c',  # Constant term
                    switching_variance=True
                )
            else:
                # Switching mean only
                model = MarkovRegression(
                    y,
                    k_regimes=k_regimes,
                    trend='c',
                    switching_variance=False
                )
            
            # Fit model
            self.ms_results = model.fit()
            self.ms_model = model
            
            # Extract regime characteristics
            regime_stats = {}
            
            for regime in range(k_regimes):
                mu = self.ms_results.params[f'const[{regime}]']
                
                if switching_variance:
                    sigma = np.sqrt(self.ms_results.params[f'sigma2[{regime}]'])
                else:
                    sigma = np.sqrt(self.ms_results.params['sigma2'])
                
                # Annualized statistics
                mu_annual = mu * 252 * 100  # Assuming daily data
                sigma_annual = sigma * np.sqrt(252) * 100
                
                regime_stats[f'regime_{regime}'] = {
                    'mean_daily': mu,
                    'std_daily': sigma,
                    'mean_annual': mu_annual,
                    'std_annual': sigma_annual,
                    'regime_type': 'High Volatility' if sigma > np.median([
                        np.sqrt(self.ms_results.params.get(f'sigma2[{r}]', self.ms_results.params.get('sigma2', 0)))
                        for r in range(k_regimes)
                    ]) else 'Low Volatility'
                }
            
            results_summary = {
                'k_regimes': k_regimes,
                'switching_variance': switching_variance,
                'n_observations': len(y),
                'log_likelihood': self.ms_results.llf,
                'aic': self.ms_results.aic,
                'bic': self.ms_results.bic,
                'regime_stats': regime_stats,
                'converged': self.ms_results.mle_retvals['converged']
            }
            
            print(f"   ✅ Model fitted successfully")
            print(f"   📊 Log-likelihood: {self.ms_results.llf:.2f}")
            print(f"   📊 AIC: {self.ms_results.aic:.2f}, BIC: {self.ms_results.bic:.2f}")
            
            # Print regime characteristics
            for regime, stats in regime_stats.items():
                print(f"   {regime}: μ={stats['mean_annual']:+.1f}% σ={stats['std_annual']:.1f}% ({stats['regime_type']})")
            
            return results_summary
            
        except Exception as e:
            print(f"   ❌ Error fitting model: {str(e)}")
            raise
    
    def get_regime_probabilities(self) -> pd.DataFrame:
        """
        Get smoothed regime probabilities.
        
        Returns:
            pd.DataFrame: Regime probabilities over time
        """
        if self.ms_results is None:
            raise ValueError("Model not fitted.")
        
        # Get smoothed probabilities
        smoothed_probs = self.ms_results.smoothed_marginal_probabilities
        
        # Create DataFrame with dates
        prob_df = pd.DataFrame(
            smoothed_probs,
            index=self.returns.index,
            columns=[f'Regime_{i}_Prob' for i in range(smoothed_probs.shape[1])]
        )
        
        # Add most likely regime
        prob_df['Most_Likely_Regime'] = prob_df.idxmax(axis=1).str.extract('(\d+)').astype(int)
        
        return prob_df
    
    def identify_regime_periods(self) -> List[Dict]:
        """
        Identify distinct regime periods.
        
        Returns:
            List[Dict]: List of regime periods with characteristics
        """
        if self.ms_results is None:
            raise ValueError("Model not fitted.")
        
        print("\n📊 Identifying regime periods...")
        
        regime_probs = self.get_regime_probabilities()
        most_likely = regime_probs['Most_Likely_Regime']
        
        # Find regime changes
        regime_changes = most_likely.diff() != 0
        change_points = most_likely[regime_changes].index.tolist()
        
        # Add start and end dates
        if len(change_points) == 0 or change_points[0] != most_likely.index[0]:
            change_points = [most_likely.index[0]] + change_points
        if change_points[-1] != most_likely.index[-1]:
            change_points.append(most_likely.index[-1])
        
        # Build regime periods
        periods = []
        
        for i in range(len(change_points) - 1):
            start_date = change_points[i]
            end_date = change_points[i + 1]
            
            period_data = self.oil_data.loc[start_date:end_date]
            period_returns = self.returns.loc[start_date:end_date]
            regime = most_likely.loc[start_date]
            
            period_info = {
                'regime': regime,
                'start_date': start_date,
                'end_date': end_date,
                'duration_days': (end_date - start_date).days,
                'mean_return_daily': period_returns.mean(),
                'std_return_daily': period_returns.std(),
                'mean_return_annual': period_returns.mean() * 252 * 100,
                'std_return_annual': period_returns.std() * np.sqrt(252) * 100,
                'min_price': period_data[self.target_column].min(),
                'max_price': period_data[self.target_column].max(),
                'start_price': period_data[self.target_column].iloc[0],
                'end_price': period_data[self.target_column].iloc[-1],
                'total_return': (period_data[self.target_column].iloc[-1] / 
                               period_data[self.target_column].iloc[0] - 1) * 100
            }
            
            periods.append(period_info)
        
        print(f"   ✅ Identified {len(periods)} regime periods")
        
        return periods
    
    def plot_markov_switching_results(self, figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
        """
        Plot Markov-Switching analysis results.
        
        Args:
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Figure object
        """
        if self.ms_results is None:
            raise ValueError("Model not fitted.")
        
        print("\n📊 Creating Markov-Switching plots...")
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        # Get regime probabilities
        regime_probs = self.get_regime_probabilities()
        
        # 1. Oil prices with regime coloring
        ax1 = axes[0, 0]
        
        # Color by most likely regime
        for regime in range(self.ms_results.k_regimes):
            mask = regime_probs['Most_Likely_Regime'] == regime
            regime_data = self.oil_data.loc[mask, self.target_column]
            ax1.scatter(regime_data.index, regime_data.values, 
                       alpha=0.6, s=10, label=f'Regime {regime}')
        
        ax1.set_title('Oil Prices by Regime')
        ax1.set_ylabel('Price (USD)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Regime probabilities over time
        ax2 = axes[0, 1]
        
        for i in range(self.ms_results.k_regimes):
            ax2.plot(regime_probs.index, regime_probs[f'Regime_{i}_Prob'], 
                    label=f'Regime {i}', alpha=0.8)
        
        ax2.set_title('Regime Probabilities Over Time')
        ax2.set_ylabel('Probability')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Returns by regime
        ax3 = axes[1, 0]
        
        for regime in range(self.ms_results.k_regimes):
            mask = regime_probs['Most_Likely_Regime'] == regime
            regime_returns = self.returns.loc[mask]
            ax3.hist(regime_returns, bins=50, alpha=0.6, 
                    label=f'Regime {regime}', density=True)
        
        ax3.set_title('Return Distributions by Regime')
        ax3.set_xlabel('Returns')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Regime duration analysis
        ax4 = axes[1, 1]
        
        periods = self.identify_regime_periods()
        regime_durations = {}
        
        for period in periods:
            regime = period['regime']
            if regime not in regime_durations:
                regime_durations[regime] = []
            regime_durations[regime].append(period['duration_days'])
        
        regime_labels = []
        durations_data = []
        
        for regime, durations in regime_durations.items():
            regime_labels.append(f'Regime {regime}')
            durations_data.append(durations)
        
        ax4.boxplot(durations_data, labels=regime_labels)
        ax4.set_title('Regime Duration Distributions')
        ax4.set_ylabel('Duration (days)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Model diagnostics
        ax5 = axes[2, 0]
        
        # Standardized residuals
        residuals = self.ms_results.resid
        ax5.plot(self.returns.index, residuals, alpha=0.7, linewidth=0.5)
        ax5.set_title('Standardized Residuals')
        ax5.set_ylabel('Residuals')
        ax5.grid(True, alpha=0.3)
        
        # 6. Model summary
        ax6 = axes[2, 1]
        ax6.axis('off')
        
        # Create summary statistics
        regime_stats = self.ms_results.summary().tables[1] if hasattr(self.ms_results, 'summary') else None
        
        summary_text = f"""
MARKOV-SWITCHING MODEL

Regimes: {self.ms_results.k_regimes}
Observations: {len(self.returns)}
Switching Variance: {hasattr(self.ms_results, 'switching_variance')}

Model Fit:
• Log-likelihood: {self.ms_results.llf:.2f}
• AIC: {self.ms_results.aic:.2f}
• BIC: {self.ms_results.bic:.2f}

Regime Characteristics:
"""
        
        # Add regime statistics
        for regime in range(self.ms_results.k_regimes):
            try:
                mu = self.ms_results.params[f'const[{regime}]']
                sigma_param = f'sigma2[{regime}]' if f'sigma2[{regime}]' in self.ms_results.params else 'sigma2'
                sigma = np.sqrt(self.ms_results.params[sigma_param])
                
                summary_text += f"""
Regime {regime}:
• Mean: {mu*252*100:+.1f}% annual
• Volatility: {sigma*np.sqrt(252)*100:.1f}% annual
"""
            except:
                summary_text += f"\nRegime {regime}: Parameters not accessible"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        return fig


def run_advanced_analysis(oil_data: pd.DataFrame, 
                         include_var: bool = True,
                         include_markov: bool = True) -> Dict:
    """
    Run comprehensive advanced model analysis.
    
    Args:
        oil_data (pd.DataFrame): Oil price data
        include_var (bool): Whether to run VAR analysis
        include_markov (bool): Whether to run Markov-switching analysis
        
    Returns:
        Dict: Combined results from advanced models
    """
    print("🚀 Running advanced econometric analysis...")
    
    results = {}
    
    if include_var:
        try:
            print("\n📊 VAR Analysis...")
            var_analyzer = VARAnalyzer(oil_data)
            
            # Fetch macro data
            macro_data = var_analyzer.fetch_macroeconomic_data()
            
            # Prepare and fit VAR
            var_data = var_analyzer.prepare_var_data()
            var_summary = var_analyzer.fit_var_model()
            
            # Granger causality
            causality = var_analyzer.granger_causality_analysis()
            
            # Impulse response
            irf = var_analyzer.impulse_response_analysis()
            
            results['var'] = {
                'analyzer': var_analyzer,
                'summary': var_summary,
                'causality': causality,
                'impulse_response': irf
            }
            
            print("   ✅ VAR analysis completed")
            
        except Exception as e:
            print(f"   ❌ VAR analysis failed: {str(e)}")
            results['var'] = {'error': str(e)}
    
    if include_markov:
        try:
            print("\n📊 Markov-Switching Analysis...")
            ms_analyzer = MarkovSwitchingAnalyzer(oil_data)
            
            # Fit model
            ms_summary = ms_analyzer.fit_markov_switching_model(k_regimes=2)
            
            # Get regime periods
            regime_periods = ms_analyzer.identify_regime_periods()
            
            # Get probabilities
            regime_probs = ms_analyzer.get_regime_probabilities()
            
            results['markov_switching'] = {
                'analyzer': ms_analyzer,
                'summary': ms_summary,
                'regime_periods': regime_periods,
                'regime_probabilities': regime_probs
            }
            
            print("   ✅ Markov-switching analysis completed")
            
        except Exception as e:
            print(f"   ❌ Markov-switching analysis failed: {str(e)}")
            results['markov_switching'] = {'error': str(e)}
    
    print("🎉 Advanced analysis completed!")
    return results


if __name__ == "__main__":
    print("🧪 Testing Advanced Models...")
    print("✅ Module loaded successfully!")