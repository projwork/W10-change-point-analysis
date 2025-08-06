"""
Bayesian Change Point Detection Module for Brent Oil Price Analysis

This module implements various Bayesian change point detection models using PyMC
for identifying statistically significant structural breaks in time series data.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Set PyMC sampling parameters
import pytensor.tensor as pt
import pytensor


class BayesianChangePointDetector:
    """
    Bayesian change point detection using PyMC for oil price analysis.
    """
    
    def __init__(self, data: pd.DataFrame, target_column: str = 'Price'):
        """
        Initialize the Bayesian change point detector.
        
        Args:
            data (pd.DataFrame): Time series data with datetime index
            target_column (str): Column to analyze for change points
        """
        self.data = data.copy()
        self.target_column = target_column
        self.y = data[target_column].values
        self.dates = data.index
        self.n_obs = len(self.y)
        
        # Store models and traces
        self.models = {}
        self.traces = {}
        self.results = {}
        
        print(f"📊 Initialized Bayesian detector with {self.n_obs} observations")
        print(f"📅 Date range: {self.dates[0].strftime('%Y-%m-%d')} to {self.dates[-1].strftime('%Y-%m-%d')}")
    
    def prepare_data(self, use_log_returns: bool = False) -> np.ndarray:
        """
        Prepare data for modeling (optional log transformation).
        
        Args:
            use_log_returns (bool): Whether to use log returns instead of prices
            
        Returns:
            np.ndarray: Prepared data for modeling
        """
        if use_log_returns:
            log_prices = np.log(self.y)
            log_returns = np.diff(log_prices)
            print(f"📈 Using log returns: {len(log_returns)} observations")
            return log_returns
        else:
            print(f"💰 Using price levels: {len(self.y)} observations")
            return self.y
    
    def single_changepoint_model(self, use_log_returns: bool = False, 
                                draws: int = 2000, tune: int = 1000,
                                target_accept: float = 0.95) -> az.InferenceData:
        """
        Single change point model for mean shift detection.
        
        Args:
            use_log_returns (bool): Whether to use log returns
            draws (int): Number of MCMC draws
            tune (int): Number of tuning steps
            target_accept (float): Target acceptance rate
            
        Returns:
            az.InferenceData: MCMC trace data
        """
        print("\n🔍 Building Single Change Point Model...")
        
        # Prepare data
        y_data = self.prepare_data(use_log_returns)
        n_obs = len(y_data)
        
        with pm.Model() as model:
            # Prior for change point location (discrete uniform)
            tau = pm.DiscreteUniform('tau', lower=1, upper=n_obs-2)
            
            # Priors for means before and after change point
            mu_1 = pm.Normal('mu_1', mu=np.mean(y_data), sigma=np.std(y_data))
            mu_2 = pm.Normal('mu_2', mu=np.mean(y_data), sigma=np.std(y_data))
            
            # Prior for precision (inverse variance)
            tau_prec = pm.Gamma('tau_prec', alpha=1, beta=1)
            
            # Switch function to select appropriate mean
            mu = pm.math.switch(tau >= np.arange(n_obs), mu_1, mu_2)
            
            # Likelihood
            likelihood = pm.Normal('likelihood', mu=mu, tau=tau_prec, observed=y_data)
            
            # Sample from posterior
            print(f"⚡ Sampling {draws} draws with {tune} tuning steps...")
            trace = pm.sample(draws=draws, tune=tune, 
                            target_accept=target_accept, 
                            return_inferencedata=True,
                            progressbar=True)
        
        # Store results
        model_name = f"single_cp_{'returns' if use_log_returns else 'prices'}"
        self.models[model_name] = model
        self.traces[model_name] = trace
        
        print("✅ Single change point model completed!")
        return trace
    
    def multiple_changepoints_model(self, max_changepoints: int = 3, 
                                   use_log_returns: bool = False,
                                   draws: int = 2000, tune: int = 1000) -> az.InferenceData:
        """
        Multiple change points model with variable number of change points.
        
        Args:
            max_changepoints (int): Maximum number of change points
            use_log_returns (bool): Whether to use log returns
            draws (int): Number of MCMC draws
            tune (int): Number of tuning steps
            
        Returns:
            az.InferenceData: MCMC trace data
        """
        print(f"\n🔍 Building Multiple Change Points Model (max={max_changepoints})...")
        
        # Prepare data
        y_data = self.prepare_data(use_log_returns)
        n_obs = len(y_data)
        
        with pm.Model() as model:
            # Number of change points (0 to max_changepoints)
            n_cp = pm.DiscreteUniform('n_cp', lower=0, upper=max_changepoints)
            
            # Change point locations
            tau = pm.DiscreteUniform('tau', lower=1, upper=n_obs-2, 
                                   shape=max_changepoints)
            
            # Sort change points
            tau_sorted = pm.math.sort(tau)
            
            # Means for each segment
            mu = pm.Normal('mu', mu=np.mean(y_data), sigma=np.std(y_data), 
                          shape=max_changepoints + 1)
            
            # Precision
            tau_prec = pm.Gamma('tau_prec', alpha=1, beta=1)
            
            # Build segment means
            segment_means = pm.math.zeros(n_obs)
            for i in range(max_changepoints + 1):
                if i == 0:
                    # First segment
                    mask = pm.math.lt(np.arange(n_obs), tau_sorted[0])
                elif i == max_changepoints:
                    # Last segment
                    mask = pm.math.ge(np.arange(n_obs), tau_sorted[i-1])
                else:
                    # Middle segments
                    mask = pm.math.and_(pm.math.ge(np.arange(n_obs), tau_sorted[i-1]),
                                       pm.math.lt(np.arange(n_obs), tau_sorted[i]))
                
                segment_means = segment_means + mu[i] * mask
            
            # Likelihood
            likelihood = pm.Normal('likelihood', mu=segment_means, 
                                 tau=tau_prec, observed=y_data)
            
            # Sample
            print(f"⚡ Sampling {draws} draws with {tune} tuning steps...")
            trace = pm.sample(draws=draws, tune=tune, 
                            target_accept=0.9,
                            return_inferencedata=True,
                            progressbar=True)
        
        # Store results
        model_name = f"multiple_cp_{'returns' if use_log_returns else 'prices'}"
        self.models[model_name] = model
        self.traces[model_name] = trace
        
        print("✅ Multiple change points model completed!")
        return trace
    
    def variance_changepoint_model(self, use_log_returns: bool = True,
                                  draws: int = 2000, tune: int = 1000) -> az.InferenceData:
        """
        Change point model for variance/volatility regime changes.
        
        Args:
            use_log_returns (bool): Whether to use log returns (recommended for variance)
            draws (int): Number of MCMC draws
            tune (int): Number of tuning steps
            
        Returns:
            az.InferenceData: MCMC trace data
        """
        print("\n🔍 Building Variance Change Point Model...")
        
        # Prepare data (force log returns for variance modeling)
        y_data = self.prepare_data(use_log_returns=True)
        n_obs = len(y_data)
        
        with pm.Model() as model:
            # Change point location
            tau = pm.DiscreteUniform('tau', lower=10, upper=n_obs-10)
            
            # Mean (assume constant)
            mu = pm.Normal('mu', mu=0, sigma=0.1)
            
            # Variances before and after change point
            sigma_1 = pm.HalfNormal('sigma_1', sigma=0.1)
            sigma_2 = pm.HalfNormal('sigma_2', sigma=0.1)
            
            # Switch function for variance
            sigma = pm.math.switch(tau >= np.arange(n_obs), sigma_1, sigma_2)
            
            # Likelihood
            likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=y_data)
            
            # Sample
            print(f"⚡ Sampling {draws} draws with {tune} tuning steps...")
            trace = pm.sample(draws=draws, tune=tune,
                            target_accept=0.95,
                            return_inferencedata=True,
                            progressbar=True)
        
        # Store results
        model_name = "variance_cp"
        self.models[model_name] = model
        self.traces[model_name] = trace
        
        print("✅ Variance change point model completed!")
        return trace
    
    def check_convergence(self, trace: az.InferenceData, model_name: str = None) -> Dict:
        """
        Check MCMC convergence diagnostics.
        
        Args:
            trace (az.InferenceData): MCMC trace
            model_name (str): Name of the model
            
        Returns:
            Dict: Convergence diagnostics
        """
        print(f"\n🔍 Checking convergence for {model_name or 'model'}...")
        
        # Get summary statistics
        summary = az.summary(trace)
        
        # Check R-hat values
        rhat_values = summary['r_hat'].dropna()
        good_rhat = (rhat_values < 1.1).all()
        
        # Check effective sample size
        ess_bulk = summary['ess_bulk'].dropna()
        ess_tail = summary['ess_tail'].dropna()
        good_ess = (ess_bulk > 400).all() and (ess_tail > 400).all()
        
        # Monte Carlo standard error
        mcse_mean = summary['mcse_mean'].dropna()
        mcse_sd = summary['mcse_sd'].dropna()
        
        diagnostics = {
            'converged': good_rhat and good_ess,
            'rhat_max': rhat_values.max(),
            'rhat_good': good_rhat,
            'ess_bulk_min': ess_bulk.min(),
            'ess_tail_min': ess_tail.min(),
            'ess_good': good_ess,
            'mcse_mean_max': mcse_mean.max(),
            'mcse_sd_max': mcse_sd.max(),
            'summary': summary
        }
        
        # Print results
        if diagnostics['converged']:
            print("✅ Model converged successfully!")
        else:
            print("⚠️ Convergence issues detected!")
            
        print(f"   R-hat max: {diagnostics['rhat_max']:.4f} (should be < 1.1)")
        print(f"   ESS bulk min: {diagnostics['ess_bulk_min']:.0f} (should be > 400)")
        print(f"   ESS tail min: {diagnostics['ess_tail_min']:.0f} (should be > 400)")
        
        return diagnostics
    
    def extract_changepoints(self, trace: az.InferenceData, model_name: str,
                           credible_interval: float = 0.95) -> Dict:
        """
        Extract change point estimates from MCMC trace.
        
        Args:
            trace (az.InferenceData): MCMC trace
            model_name (str): Name of the model
            credible_interval (float): Credible interval level
            
        Returns:
            Dict: Change point analysis results
        """
        print(f"\n📊 Extracting change points from {model_name}...")
        
        # Get posterior samples
        posterior = trace.posterior
        
        results = {}
        
        if 'tau' in posterior:
            # Single or multiple change points
            tau_samples = posterior['tau'].values.flatten()
            
            # Calculate statistics
            tau_mean = np.mean(tau_samples)
            tau_median = np.median(tau_samples)
            tau_mode = float(pd.Series(tau_samples).mode().iloc[0])
            
            # Credible interval
            alpha = 1 - credible_interval
            tau_ci = np.percentile(tau_samples, [100*alpha/2, 100*(1-alpha/2)])
            
            # Convert to dates
            tau_date_mean = self.dates[int(tau_mean)]
            tau_date_median = self.dates[int(tau_median)]
            tau_date_mode = self.dates[int(tau_mode)]
            
            results['single_changepoint'] = {
                'tau_samples': tau_samples,
                'tau_mean': tau_mean,
                'tau_median': tau_median,
                'tau_mode': tau_mode,
                'tau_credible_interval': tau_ci,
                'date_mean': tau_date_mean,
                'date_median': tau_date_median,
                'date_mode': tau_date_mode,
                'uncertainty': tau_ci[1] - tau_ci[0]
            }
            
            print(f"   📅 Change point (mode): {tau_date_mode.strftime('%Y-%m-%d')}")
            print(f"   📅 Change point (median): {tau_date_median.strftime('%Y-%m-%d')}")
            print(f"   🎯 Uncertainty: ±{(tau_ci[1] - tau_ci[0])/2:.1f} days")
        
        # Extract parameter estimates
        if 'mu_1' in posterior and 'mu_2' in posterior:
            mu_1_samples = posterior['mu_1'].values.flatten()
            mu_2_samples = posterior['mu_2'].values.flatten()
            
            results['parameters'] = {
                'mu_1_mean': np.mean(mu_1_samples),
                'mu_1_ci': np.percentile(mu_1_samples, [2.5, 97.5]),
                'mu_2_mean': np.mean(mu_2_samples),
                'mu_2_ci': np.percentile(mu_2_samples, [2.5, 97.5]),
                'difference_mean': np.mean(mu_2_samples - mu_1_samples),
                'difference_ci': np.percentile(mu_2_samples - mu_1_samples, [2.5, 97.5])
            }
            
            # Calculate impact
            impact_pct = (results['parameters']['mu_2_mean'] / 
                         results['parameters']['mu_1_mean'] - 1) * 100
            
            print(f"   📈 Before change: ${results['parameters']['mu_1_mean']:.2f}")
            print(f"   📈 After change: ${results['parameters']['mu_2_mean']:.2f}")
            print(f"   📊 Impact: {impact_pct:+.1f}%")
        
        # Store results
        self.results[model_name] = results
        return results
    
    def plot_changepoint_results(self, trace: az.InferenceData, model_name: str,
                               figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot change point detection results.
        
        Args:
            trace (az.InferenceData): MCMC trace
            model_name (str): Name of the model
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Figure object
        """
        print(f"\n📊 Creating plots for {model_name}...")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Original time series with change point
        axes[0, 0].plot(self.dates, self.y, 'b-', alpha=0.7, linewidth=1)
        
        if model_name in self.results and 'single_changepoint' in self.results[model_name]:
            cp_result = self.results[model_name]['single_changepoint']
            cp_date = cp_result['date_mode']
            axes[0, 0].axvline(x=cp_date, color='red', linestyle='--', 
                             linewidth=2, label=f"Change Point: {cp_date.strftime('%Y-%m-%d')}")
            axes[0, 0].legend()
        
        axes[0, 0].set_title(f'Oil Prices with Detected Change Point ({model_name})')
        axes[0, 0].set_ylabel('Price (USD)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Posterior distribution of change point
        if 'tau' in trace.posterior:
            tau_samples = trace.posterior['tau'].values.flatten()
            axes[0, 1].hist(tau_samples, bins=50, alpha=0.7, density=True, color='blue')
            axes[0, 1].axvline(x=np.median(tau_samples), color='red', linestyle='--',
                             label=f"Median: {np.median(tau_samples):.0f}")
            axes[0, 1].set_title('Posterior Distribution of Change Point Location')
            axes[0, 1].set_xlabel('Time Index')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Parameter posterior distributions
        if 'mu_1' in trace.posterior and 'mu_2' in trace.posterior:
            mu_1_samples = trace.posterior['mu_1'].values.flatten()
            mu_2_samples = trace.posterior['mu_2'].values.flatten()
            
            axes[1, 0].hist(mu_1_samples, bins=50, alpha=0.7, label='Before (μ₁)', color='blue')
            axes[1, 0].hist(mu_2_samples, bins=50, alpha=0.7, label='After (μ₂)', color='red')
            axes[1, 0].set_title('Posterior Distributions of Mean Parameters')
            axes[1, 0].set_xlabel('Value')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Difference distribution
            diff_samples = mu_2_samples - mu_1_samples
            axes[1, 1].hist(diff_samples, bins=50, alpha=0.7, color='green')
            axes[1, 1].axvline(x=0, color='black', linestyle='-', alpha=0.5)
            axes[1, 1].axvline(x=np.median(diff_samples), color='red', linestyle='--',
                             label=f"Median: {np.median(diff_samples):.2f}")
            axes[1, 1].set_title('Posterior Distribution of Mean Difference (μ₂ - μ₁)')
            axes[1, 1].set_xlabel('Difference')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_trace_diagnostics(self, trace: az.InferenceData, model_name: str) -> plt.Figure:
        """
        Plot MCMC trace diagnostics.
        
        Args:
            trace (az.InferenceData): MCMC trace
            model_name (str): Name of the model
            
        Returns:
            plt.Figure: Figure object
        """
        print(f"\n📊 Creating trace diagnostics for {model_name}...")
        
        # Use ArviZ for trace plots
        fig = az.plot_trace(trace, figsize=(12, 8))
        fig.suptitle(f'MCMC Trace Diagnostics - {model_name}', fontsize=16)
        plt.tight_layout()
        
        return fig
    
    def compare_models(self, traces: Dict[str, az.InferenceData]) -> Dict:
        """
        Compare multiple models using information criteria.
        
        Args:
            traces (Dict[str, az.InferenceData]): Dictionary of model traces
            
        Returns:
            Dict: Model comparison results
        """
        print("\n📊 Comparing models using information criteria...")
        
        comparison_data = {}
        
        for name, trace in traces.items():
            try:
                # Calculate WAIC and LOO
                waic = az.waic(trace)
                loo = az.loo(trace)
                
                comparison_data[name] = {
                    'waic': waic.waic,
                    'waic_se': waic.se,
                    'loo': loo.loo,
                    'loo_se': loo.se,
                    'p_waic': waic.p_waic,
                    'p_loo': loo.p_loo
                }
                
                print(f"   {name}:")
                print(f"      WAIC: {waic.waic:.2f} (±{waic.se:.2f})")
                print(f"      LOO: {loo.loo:.2f} (±{loo.se:.2f})")
                
            except Exception as e:
                print(f"   ⚠️ Could not calculate IC for {name}: {str(e)}")
                comparison_data[name] = None
        
        return comparison_data
    
    def generate_report(self, model_name: str) -> Dict:
        """
        Generate comprehensive analysis report for a model.
        
        Args:
            model_name (str): Name of the model to report on
            
        Returns:
            Dict: Comprehensive report
        """
        if model_name not in self.traces:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.traces.keys())}")
        
        trace = self.traces[model_name]
        
        print(f"\n📋 Generating comprehensive report for {model_name}...")
        
        # Check convergence
        convergence = self.check_convergence(trace, model_name)
        
        # Extract change points
        if model_name not in self.results:
            self.extract_changepoints(trace, model_name)
        
        results = self.results[model_name]
        
        # Generate report
        report = {
            'model_name': model_name,
            'data_info': {
                'n_observations': self.n_obs,
                'date_range': {
                    'start': self.dates[0].strftime('%Y-%m-%d'),
                    'end': self.dates[-1].strftime('%Y-%m-%d')
                },
                'price_range': {
                    'min': float(np.min(self.y)),
                    'max': float(np.max(self.y)),
                    'mean': float(np.mean(self.y))
                }
            },
            'convergence': convergence,
            'change_points': results,
            'recommendations': self._generate_recommendations(convergence, results)
        }
        
        return report
    
    def _generate_recommendations(self, convergence: Dict, results: Dict) -> List[str]:
        """Generate analysis recommendations based on results."""
        recommendations = []
        
        if not convergence['converged']:
            recommendations.append("⚠️ Model did not converge properly. Consider increasing sampling steps or adjusting priors.")
        
        if 'single_changepoint' in results:
            cp = results['single_changepoint']
            if cp['uncertainty'] > 100:  # More than 100 days uncertainty
                recommendations.append("📊 High uncertainty in change point location. Consider using more informative priors.")
            
            if 'parameters' in results:
                diff = results['parameters']['difference_mean']
                if abs(diff) < 0.01 * np.mean(self.y):  # Less than 1% change
                    recommendations.append("📈 Small magnitude change detected. Verify economic significance.")
        
        recommendations.append("✅ Use change point results to correlate with major market events.")
        recommendations.append("📊 Consider running variance change point model for volatility analysis.")
        
        return recommendations


def run_bayesian_analysis(data: pd.DataFrame, target_column: str = 'Price',
                         models: List[str] = None) -> BayesianChangePointDetector:
    """
    Convenience function to run comprehensive Bayesian change point analysis.
    
    Args:
        data (pd.DataFrame): Time series data
        target_column (str): Column to analyze
        models (List[str]): Models to run
        
    Returns:
        BayesianChangePointDetector: Fitted detector with results
    """
    if models is None:
        models = ['single_cp_prices', 'single_cp_returns', 'variance_cp']
    
    detector = BayesianChangePointDetector(data, target_column)
    
    print("🚀 Running comprehensive Bayesian change point analysis...")
    
    for model in models:
        try:
            if model == 'single_cp_prices':
                trace = detector.single_changepoint_model(use_log_returns=False)
                detector.extract_changepoints(trace, model)
            elif model == 'single_cp_returns':
                trace = detector.single_changepoint_model(use_log_returns=True)
                detector.extract_changepoints(trace, model)
            elif model == 'variance_cp':
                trace = detector.variance_changepoint_model()
                detector.extract_changepoints(trace, model)
            elif model == 'multiple_cp':
                trace = detector.multiple_changepoints_model()
                detector.extract_changepoints(trace, model)
                
        except Exception as e:
            print(f"❌ Error running {model}: {str(e)}")
    
    print("🎉 Bayesian analysis completed!")
    return detector


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing Bayesian Change Point Detection...")
    
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    
    # Simulate data with a change point at day 100
    prices_1 = 50 + np.random.normal(0, 2, 100)  # Mean 50, std 2
    prices_2 = 60 + np.random.normal(0, 3, 100)  # Mean 60, std 3
    prices = np.concatenate([prices_1, prices_2])
    
    test_data = pd.DataFrame({
        'Price': prices
    }, index=dates)
    
    # Run analysis
    detector = BayesianChangePointDetector(test_data)
    trace = detector.single_changepoint_model(draws=1000, tune=500)
    
    # Check results
    convergence = detector.check_convergence(trace, 'test_model')
    results = detector.extract_changepoints(trace, 'test_model')
    
    print("✅ Test completed successfully!")