"""
Change Point Detection Models for Brent Oil Price Analysis

This module implements various change point detection algorithms for identifying
structural breaks in oil price time series data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import ruptures as rpt
from scipy import stats
# from sklearn.preprocessing import StandardScaler  # Not used in current implementation
import warnings
warnings.filterwarnings('ignore')


class ChangePointDetector:
    """
    Comprehensive change point detection for oil price time series.
    """
    
    def __init__(self, data: pd.DataFrame, target_column: str = 'Price'):
        """
        Initialize the change point detector.
        
        Args:
            data (pd.DataFrame): Time series data with datetime index
            target_column (str): Column to analyze for change points
        """
        self.data = data.copy()
        self.target_column = target_column
        self.signal = data[target_column].values
        self.dates = data.index
        self.change_points = {}
        self.models = {}
        
    def detect_mean_shifts(self, method: str = 'pelt', penalty: float = None, 
                          min_size: int = 30) -> List[int]:
        """
        Detect change points in mean using ruptures library.
        
        Args:
            method (str): Detection method ('pelt', 'binseg', 'window')
            penalty (float): Penalty parameter for model selection
            min_size (int): Minimum segment size
            
        Returns:
            List[int]: Indices of detected change points
        """
        # Prepare signal (remove NaN values and convert to numpy array)
        signal = pd.Series(self.signal).dropna().values
        
        if penalty is None:
            penalty = np.log(len(signal)) * np.var(signal)
        
        # Choose detection algorithm with optimized models for large datasets
        if method == 'pelt':
            # Use normal model which is faster than RBF for large datasets
            model = rpt.Pelt(model="normal", min_size=min_size).fit(signal)
            change_points = model.predict(pen=penalty)
        elif method == 'binseg':
            # Use normal model and limit number of change points for efficiency
            model = rpt.Binseg(model="normal", min_size=min_size).fit(signal)
            change_points = model.predict(n_bkps=10)  # Limit to 10 change points
        elif method == 'window':
            model = rpt.Window(width=100, model="normal", min_size=min_size).fit(signal)
            change_points = model.predict(pen=penalty)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Remove the last point (end of series)
        if change_points and change_points[-1] == len(signal):
            change_points = change_points[:-1]
        
        self.change_points[f'mean_shift_{method}'] = change_points
        self.models[f'mean_shift_{method}'] = model
        
        return change_points
    
    def detect_variance_changes(self, method: str = 'pelt', penalty: float = None,
                               min_size: int = 30) -> List[int]:
        """
        Detect change points in variance.
        
        Args:
            method (str): Detection method
            penalty (float): Penalty parameter
            min_size (int): Minimum segment size
            
        Returns:
            List[int]: Indices of detected change points
        """
        signal = pd.Series(self.signal).dropna().values
        
        if penalty is None:
            penalty = np.log(len(signal)) * 2
        
        # Use squared deviations from rolling mean for variance detection
        rolling_mean = pd.Series(signal).rolling(window=min_size, center=True).mean()
        var_signal = (signal - rolling_mean.fillna(method='bfill').fillna(method='ffill')) ** 2
        
        # Convert to numpy array for ruptures library
        var_signal = var_signal.values
        
        if method == 'pelt':
            model = rpt.Pelt(model="normal", min_size=min_size).fit(var_signal)
            change_points = model.predict(pen=penalty)
        elif method == 'binseg':
            model = rpt.Binseg(model="normal", min_size=min_size).fit(var_signal)
            change_points = model.predict(n_bkps=10)  # Limit to 10 change points
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if change_points and change_points[-1] == len(var_signal):
            change_points = change_points[:-1]
        
        self.change_points[f'variance_{method}'] = change_points
        self.models[f'variance_{method}'] = model
        
        return change_points
    
    def detect_trend_changes(self, window_size: int = 100, 
                           significance_level: float = 0.05) -> List[int]:
        """
        Detect trend change points using rolling regression slopes.
        
        Args:
            window_size (int): Size of rolling window
            significance_level (float): Significance level for trend change
            
        Returns:
            List[int]: Indices of detected change points
        """
        signal = pd.Series(self.signal).dropna()
        slopes = []
        
        # Calculate rolling slopes
        for i in range(window_size, len(signal) - window_size):
            start_idx = i - window_size // 2
            end_idx = i + window_size // 2
            
            x = np.arange(end_idx - start_idx)
            y = signal.iloc[start_idx:end_idx].values
            
            slope, _, _, p_value, _ = stats.linregress(x, y)
            slopes.append((i, slope, p_value))
        
        # Detect significant slope changes
        change_points = []
        prev_slope = None
        
        for i, slope, p_value in slopes:
            if prev_slope is not None and p_value < significance_level:
                # Check if slope direction changed significantly
                if (prev_slope > 0 and slope < 0) or (prev_slope < 0 and slope > 0):
                    change_points.append(i)
            prev_slope = slope
        
        self.change_points['trend_changes'] = change_points
        return change_points
    
    def detect_cusum_changes(self, threshold: float = None) -> List[int]:
        """
        Detect change points using CUSUM (Cumulative Sum) algorithm.
        
        Args:
            threshold (float): Detection threshold
            
        Returns:
            List[int]: Indices of detected change points
        """
        signal = pd.Series(self.signal).dropna().values
        
        if threshold is None:
            threshold = 3 * np.std(signal)
        
        # Calculate CUSUM
        mean_estimate = np.mean(signal)
        cusum_pos = np.zeros(len(signal))
        cusum_neg = np.zeros(len(signal))
        change_points = []
        
        for i in range(1, len(signal)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + signal[i] - mean_estimate)
            cusum_neg[i] = min(0, cusum_neg[i-1] + signal[i] - mean_estimate)
            
            if cusum_pos[i] > threshold or cusum_neg[i] < -threshold:
                change_points.append(i)
                # Reset CUSUM after detection
                cusum_pos[i] = 0
                cusum_neg[i] = 0
                mean_estimate = np.mean(signal[max(0, i-50):i+1])  # Update mean estimate
        
        self.change_points['cusum'] = change_points
        return change_points
    
    def comprehensive_detection(self, methods: List[str] = None) -> Dict[str, List[int]]:
        """
        Run multiple change point detection methods.
        
        Args:
            methods (List[str]): Methods to run (default: all available)
            
        Returns:
            Dict[str, List[int]]: Change points detected by each method
        """
        if methods is None:
            methods = ['pelt_mean', 'binseg_mean', 'pelt_variance', 'trend', 'cusum']
        
        results = {}
        
        for method in methods:
            if method == 'pelt_mean':
                results['pelt_mean'] = self.detect_mean_shifts('pelt')
            elif method == 'binseg_mean':
                results['binseg_mean'] = self.detect_mean_shifts('binseg')
            elif method == 'pelt_variance':
                results['pelt_variance'] = self.detect_variance_changes('pelt')
            elif method == 'binseg_variance':
                results['binseg_variance'] = self.detect_variance_changes('binseg')
            elif method == 'trend':
                results['trend'] = self.detect_trend_changes()
            elif method == 'cusum':
                results['cusum'] = self.detect_cusum_changes()
        
        return results
    
    def get_consensus_change_points(self, min_methods: int = 2, 
                                   tolerance: int = 30) -> List[int]:
        """
        Get consensus change points that are detected by multiple methods.
        
        Args:
            min_methods (int): Minimum number of methods that must agree
            tolerance (int): Tolerance in days for considering points as same
            
        Returns:
            List[int]: Consensus change points
        """
        all_points = []
        for method, points in self.change_points.items():
            all_points.extend(points)
        
        if not all_points:
            return []
        
        # Group nearby points
        all_points = sorted(set(all_points))
        consensus_points = []
        
        i = 0
        while i < len(all_points):
            current_group = [all_points[i]]
            j = i + 1
            
            # Find all points within tolerance
            while j < len(all_points) and all_points[j] - all_points[i] <= tolerance:
                current_group.append(all_points[j])
                j += 1
            
            # If enough methods detected points in this group
            if len(current_group) >= min_methods:
                # Use median as consensus point
                consensus_points.append(int(np.median(current_group)))
            
            i = j
        
        return consensus_points
    
    def analyze_segments(self, change_points: List[int] = None) -> List[Dict]:
        """
        Analyze segments between change points.
        
        Args:
            change_points (List[int]): Change points to use for segmentation
            
        Returns:
            List[Dict]: Analysis of each segment
        """
        if change_points is None:
            change_points = self.get_consensus_change_points()
        
        signal = pd.Series(self.signal).dropna()
        segments = []
        
        # Add start and end points
        boundaries = [0] + change_points + [len(signal)]
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            segment_data = signal.iloc[start_idx:end_idx]
            segment_dates = self.dates[start_idx:end_idx]
            
            # Calculate segment statistics
            segment_stats = {
                'start_date': segment_dates[0],
                'end_date': segment_dates[-1],
                'duration_days': (segment_dates[-1] - segment_dates[0]).days,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'mean': segment_data.mean(),
                'std': segment_data.std(),
                'min': segment_data.min(),
                'max': segment_data.max(),
                'start_value': segment_data.iloc[0],
                'end_value': segment_data.iloc[-1],
                'total_change': segment_data.iloc[-1] - segment_data.iloc[0],
                'pct_change': ((segment_data.iloc[-1] - segment_data.iloc[0]) / segment_data.iloc[0]) * 100
            }
            
            # Calculate trend
            x = np.arange(len(segment_data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, segment_data.values)
            
            segment_stats.update({
                'trend_slope': slope,
                'trend_r_squared': r_value**2,
                'trend_p_value': p_value,
                'trend_direction': 'Increasing' if slope > 0 else 'Decreasing' if slope < 0 else 'Flat'
            })
            
            segments.append(segment_stats)
        
        return segments
    
    def plot_change_points(self, methods: List[str] = None, 
                          figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot detected change points on the time series.
        
        Args:
            methods (List[str]): Methods to plot (default: all)
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Figure object
        """
        if methods is None:
            methods = list(self.change_points.keys())
        
        fig, axes = plt.subplots(len(methods) + 1, 1, figsize=figsize, sharex=True)
        if len(methods) == 0:
            axes = [axes]
        
        # Plot original signal
        signal = pd.Series(self.signal, index=self.dates).dropna()
        axes[0].plot(signal.index, signal.values, 'b-', alpha=0.7, label='Oil Price')
        axes[0].set_title('Brent Oil Price with Change Points')
        axes[0].set_ylabel('Price (USD)')
        axes[0].grid(True)
        axes[0].legend()
        
        # Plot each method's change points
        colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
        
        for i, (method, color) in enumerate(zip(methods, colors)):
            if method in self.change_points:
                change_points = self.change_points[method]
                
                # Plot signal
                axes[i+1].plot(signal.index, signal.values, 'b-', alpha=0.5)
                
                # Plot change points
                for cp in change_points:
                    if cp < len(signal):
                        axes[i+1].axvline(x=signal.index[cp], color=color, 
                                        linestyle='--', alpha=0.8, linewidth=2)
                
                axes[i+1].set_title(f'Change Points - {method.replace("_", " ").title()}')
                axes[i+1].set_ylabel('Price (USD)')
                axes[i+1].grid(True)
        
        plt.xlabel('Date')
        plt.tight_layout()
        return fig
    
    def get_results_summary(self) -> Dict:
        """
        Get summary of all change point detection results.
        
        Returns:
            Dict: Summary of results including change points and segments
        """
        consensus_points = self.get_consensus_change_points()
        segments = self.analyze_segments(consensus_points)
        
        summary = {
            'methods_used': list(self.change_points.keys()),
            'change_points_by_method': {
                method: {
                    'count': len(points),
                    'dates': [self.dates[p].strftime('%Y-%m-%d') for p in points if p < len(self.dates)]
                }
                for method, points in self.change_points.items()
            },
            'consensus_change_points': {
                'count': len(consensus_points),
                'dates': [self.dates[p].strftime('%Y-%m-%d') for p in consensus_points if p < len(self.dates)],
                'indices': consensus_points
            },
            'segments_analysis': {
                'total_segments': len(segments),
                'average_duration_days': np.mean([s['duration_days'] for s in segments]),
                'segments': segments
            }
        }
        
        return summary


def detect_oil_price_changes(data: pd.DataFrame, target_column: str = 'Price',
                           methods: List[str] = None) -> Dict:
    """
    Convenience function for comprehensive change point detection.
    
    Args:
        data (pd.DataFrame): Time series data
        target_column (str): Column to analyze
        methods (List[str]): Detection methods to use
        
    Returns:
        Dict: Detection results
    """
    detector = ChangePointDetector(data, target_column)
    detector.comprehensive_detection(methods)
    return detector.get_results_summary()


if __name__ == "__main__":
    # Example usage
    from data_loader import load_brent_oil_data
    
    # Load data
    data = load_brent_oil_data()
    
    # Detect change points
    detector = ChangePointDetector(data)
    results = detector.comprehensive_detection()
    
    print("Change Point Detection Results:")
    for method, points in results.items():
        print(f"{method}: {len(points)} change points detected")
    
    # Get consensus points
    consensus = detector.get_consensus_change_points()
    print(f"\nConsensus change points: {len(consensus)}")
    
    # Analyze segments
    segments = detector.analyze_segments(consensus)
    print(f"Number of segments: {len(segments)}")