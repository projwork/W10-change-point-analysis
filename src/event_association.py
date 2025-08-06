"""
Event Association and Quantitative Impact Analysis Module

This module associates detected Bayesian change points with major market events
and quantifies their impacts on oil prices.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class EventChangePointAssociator:
    """
    Associate Bayesian change points with major market events and quantify impacts.
    """
    
    def __init__(self, oil_data: pd.DataFrame, events_data: pd.DataFrame, 
                 bayesian_results: Dict):
        """
        Initialize the event associator.
        
        Args:
            oil_data (pd.DataFrame): Oil price time series data
            events_data (pd.DataFrame): Major market events data
            bayesian_results (Dict): Results from Bayesian change point analysis
        """
        self.oil_data = oil_data.copy()
        self.events_data = events_data.copy()
        self.bayesian_results = bayesian_results
        self.associations = {}
        self.impact_analysis = {}
        
        print(f"📊 Initialized Event Associator")
        print(f"   Oil data: {len(oil_data)} observations")
        print(f"   Events: {len(events_data)} major events")
        print(f"   Bayesian models: {list(bayesian_results.keys())}")
    
    def associate_events_with_changepoints(self, model_name: str, 
                                         tolerance_days: int = 90,
                                         confidence_threshold: float = 0.8) -> Dict:
        """
        Associate detected change points with major market events.
        
        Args:
            model_name (str): Name of the Bayesian model to analyze
            tolerance_days (int): Days tolerance for event association
            confidence_threshold (float): Minimum confidence for association
            
        Returns:
            Dict: Event-change point associations
        """
        print(f"\n🔍 Associating events with change points for {model_name}...")
        print(f"   Tolerance: ±{tolerance_days} days")
        
        if model_name not in self.bayesian_results:
            raise ValueError(f"Model {model_name} not found in Bayesian results")
        
        # Get change point results
        cp_results = self.bayesian_results[model_name]
        
        if 'single_changepoint' not in cp_results:
            print("⚠️ No single change point found in results")
            return {}
        
        cp_info = cp_results['single_changepoint']
        cp_date = cp_info['date_mode']  # Use mode as best estimate
        cp_uncertainty = cp_info['uncertainty']
        
        print(f"   📅 Change point: {cp_date.strftime('%Y-%m-%d')}")
        print(f"   🎯 Uncertainty: ±{cp_uncertainty/2:.1f} days")
        
        # Find events within tolerance window
        tolerance_window = timedelta(days=tolerance_days)
        start_window = cp_date - tolerance_window
        end_window = cp_date + tolerance_window
        
        nearby_events = self.events_data[
            (self.events_data.index >= start_window) & 
            (self.events_data.index <= end_window)
        ].copy()
        
        print(f"   📋 Found {len(nearby_events)} events within tolerance window")
        
        # Calculate association scores
        associations = []
        
        for event_date, event_row in nearby_events.iterrows():
            days_difference = abs((event_date - cp_date).days)
            
            # Calculate association confidence based on proximity and uncertainty
            proximity_score = max(0, 1 - (days_difference / tolerance_days))
            uncertainty_penalty = min(1, cp_uncertainty / (2 * tolerance_days))
            confidence = proximity_score * (1 - uncertainty_penalty * 0.5)
            
            # Expected impact direction
            expected_impact = event_row.get('expected_impact', 'Unknown')
            magnitude = event_row.get('magnitude', 'Unknown')
            
            association = {
                'event_date': event_date,
                'event_name': event_row['event'],
                'event_type': event_row['type'],
                'description': event_row['description'],
                'expected_impact': expected_impact,
                'magnitude': magnitude,
                'days_difference': days_difference,
                'proximity_score': proximity_score,
                'confidence': confidence,
                'associated': confidence >= confidence_threshold
            }
            
            associations.append(association)
        
        # Sort by confidence
        associations.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Store results
        self.associations[model_name] = {
            'change_point_date': cp_date,
            'change_point_uncertainty': cp_uncertainty,
            'tolerance_days': tolerance_days,
            'total_nearby_events': len(nearby_events),
            'associations': associations,
            'best_association': associations[0] if associations else None,
            'high_confidence_associations': [a for a in associations if a['associated']]
        }
        
        # Print results
        if associations:
            best = associations[0]
            print(f"\n   🎯 Best Association:")
            print(f"      Event: {best['event_name']}")
            print(f"      Date: {best['event_date'].strftime('%Y-%m-%d')}")
            print(f"      Confidence: {best['confidence']:.2f}")
            print(f"      Days difference: {best['days_difference']}")
            
            high_conf = [a for a in associations if a['associated']]
            print(f"\n   ✅ High confidence associations: {len(high_conf)}")
        else:
            print("   ❌ No events found within tolerance window")
        
        return self.associations[model_name]
    
    def quantify_impact(self, model_name: str, 
                       pre_period_days: int = 30,
                       post_period_days: int = 30) -> Dict:
        """
        Quantify the impact of change points on oil prices.
        
        Args:
            model_name (str): Name of the Bayesian model
            pre_period_days (int): Days before change point for baseline
            post_period_days (int): Days after change point for impact measurement
            
        Returns:
            Dict: Quantified impact analysis
        """
        print(f"\n📊 Quantifying impact for {model_name}...")
        
        if model_name not in self.bayesian_results:
            raise ValueError(f"Model {model_name} not found")
        
        cp_results = self.bayesian_results[model_name]
        
        if 'single_changepoint' not in cp_results:
            print("⚠️ No single change point found")
            return {}
        
        cp_info = cp_results['single_changepoint']
        cp_date = cp_info['date_mode']
        cp_index = cp_info['tau_mode']
        
        # Get parameter estimates from Bayesian model
        params = cp_results.get('parameters', {})
        
        # Calculate periods
        pre_start = cp_date - timedelta(days=pre_period_days)
        post_end = cp_date + timedelta(days=post_period_days)
        
        # Get price data for periods
        pre_data = self.oil_data[
            (self.oil_data.index >= pre_start) & 
            (self.oil_data.index < cp_date)
        ]['Price']
        
        post_data = self.oil_data[
            (self.oil_data.index >= cp_date) & 
            (self.oil_data.index <= post_end)
        ]['Price']
        
        # Calculate statistics
        pre_mean = pre_data.mean()
        pre_std = pre_data.std()
        pre_min = pre_data.min()
        pre_max = pre_data.max()
        
        post_mean = post_data.mean()
        post_std = post_data.std()
        post_min = post_data.min()
        post_max = post_data.max()
        
        # Calculate impacts
        absolute_change = post_mean - pre_mean
        percentage_change = (absolute_change / pre_mean) * 100
        volatility_change = post_std - pre_std
        volatility_change_pct = (volatility_change / pre_std) * 100
        
        # Statistical significance (simple t-test)
        from scipy import stats as scipy_stats
        t_stat, p_value = scipy_stats.ttest_ind(pre_data, post_data)
        significant = p_value < 0.05
        
        # Calculate cumulative returns
        pre_return = (pre_data.iloc[-1] / pre_data.iloc[0] - 1) * 100
        post_return = (post_data.iloc[-1] / post_data.iloc[0] - 1) * 100
        
        impact_analysis = {
            'change_point_date': cp_date,
            'analysis_periods': {
                'pre_start': pre_start,
                'pre_end': cp_date,
                'post_start': cp_date,
                'post_end': post_end,
                'pre_days': len(pre_data),
                'post_days': len(post_data)
            },
            'price_impact': {
                'pre_mean': pre_mean,
                'post_mean': post_mean,
                'absolute_change': absolute_change,
                'percentage_change': percentage_change,
                'direction': 'Increase' if absolute_change > 0 else 'Decrease'
            },
            'volatility_impact': {
                'pre_volatility': pre_std,
                'post_volatility': post_std,
                'volatility_change': volatility_change,
                'volatility_change_pct': volatility_change_pct,
                'direction': 'Increase' if volatility_change > 0 else 'Decrease'
            },
            'statistical_significance': {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': significant,
                'confidence_level': '95%'
            },
            'period_performance': {
                'pre_return': pre_return,
                'post_return': post_return,
                'pre_range': {'min': pre_min, 'max': pre_max},
                'post_range': {'min': post_min, 'max': post_max}
            },
            'bayesian_estimates': params
        }
        
        # Store results
        self.impact_analysis[model_name] = impact_analysis
        
        # Print summary
        print(f"   📅 Change point: {cp_date.strftime('%Y-%m-%d')}")
        print(f"   📊 Price impact: {absolute_change:+.2f} USD ({percentage_change:+.1f}%)")
        print(f"   📈 Before: ${pre_mean:.2f} ± ${pre_std:.2f}")
        print(f"   📈 After: ${post_mean:.2f} ± ${post_std:.2f}")
        print(f"   📊 Volatility impact: {volatility_change:+.2f} USD ({volatility_change_pct:+.1f}%)")
        print(f"   🧪 Statistical significance: {'Yes' if significant else 'No'} (p={p_value:.4f})")
        
        return impact_analysis
    
    def create_impact_hypothesis(self, model_name: str) -> str:
        """
        Create a structured hypothesis about event-change point relationships.
        
        Args:
            model_name (str): Name of the Bayesian model
            
        Returns:
            str: Formatted hypothesis statement
        """
        if model_name not in self.associations or model_name not in self.impact_analysis:
            return "Insufficient data for hypothesis generation."
        
        association = self.associations[model_name]
        impact = self.impact_analysis[model_name]
        
        cp_date = association['change_point_date']
        best_event = association['best_association']
        
        if not best_event or not best_event['associated']:
            return f"""
**Change Point Analysis - {model_name}**

📅 **Detected Change Point**: {cp_date.strftime('%Y-%m-%d')}

📊 **Quantified Impact**:
- Price shift: {impact['price_impact']['absolute_change']:+.2f} USD ({impact['price_impact']['percentage_change']:+.1f}%)
- From: ${impact['price_impact']['pre_mean']:.2f} to ${impact['price_impact']['post_mean']:.2f}
- Volatility change: {impact['volatility_impact']['volatility_change_pct']:+.1f}%
- Statistical significance: {'Yes' if impact['statistical_significance']['significant'] else 'No'}

❓ **Event Association**: No high-confidence event association found within tolerance window.

🔍 **Hypothesis**: The structural break detected around {cp_date.strftime('%Y-%m-%d')} represents an endogenous market regime change or response to unobserved factors, rather than a direct response to major documented events.
"""
        
        event = best_event
        
        # Determine impact direction consistency
        expected_direction = event['expected_impact'].lower()
        actual_direction = impact['price_impact']['direction'].lower()
        
        if expected_direction == 'positive' and actual_direction.startswith('inc'):
            consistency = "✅ **Consistent**: Expected positive impact aligns with observed price increase"
        elif expected_direction == 'negative' and actual_direction.startswith('dec'):
            consistency = "✅ **Consistent**: Expected negative impact aligns with observed price decrease"
        else:
            consistency = "⚠️ **Inconsistent**: Expected and observed impacts do not align"
        
        hypothesis = f"""
**Change Point Analysis - {model_name}**

📅 **Detected Change Point**: {cp_date.strftime('%Y-%m-%d')}

🌍 **Associated Event**: {event['event_name']}
- Date: {event['event_date'].strftime('%Y-%m-%d')} ({event['days_difference']:+d} days from change point)
- Type: {event['event_type']}
- Confidence: {event['confidence']:.2f}
- Expected Impact: {event['expected_impact']} ({event['magnitude']} magnitude)

📊 **Quantified Impact**:
- Price shift: {impact['price_impact']['absolute_change']:+.2f} USD ({impact['price_impact']['percentage_change']:+.1f}%)
- Before change: ${impact['price_impact']['pre_mean']:.2f} (±${impact['volatility_impact']['pre_volatility']:.2f})
- After change: ${impact['price_impact']['post_mean']:.2f} (±${impact['volatility_impact']['post_volatility']:.2f})
- Volatility change: {impact['volatility_impact']['volatility_change_pct']:+.1f}%
- Statistical significance: {'Yes' if impact['statistical_significance']['significant'] else 'No'} (p={impact['statistical_significance']['p_value']:.4f})

{consistency}

🔍 **Hypothesis**: Following the {event['event_name']} around {event['event_date'].strftime('%Y-%m-%d')}, the Bayesian model detects a statistically significant structural break in Brent oil prices. The average daily price shifted from ${impact['price_impact']['pre_mean']:.2f} to ${impact['price_impact']['post_mean']:.2f}, representing a {impact['price_impact']['percentage_change']:+.1f}% change. This {impact['price_impact']['direction'].lower()} {'aligns with' if 'Consistent' in consistency else 'contradicts'} the expected {event['expected_impact'].lower()} impact of such {event['event_type'].lower()} events.

📋 **Event Description**: {event['description']}
"""
        
        return hypothesis
    
    def plot_event_impact_analysis(self, model_name: str, 
                                  figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create comprehensive visualization of event impact analysis.
        
        Args:
            model_name (str): Name of the Bayesian model
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Figure object
        """
        print(f"\n📊 Creating event impact visualization for {model_name}...")
        
        if model_name not in self.impact_analysis:
            raise ValueError(f"No impact analysis found for {model_name}")
        
        impact = self.impact_analysis[model_name]
        cp_date = impact['change_point_date']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Price series with change point and event
        ax1 = axes[0, 0]
        
        # Plot full price series
        ax1.plot(self.oil_data.index, self.oil_data['Price'], 'b-', alpha=0.6, linewidth=1)
        
        # Highlight analysis periods
        pre_start = impact['analysis_periods']['pre_start']
        post_end = impact['analysis_periods']['post_end']
        
        pre_data = self.oil_data[
            (self.oil_data.index >= pre_start) & 
            (self.oil_data.index < cp_date)
        ]
        post_data = self.oil_data[
            (self.oil_data.index >= cp_date) & 
            (self.oil_data.index <= post_end)
        ]
        
        ax1.plot(pre_data.index, pre_data['Price'], 'g-', linewidth=2, alpha=0.8, label='Pre-change period')
        ax1.plot(post_data.index, post_data['Price'], 'r-', linewidth=2, alpha=0.8, label='Post-change period')
        
        # Mark change point
        ax1.axvline(x=cp_date, color='black', linestyle='--', linewidth=2, label=f'Change Point: {cp_date.strftime("%Y-%m-%d")}')
        
        # Mark associated events
        if model_name in self.associations:
            associations = self.associations[model_name]['associations']
            for assoc in associations[:3]:  # Show top 3 events
                if assoc['associated']:
                    ax1.axvline(x=assoc['event_date'], color='orange', linestyle=':', 
                              alpha=0.7, label=f"Event: {assoc['event_name'][:20]}...")
        
        ax1.set_title('Oil Prices: Change Point and Event Analysis')
        ax1.set_ylabel('Price (USD/barrel)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Before/After comparison
        ax2 = axes[0, 1]
        
        periods = ['Before\nChange Point', 'After\nChange Point']
        means = [impact['price_impact']['pre_mean'], impact['price_impact']['post_mean']]
        stds = [impact['volatility_impact']['pre_volatility'], impact['volatility_impact']['post_volatility']]
        
        bars = ax2.bar(periods, means, yerr=stds, capsize=5, alpha=0.7, 
                      color=['green', 'red'], edgecolor='black')
        
        # Add value labels
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(stds) * 0.1,
                    f'${mean:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_title('Mean Price Comparison')
        ax2.set_ylabel('Price (USD/barrel)')
        ax2.grid(True, alpha=0.3)
        
        # Add significance indicator
        if impact['statistical_significance']['significant']:
            ax2.text(0.5, max(means) + max(stds) * 1.5, 
                    f"Statistically Significant\n(p={impact['statistical_significance']['p_value']:.4f})",
                    ha='center', va='center', 
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # 3. Volatility comparison
        ax3 = axes[1, 0]
        
        volatilities = [impact['volatility_impact']['pre_volatility'], 
                       impact['volatility_impact']['post_volatility']]
        bars = ax3.bar(periods, volatilities, alpha=0.7, 
                      color=['blue', 'purple'], edgecolor='black')
        
        for bar, vol in zip(bars, volatilities):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                    f'${vol:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax3.set_title('Volatility Comparison')
        ax3.set_ylabel('Standard Deviation (USD)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Impact summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary text
        summary_text = f"""
IMPACT SUMMARY

Price Change:
• Absolute: {impact['price_impact']['absolute_change']:+.2f} USD
• Percentage: {impact['price_impact']['percentage_change']:+.1f}%
• Direction: {impact['price_impact']['direction']}

Volatility Change:
• Absolute: {impact['volatility_impact']['volatility_change']:+.2f} USD
• Percentage: {impact['volatility_impact']['volatility_change_pct']:+.1f}%
• Direction: {impact['volatility_impact']['direction']}

Statistical Test:
• t-statistic: {impact['statistical_significance']['t_statistic']:.3f}
• p-value: {impact['statistical_significance']['p_value']:.4f}
• Significant: {'Yes' if impact['statistical_significance']['significant'] else 'No'}

Analysis Periods:
• Pre-change: {impact['analysis_periods']['pre_days']} days
• Post-change: {impact['analysis_periods']['post_days']} days
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def generate_comprehensive_report(self, model_names: List[str] = None) -> Dict:
        """
        Generate comprehensive report for all models.
        
        Args:
            model_names (List[str]): Models to include in report
            
        Returns:
            Dict: Comprehensive analysis report
        """
        if model_names is None:
            model_names = list(self.bayesian_results.keys())
        
        print("\n📋 Generating comprehensive event association report...")
        
        report = {
            'analysis_summary': {
                'total_models': len(model_names),
                'total_events': len(self.events_data),
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'oil_data_period': {
                    'start': self.oil_data.index.min().strftime('%Y-%m-%d'),
                    'end': self.oil_data.index.max().strftime('%Y-%m-%d'),
                    'total_days': len(self.oil_data)
                }
            },
            'model_results': {},
            'event_associations': {},
            'impact_quantifications': {},
            'hypotheses': {}
        }
        
        for model_name in model_names:
            try:
                print(f"\n   Processing {model_name}...")
                
                # Run association analysis
                associations = self.associate_events_with_changepoints(model_name)
                impact = self.quantify_impact(model_name)
                hypothesis = self.create_impact_hypothesis(model_name)
                
                report['model_results'][model_name] = {
                    'change_point_detected': len(associations) > 0,
                    'events_associated': len(associations.get('high_confidence_associations', [])),
                    'best_association_confidence': associations['best_association']['confidence'] if associations.get('best_association') else 0
                }
                
                report['event_associations'][model_name] = associations
                report['impact_quantifications'][model_name] = impact
                report['hypotheses'][model_name] = hypothesis
                
            except Exception as e:
                print(f"   ⚠️ Error processing {model_name}: {str(e)}")
                report['model_results'][model_name] = {'error': str(e)}
        
        return report


def analyze_event_impacts(oil_data: pd.DataFrame, events_data: pd.DataFrame,
                         bayesian_results: Dict, models: List[str] = None) -> EventChangePointAssociator:
    """
    Convenience function for comprehensive event impact analysis.
    
    Args:
        oil_data (pd.DataFrame): Oil price data
        events_data (pd.DataFrame): Events data
        bayesian_results (Dict): Bayesian analysis results
        models (List[str]): Models to analyze
        
    Returns:
        EventChangePointAssociator: Fitted associator with results
    """
    associator = EventChangePointAssociator(oil_data, events_data, bayesian_results)
    
    if models is None:
        models = list(bayesian_results.keys())
    
    for model in models:
        try:
            associator.associate_events_with_changepoints(model)
            associator.quantify_impact(model)
        except Exception as e:
            print(f"❌ Error analyzing {model}: {str(e)}")
    
    return associator


if __name__ == "__main__":
    print("🧪 Testing Event Association Analysis...")
    print("✅ Module loaded successfully!")