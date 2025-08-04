#!/usr/bin/env python3
"""
Brent Oil Price Change Point Analysis Workflow

This script demonstrates the complete analysis workflow for detecting change points
in Brent oil prices and correlating them with major market events.

Usage:
    python workflow_analysis.py [--output-dir OUTPUT_DIR] [--plot]

Example:
    python workflow_analysis.py --output-dir results --plot
"""

import argparse
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import BrentOilDataLoader
from time_series_analysis import TimeSeriesAnalyzer
from events_data import OilMarketEvents
from changepoint_models import ChangePointDetector
from visualization import OilPriceVisualizer, create_summary_report
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from datetime import datetime


class ChangePointAnalysisWorkflow:
    """
    Complete workflow for Brent oil price change point analysis.
    """
    
    def __init__(self, data_path: str = "data/BrentOilPrices.csv", output_dir: str = "results"):
        """
        Initialize the analysis workflow.
        
        Args:
            data_path (str): Path to the oil price data
            output_dir (str): Directory for output files
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_loader = None
        self.oil_data = None
        self.events_manager = None
        self.events_data = None
        self.ts_analyzer = None
        self.cp_detector = None
        self.visualizer = None
        
        # Results storage
        self.results = {}
        
    def step_1_load_data(self):
        """Step 1: Load and preprocess data."""
        print("📥 Step 1: Loading and preprocessing data...")
        
        # Load oil price data
        self.data_loader = BrentOilDataLoader(self.data_path)
        self.oil_data = self.data_loader.preprocess_data()
        
        # Load events data
        self.events_manager = OilMarketEvents()
        self.events_data = self.events_manager.get_events_dataframe()
        
        # Store basic statistics
        data_summary = self.data_loader.get_data_summary()
        events_summary = self.events_manager.get_summary_statistics()
        
        self.results['data_overview'] = {
            'oil_data_summary': data_summary,
            'events_summary': events_summary,
            'data_shape': self.oil_data.shape,
            'date_range': {
                'start': self.oil_data.index.min().isoformat(),
                'end': self.oil_data.index.max().isoformat()
            }
        }
        
        print(f"   ✅ Loaded {len(self.oil_data):,} oil price records")
        print(f"   ✅ Loaded {len(self.events_data)} major market events")
        
    def step_2_time_series_analysis(self):
        """Step 2: Analyze time series properties."""
        print("🔍 Step 2: Analyzing time series properties...")
        
        self.ts_analyzer = TimeSeriesAnalyzer(self.oil_data, 'Price')
        ts_results = self.ts_analyzer.comprehensive_analysis()
        
        self.results['time_series_analysis'] = ts_results
        
        # Print key findings
        stationarity = ts_results['stationarity']
        trend = ts_results['trend_analysis']['linear_trend']
        volatility = ts_results['volatility_analysis']['returns_stats']
        
        print(f"   📊 Stationarity: {stationarity['conclusion']}")
        print(f"   📈 Trend: {trend['direction']} (R²={trend['r_squared']:.4f})")
        print(f"   📊 Annual volatility: {volatility['std']*np.sqrt(252)*100:.1f}%")
        
    def step_3_change_point_detection(self):
        """Step 3: Detect change points using multiple methods."""
        print("⚡ Step 3: Detecting change points...")
        
        self.cp_detector = ChangePointDetector(self.oil_data, 'Price')
        
        # Run multiple detection methods
        methods = ['pelt_mean', 'binseg_mean', 'pelt_variance', 'trend', 'cusum']
        cp_results = self.cp_detector.comprehensive_detection(methods)
        
        # Get consensus change points
        consensus_points = self.cp_detector.get_consensus_change_points(min_methods=2, tolerance=30)
        
        # Analyze segments
        segments = self.cp_detector.analyze_segments(consensus_points)
        
        self.results['change_point_detection'] = {
            'methods_results': {
                method: {
                    'count': len(points),
                    'points': points
                }
                for method, points in cp_results.items()
            },
            'consensus_points': consensus_points,
            'segments': segments
        }
        
        print(f"   🎯 Found {len(consensus_points)} consensus change points")
        print(f"   📊 Identified {len(segments)} market segments")
        
    def step_4_event_correlation(self):
        """Step 4: Analyze correlation between change points and events."""
        print("🌍 Step 4: Analyzing event correlations...")
        
        consensus_points = self.results['change_point_detection']['consensus_points']
        tolerance_days = 90
        
        # Find correlations
        correlations = []
        events_found = 0
        
        for i, cp in enumerate(consensus_points):
            if cp < len(self.oil_data):
                cp_date = self.oil_data.index[cp]
                
                # Find events within tolerance
                start_window = cp_date - pd.Timedelta(days=tolerance_days)
                end_window = cp_date + pd.Timedelta(days=tolerance_days)
                
                nearby_events = self.events_data[
                    (self.events_data.index >= start_window) & 
                    (self.events_data.index <= end_window)
                ]
                
                if len(nearby_events) > 0:
                    events_found += 1
                
                correlations.append({
                    'change_point': i + 1,
                    'date': cp_date.isoformat(),
                    'nearby_events_count': len(nearby_events),
                    'nearby_events': nearby_events.to_dict('records') if len(nearby_events) > 0 else []
                })
        
        correlation_rate = events_found / len(consensus_points) * 100 if consensus_points else 0
        
        self.results['event_correlation'] = {
            'tolerance_days': tolerance_days,
            'correlations': correlations,
            'summary': {
                'total_change_points': len(consensus_points),
                'change_points_with_events': events_found,
                'correlation_rate': correlation_rate
            }
        }
        
        print(f"   📈 Correlation rate: {correlation_rate:.1f}% (±{tolerance_days} days)")
        
    def step_5_generate_visualizations(self, save_plots: bool = True):
        """Step 5: Generate comprehensive visualizations."""
        print("📊 Step 5: Generating visualizations...")
        
        self.visualizer = OilPriceVisualizer(self.oil_data, self.events_data)
        consensus_points = self.results['change_point_detection']['consensus_points']
        segments = self.results['change_point_detection']['segments']
        
        if save_plots:
            # Price overview
            fig1 = self.visualizer.plot_price_overview(figsize=(15, 10))
            fig1.savefig(self.output_dir / 'price_overview.png', dpi=300, bbox_inches='tight')
            plt.close(fig1)
            
            # Change points analysis
            fig2 = self.visualizer.plot_change_points_analysis(
                consensus_points, "Consensus Change Points", figsize=(15, 8)
            )
            fig2.savefig(self.output_dir / 'change_points_analysis.png', dpi=300, bbox_inches='tight')
            plt.close(fig2)
            
            # Events correlation
            fig3 = self.visualizer.plot_events_correlation(
                consensus_points, tolerance_days=90, figsize=(18, 10)
            )
            fig3.savefig(self.output_dir / 'events_correlation.png', dpi=300, bbox_inches='tight')
            plt.close(fig3)
            
            # Segments analysis
            fig4 = self.visualizer.plot_segments_analysis(segments, figsize=(15, 10))
            fig4.savefig(self.output_dir / 'segments_analysis.png', dpi=300, bbox_inches='tight')
            plt.close(fig4)
            
            # Summary report
            fig5 = create_summary_report(self.oil_data, consensus_points, segments, self.events_data)
            fig5.savefig(self.output_dir / 'summary_report.png', dpi=300, bbox_inches='tight')
            plt.close(fig5)
            
            print(f"   💾 Saved 5 visualization files to {self.output_dir}")
        
    def step_6_export_results(self):
        """Step 6: Export results and data."""
        print("💾 Step 6: Exporting results...")
        
        # Export processed data
        self.oil_data.to_csv(self.output_dir / 'processed_oil_data.csv')
        
        # Export events data
        self.events_data.to_csv(self.output_dir / 'oil_market_events.csv')
        
        # Export change points
        consensus_points = self.results['change_point_detection']['consensus_points']
        if consensus_points:
            cp_df = pd.DataFrame({
                'index': consensus_points,
                'date': [self.oil_data.index[cp].strftime('%Y-%m-%d') for cp in consensus_points if cp < len(self.oil_data)],
                'price': [self.oil_data['Price'].iloc[cp] for cp in consensus_points if cp < len(self.oil_data)]
            })
            cp_df.to_csv(self.output_dir / 'change_points.csv', index=False)
        
        # Export segments
        segments = self.results['change_point_detection']['segments']
        if segments:
            segments_df = pd.DataFrame(segments)
            segments_df.to_csv(self.output_dir / 'segments_analysis.csv', index=False)
        
        # Export complete results as JSON
        # Convert datetime objects to strings for JSON serialization
        results_copy = self._prepare_results_for_json(self.results.copy())
        
        with open(self.output_dir / 'analysis_results.json', 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
        
        print(f"   📄 Exported data files to {self.output_dir}")
        print(f"   📋 Complete results saved to analysis_results.json")
        
    def _prepare_results_for_json(self, obj):
        """Recursively convert datetime and numpy objects for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._prepare_results_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_results_for_json(item) for item in obj]
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def run_complete_workflow(self, save_plots: bool = True):
        """Run the complete analysis workflow."""
        print("🚀 Starting Brent Oil Change Point Analysis Workflow")
        print("=" * 60)
        
        start_time = datetime.now()
        
        try:
            self.step_1_load_data()
            self.step_2_time_series_analysis()
            self.step_3_change_point_detection()
            self.step_4_event_correlation()
            self.step_5_generate_visualizations(save_plots)
            self.step_6_export_results()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            print("\n🎉 Analysis workflow completed successfully!")
            print(f"⏱️  Total execution time: {duration}")
            print(f"📁 Results saved to: {self.output_dir.absolute()}")
            
            # Print summary
            self._print_summary()
            
        except Exception as e:
            print(f"\n❌ Error occurred during analysis: {str(e)}")
            raise
    
    def _print_summary(self):
        """Print analysis summary."""
        print("\n📊 ANALYSIS SUMMARY")
        print("=" * 40)
        
        # Data overview
        data_summary = self.results['data_overview']
        print(f"📅 Analysis period: {data_summary['date_range']['start'][:10]} to {data_summary['date_range']['end'][:10]}")
        print(f"📊 Total data points: {data_summary['data_shape'][0]:,}")
        
        # Change points
        cp_results = self.results['change_point_detection']
        print(f"🎯 Consensus change points: {len(cp_results['consensus_points'])}")
        print(f"📊 Market segments: {len(cp_results['segments'])}")
        
        # Event correlation
        event_corr = self.results['event_correlation']
        print(f"🌍 Event correlation rate: {event_corr['summary']['correlation_rate']:.1f}%")
        
        # Key statistics
        if cp_results['segments']:
            segments = cp_results['segments']
            durations = [s['duration_days'] for s in segments]
            returns = [s['pct_change'] for s in segments]
            
            print(f"📈 Average segment duration: {np.mean(durations):.0f} days")
            print(f"💹 Best segment return: {max(returns):.1f}%")
            print(f"📉 Worst segment return: {min(returns):.1f}%")


def main():
    """Main function to run the analysis workflow."""
    parser = argparse.ArgumentParser(description='Brent Oil Change Point Analysis Workflow')
    parser.add_argument('--data-path', default='../data/BrentOilPrices.csv',
                       help='Path to the oil price data file')
    parser.add_argument('--output-dir', default='../results',
                       help='Output directory for results')
    parser.add_argument('--plot', action='store_true',
                       help='Generate and save plots')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plot generation (faster execution)')
    
    args = parser.parse_args()
    
    # Determine whether to save plots
    save_plots = args.plot or not args.no_plot
    
    # Initialize and run workflow
    workflow = ChangePointAnalysisWorkflow(
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    workflow.run_complete_workflow(save_plots=save_plots)


if __name__ == "__main__":
    main()