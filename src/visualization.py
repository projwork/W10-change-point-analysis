"""
Visualization Module for Brent Oil Price Change Point Analysis

This module provides comprehensive visualization functions for time series analysis,
change point detection results, and event correlation analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class OilPriceVisualizer:
    """
    Comprehensive visualization suite for oil price analysis.
    """
    
    def __init__(self, price_data: pd.DataFrame, events_data: pd.DataFrame = None):
        """
        Initialize the visualizer.
        
        Args:
            price_data (pd.DataFrame): Oil price time series data
            events_data (pd.DataFrame): Events data (optional)
        """
        self.price_data = price_data.copy()
        self.events_data = events_data.copy() if events_data is not None else None
        
    def plot_price_overview(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create comprehensive price overview plot.
        
        Args:
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Figure object
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # Main price plot
        axes[0].plot(self.price_data.index, self.price_data['Price'], 
                    linewidth=1, color='darkblue', alpha=0.8)
        
        # Add moving averages if available
        if 'MA_30' in self.price_data.columns:
            axes[0].plot(self.price_data.index, self.price_data['MA_30'], 
                        label='30-day MA', color='orange', alpha=0.7)
        if 'MA_365' in self.price_data.columns:
            axes[0].plot(self.price_data.index, self.price_data['MA_365'], 
                        label='365-day MA', color='red', alpha=0.7)
        
        axes[0].set_title('Brent Oil Prices Over Time', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Price (USD/barrel)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Returns plot
        if 'Returns' in self.price_data.columns:
            returns = self.price_data['Returns'].dropna()
            axes[1].plot(returns.index, returns.values, 
                        linewidth=0.5, color='green', alpha=0.7)
            axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[1].set_title('Daily Returns', fontsize=12)
            axes[1].set_ylabel('Returns (%)')
            axes[1].grid(True, alpha=0.3)
        
        # Volatility plot
        if 'Returns' in self.price_data.columns:
            rolling_vol = self.price_data['Returns'].rolling(window=30).std()
            axes[2].plot(rolling_vol.index, rolling_vol.values, 
                        linewidth=1, color='red', alpha=0.8)
            axes[2].set_title('30-Day Rolling Volatility', fontsize=12)
            axes[2].set_ylabel('Volatility')
            axes[2].set_xlabel('Date')
            axes[2].grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
        
        plt.tight_layout()
        return fig
    
    def plot_change_points_analysis(self, change_points: List[int], 
                                   method_name: str = "Change Points",
                                   figsize: Tuple[int, int] = (15, 8)) -> plt.Figure:
        """
        Plot change points on price series with detailed analysis.
        
        Args:
            change_points (List[int]): Indices of change points
            method_name (str): Name of the detection method
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        
        # Main price plot with change points
        ax1.plot(self.price_data.index, self.price_data['Price'], 
                linewidth=1, color='darkblue', alpha=0.8, label='Brent Oil Price')
        
        # Add change points
        for i, cp in enumerate(change_points):
            if cp < len(self.price_data):
                date = self.price_data.index[cp]
                price = self.price_data['Price'].iloc[cp]
                ax1.axvline(x=date, color='red', linestyle='--', alpha=0.8, linewidth=2)
                ax1.scatter(date, price, color='red', s=50, zorder=5)
                ax1.annotate(f'CP{i+1}', (date, price), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color='red', fontweight='bold')
        
        ax1.set_title(f'Brent Oil Prices with {method_name}', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (USD/barrel)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Returns plot with change points
        if 'Returns' in self.price_data.columns:
            returns = self.price_data['Returns'].dropna()
            ax2.plot(returns.index, returns.values, 
                    linewidth=0.5, color='green', alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Add change points to returns plot
            for cp in change_points:
                if cp < len(self.price_data) and cp < len(returns):
                    ax2.axvline(x=self.price_data.index[cp], color='red', 
                              linestyle='--', alpha=0.8, linewidth=2)
            
            ax2.set_title('Daily Returns with Change Points', fontsize=12)
            ax2.set_ylabel('Returns (%)')
            ax2.set_xlabel('Date')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_events_correlation(self, change_points: List[int] = None,
                               tolerance_days: int = 90,
                               figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot correlation between events and change points.
        
        Args:
            change_points (List[int]): Change point indices
            tolerance_days (int): Days tolerance for event-change point matching
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Figure object
        """
        if self.events_data is None:
            raise ValueError("Events data not provided")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot price series
        ax.plot(self.price_data.index, self.price_data['Price'], 
               linewidth=1, color='darkblue', alpha=0.8, label='Brent Oil Price')
        
        # Plot events
        event_colors = {
            'Geopolitical': 'red',
            'Economic Crisis': 'orange', 
            'OPEC Decision': 'green',
            'Natural Disaster': 'purple',
            'Supply Shock': 'brown'
        }
        
        for event_type in self.events_data['type'].unique():
            events_subset = self.events_data[self.events_data['type'] == event_type]
            
            for date, row in events_subset.iterrows():
                if date in self.price_data.index:
                    price = self.price_data.loc[date, 'Price']
                    color = event_colors.get(event_type, 'gray')
                    
                    # Plot event marker
                    ax.scatter(date, price, color=color, s=100, 
                             marker='v', alpha=0.8, edgecolors='black', linewidth=1)
                    
                    # Add event label
                    ax.annotate(row['event'][:20] + '...', (date, price),
                              xytext=(0, 20), textcoords='offset points',
                              fontsize=8, rotation=45, ha='left',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
        
        # Plot change points if provided
        if change_points:
            for i, cp in enumerate(change_points):
                if cp < len(self.price_data):
                    date = self.price_data.index[cp]
                    price = self.price_data['Price'].iloc[cp]
                    ax.axvline(x=date, color='black', linestyle='--', alpha=0.6, linewidth=2)
                    ax.annotate(f'CP{i+1}', (date, price),
                              xytext=(5, -15), textcoords='offset points',
                              fontsize=10, color='black', fontweight='bold')
        
        # Create legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, markersize=10, label=event_type)
                         for event_type, color in event_colors.items()
                         if event_type in self.events_data['type'].values]
        
        legend_elements.append(plt.Line2D([0], [0], color='darkblue', linewidth=2, label='Oil Price'))
        
        if change_points:
            legend_elements.append(plt.Line2D([0], [0], color='black', 
                                            linestyle='--', linewidth=2, label='Change Points'))
        
        ax.legend(handles=legend_elements, loc='upper left')
        ax.set_title('Oil Price, Major Events, and Change Points', fontsize=14, fontweight='bold')
        ax.set_ylabel('Price (USD/barrel)')
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_segments_analysis(self, segments: List[Dict],
                              figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot analysis of segments between change points.
        
        Args:
            segments (List[Dict]): Segment analysis results
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Segment duration distribution
        durations = [s['duration_days'] for s in segments]
        axes[0, 0].hist(durations, bins=min(20, len(segments)), alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Distribution of Segment Durations')
        axes[0, 0].set_xlabel('Duration (days)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Segment mean prices
        means = [s['mean'] for s in segments]
        dates = [s['start_date'] for s in segments]
        axes[0, 1].plot(dates, means, marker='o', linewidth=2, markersize=6)
        axes[0, 1].set_title('Average Price by Segment')
        axes[0, 1].set_xlabel('Segment Start Date')
        axes[0, 1].set_ylabel('Average Price (USD)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Segment volatility (std)
        volatilities = [s['std'] for s in segments]
        axes[1, 0].plot(dates, volatilities, marker='s', linewidth=2, markersize=6, color='red')
        axes[1, 0].set_title('Volatility by Segment')
        axes[1, 0].set_xlabel('Segment Start Date')
        axes[1, 0].set_ylabel('Standard Deviation')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Segment returns
        returns = [s['pct_change'] for s in segments]
        colors = ['green' if r > 0 else 'red' for r in returns]
        axes[1, 1].bar(range(len(returns)), returns, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Total Return by Segment')
        axes[1, 1].set_xlabel('Segment Number')
        axes[1, 1].set_ylabel('Total Return (%)')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_dashboard(self, change_points: List[int] = None) -> go.Figure:
        """
        Create interactive dashboard using Plotly.
        
        Args:
            change_points (List[int]): Change point indices
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Oil Price with Change Points', 'Daily Returns', 'Rolling Volatility'),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Main price plot
        fig.add_trace(
            go.Scatter(
                x=self.price_data.index,
                y=self.price_data['Price'],
                mode='lines',
                name='Brent Oil Price',
                line=dict(color='darkblue', width=1),
                hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add moving averages if available
        if 'MA_30' in self.price_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.price_data.index,
                    y=self.price_data['MA_30'],
                    mode='lines',
                    name='30-day MA',
                    line=dict(color='orange', width=1),
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        # Add change points
        if change_points:
            for i, cp in enumerate(change_points):
                if cp < len(self.price_data):
                    date = self.price_data.index[cp]
                    price = self.price_data['Price'].iloc[cp]
                    
                    # Vertical line
                    fig.add_vline(
                        x=date,
                        line=dict(color='red', width=2, dash='dash'),
                        opacity=0.8,
                        row=1, col=1
                    )
                    
                    # Change point marker
                    fig.add_trace(
                        go.Scatter(
                            x=[date],
                            y=[price],
                            mode='markers',
                            name=f'Change Point {i+1}',
                            marker=dict(color='red', size=8, symbol='diamond'),
                            hovertemplate=f'Change Point {i+1}<br>Date: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>'
                        ),
                        row=1, col=1
                    )
        
        # Returns plot
        if 'Returns' in self.price_data.columns:
            returns = self.price_data['Returns'].dropna()
            fig.add_trace(
                go.Scatter(
                    x=returns.index,
                    y=returns.values * 100,  # Convert to percentage
                    mode='lines',
                    name='Daily Returns',
                    line=dict(color='green', width=0.5),
                    hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add change points to returns plot
            if change_points:
                for cp in change_points:
                    if cp < len(self.price_data):
                        fig.add_vline(
                            x=self.price_data.index[cp],
                            line=dict(color='red', width=2, dash='dash'),
                            opacity=0.8,
                            row=2, col=1
                        )
        
        # Volatility plot
        if 'Returns' in self.price_data.columns:
            rolling_vol = self.price_data['Returns'].rolling(window=30).std() * 100
            fig.add_trace(
                go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol.values,
                    mode='lines',
                    name='30-day Volatility',
                    line=dict(color='red', width=1),
                    hovertemplate='Date: %{x}<br>Volatility: %{y:.2f}%<extra></extra>'
                ),
                row=3, col=1
            )
            
            # Add change points to volatility plot
            if change_points:
                for cp in change_points:
                    if cp < len(self.price_data):
                        fig.add_vline(
                            x=self.price_data.index[cp],
                            line=dict(color='red', width=2, dash='dash'),
                            opacity=0.8,
                            row=3, col=1
                        )
        
        # Update layout
        fig.update_layout(
            title='Brent Oil Price Analysis Dashboard',
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Price (USD/barrel)", row=1, col=1)
        fig.update_yaxes(title_text="Returns (%)", row=2, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=3, col=1)
        
        # Update x-axis label
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        return fig
    
    def plot_comparison_methods(self, change_points_dict: Dict[str, List[int]],
                               figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
        """
        Compare change points detected by different methods.
        
        Args:
            change_points_dict (Dict[str, List[int]]): Change points by method
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Figure object
        """
        n_methods = len(change_points_dict)
        fig, axes = plt.subplots(n_methods + 1, 1, figsize=figsize, sharex=True)
        
        if n_methods == 0:
            axes = [axes]
        
        # Plot original series
        axes[0].plot(self.price_data.index, self.price_data['Price'], 
                    'b-', alpha=0.7, linewidth=1)
        axes[0].set_title('Brent Oil Prices', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Price (USD)')
        axes[0].grid(True, alpha=0.3)
        
        # Plot each method
        colors = plt.cm.Set1(np.linspace(0, 1, n_methods))
        
        for i, ((method, change_points), color) in enumerate(zip(change_points_dict.items(), colors)):
            # Plot price series
            axes[i+1].plot(self.price_data.index, self.price_data['Price'], 
                          'b-', alpha=0.3, linewidth=1)
            
            # Plot change points
            for cp in change_points:
                if cp < len(self.price_data):
                    date = self.price_data.index[cp]
                    axes[i+1].axvline(x=date, color=color, linestyle='--', 
                                    alpha=0.8, linewidth=2)
            
            axes[i+1].set_title(f'{method.replace("_", " ").title()} - {len(change_points)} change points')
            axes[i+1].set_ylabel('Price (USD)')
            axes[i+1].grid(True, alpha=0.3)
        
        plt.xlabel('Date')
        plt.tight_layout()
        return fig


def create_summary_report(price_data: pd.DataFrame, change_points: List[int],
                         segments: List[Dict], events_data: pd.DataFrame = None) -> plt.Figure:
    """
    Create a comprehensive summary report figure.
    
    Args:
        price_data (pd.DataFrame): Price data
        change_points (List[int]): Detected change points
        segments (List[Dict]): Segment analysis
        events_data (pd.DataFrame): Events data
        
    Returns:
        plt.Figure: Summary report figure
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main price plot with change points
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(price_data.index, price_data['Price'], 'b-', linewidth=1, alpha=0.8)
    
    for i, cp in enumerate(change_points):
        if cp < len(price_data):
            date = price_data.index[cp]
            price = price_data['Price'].iloc[cp]
            ax1.axvline(x=date, color='red', linestyle='--', alpha=0.8, linewidth=2)
            ax1.annotate(f'CP{i+1}', (date, price), xytext=(0, 10), 
                        textcoords='offset points', ha='center', fontsize=8)
    
    ax1.set_title('Brent Oil Prices with Detected Change Points', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USD/barrel)')
    ax1.grid(True, alpha=0.3)
    
    # Segment statistics
    ax2 = fig.add_subplot(gs[1, 0])
    durations = [s['duration_days'] for s in segments]
    ax2.hist(durations, bins=min(15, len(segments)), alpha=0.7, edgecolor='black')
    ax2.set_title('Segment Durations')
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3)
    
    # Segment returns
    ax3 = fig.add_subplot(gs[1, 1])
    returns = [s['pct_change'] for s in segments]
    colors = ['green' if r > 0 else 'red' for r in returns]
    ax3.bar(range(len(returns)), returns, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_title('Segment Returns')
    ax3.set_xlabel('Segment')
    ax3.set_ylabel('Return (%)')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    # Volatility analysis
    ax4 = fig.add_subplot(gs[1, 2])
    volatilities = [s['std'] for s in segments]
    ax4.plot(range(len(volatilities)), volatilities, 'ro-', markersize=6)
    ax4.set_title('Segment Volatilities')
    ax4.set_xlabel('Segment')
    ax4.set_ylabel('Std Dev')
    ax4.grid(True, alpha=0.3)
    
    # Summary statistics
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Create summary text
    summary_text = f"""
    CHANGE POINT ANALYSIS SUMMARY
    ═══════════════════════════════════════════════════════════════════════════
    
    Total Change Points Detected: {len(change_points)}
    Analysis Period: {price_data.index.min().strftime('%Y-%m-%d')} to {price_data.index.max().strftime('%Y-%m-%d')}
    Total Days: {(price_data.index.max() - price_data.index.min()).days:,}
    
    Segment Analysis:
    • Number of Segments: {len(segments)}
    • Average Segment Duration: {np.mean(durations):.0f} days
    • Shortest Segment: {min(durations):.0f} days
    • Longest Segment: {max(durations):.0f} days
    
    Price Statistics:
    • Overall Price Range: ${price_data['Price'].min():.2f} - ${price_data['Price'].max():.2f}
    • Average Price: ${price_data['Price'].mean():.2f}
    • Price Volatility (Std): ${price_data['Price'].std():.2f}
    
    Segment Performance:
    • Positive Return Segments: {sum(1 for r in returns if r > 0)} ({sum(1 for r in returns if r > 0)/len(returns)*100:.1f}%)
    • Negative Return Segments: {sum(1 for r in returns if r < 0)} ({sum(1 for r in returns if r < 0)/len(returns)*100:.1f}%)
    • Best Segment Return: {max(returns):.1f}%
    • Worst Segment Return: {min(returns):.1f}%
    """
    
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
    
    return fig


if __name__ == "__main__":
    # Example usage
    from data_loader import load_brent_oil_data
    from events_data import load_oil_market_events
    
    # Load data
    price_data = load_brent_oil_data()
    events_data = load_oil_market_events()
    
    # Create visualizer
    visualizer = OilPriceVisualizer(price_data, events_data)
    
    # Create plots
    fig1 = visualizer.plot_price_overview()
    fig2 = visualizer.plot_events_correlation()
    
    plt.show()