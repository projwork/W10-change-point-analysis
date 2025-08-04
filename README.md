# Brent Oil Price Change Point Analysis

A comprehensive analysis framework for detecting structural breaks in Brent oil prices and correlating them with major market events.

## 📋 Project Overview

This project implements a systematic approach to identify change points in historical Brent oil prices (May 1987 - September 2022) and analyze their correlation with major geopolitical events, OPEC decisions, and economic shocks.

### 🎯 Objectives

1. **Data Analysis Workflow**: Define and implement a systematic analysis process
2. **Time Series Properties**: Investigate stationarity, trends, and volatility patterns
3. **Change Point Detection**: Apply multiple algorithms to identify structural breaks
4. **Event Correlation**: Analyze temporal relationships between breaks and market events
5. **Statistical vs Causal Analysis**: Discuss limitations and assumptions

### 🔑 Key Features

- **Modular Design**: Clean, reusable modules for different analysis components
- **Multiple Algorithms**: PELT, Binary Segmentation, CUSUM, and trend-based detection
- **Comprehensive Visualization**: Interactive dashboards and detailed plots
- **Event Database**: Curated dataset of 20+ major oil market events
- **Statistical Rigor**: Proper handling of assumptions and limitations

## 📁 Project Structure

```
change-point-analysis/
├── data/
│   ├── BrentOilPrices.csv           # Raw oil price data
│   └── oil_market_events.csv        # Major market events (generated)
├── src/                             # Core analysis modules
│   ├── __init__.py
│   ├── data_loader.py              # Data loading and preprocessing
│   ├── time_series_analysis.py     # Time series properties analysis
│   ├── events_data.py              # Oil market events database
│   ├── changepoint_models.py       # Change point detection algorithms
│   └── visualization.py            # Comprehensive visualization suite
├── scripts/                        # Analysis workflows
│   ├── __init__.py
│   └── workflow_analysis.py        # Complete analysis workflow
├── notebooks/                      # Jupyter notebooks
│   └── brent_oil_changepoint_analysis.ipynb
├── tests/                          # Unit tests
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Virtual environment activated

### Installation

1. **Clone the repository** (if applicable) or navigate to the project directory:

```bash
cd change-point-analysis
```

2. **Activate the virtual environment**:

```bash
.\.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

3. **Install dependencies** (already done):

```bash
pip install pandas numpy matplotlib seaborn ruptures statsmodels scipy jupyter plotly yfinance requests beautifulsoup4
```

### Usage Options

#### Option 1: Jupyter Notebook (Recommended)

```bash
jupyter lab notebooks/brent_oil_changepoint_analysis.ipynb
```

#### Option 2: Command Line Workflow

```bash
python scripts/workflow_analysis.py --output-dir results --plot
```

#### Option 3: Python Scripts

```python
from src.data_loader import load_brent_oil_data
from src.changepoint_models import ChangePointDetector

# Load data
data = load_brent_oil_data()

# Detect change points
detector = ChangePointDetector(data)
results = detector.comprehensive_detection()
```

## 📊 Analysis Methodology

### 1. Data Analysis Workflow

1. **Data Loading & Preprocessing**

   - Load raw Brent oil price data
   - Handle missing values and date parsing
   - Calculate derived features (returns, moving averages)

2. **Time Series Analysis**

   - Stationarity testing (ADF, KPSS)
   - Trend analysis (linear regression)
   - Volatility analysis (rolling statistics)

3. **Change Point Detection**

   - PELT (Pruned Exact Linear Time)
   - Binary Segmentation
   - CUSUM (Cumulative Sum)
   - Trend change detection
   - Variance change detection

4. **Consensus Building**

   - Combine results from multiple methods
   - Apply tolerance-based clustering
   - Generate consensus change points

5. **Event Correlation**

   - Match change points with major events
   - Analyze temporal proximity
   - Calculate correlation statistics

6. **Visualization & Reporting**
   - Interactive dashboards
   - Comprehensive plots
   - Statistical summaries

### 2. Change Point Models

**Purpose**: Identify locations in time series where statistical properties change

**Expected Outputs**:

- Change point dates and indices
- Segment-wise statistics (mean, variance, trend)
- Statistical significance measures

**Applications**:

- Market regime identification
- Risk assessment periods
- Event impact analysis
- Portfolio strategy timing

### 3. Event Database

Curated dataset of major oil market events including:

- **Geopolitical Events**: Wars, conflicts, sanctions
- **OPEC Decisions**: Production cuts/increases, policy changes
- **Economic Crises**: Financial crashes, recessions
- **Natural Disasters**: Hurricanes, oil spill incidents
- **Supply Shocks**: Infrastructure disruptions

## 📈 Key Results & Insights

### Statistical Properties

- **Non-stationarity**: Oil prices exhibit non-stationary behavior
- **Volatility Clustering**: Periods of high/low volatility cluster together
- **Trend Reversals**: Multiple regime changes over the analysis period

### Change Point Detection

- **Multiple Regimes**: Clear structural breaks identified across different periods
- **Event Correlation**: Significant temporal proximity between major events and change points
- **Algorithm Consensus**: Different methods show reasonable agreement on major breaks

### Market Insights

- **Crisis Periods**: Major economic crises often coincide with structural breaks
- **OPEC Impact**: Production decisions show mixed immediate correlation with price changes
- **Geopolitical Events**: Wars and conflicts typically associated with price volatility increases

## ⚠️ Limitations & Assumptions

### Statistical vs. Causal Analysis

**CRITICAL**: This analysis identifies statistical correlations, NOT causal relationships.

**What We Can Conclude**:

- Temporal proximity between events and change points
- Statistical patterns in historical data
- Market timing relationships

**What We CANNOT Conclude**:

- Direct causation between events and price changes
- Predictive power for future events
- Quantitative impact magnitudes

### Technical Limitations

- **Retrospective Analysis**: Change point detection is backward-looking
- **Parameter Sensitivity**: Results depend on algorithm parameters
- **False Positives**: May detect changes due to random variation
- **Data Quality**: Limited to available daily price data

### Methodological Assumptions

- **Event Selection**: Limited to documented, major events
- **Market Efficiency**: Assumes prices reflect available information
- **Linear Relationships**: Some models assume linear trend changes
- **Independence**: Assumes events are independent (often not true)

## 📚 References & Further Reading

### Academic Literature

- Killick, R., & Eckley, I. (2014). changepoint: An R package for changepoint analysis
- Truong, C., et al. (2020). Selective review of offline change point detection methods
- Hamilton, J. D. (2008). Oil and the macroeconomy since World War II

### Technical Documentation

- [Ruptures Library](https://centre-borelli.github.io/ruptures-docs/)
- [Statsmodels Documentation](https://www.statsmodels.org/)
- [Time Series Analysis Methods](https://otexts.com/fpp3/)

## 🤝 Contributing

This project follows academic research standards. Contributions should include:

1. **Documentation**: Clear methodology and assumptions
2. **Testing**: Unit tests for new functionality
3. **Validation**: Cross-validation with alternative datasets
4. **Reproducibility**: Detailed steps for result reproduction

## 📄 License

This project is for academic and research purposes. Please cite appropriately if used in publications.

## 📞 Contact & Support

For questions about methodology, implementation, or results interpretation, please refer to:

1. **Documentation**: Comprehensive comments in source code
2. **Notebook**: Detailed explanations in Jupyter notebook
3. **Academic Literature**: Referenced papers and methodologies

---

**Disclaimer**: This analysis is for research and educational purposes only. Results should not be used for financial decision-making without proper risk assessment and additional analysis.
