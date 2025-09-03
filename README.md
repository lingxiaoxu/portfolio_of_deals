# Institutional-Quality Private Credit Portfolio Analytics

## Overview

Professional private credit portfolio analysis system with sophisticated Monte Carlo simulation, dynamic prepayment modeling, and institutional-quality risk management. Features comprehensive credit risk analytics, FRED data integration, and multi-factor cashflow modeling.

## Core System Architecture

### Main Analysis Scripts
- `run_synthetic.py` - Adaptive portfolio analysis (10 private loans or 150 synthetic instruments)
- `run_deals.py` - Specialized private credit analytics with AI-powered PDF processing

### Risk Modeling Modules
- `prepayment_risk_module.py` - Centralized prepayment risk engine with multi-factor modeling
- `credit_risk_module.py` - Advanced credit risk integration with fundamental analysis
- `enriched_bond_portfolio.py` - Sophisticated fixed income portfolio modeling with Monte Carlo simulation

### Core Analytics Infrastructure
- `bond_utilities.py` - Mathematical functions for bonds and loans (IRR, duration, pricing)
- `forward_rate_lookup.py` - FRED data integration with historical and forward rate projections
- `forward_rate_projections.py` - Nelson-Siegel yield curve modeling
- `yield_curve_modeling.py` - Professional yield curve implementation
- `cashflow_exporter.py` - Institutional-quality cashflow export functions
- `download_fred_data.py` - FRED economic data download and integration

### Configuration
- `config_template.py` - Template for API key configuration

## Project Structure
```
├── run_synthetic.py                 # Adaptive portfolio analysis (10 loans or 150 assets)
├── run_deals.py                     # Specialized private credit analytics
├── prepayment_risk_module.py        # Centralized prepayment modeling
├── credit_risk_module.py            # Advanced credit risk analytics
├── enriched_bond_portfolio.py       # Monte Carlo portfolio simulation
├── bond_utilities.py               # Core financial calculations
├── forward_rate_*.py               # Professional rate modeling
├── requirements.txt                # Organized dependencies
├── deals_data/                     # Real private credit data
├── plots/                          # Organized visualization output
├── logs/                           # Timestamped analysis logs
├── cashflows_YYYYMMDD_HHMMSS/      # Generated cashflow schedules
└── forward_rates_YYYYMMDD_HHMMSS/  # Nelson-Siegel rate projections
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Configuration
```bash
cp config_template.py config.py
# Edit config.py and add your OpenAI API key for PDF processing
```

### 3. Run Analysis

#### Private Credit Portfolio Analysis (Primary Mode)
```bash
python run_synthetic.py
```
- **Adaptive Analysis**: 10 private loans (with deals_data) or 150 synthetic instruments
- **Monte Carlo Simulation**: 10,000 cashflow scenarios per loan
- **Professional Output**: Individual loan distributions + portfolio efficient frontier
- **Dynamic Risk Modeling**: Hazard-based defaults, sophisticated prepayment, SOFR path simulation

#### Specialized Deals Analysis
```bash
python run_deals.py
```
- **Advanced Credit Modeling**: Fundamental-based default/recovery modeling
- **PDF Integration**: AI-powered memo processing
- **Stress Testing**: Comprehensive forward rate stress scenarios
- **Professional Dashboard**: 9-panel analytics visualization

## Key Features

### Monte Carlo Private Credit Simulation
- **100,000 total scenarios**: 10,000 paths per loan across 10 loans
- **Proper Fixed Income Methodology**: Cashflow-based IRR calculation (not equity-like returns)
- **Hazard-Based Default Modeling**: Professional survival analysis with proper hazard rates
- **Dynamic Prepayment**: Multi-factor prepayment speed based on LTV, DSCR, rating, sector
- **Interest Rate Path Dependency**: Mean-reverting SOFR paths for floating rate loans
- **Loan-Specific Amortization**: Respects actual IO periods and amortization schedules

### Centralized Risk Management
- **Unified Prepayment Engine**: Single source of truth across all scripts
- **Credit Quality-Based Adjustments**: Dynamic volatility caps and prepayment rates
- **FRED Data Integration**: All rates from real market data, no hardcoded assumptions
- **Professional Risk Framework**: IRR ranges 5.43% - 11.00%, all above 4.33% EFFR

### Advanced Portfolio Optimization
- **Efficient Frontier**: Portfolio combinations using Monte Carlo-derived statistics
- **Capital Allocation Line (CAL)**: Properly intersects y-axis at 4.33% EFFR
- **Dynamic Correlation**: Sector-based, rating-based, geographic correlations
- **Professional Results**: Positive Sharpe ratios, realistic risk/return profiles

## Expected Output

### Console Output
```
✅ Using current EFFR as risk-free rate: 4.33%
Loaded 10 loan specifications
Using deals_data mode - skipping bond loading (portfolio.csv used for loans)

MONTE CARLO PATH SIMULATION FOR INDIVIDUAL LOANS
Running stochastic simulations for each loan (10,000 scenarios each)...

Simulating 10000 FIXED INCOME paths for PORT_001 (Zeta Industrials plc)
  Base coupon: 7.48%, Default prob: 3.5%
  Mean IRR: 7.45%, IRR Volatility: 1.85%, Default Rate: 16.2%, Prepay Rate: 3.0%

Max-Sharpe Portfolio: 8.47% return, 1.51 Sharpe
```

### Generated Files
```
plots/
├── loan_path_simulation_YYYYMMDD_HHMMSS.png        # Individual loan Monte Carlo distributions
├── efficient_frontier_YYYYMMDD_HHMMSS.png          # Portfolio optimization results
└── private_loan_portfolio_analytics_YYYYMMDD.png   # Comprehensive dashboard

cashflows_YYYYMMDD_HHMMSS/
├── DEAL_001_Echo_Foods_Co._cashflows.txt           # Detailed cashflow analysis
├── PORT_001_Zeta_Industrials_plc_cashflows.txt     # With IRR calculations
└── [8 more loan cashflow files...]

logs/
├── synthetic_analysis_YYYYMMDD_HHMMSS.log          # Complete audit trail
└── deals_analysis_YYYYMMDD_HHMMSS.log              # Deals-specific analysis
```

## Portfolio Analytics

### Risk Metrics
- **Expected Returns**: 5.43% - 11.00% IRR range (all above risk-free rate)
- **Risk-Adjusted Returns**: Sharpe ratios 1.37 - 1.51 
- **Credit Risk**: 3% - 5.3% default probabilities with sophisticated recovery modeling
- **Prepayment Risk**: 2% - 4% annual prepayment speeds based on loan fundamentals

### Visualization Output
- **Individual Loan Histograms**: IRR distributions from 10,000 scenarios each
- **Portfolio Efficient Frontier**: 5,000 portfolio combinations with proper CAL line
- **Professional Dashboard**: Sector analysis, credit metrics, stress testing results

## Technical Architecture

### Adaptive Data Processing
- **Deals Data Mode**: Uses enhanced metadata files for 10 private credit loans
- **Synthetic Mode**: Generates 150-asset portfolio for broader analysis
- **Dynamic Detection**: Automatically switches based on data availability

### Professional Risk Modeling
- **Prepayment Engine**: Multi-factor speed calculation (LTV, DSCR, rating, sector, seniority)
- **Default Modeling**: Hazard-based survival analysis with proper probability distributions
- **Interest Rate Risk**: Mean-reverting SOFR paths with quarterly resets
- **Correlation Modeling**: Fundamental factor-based correlation matrices

### Market Data Integration
- **FRED Data**: Real-time SOFR, Treasury rates, economic indicators
- **Nelson-Siegel Curves**: Professional yield curve modeling and forward rate projections
- **Historical Lookups**: Vintage rate comparison for duration/convexity analysis
- **Dynamic Risk-Free Rate**: Uses current EFFR (4.33%) from FRED data

## Dependencies

Core packages for institutional analysis:
```
pandas>=2.2.2         # Data analysis and portfolio modeling
numpy>=1.26.4         # Mathematical calculations
matplotlib>=3.9.1     # Professional visualizations
scipy>=1.14.0         # Advanced statistical modeling
scikit-learn>=1.5.1   # Machine learning for credit modeling
statsmodels>=0.14.2   # Econometric analysis
fredapi>=0.5.2        # Federal Reserve data integration
openai==0.28.0        # AI-powered PDF processing
pdfplumber>=0.11.7    # PDF text extraction
seaborn>=0.13.2       # Statistical visualization
```

## API Reference

### Core Functions
```python
# Adaptive Portfolio Analysis
python run_synthetic.py  # Automatically detects data source

# Specialized Credit Analysis
python run_deals.py [data_source]

# Direct Module Usage
from prepayment_risk_module import PrepaymentRiskEngine, PrepaymentFactors
from credit_risk_module import get_cashflow_attributes, compare_cashflow_modes
from enriched_bond_portfolio import create_enhanced_bond_portfolio

# Prepayment Analysis
engine = PrepaymentRiskEngine()
prepay_speed = engine.calculate_prepayment_speed(loan_factors)
```

### Key Configuration Options
- **Data Source**: Automatically detected (`deals_data/` vs `synthetic_data/`)
- **Simulation Paths**: 10,000 scenarios per loan (configurable)
- **Risk-Free Rate**: Dynamic from FRED EFFR data
- **Advanced Credit Mode**: Fundamental-based vs static assumptions

## Performance Specifications
- **Portfolio Capacity**: 10 private loans (primary) or 150+ synthetic instruments
- **Processing Time**: ~45 seconds for 100,000 Monte Carlo scenarios
- **Memory Usage**: ~300MB for comprehensive analysis
- **Output Generation**: 15+ analysis files per run including plots, cashflows, logs

## Troubleshooting

### Data Issues
```bash
# Verify data structure
ls deals_data/  # Should show enhanced_*.txt files
ls synthetic_data/  # Should show *.csv files
```

### API Configuration
```bash
# Setup OpenAI key for PDF processing
cp config_template.py config.py
# Edit config.py with your API key
```

### Rate Data Issues
```bash
# Download latest FRED data
python download_fred_data.py
```

### Memory Optimization
- Reduce simulation paths in `simulate_loan_paths()` function
- Use synthetic mode for lighter analysis
- Monitor `plots/` directory for output file management

---

*Professional private credit analytics with institutional-quality Monte Carlo simulation, dynamic risk modeling, and comprehensive portfolio optimization.*