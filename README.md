# Asset Management Backtesting Notebook

Jupyter notebook for an asset management assignment: portfolio construction + backtesting, performance statistics, turnover, and factor-model evaluation (CAPM and Fama–French 5-factor).  

## What’s inside
The notebook includes:
- Utility functions for performance analytics: annualized return/volatility, Sharpe ratio, drawdowns, skewness, kurtosis, and summary statistics.  
- Rolling backtest engine (`backtest_ws`) to evaluate portfolio weighting schemes over time.  
- Portfolio strategies on a 6-country universe: Equal Weight, Risk Parity, 60/40 with risk-free, and a volatility-timing allocation overlay.  
- Stock portfolio exercise using tickers and ESG scores, including an “EW” portfolio and a “Green” portfolio tilted toward higher ESG scores.  
- Performance attribution / regression against CAPM and Fama–French 5 factors; alpha, tracking error, and information ratio.  

## Repository files
- `55724.ipynb` — main notebook with code + discussion.  
- `am2023.py` — helper code (if used/imported in your workflow).  
- `F-F_Research_Data_Factors.CSV` — Fama–French factor data (local copy).  

## Data dependencies (important)
The notebook also references local Excel files (e.g., *Countries.xlsx*, *Tickers*, *ESG-score*) using absolute paths and will error if those files are not present.  
To make this repo reproducible, either:
- Add the required Excel files to a `/data` folder and update paths in the notebook, or
- Replace Excel inputs with included CSVs (or download links) and document the process.

## Setup
### Requirements
Python packages used in the notebook include:
- `numpy`, `pandas`, `scipy`, `matplotlib`
- `statsmodels`
- `pandas-datareader`
- `openpyxl` (for Excel)
- `yfinance` (for stock return retrieval)

Install (example):
```bash
pip install numpy pandas scipy matplotlib statsmodels pandas-datareader openpyxl yfinance
