# Stock Portfolio Risk Analyzer - Architecture & Implementation Summary

This document serves as a comprehensive "catch-up" guide for any AI reviewing the `Stock Portfolio Risk Analyzer` project. It details the core architectural decisions, the mathematical implementations of our risk models, and the recent refactoring history.

## 1. Project Paradigm Shift: From "Historical Uploads" to "Broker CSV + YFinance"
Historically, this app required users to manually upload a CSV containing years of daily closing prices for every asset in their portfolio. 
**We have completely refactored this.**
- **New Flow:** The user now uploads a simple **"Broker Holdings Export"** CSV containing just `Ticker` and `Portfolio Weight %`.
- **backend/main.py**: The FastAPI `/analyze` endpoint intercepts this CSV and passes it to `risk_engine.py`.
- **YFinance Integration (`fetch_price_history`)**: The backend dynamically fetches the last 365 days of adjusted closing prices for all tickers in the uploaded CSV via the `yfinance` library.
  - *Ticker Resolution:* Bare tickers (like `TCS` or `RELIANCE`) are automatically appended with `.NS` (NSE India) if they lack a suffix, enabling seamless Indian stock market analysis.
  - *Rate Limiting:* We implemented a robust retry mechanism with exponential backoff (up to 4 retries) to handle Yahoo Finance's strict rate limits, alongside `ffill()` and `dropna()` to handle silent NaNs and holiday data gaps.
- **Benchmark Alignment**: The user specifies a benchmark (default `^NSEI`). The engine fetches the benchmark, calculates daily returns for both the portfolio and the bench, and performs an *inner join* to perfectly align trading days before computing Beta and covariance.

## 2. Quantitative Risk Engine (`risk_engine.py`)
This is the mathematical core of the application. It uses vectorized `numpy` and `pandas` operations exclusively (no slow Python loops for math).

### Core Computations:
- **Portfolio Daily Expected Return ($E[R_p]$):** `np.dot(weights, mean_returns)`
- **Portfolio Volatility ($\sigma_p$):** `np.sqrt(weights.T @ cov_matrix @ weights)`
- **Portfolio Beta ($\beta$):** `Cov(R_p, R_m) / Var(R_m)`. We ensure `np.var` uses `ddof=1` (sample variance) to match `np.cov`'s default behavior for accurate Beta scaling.
- **Correlation Matrix:** Standard Pearson correlation shipped to the frontend for the heatmap.
- **Parametric VaR:** $E[R_p] + Z * \sigma_p$
- **Historical VaR:** `np.percentile(historical_returns, 5)`

*(Note: We deliberately stripped out all Sharpe Ratio calculations from the engine and frontend UI to reduce noise and focus strictly on downside tail risk).*

## 3. The Monte Carlo Simulation Engine
We discarded the previous 1-day simplistic Monte Carlo model and rebuilt a rigorous **30-Day Compounded Geometric Brownian Motion (GBM)** simulator using multivariate normal sampling.

### Mathematical Implementation (`monte_carlo_simulation` & `generate_mc_forward_paths`):
1. **Multivariate Normal Sampling:** We use `np.random.multivariate_normal(mean, cov, size=(simulations, horizon_days))` to generate 10,000 distinct paths, each containing 30 days of correlated asset returns.
2. **Portfolio Aggregation:** `np.einsum('ijk,k->ij')` rapidly multiplies the 3D asset return matrix by the 1D weight array to yield a 2D matrix of shape `(simulations, horizon_days)` representing daily portfolio returns.
3. **True Compounding:** Multi-day returns are calculated correctly using `np.prod(1 + R) - 1` rather than simplistic $T \times \mu$ scaling.
4. **Volatility Drag Adjustment:** The arithmetic mean of compounded returns inherently suffers from volatility drag ($-\sigma^2/2$). For the frontend chart (`MonteCarloChart.jsx`), the *Expected* line tracks the **Mean ($E[X]$)** of the simulated paths, NOT the median, as the median gets pulled down heavily by this drag, creating a visually deceptive downward trend in highly volatile portfolios.
5. **Chart Path Rendering (`generate_mc_forward_paths`):** Instead of calculating the 50th percentile *independently on every single day*, the engine specifically identifies the 3 exact simulation runs that *terminated* closest to the 10th (Stress), 50th (Median), and 90th (Bull) percentiles on Day 30. It then returns the historical daily trace of *those three specific paths* to the frontend. This ensures the UI renders realistic, jagged random walks rather than artificially smooth, identical "average" curves.

## 4. Frontend Architecture (`/frontend/src`)
The React frontend (styled with Tailwind CSS and animated with GSAP) was overhauled to handle the new 30-Day metrics.

- **`UploadPortfolio.jsx`**: Rebuilt to accept the new Broker CSV format (Ticker + Weight). Includes dynamic loading states while the backend reaches out to Yahoo Finance.
- **`RiskMetrics.jsx`**: 
  - Precision display: Small expected returns (e.g., `0.06%` daily) are formatted to 4 decimal places before rounding, preventing them from snapping to a false `0.0%`.
  - GSAP `.kill()` resets: The `AnimatedNumber` counter correctly resets its internal tweens on data refresh so numbers don't spasm when uploading a new portfolio.
  - Negative values: The regex in the number formatter was patched to explicitly preserve `-` signs so short-term losses chart correctly.
- **`ScenarioPanel.jsx`**: An interactive stress-testing panel. It takes the backend's computed Portfolio Volatility, Base VaR, and Beta. When the user slides the "Market Drop" or "Asset Shock" sliders, it mathematically calculates theoretical "Stressed VaR" and portfolio P&L impacts in real-time on the client side without needing another API call.
- **`RiskGauge.jsx`**: Re-calibrated the 0-100 scoring needle to handle the new, much larger **30-Day VaR** magnitudes (a 10% 30-day loss is normal, whereas a 10% 1-day loss is catastrophic).

## Current Status
The app is fully stable. Uploading a 2-column CSV immediately routes through YFinance, calculates all 30-day compound parameters, and renders smooth, accurate React charts. No outstanding bugs remain.
