import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
from fastapi import HTTPException
from typing import List, Optional

def fetch_benchmark_returns(ticker: str, start_date: str, end_date: str) -> pd.Series:
    """
    Fetches historical adjusted close prices for a given ticker and returns daily percentage returns.
    """
    try:
        # Fetching data using yfinance
        # Use a slightly wider buffer to ensure returns can be calculated for the first date
        start_buffer = (pd.to_datetime(start_date) - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        df = yf.download(ticker, start=start_buffer, end=end_date, progress=False)
        
        if df.empty:
            raise ValueError(f"No price data found for benchmark ticker: {ticker}")
        
        # Extract 'Adj Close'
        if 'Adj Close' in df:
            benchmark_series = df['Adj Close']
        else:
            # Check for MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                if 'Adj Close' in df.columns.get_level_values(0):
                    benchmark_series = df['Adj Close']
                else:
                    benchmark_series = df['Close']
            else:
                benchmark_series = df['Close']
                
        # Handle cases where multiple tickers were somehow downloaded (flatten)
        if isinstance(benchmark_series, pd.DataFrame):
            benchmark_series = benchmark_series.iloc[:, 0]
            
        # Calculate returns and drop NaNs
        returns = benchmark_series.pct_change().dropna()
        returns.name = "Benchmark_Returns"
        return returns
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch benchmark data for {ticker}: {str(e)}")

def preprocess_portfolio_data(df: pd.DataFrame, benchmark_ticker: str):
    """
    Standardizes CSV data (assets only) and fetches benchmark returns.
    Aligns both series by date.
    """
    # 1. Identify and parse the Date column (always the first column)
    date_col_name = df.columns[0]
    
    # Convert to datetime, coerce errors to NaT
    df[date_col_name] = pd.to_datetime(df[date_col_name], errors='coerce')
    
    # Drop any rows where the date couldn't be parsed
    df = df.dropna(subset=[date_col_name])
    
    # Sort by date ascending
    df = df.sort_values(by=date_col_name)
    
    # Set date as index and remove from columns
    df = df.set_index(date_col_name)
    
    # Convert all asset columns to numeric, drop non-numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Forward fill small gaps, then drop remaining NaNs
    df = df.ffill().dropna()
    
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV is empty after cleaning. Check your date and price columns.")

    # 2. Extract scalar start/end dates
    # Use Python's built-in string conversion to guarantee scalar str output
    start_date = str(df.index.min().date())
    end_date   = str((df.index.max() + pd.Timedelta(days=1)).date())
    
    # Capture asset names before any transformation
    asset_names = df.columns.tolist()
    
    # 3. Calculate Asset Returns
    asset_returns = df.pct_change().dropna()
    
    # 4. Fetch Benchmark Returns from yfinance
    bench_returns = fetch_benchmark_returns(benchmark_ticker, start_date, end_date)
    
    # 5. Date Alignment — inner join on common trading days
    combined = pd.concat([asset_returns, bench_returns], axis=1).dropna()
    
    if len(combined) < 30:
        raise HTTPException(
            status_code=400, 
            detail=f"Sample size too small after date alignment ({len(combined)} overlapping days). Minimum 30 required."
        )
    
    # Separate back into assets and benchmark
    final_asset_returns = combined.drop(columns=["Benchmark_Returns"])
    final_bench_returns = combined["Benchmark_Returns"]
    
    return final_asset_returns, final_bench_returns, asset_names


def get_risk_metrics(
    asset_returns: pd.DataFrame, 
    bench_returns: pd.Series, 
    benchmark_ticker: str,
    asset_names: List[str], 
    weights: Optional[List[float]] = None, 
    risk_free_rate: float = 0.0, 
    confidence_level: float = 0.95, 
    simulations: int = 10000
):
    """
    Computation engine for portfolio risk analytics using standardized Beta formula.
    """
    num_assets = asset_returns.shape[1]
    
    # 1. Weights Handling
    if weights is None:
        weights = np.array([1.0 / num_assets] * num_assets)
    else:
        weights = np.array(weights)
        # Ensure weights sum to 1.0
        if not np.isclose(np.sum(weights), 1.0):
            weights = weights / np.sum(weights)

    # 2. Statistical Components
    mean_returns = asset_returns.mean()
    cov_matrix = asset_returns.cov()
    corr_matrix = asset_returns.corr()
    
    # 3. Portfolio Return Series (Rp)
    # Required for Beta computation: Rp = Sum(wi * Ri)
    port_hist_returns = np.dot(asset_returns, weights)
    
    # 4. Performance Metrics
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Annualized Sharpe (assuming daily returns)
    r_f_daily = risk_free_rate / 252
    sharpe = (port_return - r_f_daily) / port_vol if port_vol > 0 else 0
    
    # 5. Value at Risk (VaR)
    # Historical VaR
    hist_var = np.percentile(port_hist_returns, (1 - confidence_level) * 100)
    
    # Parametric VaR
    z_score = norm.ppf(1 - confidence_level)
    parametric_var = port_return + z_score * port_vol
    
    # Monte Carlo VaR (with Cholesky for correlation preservation)
    try:
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        # Fallback for near-singular matrices
        clean_cov = cov_matrix + np.eye(len(cov_matrix)) * 1e-9
        L = np.linalg.cholesky(clean_cov)
        
    Z = np.random.standard_normal((simulations, num_assets))
    sim_asset_returns = mean_returns.values + np.dot(Z, L.T)
    port_sim_returns = np.dot(sim_asset_returns, weights)
    mc_var = np.percentile(port_sim_returns, (1 - confidence_level) * 100)
    
    # 6. Beta Calculation (Strict Cov/Var Formula)
    # Beta = Cov(Rp, Rm) / Var(Rm)
    # Rp is port_hist_returns, Rm is bench_returns
    covariance_matrix = np.cov(port_hist_returns, bench_returns)
    cov_rp_rm = covariance_matrix[0, 1]
    var_rm = np.var(bench_returns)
    
    beta = cov_rp_rm / var_rm if var_rm > 0 else 1.0
    
    return {
        "portfolio_expected_return": float(port_return),
        "portfolio_volatility": float(port_vol),
        "sharpe_ratio": float(sharpe),
        "historical_var_95": float(hist_var),
        "parametric_var_95": float(parametric_var),
        "monte_carlo_var_95": float(mc_var),
        "beta": float(beta),
        "correlation_matrix": corr_matrix.values.tolist(),
        "asset_names": asset_names,
        "benchmark_ticker": benchmark_ticker
    }