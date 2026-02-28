import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
from fastapi import HTTPException
from typing import List, Optional

def fetch_benchmark_from_yfinance(ticker: str, start_date: str, end_date: str) -> pd.Series:
    """
    Fetches historical adjusted close prices for a given ticker or index.
    """
    try:
        # Fetching data using yfinance
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
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
            
        return benchmark_series.dropna()
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch benchmark data for {ticker}: {str(e)}")

def build_blended_benchmark_returns(
    tickers: List[str],
    weights: List[float],
    start_date: str,
    end_date: str
) -> pd.Series:
    """
    Fetches and blends multiple index returns based on weights.
    """
    returns_list = []
    
    for ticker in tickers:
        prices = fetch_benchmark_from_yfinance(ticker, start_date, end_date)
        returns = prices.pct_change().dropna()
        returns.name = ticker
        returns_list.append(returns)
        
    # Align all return series
    combined_bench = pd.concat(returns_list, axis=1).dropna()
    
    if combined_bench.empty:
        raise HTTPException(status_code=400, detail="No overlapping dates found for the specified benchmark tickers.")

    # Apply weights and sum
    weights_arr = np.array(weights)
    blended_returns = (combined_bench * weights_arr).sum(axis=1)
    
    return blended_returns

def preprocess_blended_data(
    df: pd.DataFrame, 
    benchmark_tickers: Optional[List[str]] = None, 
    benchmark_weights: Optional[List[float]] = None
):
    """
    Implements Blended Benchmark Logic.
    """
    # 1. Standardize Dates
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df = df.sort_values(by=df.columns[0])
    df = df.set_index(df.columns[0])
    df = df.ffill().dropna()

    benchmark_type = "single"
    benchmark_source = None
    bench_returns_series = None
    benchmark_components = None
    
    # 2. Benchmark Logic
    # A) Check CSV for 'Benchmark' column
    benchmark_col = next((c for c in df.columns if c.lower() == 'benchmark'), None)
    
    start_date_str = df.index.min().strftime('%Y-%m-%d')
    end_date_str = (df.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    if benchmark_col:
        # CSV-based Benchmark
        bench_prices = df[benchmark_col]
        bench_returns_series = bench_prices.pct_change().dropna()
        assets_df = df.drop(columns=[benchmark_col])
        benchmark_source = "csv"
    elif benchmark_tickers and benchmark_weights:
        # Blended yfinance Benchmark
        if len(benchmark_tickers) != len(benchmark_weights):
            raise HTTPException(status_code=400, detail="Benchmark tickers and weights must have the same length.")
        
        if not np.isclose(np.sum(benchmark_weights), 1.0, atol=1e-6):
            raise HTTPException(status_code=400, detail="Benchmark weights must sum to 1.0.")

        bench_returns_series = build_blended_benchmark_returns(benchmark_tickers, benchmark_weights, start_date_str, end_date_str)
        assets_df = df
        benchmark_source = "yfinance"
        benchmark_type = "blended"
        benchmark_components = [{"ticker": t, "weight": w} for t, w in zip(benchmark_tickers, benchmark_weights)]
    elif benchmark_tickers: # Single ticker optimization
        # Handle as single ticker
        bench_prices = fetch_benchmark_from_yfinance(benchmark_tickers[0], start_date_str, end_date_str)
        bench_returns_series = bench_prices.pct_change().dropna()
        assets_df = df
        benchmark_source = "yfinance"
        benchmark_type = "single"
        benchmark_components = [{"ticker": benchmark_tickers[0], "weight": 1.0}]
    else:
        raise HTTPException(
            status_code=400, 
            detail="No benchmark provided. Provide a 'Benchmark' column in CSV or 'benchmark_tickers'."
        )

    # 3. Clean and Align
    asset_returns = assets_df.pct_change().dropna()
    
    # Align dates
    bench_returns_df = bench_returns_series.to_frame(name='Benchmark_Returns')
    combined = pd.concat([asset_returns, bench_returns_df], axis=1).dropna()
    
    if len(combined) < 30:
        raise HTTPException(
            status_code=400, 
            detail=f"Sample size too small after alignment ({len(combined)} observations). Minimum 30 required."
        )
    
    final_asset_returns = combined.drop(columns=['Benchmark_Returns'])
    final_bench_returns = combined['Benchmark_Returns']
    
    return final_asset_returns, final_bench_returns, benchmark_source, benchmark_type, benchmark_components, assets_df.columns.tolist()

def get_risk_metrics(
    asset_returns: pd.DataFrame, 
    bench_returns: pd.Series, 
    benchmark_source: str, 
    benchmark_type: str,
    benchmark_components: Optional[List[dict]],
    asset_names: List[str], 
    weights: Optional[List[float]] = None, 
    risk_free_rate: float = 0.0, 
    confidence_level: float = 0.95, 
    simulations: int = 10000
):
    """
    Computation engine for blended benchmark beta and tail risk.
    """
    num_assets = asset_returns.shape[1]
    
    if weights is None:
        weights = np.array([1.0 / num_assets] * num_assets)
    else:
        weights = np.array(weights)
        weights = weights / np.sum(weights)

    mean_returns = asset_returns.mean()
    cov_matrix = asset_returns.cov()
    corr_matrix = asset_returns.corr()
    
    port_hist_returns = np.dot(asset_returns, weights)
    
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    r_f_daily = risk_free_rate / 252
    sharpe = (port_return - r_f_daily) / port_vol if port_vol > 0 else 0
    
    hist_var = np.percentile(port_hist_returns, (1 - confidence_level) * 100)
    
    z_score = norm.ppf(1 - confidence_level)
    parametric_var = port_return + z_score * port_vol
    
    try:
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        clean_cov = cov_matrix + np.eye(len(cov_matrix)) * 1e-9
        L = np.linalg.cholesky(clean_cov)
        
    Z = np.random.standard_normal((simulations, num_assets))
    sim_asset_returns = mean_returns.values + np.dot(Z, L.T)
    port_sim_returns = np.dot(sim_asset_returns, weights)
    mc_var = np.percentile(port_sim_returns, (1 - confidence_level) * 100)
    
    # Beta = Cov(Rp, Rb) / Var(Rb)
    cov_rp_rb = np.cov(port_hist_returns, bench_returns)[0, 1]
    var_rb = np.var(bench_returns)
    beta = cov_rp_rb / var_rb if var_rb != 0 else 1.0
    
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
        "benchmark_source": benchmark_source,
        "benchmark_type": benchmark_type,
        "benchmark_components": benchmark_components
    }