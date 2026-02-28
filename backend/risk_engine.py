import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
from fastapi import HTTPException

def fetch_benchmark_from_yfinance(ticker, start_date, end_date):
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
        raise HTTPException(status_code=400, detail=f"Failed to fetch benchmark data from yfinance: {str(e)}")

def preprocess_hybrid_data(df, benchmark_ticker=None):
    """
    Implements Hybrid Benchmark Logic:
    1. Check for 'Benchmark' column in CSV.
    2. If not found, fetch from yfinance using ticker.
    3. Aligns dates and filters assets.
    """
    # 1. Standardize Dates
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df = df.sort_values(by=df.columns[0])
    df = df.set_index(df.columns[0])
    df = df.ffill().dropna()

    benchmark_source = None
    benchmark_series = None
    
    # 2. Case-insensitive search for 'Benchmark' column
    benchmark_col = next((c for c in df.columns if c.lower() == 'benchmark'), None)
    
    if benchmark_col:
        # CSV-based Benchmark (Primary)
        benchmark_series = df[benchmark_col]
        assets_df = df.drop(columns=[benchmark_col])
        benchmark_source = "csv"
    elif benchmark_ticker:
        # yfinance Fallback
        start_date = df.index.min().strftime('%Y-%m-%d')
        # Add 1 day to end date to ensure full coverage
        end_date = (df.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        
        benchmark_series = fetch_benchmark_from_yfinance(benchmark_ticker, start_date, end_date)
        assets_df = df
        benchmark_source = "yfinance"
    else:
        # Neither provided
        raise HTTPException(
            status_code=400, 
            detail="Zero Benchmark inputs. CSV must contain a 'Benchmark' column or provide a 'benchmark_ticker'."
        )

    # 3. Clean and Align series
    # Convert Asset prices to returns, then align with benchmark
    asset_returns = assets_df.pct_change().dropna()
    bench_returns = benchmark_series.to_frame().pct_change().dropna()
    
    # Force column name for benchmark for alignment
    bench_returns.columns = ['Benchmark_Returns']
    
    # Alignment: Combine all to shared dates
    combined = pd.concat([asset_returns, bench_returns], axis=1).dropna()
    
    if len(combined) < 30:
        raise HTTPException(
            status_code=400, 
            detail=f"Sample size too small after date alignment ({len(combined)} rows). Minimum 30 rows required."
        )
    
    # Final data extraction
    final_asset_returns = combined.drop(columns=['Benchmark_Returns'])
    final_bench_returns = combined['Benchmark_Returns']
    
    return final_asset_returns, final_bench_returns, benchmark_source, assets_df.columns.tolist()

def get_risk_metrics(asset_returns, bench_returns, benchmark_source, asset_names, weights=None, risk_free_rate=0.0, confidence_level=0.95, simulations=10000):
    """
    Updated computation engine following exact Beta formula: Cov(Rp, Rm) / Var(Rm).
    """
    num_assets = asset_returns.shape[1]
    
    # 1. Weights Handling
    if weights is None:
        weights = np.array([1.0 / num_assets] * num_assets)
    else:
        weights = np.array(weights)
        if len(weights) != num_assets:
            raise ValueError(f"Weight length {len(weights)} does not match asset count {num_assets}")
        # Ensure weights sum to 1.0
        weights = weights / np.sum(weights)

    # 2. Components
    mean_returns = asset_returns.mean()
    cov_matrix = asset_returns.cov()
    corr_matrix = asset_returns.corr()
    
    # 3. Portfolio Return Series (Rp)
    # Required for Beta computation
    port_hist_returns = np.dot(asset_returns, weights)
    
    # 4. Performance Metrics
    # Note: Using daily metrics, sharpe adjusted for risk free rate
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    r_f_daily = risk_free_rate / 252
    sharpe = (port_return - r_f_daily) / port_vol if port_vol > 0 else 0
    
    # 5. Value at Risk (VaR)
    # Historical
    hist_var = np.percentile(port_hist_returns, (1 - confidence_level) * 100)
    
    # Parametric
    z_score = norm.ppf(1 - confidence_level)
    parametric_var = port_return + z_score * port_vol
    
    # Monte Carlo (with Cholesky)
    try:
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        # Standard fallback for non-PSD matrix
        clean_cov = cov_matrix + np.eye(len(cov_matrix)) * 1e-9
        L = np.linalg.cholesky(clean_cov)
        
    Z = np.random.standard_normal((simulations, num_assets))
    sim_asset_returns = mean_returns.values + np.dot(Z, L.T)
    port_sim_returns = np.dot(sim_asset_returns, weights)
    mc_var = np.percentile(port_sim_returns, (1 - confidence_level) * 100)
    
    # 6. Beta Computation (Strict Cov/Var Formula)
    # Beta = Cov(Rp, Rm) / Var(Rm)
    # port_hist_returns is Rp
    # bench_returns is Rm
    cov_rp_rm = np.cov(port_hist_returns, bench_returns)[0, 1]
    var_rm = np.var(bench_returns)
    
    beta = cov_rp_rm / var_rm if var_rm != 0 else 1.0
    
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
        "benchmark_source": benchmark_source
    }