import numpy as np
import pandas as pd
from scipy.stats import norm

def preprocess_data(df):
    """
    Standardizes CSV data:
    1. Sets Date as index
    2. Sorts by Date
    3. Handles missing values
    4. Separates Assets from Benchmark (last column)
    """
    # Assume first column is Date
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df = df.sort_values(by=df.columns[0])
    df = df.set_index(df.columns[0])
    
    # Handle NaNs
    df = df.ffill().dropna()
    
    # Separate Benchmark (last column)
    benchmark_col = df.columns[-1]
    assets_df = df.iloc[:, :-1]
    benchmark_series = df[benchmark_col]
    
    return assets_df, benchmark_series

def calculate_returns(df):
    return df.pct_change().dropna()

def get_risk_metrics(assets_df, benchmark_series, weights=None, risk_free_rate=0.0, confidence_level=0.95, simulations=10000):
    """
    Core math engine for portfolio risk analytics.
    """
    # 1. Compute Returns
    asset_returns = calculate_returns(assets_df)
    bench_returns = calculate_returns(benchmark_series.to_frame())
    
    # Alignment (ensure return series match)
    combined = pd.concat([asset_returns, bench_returns], axis=1).dropna()
    asset_returns = combined.iloc[:, :-1]
    bench_returns = combined.iloc[:, -1]
    
    num_assets = asset_returns.shape[1]
    
    # 2. Weights Handling
    if weights is None:
        weights = np.array([1.0 / num_assets] * num_assets)
    else:
        weights = np.array(weights)
        if len(weights) != num_assets:
            raise ValueError(f"Weight length {len(weights)} does not match asset count {num_assets}")
        if not np.isclose(np.sum(weights), 1.0):
            weights = weights / np.sum(weights) # Normalize to 1.0

    # 3. Statistical Components
    mean_returns = asset_returns.mean()
    cov_matrix = asset_returns.cov()
    corr_matrix = asset_returns.corr()
    
    # 4. Portfolio Performance
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (port_return - (risk_free_rate / 252)) / port_vol if port_vol > 0 else 0
    
    # 5. Historical VaR (95%)
    port_hist_returns = np.dot(asset_returns, weights)
    hist_var = np.percentile(port_hist_returns, (1 - confidence_level) * 100)
    
    # 6. Parametric VaR (Variance-Covariance)
    z_score = norm.ppf(1 - confidence_level)
    parametric_var = port_return + z_score * port_vol
    
    # 7. Monte Carlo VaR
    # Cholesky Decomposition for correlation preservation
    try:
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        # Handle non-positive definite matrix by adding small jitter to diagonal
        clean_cov = cov_matrix + np.eye(len(cov_matrix)) * 1e-8
        L = np.linalg.cholesky(clean_cov)
        
    # Generate Correlated Random Normals
    Z = np.random.standard_normal((simulations, num_assets))
    simulated_returns = mean_returns.values + np.dot(Z, L.T)
    
    # Map to Portfolio returns
    port_sim_returns = np.dot(simulated_returns, weights)
    mc_var = np.percentile(port_sim_returns, (1 - confidence_level) * 100)
    
    # 8. Beta Calculation
    # Beta = Cov(Rp, Rm) / Var(Rm)
    covariance_with_bench = np.cov(port_hist_returns, bench_returns)[0, 1]
    bench_vol = np.var(bench_returns)
    beta = covariance_with_bench / bench_vol if bench_vol > 0 else 1.0
    
    return {
        "portfolio_expected_return": float(port_return),
        "portfolio_volatility": float(port_vol),
        "sharpe_ratio": float(sharpe),
        "historical_var_95": float(hist_var),
        "parametric_var_95": float(parametric_var),
        "monte_carlo_var_95": float(mc_var),
        "beta": float(beta),
        "correlation_matrix": corr_matrix.values.tolist(),
        "asset_names": assets_df.columns.tolist()
    }