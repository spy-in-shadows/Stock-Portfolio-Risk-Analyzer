import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
import time
from datetime import datetime, timedelta
from fastapi import HTTPException
from typing import List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════
# MODULE 1: Holdings CSV Parser
# Extracts tickers and weights from a broker-style export CSV.
# ═══════════════════════════════════════════════════════════════════

def parse_holdings_csv(df: pd.DataFrame) -> Tuple[List[str], List[float]]:
    """
    Parses a broker-style holdings CSV.
    Required columns: 'Ticker', 'Portfolio Weight %'
    Returns: (tickers: List[str], weights: List[float]) — weights are decimals summing to 1.0
    """
    # Normalize column names — strip surrounding whitespace
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = ['Ticker', 'Portfolio Weight %']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Missing required columns: {missing}. "
                f"Your CSV has: {list(df.columns)}. "
                "Ensure 'Ticker' and 'Portfolio Weight %' columns are present."
            )
        )

    # Work only with the two critical columns
    holdings = df[['Ticker', 'Portfolio Weight %']].copy()

    # Sanitize tickers
    holdings['Ticker'] = holdings['Ticker'].astype(str).str.strip().str.upper()
    holdings = holdings[holdings['Ticker'].notna() & (holdings['Ticker'] != '') & (holdings['Ticker'] != 'NAN')]

    # Coerce weights to numeric and drop unparseable rows
    holdings['weight'] = pd.to_numeric(holdings['Portfolio Weight %'], errors='coerce')
    holdings = holdings.dropna(subset=['weight'])

    # Drop zero or negative weight positions
    holdings = holdings[holdings['weight'] > 0]

    if holdings.empty:
        raise HTTPException(
            status_code=400,
            detail="No valid holdings found. Ensure 'Portfolio Weight %' has positive numeric values."
        )

    # Aggregate duplicate tickers by summing weights
    holdings = holdings.groupby('Ticker', as_index=False)['weight'].sum()

    if len(holdings) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"At least 2 assets required. Got {len(holdings)} unique ticker(s) after parsing."
        )

    # Normalize to decimal weights summing to exactly 1.0
    holdings['weight_decimal'] = holdings['weight'] / holdings['weight'].sum()

    tickers = holdings['Ticker'].tolist()
    weights = holdings['weight_decimal'].round(8).tolist()

    return tickers, weights


# ═══════════════════════════════════════════════════════════════════
# MODULE 2: Multi-Ticker Historical Price Fetcher
# Fetches adjusted close prices for a list of tickers via yfinance.
# ═══════════════════════════════════════════════════════════════════

def _build_yf_ticker(ticker: str) -> str:
    """
    Maps a bare ticker symbol to its Yahoo Finance equivalent.
    - Already has '.' or starts with '^' → use as-is
    - Otherwise → append '.NS' (NSE India default)
    Zero API calls — pure string logic.
    """
    if '.' in ticker or ticker.startswith('^'):
        return ticker
    return ticker + '.NS'


def fetch_price_history(
    tickers: List[str],
    start_date: str,
    end_date: str
) -> tuple:
    """
    Fetches historical adjusted close prices for all tickers in ONE batch call.
    Auto-appends .NS suffix for bare Indian tickers (no probe calls).
    Returns: (price_df, resolved_map)
      - price_df: DatetimeIndex × original_ticker columns
      - resolved_map: {original_ticker: yf_ticker}
    """
    # Step 1: Build YF ticker list (zero API calls)
    resolved_map = {t: _build_yf_ticker(t) for t in tickers}
    yf_tickers   = [resolved_map[t] for t in tickers]

    # Step 2: Single batch download with retry
    max_retries = 4
    df = None
    for attempt in range(max_retries):
        if attempt > 0:
            time.sleep(2 ** (attempt - 1))   # 1s, 2s, 4s gaps
        df = yf.download(
            yf_tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True
        )
        if df is not None and not df.empty:
            break

    if df is None or df.empty:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not fetch market data for: {yf_tickers}. "
                "Yahoo Finance may be rate-limiting. Please wait ~30 seconds and retry."
            )
        )

    # Step 3: Extract price column (handle yfinance 0.2.x MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0)
        price_df = df['Close'] if 'Close' in level0 else df['Adj Close']
    else:
        price_df = df[['Close']] if 'Close' in df.columns else df[['Adj Close']]

    if isinstance(price_df, pd.Series):
        price_df = price_df.to_frame(name=yf_tickers[0])

    # Step 4: Rename columns from YF tickers back to original names
    reverse_map = {v: k for k, v in resolved_map.items()}
    price_df.columns = [reverse_map.get(str(c).strip(), str(c).strip()) for c in price_df.columns]

    # Step 5: Select and reorder to match original tickers order
    missing = [t for t in tickers if t not in price_df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No price data returned for: {missing} "
                f"(tried as: {[resolved_map[t] for t in missing]}). "
                "Check that the ticker exists on Yahoo Finance."
            )
        )
    price_df = price_df[tickers]

    # Step 6: Guard against all-NaN columns (silent rate-limit response)
    all_nan_cols = [c for c in price_df.columns if price_df[c].isna().all()]
    if all_nan_cols:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Received empty data for: {all_nan_cols}. "
                "Yahoo Finance may be rate-limiting. Wait ~30 seconds and retry."
            )
        )

    # Step 7: Forward-fill small holiday gaps then drop residual NaNs
    price_df = price_df.ffill().dropna()

    if len(price_df) < 60:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Only {len(price_df)} aligned trading days. "
                "Minimum 60 required. Try increasing history_days."
            )
        )

    return price_df, resolved_map




# ═══════════════════════════════════════════════════════════════════
# MODULE 3: Benchmark Returns Fetcher
# Single ticker benchmark via yfinance, with retry & column guards.
# ═══════════════════════════════════════════════════════════════════

def fetch_benchmark_returns(ticker: str, start_date: str, end_date: str) -> pd.Series:
    """
    Fetches adjusted close prices for a benchmark ticker.
    Returns daily percentage returns as a named pd.Series.
    """
    try:
        max_retries = 3
        df = None

        for attempt in range(max_retries):
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True
            )
            if df is not None and not df.empty:
                break
            wait = 2 ** attempt
            time.sleep(wait)

        if df is None or df.empty:
            raise ValueError(
                f"No data returned for benchmark '{ticker}'. "
                "Yahoo Finance may be rate-limiting. Wait and retry."
            )

        # Robust column extraction
        if isinstance(df.columns, pd.MultiIndex):
            level0 = df.columns.get_level_values(0)
            if 'Close' in level0:
                price_series = df['Close'].iloc[:, 0]
            elif 'Adj Close' in level0:
                price_series = df['Adj Close'].iloc[:, 0]
            else:
                raise ValueError(f"No price column found. Available: {list(level0.unique())}")
        else:
            if 'Close' in df.columns:
                price_series = df['Close']
            elif 'Adj Close' in df.columns:
                price_series = df['Adj Close']
            else:
                raise ValueError(f"No price column found. Available: {list(df.columns)}")

        if isinstance(price_series, pd.DataFrame):
            price_series = price_series.iloc[:, 0]

        price_series = price_series.dropna()

        if price_series.empty:
            raise ValueError(f"Benchmark '{ticker}' price series is empty after cleaning.")

        returns = price_series.pct_change().dropna()
        returns.name = "Benchmark_Returns"
        return returns

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Benchmark fetch failed for '{ticker}': {str(e)}"
        )


# ═══════════════════════════════════════════════════════════════════
# MODULE 4: Broker CSV → Risk-Ready Data Pipeline
# Orchestrates parse → fetch → align
# ═══════════════════════════════════════════════════════════════════

def preprocess_broker_data(
    df: pd.DataFrame,
    benchmark_ticker: str,
    history_days: int = 365
):
    """
    Full pipeline: parse broker CSV → fetch prices → fetch benchmark → align.
    Returns: (asset_returns, bench_returns, weights, asset_names)
    """
    # Step 1: Parse holdings from the broker CSV
    tickers, weights = parse_holdings_csv(df)

    # Step 2: Define date range for historical data fetch
    end_date   = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=history_days)).strftime('%Y-%m-%d')

    # Step 3: Fetch historical portfolio asset prices
    price_df, resolved_map = fetch_price_history(tickers, start_date, end_date)


    # Step 4: Fetch benchmark returns across the same date range
    bench_returns = fetch_benchmark_returns(benchmark_ticker, start_date, end_date)

    # Step 5: Compute asset returns
    asset_returns = price_df.pct_change().dropna()

    # Step 6: Align asset returns with benchmark returns (inner join on common dates)
    combined = pd.concat([asset_returns, bench_returns], axis=1).dropna()

    if len(combined) < 60:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Only {len(combined)} overlapping trading days between portfolio and benchmark. "
                "Minimum 60 required."
            )
        )

    final_asset_returns = combined.drop(columns=["Benchmark_Returns"])
    final_bench_returns = combined["Benchmark_Returns"]

    # Reorder weights to match final asset columns (after any column drops in alignment)
    final_tickers  = final_asset_returns.columns.tolist()
    ticker_to_weight = dict(zip(tickers, weights))
    final_weights = [ticker_to_weight[t] for t in final_tickers]

    # Re-normalize in case any tickers were dropped  
    total = sum(final_weights)
    final_weights = [w / total for w in final_weights]

    return final_asset_returns, final_bench_returns, final_weights, final_tickers


# ═══════════════════════════════════════════════════════════════════
# MODULE 5: Core Risk Computation Engine (Unchanged)
# All formula-level math lives here.
# ═══════════════════════════════════════════════════════════════════

def get_risk_metrics(
    asset_returns: pd.DataFrame,
    bench_returns: pd.Series,
    benchmark_ticker: str,
    asset_names: List[str],
    weights: List[float],
    risk_free_rate: float = 0.0,
    confidence_level: float = 0.95,
    simulations: int = 10000
) -> dict:
    """
    Vectorized portfolio risk engine.
    Computes: volatility, Sharpe, VaR (3 methods), Beta, correlation matrix, MC paths.
    """
    weights = np.array(weights)

    # Ensure weights sum to 1.0
    if not np.isclose(weights.sum(), 1.0):
        weights = weights / weights.sum()

    # ── Statistical Components ──────────────────────────────────────
    mean_returns = asset_returns.mean()
    cov_matrix   = asset_returns.cov()
    corr_matrix  = asset_returns.corr()

    # ── Portfolio Historical Returns: Rp = w · R ───────────────────
    port_hist_returns = np.dot(asset_returns.values, weights)

    # ── Performance Metrics ─────────────────────────────────────────
    port_return = float(np.dot(weights, mean_returns))
    port_vol    = float(np.sqrt(weights.T @ cov_matrix.values @ weights))
    r_f_daily   = risk_free_rate / 252
    sharpe      = (port_return - r_f_daily) / port_vol if port_vol > 0 else 0.0

    # ── Value at Risk ───────────────────────────────────────────────
    hist_var      = float(np.percentile(port_hist_returns, (1 - confidence_level) * 100))
    z_score       = norm.ppf(1 - confidence_level)
    parametric_var = port_return + z_score * port_vol

    # ── Monte Carlo 30-Day Simulation ──────────────────────────────
    mc = monte_carlo_simulation(
        mean=mean_returns.values,
        cov=cov_matrix.values,
        weights=weights,
        simulations=simulations,
        horizon_days=30
    )

    # ── Monte Carlo Forward Paths (for chart) ───────────────────────
    mc_chart_data = generate_mc_forward_paths(
        mean=mean_returns.values,
        cov=cov_matrix.values,
        weights=weights,
        horizon=31,
        path_simulations=5000
    )

    # ── Beta = Cov(Rp, Rm) / Var(Rm) ───────────────────────────────
    bench_arr = bench_returns.values
    cov_mat   = np.cov(port_hist_returns, bench_arr)
    cov_rp_rm = cov_mat[0, 1]
    var_rm    = np.var(bench_arr)
    beta      = cov_rp_rm / var_rm if var_rm > 0 else 1.0

    return {
        # Empirical daily metrics (from historical data)
        "portfolio_expected_return":       port_return,
        "portfolio_volatility":            port_vol,
        "sharpe_ratio":                    float(sharpe),
        "historical_var_95":               hist_var,
        "parametric_var_95":               float(parametric_var),
        # Monte Carlo 30-day compounded metrics
        "monte_carlo_var_95":              mc["var95"],        # Alias for risk gauge compat
        "monte_carlo_expected_return_30d": mc["expected"],
        "monte_carlo_volatility_30d":      mc["volatility"],
        "monte_carlo_sharpe_30d":          mc["sharpe"],
        "monte_carlo_var95_30d":           mc["var95"],
        # Meta
        "beta":                            float(beta),
        "correlation_matrix":              corr_matrix.values.tolist(),
        "asset_names":                     asset_names,
        "benchmark_ticker":                benchmark_ticker,
        "mc_chart_data":                   mc_chart_data
    }




# ═══════════════════════════════════════════════════════════════════
# MODULE 6A: Monte Carlo Simulation — 30-Day Compounded
# Proper multi-day compounding via multivariate_normal sampling.
# ═══════════════════════════════════════════════════════════════════

def monte_carlo_simulation(
    mean: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    simulations: int = 10000,
    horizon_days: int = 30
) -> dict:
    """
    Runs a proper multi-day Monte Carlo simulation.

    Step 1: Sample (simulations, horizon_days, num_assets) daily returns
            from multivariate_normal(mean, cov).
    Step 2: Compute portfolio daily returns via einsum.
    Step 3: Compound across horizon_days → (simulations,) end-of-period returns.
    Step 4: Extract expected, volatility, Sharpe, VaR95 from the terminal distribution.

    No sqrt(T) scaling. No 1-day shortcut. Pure compounding.
    """
    # Step 1: Simulated daily asset returns — shape (simulations, horizon_days, num_assets)
    simulated_daily = np.random.multivariate_normal(
        mean=mean,
        cov=cov,
        size=(simulations, horizon_days)       # ← correct shape
    )

    # Step 2: Portfolio daily returns — shape (simulations, horizon_days)
    portfolio_daily = np.einsum('ijk,k->ij', simulated_daily, weights)

    # Step 3: 30-day compounded terminal return — shape (simulations,)
    # prod(1 + r_t) - 1 over the horizon for each simulation path
    simulated_end_returns = np.prod(1 + portfolio_daily, axis=1) - 1

    # Step 4: Terminal distribution statistics
    expected   = float(np.mean(simulated_end_returns))
    volatility = float(np.std(simulated_end_returns))
    sharpe     = expected / volatility if volatility > 0 else 0.0
    var95      = float(np.percentile(simulated_end_returns, 5))

    return {
        "expected":   expected,
        "volatility": volatility,
        "sharpe":     sharpe,
        "var95":      var95
    }


# ═══════════════════════════════════════════════════════════════════
# MODULE 6B: Monte Carlo Forward Path Generator (Chart Visualization)
# Produces per-day percentile bands using the same multivariate_normal.
# ═══════════════════════════════════════════════════════════════════

def generate_mc_forward_paths(
    mean: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    horizon: int = 31,
    path_simulations: int = 5000
) -> list:
    """
    Multi-step compounding GBM paths for chart visualization.
    Uses multivariate_normal directly — no Cholesky tensordot.
    Returns a list of (horizon+1) dicts, one per day T+0 to T+horizon.
    """
    # Daily asset returns — shape (path_simulations, horizon, num_assets)
    simulated_daily = np.random.multivariate_normal(
        mean=mean,
        cov=cov,
        size=(path_simulations, horizon)
    )

    # Portfolio daily returns — shape (path_simulations, horizon)
    portfolio_daily = np.einsum('ijk,k->ij', simulated_daily, weights)

    # Cumulative compounding — shape (path_simulations, horizon+1)
    # Column 0 = 100 (start), columns 1..horizon = compounded values
    ones_col    = np.ones((path_simulations, 1))
    cum_factors = np.cumprod(np.hstack([ones_col, 1 + portfolio_daily]), axis=1)
    cumulative  = cum_factors * 100.0    # Base portfolio value = 100

    result = []
    for t in range(horizon + 1):
        day_vals = cumulative[:, t]             # shape (path_simulations,)
        p5, p10, p50, p90 = np.percentile(day_vals, [5, 10, 50, 90])

        result.append({
            "day":    t,
            "path1": round(float(day_vals[int(np.argmin(np.abs(day_vals - p50)))]), 4),  # Median path
            "path2": round(float(day_vals[int(np.argmin(np.abs(day_vals - p10)))]), 4),  # Stress (P10)
            "path3": round(float(day_vals[int(np.argmin(np.abs(day_vals - p90)))]), 4),  # Bull   (P90)
            "median": round(float(p50), 4),
            "var95":  round(float(p5), 4),
            "confidenceBand": [round(float(p10), 4), round(float(p90), 4)]
        })

    return result