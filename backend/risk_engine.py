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

def _resolve_tickers(tickers: List[str], start_date: str, end_date: str) -> dict:
    """
    Resolves ticker symbols to their correct Yahoo Finance format.
    For each ticker that doesn't return data, tries appending .NS (NSE) then .BO (BSE).
    Returns: dict mapping original_ticker → resolved_ticker
    """
    resolution_map = {}
    unresolved = []

    for ticker in tickers:
        # Already has an exchange suffix — use as-is
        if '.' in ticker or ticker.startswith('^'):
            resolution_map[ticker] = ticker
            continue
        unresolved.append(ticker)

    if not unresolved:
        return resolution_map

    # Try all unresolved tickers in batch first (as-is)
    if unresolved:
        test_df = yf.download(unresolved, start=start_date, end=end_date,
                              progress=False, auto_adjust=True)
        if test_df is not None and not test_df.empty:
            if isinstance(test_df.columns, pd.MultiIndex):
                fetched_cols = set(test_df.columns.get_level_values(1))
            else:
                fetched_cols = set(unresolved)

            resolved_now = [t for t in unresolved if t in fetched_cols]
            still_missing = [t for t in unresolved if t not in fetched_cols]

            for t in resolved_now:
                resolution_map[t] = t
            unresolved = still_missing

    # For anything still unresolved, try .NS then .BO individually
    for ticker in unresolved:
        resolved = None
        for suffix in ['.NS', '.BO']:
            candidate = ticker + suffix
            try:
                test = yf.download(candidate, start=start_date, end=end_date,
                                   progress=False, auto_adjust=True)
                if test is not None and not test.empty:
                    resolved = candidate
                    break
            except Exception:
                continue
        if resolved:
            resolution_map[ticker] = resolved
        else:
            # Cannot resolve — raise with helpful message
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Could not resolve ticker '{ticker}' on Yahoo Finance. "
                    f"Tried: {ticker}, {ticker}.NS, {ticker}.BO. "
                    "Use the full Yahoo Finance ticker symbol (e.g., RELIANCE.NS, TCS.NS)."
                )
            )

    return resolution_map


def fetch_price_history(
    tickers: List[str],
    start_date: str,
    end_date: str
) -> tuple:
    """
    Fetches historical adjusted close prices for all tickers.
    Automatically resolves exchange suffixes (.NS / .BO) if missing.
    Returns: (price_df, resolved_ticker_map)
      - price_df: DatetimeIndex × OriginalTickers (columns)
      - resolved_map: {original_ticker: resolved_ticker}
    Requires at least 60 valid trading days after alignment.
    """
    # Step 1: Resolve all tickers to valid Yahoo Finance symbols
    resolved_map = _resolve_tickers(tickers, start_date, end_date)
    resolved_tickers = [resolved_map[t] for t in tickers]

    # Step 2: Fetch all resolved tickers together with retry
    max_retries = 3
    df = None
    for attempt in range(max_retries):
        df = yf.download(
            resolved_tickers,
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
        raise HTTPException(
            status_code=400,
            detail=(
                f"Failed to fetch historical data even after ticker resolution. "
                "Yahoo Finance may be rate-limiting. Wait a moment and retry."
            )
        )

    # Step 3: Extract price columns (handle MultiIndex from yfinance 0.2.x)
    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0)
        if 'Close' in level0:
            price_df = df['Close']
        elif 'Adj Close' in level0:
            price_df = df['Adj Close']
        else:
            raise HTTPException(
                status_code=400,
                detail=f"No price column found in response. Available: {list(level0.unique())}"
            )
    else:
        if 'Close' in df.columns:
            price_df = df[['Close']]
        elif 'Adj Close' in df.columns:
            price_df = df[['Adj Close']]
        else:
            raise HTTPException(status_code=400, detail="No price column found in downloaded data.")

    if isinstance(price_df, pd.Series):
        price_df = price_df.to_frame(name=resolved_tickers[0])

    # Step 4: Rename columns back to original tickers for consistency
    reverse_map = {v: k for k, v in resolved_map.items()}
    price_df.columns = [reverse_map.get(str(c).strip(), str(c).strip()) for c in price_df.columns]

    # Step 5: Reorder to match original tickers list order
    available = [t for t in tickers if t in price_df.columns]
    missing = [t for t in tickers if t not in price_df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Price data not available for: {missing}. Try adding exchange suffix (.NS / .BO)."
        )
    price_df = price_df[available]

    # Step 6: Forward-fill gaps and validate minimum data
    price_df = price_df.ffill().dropna()

    if len(price_df) < 60:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Only {len(price_df)} valid trading days found. "
                "Minimum 60 required. Increase history_days or check ticker availability."
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

    # ── Monte Carlo VaR (Cholesky for correlation preservation) ─────
    try:
        L = np.linalg.cholesky(cov_matrix.values)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(cov_matrix.values + np.eye(len(cov_matrix)) * 1e-9)

    Z = np.random.standard_normal((simulations, len(weights)))
    sim_returns  = mean_returns.values + np.dot(Z, L.T)
    port_sim_ret = np.dot(sim_returns, weights)
    mc_var       = float(np.percentile(port_sim_ret, (1 - confidence_level) * 100))

    # ── Monte Carlo Forward Paths (for chart) ───────────────────────
    mc_chart_data = generate_mc_forward_paths(
        mean_returns=mean_returns,
        L=L,
        weights=weights,
        num_assets=len(weights),
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
        "portfolio_expected_return": port_return,
        "portfolio_volatility":      port_vol,
        "sharpe_ratio":              float(sharpe),
        "historical_var_95":         hist_var,
        "parametric_var_95":         float(parametric_var),
        "monte_carlo_var_95":        mc_var,
        "beta":                      float(beta),
        "correlation_matrix":        corr_matrix.values.tolist(),
        "asset_names":               asset_names,
        "benchmark_ticker":          benchmark_ticker,
        "mc_chart_data":             mc_chart_data
    }


# ═══════════════════════════════════════════════════════════════════
# MODULE 6: Monte Carlo Forward Path Generator
# Produces 31-day cumulative GBM paths for chart visualization.
# ═══════════════════════════════════════════════════════════════════

def generate_mc_forward_paths(
    mean_returns: pd.Series,
    L: np.ndarray,
    weights: np.ndarray,
    num_assets: int,
    horizon: int = 31,
    path_simulations: int = 5000
) -> list:
    """
    Multi-step forward GBM simulation with Cholesky-correlated shocks.
    Returns a list of 32 dicts (T+0 to T+31) for direct chart consumption.
    """
    # Daily correlated shocks: (horizon, path_simulations, num_assets)
    Z_all = np.random.standard_normal((horizon, path_simulations, num_assets))
    daily_asset_ret = mean_returns.values + np.tensordot(Z_all, L.T, axes=([2], [0]))

    # Portfolio daily returns: (horizon, path_simulations)
    daily_port_ret = np.dot(daily_asset_ret, weights)

    # Cumulative compounding: (horizon+1, path_simulations), base = 100
    cumulative = np.vstack([
        np.full((1, path_simulations), 100.0),
        np.cumprod(1 + daily_port_ret, axis=0) * 100
    ])

    result = []
    for t in range(horizon + 1):
        day_vals = cumulative[t]
        p10, p50, p90 = np.percentile(day_vals, [10, 50, 90])

        result.append({
            "day": t,
            "path1": round(float(day_vals[int(np.argmin(np.abs(day_vals - p50)))]), 4),  # Median
            "path2": round(float(day_vals[int(np.argmin(np.abs(day_vals - p10)))]), 4),  # Stress
            "path3": round(float(day_vals[int(np.argmin(np.abs(day_vals - p90)))]), 4),  # Bull
            "median": round(float(p50), 4),
            "var95":  round(float(np.percentile(day_vals, 5)), 4),
            "confidenceBand": [round(float(p10), 4), round(float(p90), 4)]
        })

    return result