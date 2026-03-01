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
    benchmark_ticker: Optional[str] = None,
    history_days: int = 365
):
    """
    Full pipeline: parse broker CSV → fetch prices → fetch benchmark → align.
    Returns: (asset_returns, bench_returns, weights, asset_names, portfolio_value)
    bench_returns is None when benchmark_ticker is not provided.
    """
    # Step 1: Parse holdings from the broker CSV
    tickers, weights = parse_holdings_csv(df)

    # Step 2: Define date range for historical data fetch
    end_date   = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=history_days)).strftime('%Y-%m-%d')

    # Step 3: Fetch historical portfolio asset prices
    price_df, resolved_map = fetch_price_history(tickers, start_date, end_date)

    # Step 4: Compute asset returns
    asset_returns = price_df.pct_change().dropna()

    # Step 5: Fetch and align benchmark (if provided)
    bench_returns = None
    if benchmark_ticker:
        bench_returns = fetch_benchmark_returns(benchmark_ticker, start_date, end_date)

        # Align asset returns with benchmark returns (inner join on common dates)
        combined = pd.concat([asset_returns, bench_returns], axis=1).dropna()

        if len(combined) < 60:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Only {len(combined)} overlapping trading days between portfolio and benchmark. "
                    "Minimum 60 required."
                )
            )

        asset_returns = combined.drop(columns=["Benchmark_Returns"])
        bench_returns = combined["Benchmark_Returns"]

    # Reorder weights to match final asset columns (after any column drops in alignment)
    final_tickers  = asset_returns.columns.tolist()
    ticker_to_weight = dict(zip(tickers, weights))
    final_weights = [ticker_to_weight[t] for t in final_tickers]

    # Re-normalize in case any tickers were dropped  
    total = sum(final_weights)
    final_weights = [w / total for w in final_weights]

    # Step 6: Compute approximate portfolio value from last prices × weights
    #         This is a normalized proxy (sum of weighted last prices).
    last_prices = price_df.iloc[-1]
    portfolio_value = float(np.dot(last_prices.values, np.array(weights[:len(last_prices)])))

    return asset_returns, bench_returns, final_weights, final_tickers, portfolio_value, start_date, end_date


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
    portfolio_value: float = 100.0,
    risk_free_rate: float = 0.0,
    confidence_level: float = 0.95,
    simulations: int = 10000
) -> dict:
    """
    Vectorized portfolio risk engine.
    Computes: volatility, VaR (3 methods), Beta, correlation matrix, MC paths.
    Uses log returns for MC parameter estimation.
    """
    weights = np.array(weights)

    # Ensure weights sum to 1.0
    if not np.isclose(weights.sum(), 1.0):
        weights = weights / weights.sum()

    # ── Statistical Components (simple returns for empirical metrics) ──
    mean_returns = asset_returns.mean().fillna(0)
    cov_matrix   = asset_returns.cov().fillna(0)
    corr_matrix  = asset_returns.corr().fillna(0)

    # ── Portfolio Historical Returns: Rp = w · R ───────────────────
    port_hist_returns = np.dot(asset_returns.values, weights)

    # ── Portfolio Return & Volatility ──────────────────────────────
    port_return = float(np.dot(weights, mean_returns))
    port_vol    = float(np.sqrt(weights.T @ cov_matrix.values @ weights))

    # ── Value at Risk ───────────────────────────────────────────────
    hist_var      = float(np.percentile(port_hist_returns, (1 - confidence_level) * 100))
    z_score       = norm.ppf(1 - confidence_level)
    parametric_var = port_return + z_score * port_vol

    # ── Log returns for Monte Carlo parameter estimation ───────────
    # Using log returns (ln(1+r)) is more appropriate for GBM-style
    # compounding since log returns are additive across time.
    log_returns = np.log1p(asset_returns.values)  # ln(1 + simple_return)
    mu_log = np.mean(log_returns, axis=0)          # shape: (n_assets,)
    sigma_log = np.cov(log_returns, rowvar=False)  # shape: (n_assets, n_assets)

    # ── Unified Monte Carlo (metrics + chart from same sample) ─────
    mc_result = unified_monte_carlo(
        mu=mu_log,
        sigma=sigma_log,
        weights=weights,
        N=simulations,
        T=30,
        confidence_level=confidence_level,
    )

    # ── Beta = Cov(Rp, Rm) / Var(Rm) ───────────────────────────────
    bench_arr = bench_returns.values
    cov_mat   = np.cov(port_hist_returns, bench_arr)
    cov_rp_rm = cov_mat[0, 1]
    var_rm    = np.var(bench_arr, ddof=1)
    beta      = cov_rp_rm / var_rm if var_rm > 0 else 1.0

    return {
        # Empirical daily metrics (from historical data)
        "portfolio_expected_return":       port_return,
        "portfolio_volatility":            port_vol,
        "historical_var_95":               hist_var,
        "parametric_var_95":               float(parametric_var),
        # Monte Carlo 30-day compounded metrics
        "monte_carlo_var_95":              mc_result["var95"],
        "monte_carlo_expected_return_30d": mc_result["expected"],
        "monte_carlo_volatility_30d":      mc_result["volatility"],
        "monte_carlo_var95_30d":           mc_result["var95"],
        # Simulation diagnostics
        "mc_standard_error":               mc_result["standard_error"],
        "mc_theoretical_cross_check":      mc_result["theoretical_expected"],
        # Meta
        "beta":                            float(beta),
        "correlation_matrix":              corr_matrix.values.tolist(),
        "asset_names":                     asset_names,
        "benchmark_ticker":                benchmark_ticker,
        "mc_chart_data":                   mc_result["chart_data"]
    }


# ═══════════════════════════════════════════════════════════════════
# MODULE 6: Unified Monte Carlo Engine
#
# Single simulation produces BOTH risk metrics AND chart data from
# the same random sample.  No separate functions, no parameter drift.
#
# Uses daily LOG returns ~ MVN(μ, Σ) for proper GBM compounding.
# ═══════════════════════════════════════════════════════════════════

def unified_monte_carlo(
    mu: np.ndarray,
    sigma: np.ndarray,
    weights: np.ndarray,
    N: int = 10000,
    T: int = 30,
    confidence_level: float = 0.95,
) -> dict:
    """
    Fully consistent Monte Carlo engine.

    Produces risk metrics AND chart data from a SINGLE shared simulation.
    Both outputs use the same N, T, μ, Σ, and random matrix R.

    Parameters
    ----------
    mu      : Mean daily log returns per asset, shape (n_assets,)
    sigma   : Covariance matrix of daily log returns, shape (n_assets, n_assets)
    weights : Portfolio weights, shape (n_assets,)
    N       : Number of simulations (single source of truth)
    T       : Horizon in trading days (single source of truth)
    confidence_level : e.g. 0.95 for 95% VaR

    Returns
    -------
    dict with keys:
        expected, volatility, var95, standard_error, theoretical_expected,
        chart_data (list of T+1 daily dicts)
    """

    # ══════════════════════════════════════════════════════════════
    # STEP 1 — Sample daily log returns (SINGLE random matrix)
    # R ∈ ℝ^(N × T × n_assets) ~ MVN(μ, Σ)
    # ══════════════════════════════════════════════════════════════
    R = np.random.multivariate_normal(
        mean=mu,
        cov=sigma,
        size=(N, T)
    )
    # R.shape = (N, T, n_assets)

    # ══════════════════════════════════════════════════════════════
    # STEP 2 — Weighted portfolio daily log returns
    # r_port[s, t] = Σᵢ wᵢ · R[s, t, i]
    # ══════════════════════════════════════════════════════════════
    r_port = np.einsum('ijk,k->ij', R, weights)
    # r_port.shape = (N, T)

    # Convert log returns to simple returns for compounding:
    # simple_return = exp(log_return) - 1
    simple_daily = np.expm1(r_port)
    # simple_daily.shape = (N, T)

    # ══════════════════════════════════════════════════════════════
    # STEP 3 — Compound daily → terminal return (TRUE compounding)
    # terminal[s] = ∏(t=1 to T) (1 + simple_daily[s,t]) − 1
    # ══════════════════════════════════════════════════════════════
    terminal = np.prod(1.0 + simple_daily, axis=1) - 1.0
    # terminal.shape = (N,)

    # ══════════════════════════════════════════════════════════════
    # STEP 3b — Risk metrics from terminal distribution
    # ══════════════════════════════════════════════════════════════
    var_pct     = (1.0 - confidence_level) * 100.0
    expected    = float(np.mean(terminal))
    volatility  = float(np.std(terminal, ddof=1))
    var95       = float(np.percentile(terminal, var_pct))
    se          = volatility / np.sqrt(N)     # simulation standard error

    # Theoretical cross-check with convexity adjustment:
    # For log-normal returns: E[exp(X)] = exp(μ + σ²/2)
    # Portfolio log return per day: μ_p = w·μ, σ²_p = w'Σw
    # Over T days (IID):  E[terminal] = exp(T·(μ_p + σ²_p/2)) - 1
    mu_port_log = float(np.dot(weights, mu))
    var_port_log = float(weights @ sigma @ weights)
    theoretical_expected = float(np.exp(T * (mu_port_log + var_port_log / 2.0)) - 1.0)

    # Soft consistency check (log warning, don't crash)
    cross_check_diff = abs(expected - theoretical_expected)
    if cross_check_diff > 4.0 * se and se > 0:
        import logging
        logging.warning(
            f"MC cross-check: |E[X] - E[X]_theory| = {cross_check_diff:.6f} "
            f"> 4×SE = {4*se:.6f}. "
            f"E[X]={expected:.6f}, E[X]_theory={theoretical_expected:.6f}"
        )

    # ══════════════════════════════════════════════════════════════
    # STEP 4 — Forward path chart data (reuse same simulation)
    # cum_value[s, t] = 100 · ∏(τ=1 to t) (1 + simple_daily[s,τ])
    # ══════════════════════════════════════════════════════════════
    ones_col   = np.ones((N, 1))
    cum_factors = np.cumprod(
        np.hstack([ones_col, 1.0 + simple_daily]),
        axis=1
    )
    cum_value = cum_factors * 100.0
    # cum_value.shape = (N, T+1)   — columns 0..T

    # ══════════════════════════════════════════════════════════════
    # STEP 5 — Named path selection via RANK AVERAGING
    #
    # Instead of picking paths by terminal value only (which can
    # give paths that oscillate wildly before converging to the
    # target percentile), we rank each simulation at EVERY day
    # and then pick the path whose average rank is closest to
    # the desired percentile.
    #
    # ranks[s, t] = rank(cum_value[:, t])[s]     shape: (N, T+1)
    # avg_rank[s] = mean(ranks[s, :])             shape: (N,)
    # ══════════════════════════════════════════════════════════════

    # Compute ranks at each day (0-indexed, so rank range is [0, N-1])
    # argsort of argsort gives the rank of each element
    ranks = np.zeros_like(cum_value)
    for t in range(T + 1):
        order = np.argsort(cum_value[:, t])
        ranks[order, t] = np.arange(N)
    # ranks.shape = (N, T+1),  values in [0, N-1]

    # Average rank across all days for each simulation
    avg_rank = np.mean(ranks, axis=1)
    # avg_rank.shape = (N,)

    # Select paths whose average rank is closest to desired percentiles
    target_median = 0.50 * (N - 1)
    target_bull   = 0.90 * (N - 1)
    target_stress = 0.10 * (N - 1)

    idx_median = int(np.argmin(np.abs(avg_rank - target_median)))
    idx_bull   = int(np.argmin(np.abs(avg_rank - target_bull)))
    idx_stress = int(np.argmin(np.abs(avg_rank - target_stress)))

    # ══════════════════════════════════════════════════════════════
    # STEP 6 — Assemble chart data (T+1 data points, day 0..T)
    # ══════════════════════════════════════════════════════════════
    chart_data = []
    for t in range(T + 1):
        day_vals = cum_value[:, t]
        p_var, p10, p50, p90 = np.percentile(
            day_vals, [var_pct, 10, 50, 90]
        )
        mean_val = float(np.mean(day_vals))

        chart_data.append({
            "day":    t,
            "path1":  round(float(cum_value[idx_median, t]), 4),
            "path2":  round(float(cum_value[idx_stress, t]), 4),
            "path3":  round(float(cum_value[idx_bull, t]), 4),
            "median": round(float(p50), 4),
            "mean":   round(mean_val, 4),
            "var95":  round(float(p_var), 4),
            "confidenceBand": [round(float(p10), 4), round(float(p90), 4)]
        })

    # ══════════════════════════════════════════════════════════════
    # CONSISTENCY ASSERTIONS
    # ══════════════════════════════════════════════════════════════
    # The terminal return from the metrics path and the chart's
    # final-day mean should agree within 2×SE.
    chart_terminal_mean = (chart_data[T]["mean"] / 100.0) - 1.0
    metrics_vs_chart_diff = abs(expected - chart_terminal_mean)
    if se > 0:
        assert metrics_vs_chart_diff < 2.0 * se + 1e-6, (
            f"Consistency violation: metrics E[X]={expected:.6f} vs "
            f"chart terminal mean={chart_terminal_mean:.6f}, "
            f"diff={metrics_vs_chart_diff:.6f} > 2×SE={2*se:.6f}"
        )

    return {
        "expected":               round(expected, 6),
        "volatility":             round(volatility, 6),
        "var95":                  round(var95, 6),
        "standard_error":         round(float(se), 6),
        "theoretical_expected":   round(theoretical_expected, 6),
        "chart_data":             chart_data,
    }


# ═══════════════════════════════════════════════════════════════════
# MODULE 7: Blended Benchmark Builder
# Constructs a weighted composite benchmark from multiple tickers.
# ═══════════════════════════════════════════════════════════════════

def build_blended_benchmark_returns(
    tickers: List[str],
    weights: List[float],
    start_date: str,
    end_date: str,
) -> pd.Series:
    """
    Fetches each benchmark ticker, computes daily returns, aligns on shared
    date intersection, and returns a weighted blended return series.
    All math is vectorized — no Python loops for arithmetic.
    """
    weights_arr = np.array(weights)
    return_series_list = []

    for ticker in tickers:
        yf_ticker = _build_yf_ticker(ticker)
        max_retries = 3
        df = None

        for attempt in range(max_retries):
            if attempt > 0:
                time.sleep(2 ** (attempt - 1))
            df = yf.download(
                yf_ticker,
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
                    f"Could not fetch blended benchmark ticker '{ticker}' "
                    f"(resolved: '{yf_ticker}'). Yahoo Finance may be rate-limiting."
                )
            )

        # Robust column extraction (same pattern as fetch_benchmark_returns)
        if isinstance(df.columns, pd.MultiIndex):
            level0 = df.columns.get_level_values(0)
            if 'Close' in level0:
                price_series = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
            elif 'Adj Close' in level0:
                price_series = df['Adj Close'].iloc[:, 0] if isinstance(df['Adj Close'], pd.DataFrame) else df['Adj Close']
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"No price column for '{ticker}'. Available: {list(level0.unique())}"
                )
        else:
            if 'Close' in df.columns:
                price_series = df['Close']
            elif 'Adj Close' in df.columns:
                price_series = df['Adj Close']
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"No price column for '{ticker}'. Available: {list(df.columns)}"
                )

        if isinstance(price_series, pd.DataFrame):
            price_series = price_series.iloc[:, 0]

        price_series = price_series.dropna()
        returns = price_series.pct_change().dropna()
        returns.name = ticker
        return_series_list.append(returns)

    # Align all return series on shared date intersection (inner join)
    aligned = pd.concat(return_series_list, axis=1, join='inner').dropna()

    if aligned.empty:
        raise HTTPException(
            status_code=400,
            detail="No overlapping trading days across blended benchmark tickers."
        )

    # Vectorized blended return: R_blend = aligned.values @ weights_array
    blended_values = aligned.values @ weights_arr
    blended = pd.Series(blended_values, index=aligned.index, name="blended_benchmark")

    return blended


# ═══════════════════════════════════════════════════════════════════
# MODULE 8: Performance Comparison
# Compares portfolio vs single and/or blended benchmark.
# ═══════════════════════════════════════════════════════════════════

def _annualized_sharpe(returns: np.ndarray, rf_daily: float) -> Optional[float]:
    """Annualized Sharpe Ratio = (mean(Rp - Rf) / std(Rp - Rf, ddof=1)) * sqrt(252)."""
    excess = returns - rf_daily
    std = np.std(excess, ddof=1)
    if std == 0 or np.isnan(std):
        return None
    return float((np.mean(excess) / std) * np.sqrt(252))


def _classify_performance(gap: float) -> str:
    """Classify relative performance using 1% tolerance band."""
    if gap > 0.01:
        return "Outperforming"
    elif gap < -0.01:
        return "Underperforming"
    return "At Par"


def compare_performance(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series],
    blended_returns: Optional[pd.Series],
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Compares portfolio performance against single and/or blended benchmark.
    All series are date-aligned independently before any arithmetic.
    Returns structured comparison dict with annualized metrics.
    """
    rf_daily = risk_free_rate / 252.0

    # ── Portfolio metrics (always computed) ──────────────────────────
    port_cum = float(np.prod(1.0 + portfolio_returns.values) - 1.0)
    sharpe_portfolio = _annualized_sharpe(portfolio_returns.values, rf_daily)

    # ── Single benchmark comparison ──────────────────────────────────
    benchmark_cum = None
    perf_vs_single = None
    gap_single = None
    sharpe_benchmark = None
    relative_sharpe_single = None

    if benchmark_returns is not None:
        # Align independently via inner join
        aligned = pd.concat(
            [portfolio_returns.rename("port"), benchmark_returns.rename("bench")],
            axis=1, join='inner'
        ).dropna()

        port_aligned = aligned["port"].values
        bench_aligned = aligned["bench"].values

        port_cum_aligned = float(np.prod(1.0 + port_aligned) - 1.0)
        benchmark_cum = float(np.prod(1.0 + bench_aligned) - 1.0)
        gap_single = port_cum_aligned - benchmark_cum
        perf_vs_single = _classify_performance(gap_single)

        sharpe_benchmark = _annualized_sharpe(bench_aligned, rf_daily)
        # Recalculate portfolio Sharpe on aligned data for fair comparison
        sharpe_port_aligned = _annualized_sharpe(port_aligned, rf_daily)

        if sharpe_port_aligned is not None and sharpe_benchmark is not None:
            relative_sharpe_single = sharpe_port_aligned - sharpe_benchmark
        else:
            relative_sharpe_single = None

        # Use aligned portfolio cum for the response
        port_cum = port_cum_aligned

    # ── Blended benchmark comparison ─────────────────────────────────
    blended_cum = None
    perf_vs_blended = None
    gap_blended = None
    tracking_error_blended = None
    info_ratio_blended = None

    if blended_returns is not None:
        # Align independently via inner join
        aligned_b = pd.concat(
            [portfolio_returns.rename("port"), blended_returns.rename("blend")],
            axis=1, join='inner'
        ).dropna()

        port_aligned_b = aligned_b["port"].values
        blend_aligned = aligned_b["blend"].values

        port_cum_b = float(np.prod(1.0 + port_aligned_b) - 1.0)
        blended_cum = float(np.prod(1.0 + blend_aligned) - 1.0)
        gap_blended = port_cum_b - blended_cum
        perf_vs_blended = _classify_performance(gap_blended)

        # Tracking Error (annualized): std(Rp - Rb, ddof=1) * sqrt(252)
        active_returns = port_aligned_b - blend_aligned
        te = float(np.std(active_returns, ddof=1) * np.sqrt(252))
        tracking_error_blended = te

        # Information Ratio (annualized): (mean(Rp - Rb) * 252) / TE
        if te > 0:
            info_ratio_blended = float((np.mean(active_returns) * 252) / te)
        else:
            info_ratio_blended = None

        # If single benchmark wasn't provided, use blended-aligned port_cum
        if benchmark_returns is None:
            port_cum = port_cum_b

    return {
        "portfolio_cumulative_return":          round(port_cum, 4) if port_cum is not None else None,
        "benchmark_cumulative_return":          round(benchmark_cum, 4) if benchmark_cum is not None else None,
        "blended_benchmark_cumulative_return":  round(blended_cum, 4) if blended_cum is not None else None,
        "performance_vs_single":               perf_vs_single,
        "performance_vs_blended":              perf_vs_blended,
        "relative_gap_single":                 round(gap_single, 4) if gap_single is not None else None,
        "relative_gap_blended":                round(gap_blended, 4) if gap_blended is not None else None,
        "sharpe_portfolio":                    round(sharpe_portfolio, 4) if sharpe_portfolio is not None else None,
        "sharpe_benchmark":                    round(sharpe_benchmark, 4) if sharpe_benchmark is not None else None,
        "relative_sharpe_vs_single":           round(relative_sharpe_single, 4) if relative_sharpe_single is not None else None,
        "tracking_error_blended":              round(tracking_error_blended, 4) if tracking_error_blended is not None else None,
        "information_ratio_blended":           round(info_ratio_blended, 4) if info_ratio_blended is not None else None,
    }