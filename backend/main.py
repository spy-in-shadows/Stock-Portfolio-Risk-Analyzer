from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator
from typing import Optional, List
import pandas as pd
import numpy as np
import io
import json

from risk_engine import (
    preprocess_broker_data,
    get_risk_metrics,
    build_blended_benchmark_returns,
    compare_performance,
)

app = FastAPI(
    title="Portfolio Risk Analyzer API",
    description=(
        "Accepts broker-style holdings CSV exports. "
        "Fetches historical prices via yfinance. "
        "Computes institutional-grade quantitative risk analytics "
        "with single and blended benchmark comparison."
    ),
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════
# Pydantic v2 Request / Response Models
# ═══════════════════════════════════════════════════════════════════

class BlendedBenchmark(BaseModel):
    tickers: List[str]
    weights: List[float]

    @model_validator(mode="after")
    def validate_blended(self):
        if len(self.tickers) != len(self.weights):
            raise ValueError(
                f"tickers ({len(self.tickers)}) and weights ({len(self.weights)}) must have equal length."
            )
        if any(w < 0 for w in self.weights):
            raise ValueError("All blended benchmark weights must be non-negative.")
        if not np.isclose(sum(self.weights), 1.0, atol=1e-6):
            raise ValueError(
                f"Blended benchmark weights must sum to 1.0 (got {sum(self.weights):.6f})."
            )
        return self


class ComparisonResponse(BaseModel):
    portfolio_cumulative_return: Optional[float] = None
    benchmark_cumulative_return: Optional[float] = None
    blended_benchmark_cumulative_return: Optional[float] = None
    performance_vs_single: Optional[str] = None
    performance_vs_blended: Optional[str] = None
    relative_gap_single: Optional[float] = None
    relative_gap_blended: Optional[float] = None
    sharpe_portfolio: Optional[float] = None
    sharpe_benchmark: Optional[float] = None
    relative_sharpe_vs_single: Optional[float] = None
    tracking_error_blended: Optional[float] = None
    information_ratio_blended: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════

@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "online",
        "engine": "broker_csv_risk_analytics",
        "version": "3.0.0",
        "features": ["single_benchmark", "blended_benchmark", "monte_carlo_30d"]
    }


@app.post("/analyze")
async def analyze_portfolio(
    file: UploadFile = File(...),
    benchmark_ticker: Optional[str] = Form(None),
    blended_benchmark: Optional[str] = Form(None),   # JSON string
    risk_free_rate: float = Form(0.0),
    confidence_level: float = Form(0.95),
    simulations: int = Form(10000),
    history_days: int = Form(365),
):
    """
    Accepts a broker-style holdings CSV (Ticker, Portfolio Weight %, ...).
    Automatically fetches all historical prices from yfinance.
    Supports single benchmark, blended benchmark, or both.
    Returns full quantitative risk report with performance comparison.
    """
    try:
        # ── Parse & validate blended benchmark JSON ──────────────────
        blended_config: Optional[BlendedBenchmark] = None
        if blended_benchmark:
            try:
                blended_dict = json.loads(blended_benchmark)
                blended_config = BlendedBenchmark(**blended_dict)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="blended_benchmark must be a valid JSON string."
                )
            except Exception as ve:
                raise HTTPException(status_code=400, detail=str(ve))

        # ── Validate: at least one benchmark must be provided ────────
        if not benchmark_ticker and not blended_config:
            raise HTTPException(
                status_code=400,
                detail="At least one of 'benchmark_ticker' or 'blended_benchmark' must be provided."
            )

        # ── 1. Read uploaded CSV ─────────────────────────────────────
        contents = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception as parse_err:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(parse_err)}")

        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

        if len(df.columns) < 2:
            raise HTTPException(
                status_code=400,
                detail="CSV has fewer than 2 columns. Ensure it is a valid broker export."
            )

        # ── 2. Parse holdings + fetch prices + align ─────────────────
        (
            asset_returns, bench_returns, weights,
            asset_names, portfolio_value, start_date, end_date
        ) = preprocess_broker_data(
            df=df,
            benchmark_ticker=benchmark_ticker,
            history_days=history_days
        )

        # ── 3. Run core risk engine ──────────────────────────────────
        #    Only compute full risk metrics if single benchmark is provided
        #    (MC simulation and Beta require a benchmark)
        risk_results = {}
        if bench_returns is not None:
            risk_results = get_risk_metrics(
                asset_returns=asset_returns,
                bench_returns=bench_returns,
                benchmark_ticker=benchmark_ticker,
                asset_names=asset_names,
                weights=weights,
                portfolio_value=portfolio_value,
                risk_free_rate=risk_free_rate,
                confidence_level=confidence_level,
                simulations=simulations
            )
        else:
            # Minimal risk metrics without benchmark alignment
            w = np.array(weights)
            if not np.isclose(w.sum(), 1.0):
                w = w / w.sum()
            mean_returns = asset_returns.mean().fillna(0)
            cov_matrix = asset_returns.cov().fillna(0)
            corr_matrix = asset_returns.corr().fillna(0)
            port_return = float(np.dot(w, mean_returns))
            port_vol = float(np.sqrt(w.T @ cov_matrix.values @ w))

            risk_results = {
                "portfolio_expected_return": port_return,
                "portfolio_volatility": port_vol,
                "historical_var_95": None,
                "parametric_var_95": None,
                "monte_carlo_var_95": None,
                "monte_carlo_expected_return_30d": None,
                "monte_carlo_volatility_30d": None,
                "monte_carlo_var95_30d": None,
                "beta": None,
                "correlation_matrix": corr_matrix.values.tolist(),
                "asset_names": asset_names,
                "benchmark_ticker": benchmark_ticker,
                "mc_chart_data": []
            }

        # ── 4. Build blended benchmark returns (if requested) ────────
        blended_returns = None
        if blended_config:
            blended_returns = build_blended_benchmark_returns(
                tickers=blended_config.tickers,
                weights=blended_config.weights,
                start_date=start_date,
                end_date=end_date,
            )

        # ── 5. Compute portfolio daily returns for comparison ────────
        w_arr = np.array(weights)
        port_daily_returns = pd.Series(
            np.dot(asset_returns.values, w_arr),
            index=asset_returns.index,
            name="portfolio"
        )

        # ── 6. Performance comparison ────────────────────────────────
        comparison = compare_performance(
            portfolio_returns=port_daily_returns,
            benchmark_returns=bench_returns,
            blended_returns=blended_returns,
            risk_free_rate=risk_free_rate,
        )

        # ── 7. Build final response ──────────────────────────────────
        response = {
            "status": "success",
            "start_date": start_date,
            "end_date": end_date,
            **risk_results,
            "comparison": comparison
        }

        return response

    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": "Analytical failure", "detail": str(e)}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)