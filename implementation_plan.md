# Portfolio Risk Analyzer Implementation Plan

## 1. Roadmap & Implementation Checklist

### Phase 1: Stateless Risk Engine Implementation (`risk_engine.py`)
- [ ] **Step 1: Data Preprocessing Utility**
  - Implement CSV parsing with date sorting and NaN handling.
  - Create logic to isolate Assets from Benchmark.
- [ ] **Step 2: core Statistical Functions**
  - Portfolio Volatility (Matrix-based).
  - Expected Return vector.
  - Sharpe Ratio (Risk-free rate integration).
- [ ] **Step 3: Value at Risk (VaR) Suite**
  - Historical VaR (95th percentile).
  - Parametric VaR (Variance-Covariance).
- [ ] **Step 4: Monte Carlo Simulation Module**
  - Cholesky decomposition for correlation preservation.
  - 10,000+ simulations using multivariate normal sampling.
  - Monte Carlo VaR computation.
- [ ] **Step 5: Systematic Risk Engine**
  - Beta calculation relative to standardized benchmark.
  - Correlation Matrix generation.

### Phase 2: FastAPI Backend Development (`main.py`)
- [ ] **Endpoint: `GET /`**
  - Health check and environment verification.
- [ ] **Endpoint: `POST /analyze`**
  - Support file upload via Multipart.
  - Handle optional parameters: Weights, Risk-free Rate, Confidence level.
- [ ] **Validation Layer**
  - Weights validation (Sum = 1, Length match).
  - CSV Format/Minimum row validation.
  - Error response standard (HTTP 400).
- [ ] **JSON Response Formatting**
  - Consolidate all metrics into the specified JSON structure.

### Phase 3: Infrastructure & Deployment
- [ ] **Dependencies**: Finalize `requirements.txt`.
- [ ] **Production Config**: Configure `uvicorn` for host-binding.
- [ ] **Deployment**: Integration with Render/Vercel.

## 2. Technical Stack
- **Framework**: FastAPI
- **Computations**: NumPy, Pandas, SciPy
- **Simulation**: Multivariate Normal Sampling + Cholesky Decomposition
- **Performance**: Vectorized matrix operations for <200ms latency.
