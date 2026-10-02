# Turbojet Engine Remaining Useful Life (RUL) Prediction

A project demonstrating applied Linear Algebra for predictive maintenance on NASA turbofan jet engine telemetry (C-MAPSS dataset).

## What This Project Does (In Simple Terms)

Jet engines wear down over time as they fly more flight cycles. This project takes raw sensor data (temperatures, pressures, fan speeds) and predicts **how many flight cycles an engine has left before failure** (its Remaining Useful Life, or RUL).

Instead of treating machine learning as a black box, every stage is built using fundamental Linear Algebra concepts from scratch:

1. **Gaussian Elimination / RREF (`linalg/rref.py`)**: Identifies redundant and constant sensors. Drops 7 dead sensor channels (Nullity = 7) and confirms 14 linearly independent channels (Rank = 14).
2. **Gram-Schmidt Orthogonalization (`linalg/gram_schmidt.py`)**: Transforms sensor columns into an orthonormal coordinate system ($Q^T Q = I$) to prevent numerical instability.
3. **Spectral Decomposition / PCA (`linalg/health_index.py`)**: Computes the covariance matrix of sensors and extracts its dominant eigenvector to track overall engine degradation without supervision.
4. **Least Squares Regression (`linalg/lstsq.py`, `linalg/evaluate.py`)**: Solves the overdetermined linear system $X w \approx y$ via $QR$-decomposition ($R w = Q^T y$) to predict RUL on 100 test engines.
5. **Industry Benchmark Validation**: Validates custom linear algebra implementations against Scikit-Learn (`LinearRegression`), achieving comparable RMSE (~20.99 vs 20.82).

## Quick Start

### 1. Run the Entire Pipeline Demo
```bash
python run_demo.py --explain
```

### 2. Run Individual Modules
```bash
python linalg/loader.py        # Inspect sensor variance
python linalg/rref.py          # Compute RREF, rank, and nullity
python linalg/gram_schmidt.py  # Run Gram-Schmidt QR decomposition
python linalg/health_index.py  # Generate health index plot
python linalg/evaluate.py      # Evaluate test predictions vs Scikit-Learn
```

### 3. Run Automated Tests
```bash
pytest
```

## Generated Visualizations

- `notebooks/health_index.png`: Health index degradation curves across operating cycles.
- `notebooks/pred_vs_actual.png`: Predicted vs true RUL scatter plot for 100 test engines.
