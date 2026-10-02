import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from linalg.loader import load_fd001
from linalg.rref import rref
from linalg.gram_schmidt import gram_schmidt
from linalg.lstsq import solve_least_squares
from linalg.evaluate import load_test_data, load_true_rul
from linalg.health_index import compute_health_index

def run_pipeline(explain=False):
    print("=" * 70)
    print("TURBOJET RUL PREDICTION & LINEAR ALGEBRA PIPELINE DEMO")
    print("=" * 70)

    print("\n[STAGE 1] Data Loading & Preprocessing")
    if explain:
        print("  Concept : Time-series sensor ingestion with piecewise-linear RUL capping (125 cycles).")
        print("  Purpose : Prepare training telemetry and ground-truth targets for degradation tracking.")
    df_train = load_fd001()
    sensor_cols = [c for c in df_train.columns if c.startswith("s")]
    print(f"  Outcome : Loaded {len(df_train)} rows across {df_train['unit'].nunique()} engines ({len(sensor_cols)} sensors).")

    print("\n[STAGE 2] Rank & Nullity Analysis via RREF")
    if explain:
        print("  Concept : Reduced Row Echelon Form (Gaussian elimination) and Rank-Nullity Theorem (Rank + Nullity = N).")
        print("  Purpose : Eliminate constant/redundant sensor channels to form a full-rank design matrix.")
    stds = df_train[sensor_cols].std()
    keep_sensors = stds[stds > 0.01].index.tolist()
    dropped_sensors = [c for c in sensor_cols if c not in keep_sensors]
    X_train = df_train[keep_sensors].values
    _, pivots = rref(X_train)
    rank = len(pivots)
    nullity = len(sensor_cols) - rank
    print(f"  Dropped : {len(dropped_sensors)} constant sensors -> {dropped_sensors}")
    print(f"  Retained: {rank} sensors -> {keep_sensors}")
    print(f"  Outcome : Full matrix Rank = {rank}, Sensor Nullity = {nullity} (21 - 14 = 7).")

    print("\n[STAGE 3] Orthonormal Basis Construction via Gram-Schmidt")
    if explain:
        print("  Concept : Gram-Schmidt orthogonalization yielding Q such that Q^T Q = I.")
        print("  Purpose : Generate an orthonormal column space basis for numerical stability.")
    Q = gram_schmidt(X_train)
    ortho_error = np.max(np.abs(Q.T @ Q - np.eye(rank)))
    Q_ref, _ = np.linalg.qr(X_train)
    qr_diff = np.max(np.abs(np.abs(Q) - np.abs(Q_ref)))
    print(f"  Outcome : Orthogonality error ||Q^T Q - I||_max = {ortho_error:.2e}, Diff vs numpy QR = {qr_diff:.2e}.")

    print("\n[STAGE 4] Unsupervised Health Index via Spectral Decomposition")
    if explain:
        print("  Concept : Covariance matrix eigen-decomposition (leading principal eigenvector).")
        print("  Purpose : Unsupervised health index to monitor engine degradation over operating cycles.")
    health, top_eigval = compute_health_index(X_train)
    df_train["health"] = health
    cov = np.cov(X_train - X_train.mean(axis=0), rowvar=False)
    total_variance = np.linalg.eigh(cov)[0].sum()
    var_ratio = top_eigval / total_variance
    plt.figure(figsize=(8, 5))
    for engine_id in [1, 2, 3]:
        sub = df_train[df_train["unit"] == engine_id]
        plt.plot(sub["cycle"], sub["health"], label=f"Engine {engine_id}")
    plt.xlabel("Cycle")
    plt.ylabel("Health Index")
    plt.title("Engine Health Index Trajectory (Engines 1-3)")
    plt.legend()
    plt.savefig("notebooks/health_index.png")
    print(f"  Outcome : Dominant eigenvalue = {top_eigval:.2f} ({var_ratio * 100:.2f}% variance explained).")
    print("            Trajectory plot saved to notebooks/health_index.png.")

    print("\n[STAGE 5] Supervised RUL Prediction & Benchmark Comparison")
    if explain:
        print("  Concept : QR-based least squares projection (R w = Q^T y) validated against scikit-learn LinearRegression.")
        print("  Purpose : Predict Remaining Useful Life on 100 test engines at their final recorded cycle.")
    y_train = df_train["RUL"].values
    weights = solve_least_squares(X_train, y_train)

    test_df = load_test_data()
    last_rows = test_df.groupby("unit").tail(1).sort_values("unit")
    X_test = last_rows[keep_sensors].values
    y_true = np.clip(load_true_rul(), 0, 125)

    custom_preds = X_test @ weights
    custom_rmse = np.sqrt(np.mean((custom_preds - y_true) ** 2))

    sk_model = LinearRegression().fit(X_train, y_train)
    sk_preds = sk_model.predict(X_test)
    sk_rmse = np.sqrt(np.mean((sk_preds - y_true) ** 2))

    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, custom_preds, alpha=0.6, edgecolors="k", linewidths=0.5)
    plt.plot([0, 125], [0, 125], "r--", label="Perfect Prediction")
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("Predicted vs True RUL (Test Engines)")
    plt.legend()
    plt.savefig("notebooks/pred_vs_actual.png")

    print(f"  Custom Least Squares Test RMSE: {custom_rmse:.4f}")
    print(f"  Scikit-Learn Baseline RMSE    : {sk_rmse:.4f}")
    print(f"  RMSE Difference               : {abs(custom_rmse - sk_rmse):.4f}")
    print("            Scatter plot saved to notebooks/pred_vs_actual.png.")
    print("=" * 70)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Turbojet RUL Linear Algebra Pipeline Demo")
    parser.add_argument("--explain", action="store_true", help="Print Concept -> Purpose -> Outcome walkthrough details")
    args = parser.parse_args()
    run_pipeline(explain=args.explain)

if __name__ == "__main__":
    main()
