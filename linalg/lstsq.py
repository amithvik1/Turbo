import numpy as np

def solve_least_squares(X, y):
    Q, R = np.linalg.qr(X)
    y_proj = Q.T @ y
    weights = np.linalg.solve(R, y_proj)
    return weights

if __name__ == "__main__":
    from loader import load_fd001

    df = load_fd001()
    sensor_cols = [c for c in df.columns if c.startswith("s")]

    # Drop near-constant sensors using std over the FULL dataset, not a sample
    stds = df[sensor_cols].std()
    keep = stds[stds > 0.01].index.tolist()
    print("Dropped as near-constant:", [c for c in sensor_cols if c not in keep])
    print("Keeping", len(keep), "sensors:", keep)

    X = df[keep].values
    y = df["RUL"].values

    weights = solve_least_squares(X, y)
    predictions = X @ weights

    rmse = np.sqrt(np.mean((predictions - y) ** 2))
    print("Weights:", dict(zip(keep, weights)))
    print("Training RMSE:", rmse)
