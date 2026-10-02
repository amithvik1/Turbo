import numpy as np
import pandas as pd

COLS = ["unit", "cycle"] + [f"op{i}" for i in (1, 2, 3)] + [f"s{i}" for i in range(1, 22)]

def load_test_data(path="data/test_FD001.txt"):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
    return df

def load_true_rul(path="data/RUL_FD001.txt"):
    # one number per line, one per engine, in engine-id order
    return pd.read_csv(path, header=None, names=["RUL"])["RUL"].values

if __name__ == "__main__":
    from loader import load_fd001
    from lstsq import solve_least_squares

    # retrain on full training data (same as before)
    train_df = load_fd001()
    keep = ['s2','s3','s4','s7','s8','s9','s11','s12','s13','s14','s15','s17','s20','s21']
    X_train = train_df[keep].values
    y_train = train_df["RUL"].values
    weights = solve_least_squares(X_train, y_train)

    # get test data: last row per engine only
    test_df = load_test_data()
    last_rows = test_df.groupby("unit").tail(1)
    last_rows = last_rows.sort_values("unit")

    X_test = last_rows[keep].values
    y_true = load_true_rul()
    y_true = np.clip(y_true, 0, 125)

    predictions = X_test @ weights

    rmse = np.sqrt(np.mean((predictions - y_true) ** 2))
    print("Test RMSE:", rmse)
    print("\nFirst 10 predictions vs true:")
    for p, t in zip(predictions[:10], y_true[:10]):
        print(f"  predicted={p:.1f}   true={t}")
