import numpy as np
from sklearn.linear_model import LinearRegression
from linalg.loader import load_fd001
from linalg.rref import rref
from linalg.gram_schmidt import gram_schmidt
from linalg.lstsq import solve_least_squares
from linalg.evaluate import load_test_data, load_true_rul
from linalg.health_index import compute_health_index

def test_rref_rank_and_nullity():
    df = load_fd001()
    sensor_cols = [c for c in df.columns if c.startswith("s")]
    stds = df[sensor_cols].std()
    keep = stds[stds > 0.01].index.tolist()
    X = df[keep].values
    _, pivots = rref(X)
    rank = len(pivots)
    nullity = len(sensor_cols) - rank
    assert rank == 14
    assert nullity == 7
    assert len(keep) == 14

def test_gram_schmidt_orthonormal():
    df = load_fd001()
    keep = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    X = df[keep].values
    Q = gram_schmidt(X)
    identity = np.eye(X.shape[1])
    assert np.allclose(Q.T @ Q, identity, atol=1e-5)
    Q_ref, _ = np.linalg.qr(X)
    assert np.allclose(np.abs(Q), np.abs(Q_ref), atol=1e-5)

def test_least_squares_matches_sklearn():
    train_df = load_fd001()
    keep = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    X_train = train_df[keep].values
    y_train = train_df["RUL"].values

    weights = solve_least_squares(X_train, y_train)

    test_df = load_test_data()
    last_rows = test_df.groupby("unit").tail(1).sort_values("unit")
    X_test = last_rows[keep].values
    y_true = np.clip(load_true_rul(), 0, 125)

    preds_custom = X_test @ weights
    custom_rmse = np.sqrt(np.mean((preds_custom - y_true) ** 2))

    sk_model_no_intercept = LinearRegression(fit_intercept=False).fit(X_train, y_train)
    assert np.allclose(weights, sk_model_no_intercept.coef_, atol=1e-5)
    assert np.allclose(preds_custom, sk_model_no_intercept.predict(X_test), atol=1e-5)

    sk_model_default = LinearRegression().fit(X_train, y_train)
    sk_rmse = np.sqrt(np.mean((sk_model_default.predict(X_test) - y_true) ** 2))
    assert abs(custom_rmse - sk_rmse) < 0.5

def test_health_index_spectral():
    df = load_fd001()
    keep = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    X = df[keep].values
    health, top_eigval = compute_health_index(X)
    assert len(health) == len(df)
    assert top_eigval > 0
    cov = np.cov(X - X.mean(axis=0), rowvar=False)
    total_var = np.linalg.eigh(cov)[0].sum()
    var_fraction = top_eigval / total_var
    assert var_fraction > 0.8
