import numpy as np

def gram_schmidt(A):
    Q = np.zeros_like(A, dtype=float)
    for j in range(A.shape[1]):
        v = A[:, j].astype(float).copy()
        for i in range(j):
            v -= (Q[:, i] @ A[:, j]) * Q[:, i]
        Q[:, j] = v / np.linalg.norm(v)
    return Q

if __name__ == "__main__":
    from loader import load_fd001

    df = load_fd001()
    sensor_cols = [c for c in df.columns if c.startswith("s")]
    stds = df[sensor_cols].std()
    keep = stds[stds > 0.01].index.tolist()
    print("Using columns:", keep)

    X = df[keep].values
    Q = gram_schmidt(X)
    Q_ref, _ = np.linalg.qr(X)

    print("Max difference vs numpy QR:", np.abs(np.abs(Q) - np.abs(Q_ref)).max())
