import numpy as np

def rref(A, tol=1e-8):
    A = A.astype(float).copy()
    rows, cols = A.shape
    pivot_row = 0
    pivot_cols = []
    for col in range(cols):
        if pivot_row >= rows:
            break
        pivot = np.argmax(np.abs(A[pivot_row:, col])) + pivot_row
        if abs(A[pivot, col]) < tol:
            continue
        A[[pivot_row, pivot]] = A[[pivot, pivot_row]]
        A[pivot_row] = A[pivot_row] / A[pivot_row, col]
        for r in range(rows):
            if r != pivot_row:
                A[r] -= A[r, col] * A[pivot_row]
        pivot_cols.append(col)
        pivot_row += 1
    return A, pivot_cols

if __name__ == "__main__":
    from loader import load_fd001
    df = load_fd001()
    sensor_cols = [c for c in df.columns if c.startswith("s")]
    X = df[sensor_cols].values
    R, pivots = rref(X[:50])
    print("Rank:", len(pivots))
    print("Pivot columns (keep these):", [sensor_cols[i] for i in pivots])
