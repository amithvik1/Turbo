import numpy as np
import matplotlib.pyplot as plt

def compute_health_index(X):
    # Center the data: subtract each sensor's mean so we're measuring variation, not raw scale
    X_centered = X - X.mean(axis=0)

    # Covariance matrix: 14x14, tells us how sensors move together
    cov = np.cov(X_centered, rowvar=False)

    # Eigen-decomposition: eigvals = how much variance per direction, eigvecs = the directions
    eigvals, eigvecs = np.linalg.eigh(cov)

    # eigh returns eigenvalues in ASCENDING order, so the top direction is the LAST column
    top_direction = eigvecs[:, -1]

    # Project every row onto that direction: one number per row
    health = X_centered @ top_direction
    return health, eigvals[-1]

if __name__ == "__main__":
    from loader import load_fd001

    df = load_fd001()
    keep = ['s2','s3','s4','s7','s8','s9','s11','s12','s13','s14','s15','s17','s20','s21']

    health, top_eigval = compute_health_index(df[keep].values)
    df["health"] = health

    print("Top eigenvalue (variance explained):", top_eigval)
    print("Fraction of total variance:", top_eigval / np.linalg.eigh(np.cov(df[keep].values - df[keep].values.mean(axis=0), rowvar=False))[0].sum())

    # Plot health index over time for 3 example engines
    plt.figure(figsize=(8, 5))
    for engine_id in [1, 2, 3]:
        engine_data = df[df["unit"] == engine_id]
        plt.plot(engine_data["cycle"], engine_data["health"], label=f"Engine {engine_id}")
    plt.xlabel("Cycle")
    plt.ylabel("Health Index")
    plt.title("Health Index Over Time (should trend as engine degrades)")
    plt.legend()
    plt.savefig("notebooks/health_index.png")
    print("\nPlot saved to notebooks/health_index.png")
