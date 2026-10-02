# Turbojet Linear Algebra Project: Complete Beginner & C / C++ Developer Guide

Welcome! This guide assumes you know C or C++ (variables, `std::vector`, loops, structs/classes, functions, references/pointers) and basic linear algebra, but **zero** Python, NumPy, Pandas, or Machine Learning.

Every line of code, math concept, and design decision in this project is broken down below in clear, concise terms with direct C++ parallels.

---

## Table of Contents
1. [The Real-World Problem We Are Solving](#1-the-real-world-problem-we-are-solving)
2. [C / C++ vs Python / NumPy / Pandas: Mental Model](#2-c--c-vs-python--numpy--pandas-mental-model)
3. [Linear Algebra Concepts (With Small Math Examples)](#3-linear-algebra-concepts-with-small-math-examples)
4. [File-by-File & Line-by-Line Code Breakdown](#4-file-by-file--line-by-line-code-breakdown)
   - [linalg/loader.py](#file-1-linalgloaderpy)
   - [linalg/rref.py](#file-2-linalgrrefpy)
   - [linalg/gram_schmidt.py](#file-3-linalggram_schmidtpy)
   - [linalg/lstsq.py](#file-4-linalglstsqpy)
   - [linalg/health_index.py](#file-5-linalghealth_indexpy)
   - [linalg/evaluate.py](#file-6-linalgevaluatepy)
   - [tests/test_linalg.py](#file-7-teststest_linalgpy)
   - [run_demo.py](#file-8-run_demopy)
5. [Summary of Results & Viva Key Points](#5-summary-of-results--viva-key-points)

---

## 1. The Real-World Problem We Are Solving

### What is a Turbofan Jet Engine?
A commercial airplane jet engine has fans, compressors, combustion chambers, and turbines. Over many flights ("operating cycles"), components degrade due to thermal stress and mechanical friction.

### What is RUL (Remaining Useful Life)?
- If Engine #1 runs for **192 total cycles** before breakdown:
  - At cycle 1: RUL = 191 cycles remaining.
  - At cycle 50: RUL = 142 cycles remaining.
  - At cycle 192: RUL = 0 cycles remaining (engine fails).
- **Goal**: Given 21 sensor channels (temperatures, pressures, fan speeds) at any cycle, predict **how many cycles remain** before failure.

### The Linear System ($X w \approx y$):
We arrange sensor readings into a matrix $X \in \mathbb{R}^{N \times D}$ ($N = 20,631$ time steps, $D = 14$ active sensors) and target RUL values as vector $y \in \mathbb{R}^N$. We find optimal weight vector $w \in \mathbb{R}^D$ such that:
$$X w \approx y$$
Each weight $w_j$ quantifies the degradation contribution of sensor $j$.

---

## 2. C / C++ vs Python / NumPy / Pandas: Mental Model

### 2.1 Dynamic Arrays & Strings
- **In C++**:
  ```cpp
  #include <vector>
  #include <string>
  std::vector<std::string> names = {"s1", "s2", "s3"};
  ```
- **In Python**:
  ```python
  names = ["s1", "s2", "s3"]
  ```

### 2.2 List Comprehensions (Generating Sequences)
- **In C++**:
  ```cpp
  std::vector<std::string> cols;
  for (int i = 1; i <= 21; ++i) {
      cols.push_back("s" + std::to_string(i));
  }
  ```
- **In Python**:
  ```python
  cols = [f"s{i}" for i in range(1, 22)]
  ```

### 2.3 2D Arrays & Matrices
- **In C++ (STL / Eigen)**:
  ```cpp
  // STL vector-of-vectors:
  std::vector<std::vector<double>> A(20631, std::vector<double>(14, 0.0));
  double val = A[r][c];

  // Or Eigen C++ Linear Algebra library:
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(20631, 14);
  double val = A(r, c);
  ```
- **In NumPy**:
  ```python
  import numpy as np
  A = np.zeros((20631, 14)) # Continuous heap buffer of 64-bit C doubles
  val = A[r, c]
  ```

### 2.4 Vectorization (SIMD Loops without explicit `for`)
In C++, element-wise math requires loops or `std::transform`. In NumPy, arithmetic operations invoke precompiled SIMD C/Fortran routines across the entire buffer in one statement:
- **In C++**:
  ```cpp
  std::vector<double> result(N);
  for (size_t i = 0; i < N; ++i) {
      result[i] = X[i] * 2.0 - mean;
  }
  ```
- **In NumPy**:
  ```python
  result = X * 2.0 - mean
  ```

### 2.5 Matrix Multiplication (`@` Operator)
- **In C++ (Eigen)**:
  ```cpp
  Eigen::VectorXd y_pred = X * w;
  ```
- **In NumPy**:
  ```python
  y_pred = X @ w
  ```

### 2.6 Pandas DataFrame (In-Memory Structured Table)
- **In C++**: Represented as `struct EngineRecord { int unit; int cycle; double op[3]; double s[21]; double rul; };` in `std::vector<EngineRecord> dataset;`.
- **In Pandas**: `pd.DataFrame` is a column-oriented table that allows SQL-like grouping (`groupby`), filtering, and column aggregations.

---

## 3. Linear Algebra Concepts (With Small Math Examples)

### Concept 1: Matrix Rank, Nullity, and RREF (Gaussian Elimination)
- **Problem**: 21 sensors are provided. Some channels are constant (e.g. standard deviation $\approx 0$) or redundant.
- **Rank-Nullity Theorem**:
  $$\text{Columns} = \text{Rank}(A) + \text{Nullity}(A)$$
  - $\text{Rank}(A)$: Number of linearly independent sensor channels carrying distinct physical signals.
  - $\text{Nullity}(A)$: Number of redundant or flatlined sensor channels.
- **Concrete 3x3 Math Example**:
  $$A = \begin{bmatrix} 1 & 500 & 2 \\ 2 & 500 & 4 \\ 3 & 500 & 6 \end{bmatrix}$$
  Sensor 2 is constant ($500$), and Sensor 3 is collinear ($2 \times \text{Sensor 1}$).
  Performing Gaussian elimination to Reduced Row Echelon Form (RREF):
  $$\text{RREF}(A) = \begin{bmatrix} 1 & 0 & 2 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix}$$
  - Pivot columns = 2 $\implies \text{Rank} = 2$.
  - Zero rows / non-pivot columns = 1 $\implies \text{Nullity} = 3 - 2 = 1$.
- **In Our Dataset**:
  - Initial sensors = 21.
  - Constant sensors dropped = 7 ($s_1, s_5, s_6, s_{10}, s_{16}, s_{18}, s_{19}$).
  - Nullity = 7.
  - Full column rank on 14 retained sensors = 14.

---

### Concept 2: Gram-Schmidt Orthogonalization ($QR$ Factorization)
- **Problem**: Sensor columns are correlated (e.g. turbine temperature and exhaust pressure rise together). Inverting ill-conditioned matrices causes numerical instability and roundoff errors.
- **Orthogonal Basis**: We transform sensor columns $a_1, a_2, \dots, a_k$ into orthonormal vectors $q_1, q_2, \dots, q_k$ such that:
  $$q_i^T q_j = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases} \quad \implies \quad Q^T Q = I$$
- **Modified Gram-Schmidt Formula**:
  1. First column: $v_1 = a_1, \quad q_1 = \frac{v_1}{\|v_1\|}$
  2. For column $j = 2, \dots, k$:
     $$v_j = a_j - \sum_{i=1}^{j-1} (q_i^T a_j) q_i, \quad q_j = \frac{v_j}{\|v_j\|}$$
  Subtracting $(q_i^T a_j) q_i$ removes the projection (shadow) of $a_j$ along previous directions $q_i$.

---

### Concept 3: Least Squares via $QR$ Factorization
- **Problem**: We have 20,631 equations and 14 unknowns ($X w \approx y$). The system is overdetermined with no exact solution.
- **Optimization**: Find $w$ minimizing squared residual norm $\|X w - y\|_2^2$.
- **$QR$ Derivation**:
  1. Decompose $X = QR$, where $Q \in \mathbb{R}^{N \times D}$ ($Q^T Q = I$) and $R \in \mathbb{R}^{D \times D}$ is upper triangular:
     $$X w \approx y \implies QR w \approx y$$
  2. Multiply by $Q^T$:
     $$Q^T Q R w = Q^T y \implies I R w = Q^T y \implies R w = Q^T y$$
  3. Since $R$ is upper triangular, $R w = Q^T y$ is solved in $O(D^2)$ operations by simple back-substitution without computing any matrix inverse $(X^T X)^{-1}$.

---

### Concept 4: Unsupervised Health Index via Spectral Decomposition (PCA)
- **Problem**: How to quantify engine degradation without ground-truth labels?
- **Eigen-decomposition**:
  1. Center sensor matrix: $X_c = X - \mu$.
  2. Compute $14 \times 14$ sample covariance matrix:
     $$C = \frac{1}{N-1} X_c^T X_c$$
  3. Solve $C v = \lambda v$.
     - $\lambda_{max}$: Maximum variance captured.
     - $v_{top}$: Principal eigenvector along which the engine degrades most across time.
  4. Project each observation: $\text{Health}_t = X_{c, t} \cdot v_{top}$.

---

## 4. File-by-File & Line-by-Line Code Breakdown

---

### File 1: `linalg/loader.py`
**Goal**: Ingest raw sensor telemetry, compute target RUL per engine, and apply piecewise clipping.

```python
import pandas as pd

COLS = ["unit", "cycle"] + [f"op{i}" for i in (1, 2, 3)] + [f"s{i}" for i in range(1, 22)]
```
- **Explanation**: Creates a list of 26 column headers (`unit`, `cycle`, 3 operational settings, 21 sensors).
- **C++ parallel**:
  ```cpp
  std::vector<std::string> COLS = {"unit", "cycle", "op1", "op2", "op3"};
  for (int i = 1; i <= 21; ++i) COLS.push_back("s" + std::to_string(i));
  ```

```python
def load_fd001(path="data/train_FD001.txt"):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
    df["RUL"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=125)
    return df
```
- **Explanation**:
  - `pd.read_csv`: Reads whitespace-separated text file.
  - `groupby("unit")["cycle"].transform("max") - df["cycle"]`: Calculates remaining cycles for each engine $RUL = C_{max} - C_{current}$.
  - `.clip(upper=125)`: Caps RUL at 125 cycles (early-life piece-wise linear model).
- **C++ parallel**:
  ```cpp
  struct EngineRow { int unit; int cycle; double sensors[21]; double rul; };

  std::vector<EngineRow> load_fd001(const std::string& path) {
      std::ifstream file(path);
      std::vector<EngineRow> data;
      std::unordered_map<int, int> max_cycle;
      EngineRow row;
      while (file >> row.unit >> row.cycle /* ... read sensors ... */) {
          data.push_back(row);
          max_cycle[row.unit] = std::max(max_cycle[row.unit], row.cycle);
      }
      for (auto& r : data) {
          r.rul = std::min(125.0, static_cast<double>(max_cycle[r.unit] - r.cycle));
      }
      return data;
  }
  ```

---

### File 2: `linalg/rref.py`
**Goal**: Custom Gaussian elimination to compute Reduced Row Echelon Form, rank, and nullity.

```python
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
```
- **Explanation**:
  - `np.argmax(np.abs(...))`: Partial pivoting (finds row with largest absolute coefficient to avoid division by zero).
  - `A[[pivot_row, pivot]] = A[[pivot, pivot_row]]`: Swaps current row with pivot row ($O(1)$ pointer swap in C++).
  - `A[pivot_row] /= A[pivot_row, col]`: Normalizes pivot row so leading entry is 1.0.
  - `A[r] -= A[r, col] * A[pivot_row]`: Eliminates non-zero elements in all other rows.
- **C++ parallel**:
  ```cpp
  std::pair<std::vector<std::vector<double>>, std::vector<int>> rref(
      std::vector<std::vector<double>> A, double tol = 1e-8) {
      int rows = A.size(), cols = A[0].size();
      int pivot_row = 0;
      std::vector<int> pivot_cols;
      for (int col = 0; col < cols && pivot_row < rows; ++col) {
          int pivot = pivot_row;
          double max_val = std::abs(A[pivot_row][col]);
          for (int r = pivot_row + 1; r < rows; ++r) {
              if (std::abs(A[r][col]) > max_val) {
                  max_val = std::abs(A[r][col]);
                  pivot = r;
              }
          }
          if (max_val < tol) continue;
          std::swap(A[pivot_row], A[pivot]); // O(1) vector pointer swap
          double pivot_val = A[pivot_row][col];
          for (int c = 0; c < cols; ++c) A[pivot_row][c] /= pivot_val;
          for (int r = 0; r < rows; ++r) {
              if (r == pivot_row) continue;
              double factor = A[r][col];
              for (int c = 0; c < cols; ++c) A[r][c] -= factor * A[pivot_row][c];
          }
          pivot_cols.push_back(col);
          pivot_row++;
      }
      return {A, pivot_cols};
  }
  ```

---

### File 3: `linalg/gram_schmidt.py`
**Goal**: Modified Gram-Schmidt process generating orthonormal columns ($Q^T Q = I$).

```python
def gram_schmidt(A):
    Q = np.zeros_like(A, dtype=float)
    for j in range(A.shape[1]):
        v = A[:, j].astype(float).copy()
        for i in range(j):
            v -= (Q[:, i] @ A[:, j]) * Q[:, i]
        Q[:, j] = v / np.linalg.norm(v)
    return Q
```
- **Explanation**:
  - `v -= (Q[:, i] @ A[:, j]) * Q[:, i]`: Computes projection scalar $s = q_i \cdot a_j$ and subtracts $s \cdot q_i$.
  - `Q[:, j] = v / np.linalg.norm(v)`: Divides vector $v$ by its Euclidean length $\sqrt{\sum v_k^2}$.
- **C++ parallel**:
  ```cpp
  #include <numeric>
  #include <cmath>

  // Using Eigen C++:
  Eigen::MatrixXd gram_schmidt(const Eigen::MatrixXd& A) {
      Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(A.rows(), A.cols());
      for (int j = 0; j < A.cols(); ++j) {
          Eigen::VectorXd v = A.col(j);
          for (int i = 0; i < j; ++i) {
              double proj = Q.col(i).dot(A.col(j));
              v -= proj * Q.col(i);
          }
          Q.col(j) = v / v.norm();
      }
      return Q;
  }
  ```

---

### File 4: `linalg/lstsq.py`
**Goal**: Solve overdetermined least squares $X w \approx y$ via QR factorized back-substitution.

```python
def solve_least_squares(X, y):
    Q, R = np.linalg.qr(X)
    y_proj = Q.T @ y
    weights = np.linalg.solve(R, y_proj)
    return weights
```
- **Explanation**:
  - `np.linalg.qr(X)`: Decomposes $X = QR$.
  - `y_proj = Q.T @ y`: Projects $y$ onto column space ($Q^T y$).
  - `np.linalg.solve(R, y_proj)`: Solves upper-triangular linear system $R w = Q^T y$ via back-substitution.
- **C++ parallel**:
  ```cpp
  Eigen::VectorXd solve_least_squares(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
      Eigen::HouseholderQR<Eigen::MatrixXd> qr(X);
      Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(X.rows(), X.cols());
      Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();
      Eigen::VectorXd y_proj = Q.transpose() * y;
      Eigen::VectorXd weights = R.triangularView<Eigen::Upper>().solve(y_proj);
      return weights;
  }
  ```

---

### File 5: `linalg/health_index.py`
**Goal**: Unsupervised degradation tracking using Covariance Matrix Eigen-decomposition.

```python
def compute_health_index(X):
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    top_direction = eigvecs[:, -1]
    health = X_centered @ top_direction
    return health, eigvals[-1]
```
- **Explanation**:
  - `X - X.mean(axis=0)`: Shifts columns so each sensor has mean 0.
  - `np.cov(...)`: Computes $14 \times 14$ covariance matrix $C = \frac{1}{N-1} X_c^T X_c$.
  - `np.linalg.eigh(cov)`: Computes eigenvalues and eigenvectors for symmetric matrices.
  - `eigvecs[:, -1]`: Extracts eigenvector corresponding to largest eigenvalue.
  - `X_centered @ top_direction`: Projects 14 sensor dimensions onto 1D wear trajectory.
- **C++ parallel**:
  ```cpp
  std::pair<Eigen::VectorXd, double> compute_health_index(const Eigen::MatrixXd& X) {
      Eigen::MatrixXd X_centered = X.rowwise() - X.colwise().mean();
      Eigen::MatrixXd cov = (X_centered.transpose() * X_centered) / (X.rows() - 1.0);
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(cov);
      Eigen::VectorXd top_direction = es.eigenvectors().col(cov.cols() - 1);
      Eigen::VectorXd health = X_centered * top_direction;
      double top_eigval = es.eigenvalues()(cov.cols() - 1);
      return {health, top_eigval};
  }
  ```

---

### File 6: `linalg/evaluate.py`
**Goal**: Evaluate predictions on 100 test engines and compare against Scikit-Learn baseline.

```python
test_df = load_test_data()
last_rows = test_df.groupby("unit").tail(1).sort_values("unit")
X_test = last_rows[keep].values
y_true = np.clip(load_true_rul(), 0, 125)

predictions = X_test @ weights
rmse = np.sqrt(np.mean((predictions - y_true) ** 2))
```
- **Explanation**:
  - `last_rows = test_df.groupby("unit").tail(1)`: Selects the last recorded operating cycle for each test engine.
  - `predictions = X_test @ weights`: Computes $\hat{y} = X_{\text{test}} w$.
  - `rmse`: Calculates Root Mean Squared Error $\sqrt{\frac{1}{N}\sum (\hat{y}_i - y_i)^2}$.

```python
sk_model = LinearRegression().fit(X_train, y_train)
sk_rmse = np.sqrt(np.mean((sk_model.predict(X_test) - y_true) ** 2))
print("sklearn RMSE:", sk_rmse)
```
- **Explanation**: Compares our custom QR solver against industry-standard Scikit-Learn linear regression. Both produce near-identical test error (~20.99 vs 20.82 cycles).

---

### File 7: `tests/test_linalg.py`
**Goal**: Pytest unit tests asserting linear algebra invariants.

- `test_rref_rank_and_nullity`: Validates $\text{Rank} = 14$ and $\text{Nullity} = 21 - 14 = 7$.
- `test_gram_schmidt_orthonormal`: Asserts $\|Q^T Q - I\|_\infty < 10^{-5}$.
- `test_least_squares_matches_sklearn`: Asserts custom QR weights and predictions match Scikit-Learn within numerical tolerance.
- `test_health_index_spectral`: Asserts principal eigenvalue captures $> 80\%$ of variance.

---

### File 8: `run_demo.py`
**Goal**: Single executable demo orchestrating the end-to-end pipeline with Concept $\to$ Purpose $\to$ Outcome walkthrough.

Run with:
```bash
python run_demo.py --explain
```

---

## 5. Summary of Results & Viva Key Points

1. **Applied Linear Algebra Pipeline**:
   - **Feature Selection / Rank**: 21 sensors $\to$ 7 constant sensors dropped (Nullity = 7) $\to$ 14 linearly independent sensors (Rank = 14 via RREF).
   - **Conditioning**: Orthonormal basis constructed via Gram-Schmidt ($Q^T Q = I$, error $< 1.6 \times 10^{-8}$).
   - **Unsupervised Monitoring**: Covariance matrix eigen-decomposition yields a 1D Health Index capturing **86.83%** of engine degradation variance ($\lambda = 845.95$).
   - **Supervised Prediction**: Custom QR-based least squares achieves **Test RMSE = 20.99 cycles**, matching Scikit-Learn's baseline **20.82 cycles**.
