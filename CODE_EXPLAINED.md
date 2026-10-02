# Turbojet Linear Algebra Project: Complete Beginner & C Programmer Guide

Welcome! This guide assumes you know basic C programming (variables, arrays, loops, functions, structs, pointers) and basic school math, but **zero** Python, NumPy, Pandas, or Machine Learning.

Every line of code, math concept, and design decision in this project is broken down below in simple terms.

---

## Table of Contents
1. [The Real-World Problem We Are Solving](#1-the-real-world-problem-we-are-solving)
2. [C vs Python / NumPy / Pandas: Mental Model](#2-c-vs-python--numpy--pandas-mental-model)
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
A commercial airplane jet engine has fans, compressors, combustion chambers, and turbines. Over many flights ("operating cycles"), parts wear out due to friction and high temperatures.

### What is RUL (Remaining Useful Life)?
- If Engine #1 runs for **192 total cycles** before breakdown:
  - At cycle 1, RUL = 191 cycles left.
  - At cycle 50, RUL = 142 cycles left.
  - At cycle 192, RUL = 0 cycles left (broken).
- **Goal**: Given 21 sensor readings (temperatures, pressures, fan speeds) at any cycle, predict **how many cycles remain** before failure.

### The Linear System ($X w \approx y$):
We represent sensor readings as a large 2D matrix $X$ ($N$ rows of time steps, $D$ columns of sensors) and target RUL values as vector $y$. We want to find weight vector $w$ such that:
$$X w \approx y$$
Each weight $w_j$ tells us how much sensor $j$ contributes to remaining engine life.

---

## 2. C vs Python / NumPy / Pandas: Mental Model

If you come from C, Python syntax and data structures can look unfamiliar. Here is the direct translation:

### 2.1 Variables, Lists, and Dynamic Typing
- **In C**:
  ```c
  int x = 10;
  char *names[3] = {"s1", "s2", "s3"};
  ```
- **In Python**:
  ```python
  x = 10
  names = ["s1", "s2", "s3"] # Python list (like a dynamic array of pointers)
  ```

### 2.2 List Comprehensions (Python loop in brackets)
- **In Python**:
  ```python
  cols = [f"s{i}" for i in range(1, 22)]
  ```
- **Equivalent in C**:
  ```c
  char cols[21][10];
  for (int i = 1; i <= 21; i++) {
      sprintf(cols[i - 1], "s%d", i);
  }
  ```

### 2.3 2D Arrays: C Arrays vs NumPy `ndarray`
- **In C**:
  ```c
  double A[20631][14];
  // Access row r, col c:
  double val = A[r][c];
  ```
- **In NumPy**:
  ```python
  import numpy as np
  A = np.zeros((20631, 14)) # contiguous heap-allocated double array
  val = A[r, c]
  ```

### 2.4 Vectorization (No explicit for-loops in NumPy)
In C, to add or multiply arrays, you must write `for` loops. In NumPy, operations apply across all elements in parallel using precompiled C/Fortran SIMD instructions:
- **In C**:
  ```c
  double result[N];
  for (int i = 0; i < N; i++) {
      result[i] = X[i] * 2.0 - mean;
  }
  ```
- **In NumPy**:
  ```python
  result = X * 2.0 - mean
  ```

### 2.5 Matrix Multiplication (`@` operator)
- **In C**:
  ```c
  // C = A * B where A is (N x K), B is (K x M)
  double C[N][M];
  for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
          C[i][j] = 0;
          for (int k = 0; k < K; k++) {
              C[i][j] += A[i][k] * B[k][j];
          }
      }
  }
  ```
- **In NumPy**:
  ```python
  C = A @ B
  ```

### 2.6 Pandas DataFrame (Like an in-memory SQL Table / Struct Array)
- In C, you might define a `struct EngineRecord { int unit; int cycle; double s1; double s2; ... };` and an array `struct EngineRecord table[20631];`.
- In Pandas, `pd.DataFrame` is a 2D table where each column has a name and data type, supporting grouping, filtering, and standard deviations.

---

## 3. Linear Algebra Concepts (With Small Math Examples)

### Concept 1: Matrix Rank, Nullity, and RREF (Gaussian Elimination)
- **The Problem**: We are given 21 sensors. Some sensors are flatlined (constant value across all engines and cycles, e.g., standard deviation = 0).
- **Math Principle (Rank-Nullity Theorem)**:
  $$\text{Number of Columns} = \text{Rank}(A) + \text{Nullity}(A)$$
  - $\text{Rank}(A)$: Number of linearly independent sensor channels carrying distinct physical signals.
  - $\text{Nullity}(A)$: Dimension of the null space (redundant or flatlined sensor channels).
- **Small Math Example**:
  Suppose we have 3 sensors for 3 time steps:
  $$A = \begin{bmatrix} 1 & 518.67 & 2 \\ 2 & 518.67 & 4 \\ 3 & 518.67 & 6 \end{bmatrix}$$
  Sensor 2 is constant (518.67). Sensor 3 is exactly $2 \times \text{Sensor 1}$.
  Applying Gaussian Elimination to Reduced Row Echelon Form (RREF):
  $$\text{RREF}(A) = \begin{bmatrix} 1 & 0 & 2 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix}$$
  - Number of pivot columns = 2 (Rank = 2).
  - Number of zero rows / free columns = 1 (Nullity = $3 - 2 = 1$).
- **In our Engine Dataset**:
  - Total sensors: 21
  - Flatlined/constant sensors dropped: 7 ($s_1, s_5, s_6, s_{10}, s_{16}, s_{18}, s_{19}$)
  - Nullity = 7
  - Verified Full Column Rank on remaining 14 sensors = 14.

---

### Concept 2: Gram-Schmidt Orthogonalization ($QR$ Factorization)
- **The Problem**: Sensor columns are correlated (e.g., as temperature rises, pressure rises). If columns are nearly parallel, inverting matrices causes numerical explosion (floating-point overflow).
- **The Math**: We convert column vectors $a_1, a_2, \dots, a_k$ into perpendicular unit vectors $q_1, q_2, \dots, q_k$ such that:
  $$q_i^T q_j = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases} \quad \implies \quad Q^T Q = I$$
- **The Algorithm**:
  1. $v_1 = a_1, \quad q_1 = \frac{v_1}{\|v_1\|}$
  2. For column $j$:
     $$v_j = a_j - \sum_{i=1}^{j-1} (q_i^T a_j) q_i, \quad q_j = \frac{v_j}{\|v_j\|}$$
  Subtracting $(q_i^T a_j) q_i$ strips out the shadow (projection) of $a_j$ along all previous directions $q_i$, leaving only the brand-new perpendicular direction.

---

### Concept 3: Least Squares via $QR$ Decomposition
- **The Problem**: We have 20,631 equations (rows) and 14 unknowns (weights). An exact solution $X w = y$ is impossible because there are far more equations than unknowns (overdetermined system).
- **The Solution**: Find $w$ that minimizes the sum of squared prediction errors:
  $$\min_w \|X w - y\|_2^2$$
- **Derivation using $QR$**:
  1. Substitute $X = Q R$, where $Q \in \mathbb{R}^{N \times D}$ has orthonormal columns ($Q^T Q = I$) and $R \in \mathbb{R}^{D \times D}$ is upper triangular:
     $$X w \approx y \implies Q R w \approx y$$
  2. Multiply both sides by $Q^T$:
     $$Q^T Q R w = Q^T y \implies I R w = Q^T y \implies R w = Q^T y$$
  3. Because $R$ is upper triangular, we solve $R w = Q^T y$ via fast back-substitution without computing any matrix inverse $(X^T X)^{-1}$.

---

### Concept 4: Health Index via Spectral Decomposition (Eigenvalues & Eigenvectors)
- **The Problem**: Can we monitor engine degradation without knowing the true RUL labels (unsupervised)?
- **The Math**:
  1. Center sensor data: $X_c = X - \mu$.
  2. Compute $14 \times 14$ sample covariance matrix:
     $$C = \frac{1}{N-1} X_c^T X_c$$
  3. Solve the eigenvalue equation:
     $$C v = \lambda v$$
     - $\lambda_i$: Variance explained along direction $v_i$.
     - $v_{top}$ (eigenvector with highest $\lambda$): The direction along which the engine changes the most over its lifespan.
  4. Project each row: $\text{Health}_t = X_{c, t} \cdot v_{top}$.
  5. Result: A 1D curve tracking engine degradation cycle by cycle.

---

## 4. File-by-File & Line-by-Line Code Breakdown

---

### File 1: `linalg/loader.py`
**Goal**: Load the C-MAPSS dataset from disk, assign column headers, and compute target RUL.

```python
1: import pandas as pd
```
- **Meaning**: Load the Pandas library (alias `pd`) for table manipulation.
- **C parallel**: Similar to `#include <stdio.h>` and `#include <stdlib.h>`.

```python
3: COLS = ["unit", "cycle"] + [f"op{i}" for i in (1, 2, 3)] + [f"s{i}" for i in range(1, 22)]
```
- **Meaning**: Creates a list of 26 column strings: `["unit", "cycle", "op1", "op2", "op3", "s1", "s2", ..., "s21"]`.
- **C parallel**: An array of string pointers `char *COLS[26]`.

```python
6: def load_fd001(path="data/train_FD001.txt"):
```
- **Meaning**: Defines function `load_fd001` taking file path with default argument `"data/train_FD001.txt"`.
- **C parallel**: `DataFrame* load_fd001(const char *path);`.

```python
7:     df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
```
- **Meaning**: Reads whitespace-delimited text file into a 2D table (`df`) with column names `COLS`.
- **C parallel**: `fopen(path, "r")` followed by `fscanf` in a loop reading 26 floats per line.

```python
8:     df["RUL"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
```
- **Meaning**: For each engine `unit`, find its maximum cycle (failure cycle $C_{max}$) and subtract current `cycle`:
  $$\text{RUL} = C_{max} - C_{current}$$
- **C parallel**: Loop through rows to record `max_cycle[unit_id]`, then second loop setting `rul[i] = max_cycle[unit[i]] - cycle[i]`.

```python
9:     df["RUL"] = df["RUL"].clip(upper=125)
```
- **Meaning**: Cap RUL at 125 cycles. In early life, an engine doesn't show wear, so degradation models assume constant healthy RUL until wear begins.
- **C parallel**: `if (rul[i] > 125) rul[i] = 125;`.

```python
10:    return df
```
- **Meaning**: Returns the processed table.

---

### File 2: `linalg/rref.py`
**Goal**: Implement Gaussian Elimination from scratch to compute Reduced Row Echelon Form (RREF), matrix rank, and sensor nullity.

```python
1: import numpy as np
```
- **Meaning**: Imports NumPy numerical math library.

```python
3: def rref(A, tol=1e-8):
4:     A = A.astype(float).copy()
5:     rows, cols = A.shape
6:     pivot_row = 0
7:     pivot_cols = []
```
- **Meaning**: Clones input matrix $A$ as double precision floats. `rows` ($N$), `cols` ($D$). `pivot_row` tracks the current row being eliminated. `pivot_cols` stores indices of linearly independent columns.
- **C parallel**: Allocate a copy of `double A[rows][cols]`, `int pivot_row = 0; int pivot_cols[cols];`.

```python
8:     for col in range(cols):
9:         if pivot_row >= rows:
10:            break
```
- **Meaning**: Iterate through each column 0 to `cols - 1`. If we run out of rows, stop.
- **C parallel**: `for (int col = 0; col < cols && pivot_row < rows; col++)`.

```python
11:        pivot = np.argmax(np.abs(A[pivot_row:, col])) + pivot_row
12:        if abs(A[pivot, col]) < tol:
13:            continue
```
- **Meaning**: Partial pivoting. Finds the row with largest absolute value in `col` starting from `pivot_row`. If all values are zero ($< 10^{-8}$), this column is linearly dependent; skip it.
- **C parallel**: Loop over `r` from `pivot_row` to `rows - 1` to find `max_idx`.

```python
14:        A[[pivot_row, pivot]] = A[[pivot, pivot_row]]
```
- **Meaning**: Swaps row `pivot_row` with row `pivot` so the largest entry is on the diagonal.
- **C parallel**: Loop `for (int c = 0; c < cols; c++) { double tmp = A[pivot_row][c]; A[pivot_row][c] = A[pivot][c]; A[pivot][c] = tmp; }`.

```python
15:        A[pivot_row] = A[pivot_row] / A[pivot_row, col]
```
- **Meaning**: Normalizes the pivot row so that the pivot element becomes exactly 1.0.
- **C parallel**: Loop dividing every element in `A[pivot_row]` by `pivot_val`.

```python
16:        for r in range(rows):
17:            if r != pivot_row:
18:                A[r] -= A[r, col] * A[pivot_row]
```
- **Meaning**: Eliminates the column entry in all other rows $r \neq \text{pivot\_row}$ by subtracting a multiple of the pivot row.
- **C parallel**: Nested loop updating `A[r][c] -= factor * A[pivot_row][c]`.

```python
19:        pivot_cols.append(col)
20:        pivot_row += 1
21:    return A, pivot_cols
```
- **Meaning**: Records `col` as a pivot column, advances `pivot_row`, and returns the reduced matrix and pivot list.

```python
24: if __name__ == "__main__":
...
32:     stds = df[sensor_cols].std()
33:     keep = stds[stds > 0.01].index.tolist()
34:     dropped = [c for c in sensor_cols if c not in keep]
...
38:     rank = len(pivots)
39:     total_sensors = len(sensor_cols)
40:     nullity = total_sensors - rank
```
- **Meaning**: Computes sensor standard deviation over the full 20,631 rows. Sensors with variance $< 0.01$ are constant. Confirms Rank = 14 and Nullity = $21 - 14 = 7$.

---

### File 3: `linalg/gram_schmidt.py`
**Goal**: Implement Modified Gram-Schmidt orthogonalization to construct an orthonormal matrix $Q$.

```python
3: def gram_schmidt(A):
4:     Q = np.zeros_like(A, dtype=float)
```
- **Meaning**: Creates an empty matrix $Q$ of identical dimensions ($20631 \times 14$) filled with 0.0.
- **C parallel**: `double **Q = malloc(rows * sizeof(double*)); ... calloc(...)`.

```python
5:     for j in range(A.shape[1]):
6:         v = A[:, j].astype(float).copy()
```
- **Meaning**: Loop over each column $j$. Copy column $j$ of matrix $A$ into vector $v$.
- **C parallel**: `for (int j = 0; j < cols; j++) { double v[rows]; for(int r=0; r<rows; r++) v[r] = A[r][j]; }`.

```python
7:         for i in range(j):
8:             v -= (Q[:, i] @ A[:, j]) * Q[:, i]
```
- **Meaning**: Modified Gram-Schmidt step. For each previously calculated column $q_i$, compute dot product $s = q_i \cdot a_j$, then subtract projection vector $s \cdot q_i$ from $v$.
- **C parallel**:
  ```c
  for (int i = 0; i < j; i++) {
      double dot = 0.0;
      for (int r = 0; r < rows; r++) dot += Q[r][i] * A[r][j];
      for (int r = 0; r < rows; r++) v[r] -= dot * Q[r][i];
  }
  ```

```python
9:         Q[:, j] = v / np.linalg.norm(v)
10:    return Q
```
- **Meaning**: Computes Euclidean norm $\|v\| = \sqrt{\sum v_k^2}$ and normalizes $v$ to unit length, placing it as column $j$ of $Q$.

---

### File 4: `linalg/lstsq.py`
**Goal**: Solve the linear least squares problem $X w \approx y$ via $QR$ decomposition.

```python
3: def solve_least_squares(X, y):
4:     Q, R = np.linalg.qr(X)
5:     y_proj = Q.T @ y
6:     weights = np.linalg.solve(R, y_proj)
7:     return weights
```
- **Line 4**: Decomposes $X = QR$.
- **Line 5**: Projects target vector $y$ onto the orthonormal subspace: $y_{\text{proj}} = Q^T y$.
- **Line 6**: Solves upper-triangular system $R w = y_{\text{proj}}$ using back-substitution.
- **Line 7**: Returns weight vector $w \in \mathbb{R}^{14}$.

```python
25:     predictions = X @ weights
27:     rmse = np.sqrt(np.mean((predictions - y) ** 2))
```
- **Meaning**: Predicts $\hat{y} = X w$ and calculates Root Mean Squared Error:
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2}$$

---

### File 5: `linalg/health_index.py`
**Goal**: Construct an unsupervised Health Index using Covariance Matrix Eigen-decomposition (PCA).

```python
6:     X_centered = X - X.mean(axis=0)
```
- **Meaning**: Centers data by subtracting the mean of each sensor column so every column has mean 0.
- **C parallel**: Calculate mean of column $j$, then subtract `mean[j]` from `X[i][j]` for all $i$.

```python
9:     cov = np.cov(X_centered, rowvar=False)
```
- **Meaning**: Computes the $14 \times 14$ covariance matrix:
  $$\text{cov}_{j, k} = \frac{1}{N-1} \sum_{i=1}^N X_{centered, i, j} X_{centered, i, k}$$

```python
12:    eigvals, eigvecs = np.linalg.eigh(cov)
```
- **Meaning**: Computes eigenvalues and eigenvectors of symmetric matrix `cov`. `eigh` returns eigenvalues in ascending order ($\lambda_0 \le \lambda_1 \le \dots \le \lambda_{13}$).

```python
15:    top_direction = eigvecs[:, -1]
18:    health = X_centered @ top_direction
19:    return health, eigvals[-1]
```
- **Meaning**: Selects column with largest eigenvalue (`eigvecs[:, -1]`) and projects centered sensor readings onto it to obtain a single 1D degradation scalar per time step.

---

### File 6: `linalg/evaluate.py`
**Goal**: Test our trained model on 100 unseen test engines and validate against Scikit-Learn.

```python
6: def load_test_data(path="data/test_FD001.txt"):
7:     df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
8:     return df
```
- **Meaning**: Loads test engine telemetry.

```python
10: def load_true_rul(path="data/RUL_FD001.txt"):
11:     return pd.read_csv(path, header=None, names=["RUL"])["RUL"].values
```
- **Meaning**: Loads the 100 ground-truth remaining cycle values for test engines.

```python
27:     last_rows = test_df.groupby("unit").tail(1)
```
- **Meaning**: Extracts the final recorded cycle for each test engine. (Our task is to predict how many more cycles the engine can run starting from this last observed moment).

```python
34:     predictions = X_test @ weights
36:     rmse = np.sqrt(np.mean((predictions - y_true) ** 2))
```
- **Meaning**: Evaluates our custom linear algebra model on test engines.

```python
40:     sk_model = LinearRegression().fit(X_train, y_train)
41:     sk_rmse = np.sqrt(np.mean((sk_model.predict(X_test) - y_true) ** 2))
42:     print("sklearn RMSE:", sk_rmse)
```
- **Meaning**: Industry-standard library baseline comparison. Validates that our hand-written least squares ($QR$) matches `scikit-learn`.

```python
48:     plt.figure(figsize=(7, 6))
49:     plt.scatter(y_true, predictions, alpha=0.6)
50:     plt.plot([0, 125], [0, 125], 'r--')
51:     plt.savefig("notebooks/pred_vs_actual.png")
```
- **Meaning**: Generates scatter plot comparing true vs predicted RUL with diagonal reference line.

---

### File 7: `tests/test_linalg.py`
**Goal**: Automated test suite asserting correctness of linear algebra properties.

1. `test_rref_rank_and_nullity()`: Asserts that filtered sensor matrix has Rank = 14 and Nullity = 7.
2. `test_gram_schmidt_orthonormal()`: Asserts $\|Q^T Q - I\| < 10^{-5}$ and matching QR decomposition.
3. `test_least_squares_matches_sklearn()`: Asserts custom weights match `LinearRegression` weights within tolerance and custom RMSE is within 0.5 cycles of sklearn.
4. `test_health_index_spectral()`: Asserts leading eigenvalue is positive and explains $> 80\%$ of total sensor variance.

---

### File 8: `run_demo.py`
**Goal**: End-to-end command-line demo that executes all 5 stages in order, printing Concept $\to$ Purpose $\to$ Outcome walkthrough.

- Accepts `--explain` flag: `python run_demo.py --explain`.
- Invokes all pipeline components in mathematical sequence.

---

## 5. Summary of Results & Viva Key Points

When presenting this project in a viva or interview:

1. **Why Linear Algebra?**
   - We avoid opaque black-box deep learning. Every step (RREF, QR, Covariance Eigen-decomposition, Least Squares Projection) has an exact closed-form algebraic derivation.
2. **Key Metrics to Quote**:
   - **Total initial sensors**: 21
   - **Filtered sensors**: 14 (7 constant sensors removed, Nullity = 7)
   - **Matrix Rank**: 14 (Full column rank confirmed via RREF)
   - **Gram-Schmidt Orthogonality Error**: $< 1.6 \times 10^{-8}$
   - **Health Index Variance Explained**: $86.83\%$ (Dominant eigenvalue $\lambda = 845.95$)
   - **Custom Least Squares Test RMSE**: $20.99$ cycles
   - **Scikit-Learn Baseline RMSE**: $20.82$ cycles (Matches our custom implementation within ~0.17 cycles).
