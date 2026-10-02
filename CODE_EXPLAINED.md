# Turbojet Linear Algebra Project: Complete Beginner Guide

Welcome! This guide explains every mathematical concept, matrix operation, and line of code in this project in **plain English**.

No previous knowledge of Python, NumPy, Pandas, or Machine Learning is assumed.

---

## Table of Contents
1. [The Real-World Problem We Are Solving](#1-the-real-world-problem-we-are-solving)
2. [Quick Syntax Guide: Python vs Basic Code Concepts](#2-quick-syntax-guide-python-vs-basic-code-concepts)
3. [Linear Algebra Concepts Explained with Small Numerical Matrices](#3-linear-algebra-concepts-explained-with-small-numerical-matrices)
   - [Concept 1: Matrix Rank, Nullity, and Gaussian Elimination (RREF)](#concept-1-matrix-rank-nullity-and-gaussian-elimination-rref)
   - [Concept 2: Gram-Schmidt Orthogonalization (QR Decomposition)](#concept-2-gram-schmidt-orthogonalization-qr-decomposition)
   - [Concept 3: Overdetermined Linear Systems & Least Squares via QR](#concept-3-overdetermined-linear-systems--least-squares-via-qr)
   - [Concept 4: Covariance Matrix & Spectral Decomposition (PCA Health Index)](#concept-4-covariance-matrix--spectral-decomposition-pca-health-index)
4. [Line-by-Line Code Walkthrough](#4-line-by-line-code-walkthrough)
   - [File 1: linalg/loader.py](#file-1-linalgloaderpy)
   - [File 2: linalg/rref.py](#file-2-linalgrrefpy)
   - [File 3: linalg/gram_schmidt.py](#file-3-linalggram_schmidtpy)
   - [File 4: linalg/lstsq.py](#file-4-linalglstsqpy)
   - [File 5: linalg/health_index.py](#file-5-linalghealth_indexpy)
   - [File 6: linalg/evaluate.py](#file-6-linalgevaluatepy)
   - [File 7: tests/test_linalg.py](#file-7-teststest_linalgpy)
   - [File 8: run_demo.py](#file-8-run_demopy)
5. [Summary of Results & Viva Key Takeaways](#5-summary-of-results--viva-key-takeaways)

---

## 1. The Real-World Problem We Are Solving

### What is a Turbofan Jet Engine?
A commercial airplane jet engine contains rotating fan blades, compressors, combustion chambers, and turbines. As the engine flies trips ("operating cycles"), heat and mechanical friction cause wear and tear.

### What is RUL (Remaining Useful Life)?
- If Engine #1 runs for **192 total cycles** before breakdown:
  - At cycle 1: RUL = 191 cycles left.
  - At cycle 50: RUL = 142 cycles left.
  - At cycle 192: RUL = 0 cycles left (engine breakdown).
- **Our Goal**: Given 21 sensor measurements (temperatures, pressures, fan speeds) at any point in time, predict **how many cycles remain** before the engine fails.

### The Linear System ($X w \approx y$):
- **Matrix $X$ ($20631 \text{ rows} \times 14 \text{ sensor columns}$)**: Each row is a snapshot in time; each column is a sensor reading.
- **Vector $y$ ($20631 \text{ values}$)**: The true remaining cycles (RUL) for each row.
- **Vector $w$ ($14 \text{ weights}$)**: The unknown multipliers we need to find.
$$\begin{bmatrix} x_{1,1} & x_{1,2} & \cdots & x_{1,14} \\ x_{2,1} & x_{2,2} & \cdots & x_{2,14} \\ \vdots & \vdots & \ddots & \vdots \\ x_{N,1} & x_{N,2} & \cdots & x_{N,14} \end{bmatrix} \begin{bmatrix} w_1 \\ w_2 \\ \vdots \\ w_{14} \end{bmatrix} \approx \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_N \end{bmatrix}$$

---

## 2. Quick Syntax Guide: Python vs Basic Code Concepts

If you are new to Python, here is what each syntax element means:

| Python Syntax | Meaning in Plain English | Standard Programming Parallel |
| :--- | :--- | :--- |
| `names = ["s1", "s2"]` | A dynamic list of strings. | `std::vector<std::string>` or array of strings |
| `[f"s{i}" for i in range(1, 22)]` | Loop from 1 to 21, generating `"s1"`, `"s2"`, ..., `"s21"`. | Loop appending strings to a vector |
| `A = np.zeros((N, D))` | Allocates a 2D matrix of floating-point numbers filled with 0.0. | Heap-allocated 2D array of `double` |
| `A[r, c]` | Access row `r`, column `c`. | `A[r][c]` |
| `X * 2.0 - mean` | Performs the math on **every element** of the matrix without a manual loop. | SIMD vectorized loop over array |
| `A @ B` | Matrix multiplication (or dot product of two vectors). | Matrix multiply function |
| `df = pd.read_csv(...)` | Loads a table from a text file into an in-memory spreadsheet (`DataFrame`). | Struct array / Table parser |

---

## 3. Linear Algebra Concepts Explained with Small Numerical Matrices

---

### Concept 1: Matrix Rank, Nullity, and Gaussian Elimination (RREF)

#### Why do we need this?
We start with 21 sensors. Some sensors are flatlined (constant value across all engines) or linearly dependent (duplicates of other sensors). If we feed redundant or dead columns into our math equations, matrix operations fail or produce unstable solutions.

#### The Theorem (Rank-Nullity Theorem):
$$\text{Total Columns} = \text{Rank}(A) + \text{Nullity}(A)$$
- **$\text{Rank}(A)$**: The number of linearly independent columns (unique, useful physical signals).
- **$\text{Nullity}(A)$**: The number of redundant or dead columns (the dimension of the null space).

#### Step-by-Step 3x3 Matrix Example:
Suppose we have 3 time steps and 3 sensors:
$$A = \begin{bmatrix} 1 & 500 & 2 \\ 2 & 500 & 4 \\ 3 & 500 & 6 \end{bmatrix}$$
Notice:
- Column 1: Changes ($1, 2, 3$).
- Column 2: Completely constant ($500, 500, 500$).
- Column 3: Exact duplicate of Column 1 multiplied by 2 ($2 \times \text{Col 1}$).

Let's apply Gaussian Elimination (row operations):
1. Normalize row 1:
   $$\text{Row 1} \to \text{Row 1}$$
2. Subtract multiples of Row 1 from Row 2 and Row 3:
   $$\text{Row 2} - 2 \times \text{Row 1} \implies \begin{bmatrix} 0 & 0 & 0 \end{bmatrix}$$
   $$\text{Row 3} - 3 \times \text{Row 1} \implies \begin{bmatrix} 0 & 0 & 0 \end{bmatrix}$$
3. Result in Reduced Row Echelon Form (RREF):
   $$\text{RREF}(A) = \begin{bmatrix} 1 & 500 & 2 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}$$
- Non-zero pivot rows = 1 independent signal direction ($\text{Rank} = 1$).
- Dead / redundant columns = 2 ($\text{Nullity} = 3 - 1 = 2$).

#### In Our Engine Dataset:
- Initial sensor columns: **21**
- Constant / flatlined sensors dropped: **7** ($s_1, s_5, s_6, s_{10}, s_{16}, s_{18}, s_{19}$)
- **Nullity = 7**
- We run RREF on the remaining 14 sensors over all 20,631 rows and confirm **$\text{Rank} = 14$** (full column rank).

---

### Concept 2: Gram-Schmidt Orthogonalization (QR Decomposition)

#### Why do we need this?
Sensor readings are correlated (for example, as exhaust temperature increases, chamber pressure also increases). When columns are nearly parallel, inverting matrices causes severe rounding errors.

We transform the columns into **perpendicular (orthogonal)** unit vectors so that:
$$q_i \cdot q_j = \begin{cases} 1 & \text{if } i = j \text{ (unit length)} \\ 0 & \text{if } i \neq j \text{ (perpendicular)} \end{cases} \quad \implies \quad Q^T Q = I$$

#### Step-by-Step 2D Matrix Example:
Suppose we have two sensor column vectors:
$$a_1 = \begin{bmatrix} 3 \\ 4 \end{bmatrix}, \quad a_2 = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$$

**Step 1: Normalize $a_1$ to unit length $q_1$**:
- Length of $a_1$: $\|a_1\| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = 5$.
- Unit vector $q_1$:
  $$q_1 = \frac{a_1}{5} = \begin{bmatrix} 0.6 \\ 0.8 \end{bmatrix}$$

**Step 2: Remove the shadow (projection) of $a_2$ along $q_1$**:
- Dot product $q_1 \cdot a_2 = (0.6 \times 1) + (0.8 \times 2) = 0.6 + 1.6 = 2.2$.
- Projection vector:
  $$\text{proj}_{q_1}(a_2) = 2.2 \times q_1 = 2.2 \begin{bmatrix} 0.6 \\ 0.8 \end{bmatrix} = \begin{bmatrix} 1.32 \\ 1.76 \end{bmatrix}$$
- Subtract projection to find the perpendicular component $v_2$:
  $$v_2 = a_2 - \text{proj}_{q_1}(a_2) = \begin{bmatrix} 1 \\ 2 \end{bmatrix} - \begin{bmatrix} 1.32 \\ 1.76 \end{bmatrix} = \begin{bmatrix} -0.32 \\ 0.24 \end{bmatrix}$$

**Step 3: Normalize $v_2$ to unit length $q_2$**:
- Length of $v_2$: $\|v_2\| = \sqrt{(-0.32)^2 + (0.24)^2} = \sqrt{0.1024 + 0.0576} = \sqrt{0.16} = 0.4$.
- Unit vector $q_2$:
  $$q_2 = \frac{v_2}{0.4} = \begin{bmatrix} -0.8 \\ 0.6 \end{bmatrix}$$

**Check Orthogonality**:
$$q_1 \cdot q_2 = (0.6 \times -0.8) + (0.8 \times 0.6) = -0.48 + 0.48 = 0.0$$
The vectors are now **100% perpendicular**.

---

### Concept 3: Overdetermined Linear Systems & Least Squares via QR

#### Why do we need this?
We have **20,631 rows (equations)** but only **14 sensor weights (unknowns)**:
$$X w \approx y$$
There is no exact solution that satisfies all 20,631 rows at once. We want to find the weight vector $w$ that minimizes the total squared error $\sum (Xw - y)^2$.

#### How QR Solves Least Squares Without Inverting Matrices:
1. Decompose matrix $X$ into $Q$ (orthogonal columns) and $R$ (upper-triangular matrix):
   $$X = Q R$$
2. Substitute into our equation:
   $$Q R w \approx y$$
3. Multiply both sides from the left by $Q^T$:
   $$(Q^T Q) R w = Q^T y$$
4. Since $Q$ has orthonormal columns, $Q^T Q = I$ (Identity matrix):
   $$I R w = Q^T y \implies R w = Q^T y$$
5. Because $R$ is an **upper-triangular matrix** (all entries below the main diagonal are 0), we can solve for $w$ instantly using **back-substitution** (solve for the last weight first, then plug it into the row above, repeating up to the top). No matrix inverse $(X^T X)^{-1}$ is ever calculated.

---

### Concept 4: Covariance Matrix & Spectral Decomposition (PCA Health Index)

#### Why do we need this?
What if we do not know the true RUL labels and want to track engine health purely from the sensor data (unsupervised)?

#### How it Works:
1. **Center the data**: Subtract the mean of each sensor so every sensor column averages to 0.
2. **Covariance Matrix ($14 \times 14$)**:
   $$C = \frac{1}{N-1} X_{\text{centered}}^T X_{\text{centered}}$$
   - If sensor $j$ and sensor $k$ increase together, $C_{j,k} > 0$.
   - If they move in opposite directions, $C_{j,k} < 0$.
3. **Eigen-decomposition**:
   $$C v = \lambda v$$
   - $v$ is an **eigenvector** (a direction in 14-dimensional sensor space).
   - $\lambda$ is the **eigenvalue** (the amount of variance/energy along that direction).
4. **Dominant Direction**:
   - The eigenvector $v_{\text{top}}$ with the highest eigenvalue $\lambda_{\text{top}}$ points in the direction where engine sensor changes are strongest across its entire life.
5. **Health Index**:
   - Multiply each engine's centered sensor row by $v_{\text{top}}$ to collapse 14 sensor readings into a single 1D number:
     $$\text{Health Index} = X_{\text{centered}} \cdot v_{\text{top}}$$
   - In our project, this single direction captures **86.83%** of all sensor variance in the engine.

---

## 4. Line-by-Line Code Walkthrough

---

### File 1: `linalg/loader.py`
**Goal**: Read raw engine data from disk, name all columns, and compute target RUL.

```python
1: import pandas as pd
```
- Imports the Pandas data table library.

```python
3: COLS = ["unit", "cycle"] + [f"op{i}" for i in (1, 2, 3)] + [f"s{i}" for i in range(1, 22)]
```
- Creates a list of 26 column name strings: `unit` (engine ID), `cycle` (flight number), `op1, op2, op3` (operating settings), and `s1` through `s21` (sensor channels).

```python
6: def load_fd001(path="data/train_FD001.txt"):
```
- Defines a function `load_fd001` that loads training data from `path`.

```python
7:     df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
```
- Reads the space-separated text file into a table with the 26 column headers.

```python
8:     df["RUL"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
```
- For each engine ID (`unit`), finds the maximum cycle count (when it failed), and subtracts the current cycle to get the remaining cycles before failure.

```python
9:     df["RUL"] = df["RUL"].clip(upper=125)
```
- Caps RUL at 125 cycles. In early life, engines show no measurable degradation, so standard practice is to treat RUL as constant (125) until wear begins.

```python
10:    return df
```
- Returns the complete table.

---

### File 2: `linalg/rref.py`
**Goal**: Implement Gaussian elimination from scratch to find Reduced Row Echelon Form, Rank, and Nullity.

```python
3: def rref(A, tol=1e-8):
4:     A = A.astype(float).copy()
5:     rows, cols = A.shape
6:     pivot_row = 0
7:     pivot_cols = []
```
- Copies input matrix $A$ as floating-point numbers. `rows` = number of rows ($N$), `cols` = number of columns ($D$). Initializes `pivot_row = 0` and an empty list `pivot_cols`.

```python
8:     for col in range(cols):
9:         if pivot_row >= rows:
10:            break
```
- Iterates through each column from left to right. Stops if we have exhausted all rows.

```python
11:        pivot = np.argmax(np.abs(A[pivot_row:, col])) + pivot_row
12:        if abs(A[pivot, col]) < tol:
13:            continue
```
- **Partial Pivoting**: Finds the row with the largest absolute value in the current column to prevent division by near-zero numbers. If all values are 0 (smaller than $10^{-8}$), this column is redundant; skip to the next column.

```python
14:        A[[pivot_row, pivot]] = A[[pivot, pivot_row]]
```
- Swaps the current `pivot_row` with the row containing the largest pivot value.

```python
15:        A[pivot_row] = A[pivot_row] / A[pivot_row, col]
```
- Divides the entire pivot row by its leading value so that the pivot element becomes exactly 1.0.

```python
16:        for r in range(rows):
17:            if r != pivot_row:
18:                A[r] -= A[r, col] * A[pivot_row]
```
- Eliminates the current column's values in all other rows by subtracting a multiple of the pivot row, making all other entries in this column 0.

```python
19:        pivot_cols.append(col)
20:        pivot_row += 1
21:    return A, pivot_cols
```
- Saves the index of the pivot column, moves to the next row, and returns the simplified matrix and pivot column list.

```python
28:    stds = df[sensor_cols].std()
29:    keep = stds[stds > 0.01].index.tolist()
30:    dropped = [c for c in sensor_cols if c not in keep]
```
- Calculates standard deviation of all 21 sensors over all 20,631 records. Drops 7 sensors with variance $< 0.01$ (dead/constant channels).

```python
32:    X = df[keep].values
33:    R, pivots = rref(X)
34:    rank = len(pivots)
35:    nullity = len(sensor_cols) - rank
```
- Runs RREF on the remaining 14 sensors, confirming $\text{Rank} = 14$ and $\text{Nullity} = 21 - 14 = 7$.

---

### File 3: `linalg/gram_schmidt.py`
**Goal**: Build an orthonormal matrix $Q$ ($Q^T Q = I$) from sensor columns.

```python
3: def gram_schmidt(A):
4:     Q = np.zeros_like(A, dtype=float)
```
- Allocates a matrix $Q$ of the same shape as $A$ filled with zeros.

```python
5:     for j in range(A.shape[1]):
6:         v = A[:, j].astype(float).copy()
```
- Loops through each column $j$ of matrix $A$ and copies it into vector $v$.

```python
7:         for i in range(j):
8:             v -= (Q[:, i] @ A[:, j]) * Q[:, i]
```
- Subtracts the projection of column $a_j$ along all previously computed orthonormal columns $q_i$.

```python
9:         Q[:, j] = v / np.linalg.norm(v)
10:    return Q
```
- Normalizes vector $v$ by dividing by its Euclidean length ($\sqrt{\sum v_k^2}$) and stores it as column $j$ in $Q$.

---

### File 4: `linalg/lstsq.py`
**Goal**: Solve the least squares problem $X w \approx y$ to find sensor weights.

```python
3: def solve_least_squares(X, y):
4:     Q, R = np.linalg.qr(X)
5:     y_proj = Q.T @ y
6:     weights = np.linalg.solve(R, y_proj)
7:     return weights
```
- `np.linalg.qr(X)`: Breaks $X$ into orthogonal matrix $Q$ and upper-triangular matrix $R$.
- `y_proj = Q.T @ y`: Projects target vector $y$ onto the orthogonal space.
- `np.linalg.solve(R, y_proj)`: Solves upper-triangular system $R w = Q^T y$ via back-substitution to return weight vector $w$.

```python
25:     predictions = X @ weights
27:     rmse = np.sqrt(np.mean((predictions - y) ** 2))
```
- Multiplies sensor matrix $X$ by weights $w$ to predict RUL, and calculates Root Mean Squared Error (RMSE).

---

### File 5: `linalg/health_index.py`
**Goal**: Track engine wear using Covariance Matrix Eigen-decomposition.

```python
6:     X_centered = X - X.mean(axis=0)
```
- Centers sensor data by subtracting each column's average value.

```python
9:     cov = np.cov(X_centered, rowvar=False)
```
- Computes the $14 \times 14$ covariance matrix of sensors.

```python
12:    eigvals, eigvecs = np.linalg.eigh(cov)
```
- Computes eigenvalues (variance per direction) and eigenvectors (directions) of the covariance matrix.

```python
15:    top_direction = eigvecs[:, -1]
18:    health = X_centered @ top_direction
19:    return health, eigvals[-1]
```
- Picks the eigenvector corresponding to the largest eigenvalue (`eigvecs[:, -1]`) and projects the centered sensor rows onto it to produce a 1D health curve.

---

### File 6: `linalg/evaluate.py`
**Goal**: Test model on 100 unseen test engines and validate against Scikit-Learn.

```python
26:    test_df = load_test_data()
27:    last_rows = test_df.groupby("unit").tail(1)
28:    last_rows = last_rows.sort_values("unit")
```
- Loads test telemetry and takes only the **final recorded operating cycle** for each of the 100 test engines.

```python
34:    predictions = X_test @ weights
36:    rmse = np.sqrt(np.mean((predictions - y_true) ** 2))
```
- Predicts RUL using our custom least squares weights and computes Test RMSE.

```python
40:    sk_model = LinearRegression().fit(X_train, y_train)
41:    sk_rmse = np.sqrt(np.mean((sk_model.predict(X_test) - y_true) ** 2))
42:    print("sklearn RMSE:", sk_rmse)
```
- Fits Scikit-Learn's standard `LinearRegression` model on the exact same data to prove our hand-written linear algebra matches industry libraries.

```python
48:    plt.scatter(y_true, predictions, alpha=0.6)
49:    plt.plot([0, 125], [0, 125], 'r--')
50:    plt.savefig("notebooks/pred_vs_actual.png")
```
- Saves a scatter plot comparing predicted vs true RUL with a diagonal reference line (perfect prediction line).

---

### File 7: `tests/test_linalg.py`
**Goal**: Automated unit test suite verifying linear algebra math properties.

- `test_rref_rank_and_nullity`: Checks that rank is 14 and nullity is 7 ($21 - 14 = 7$).
- `test_gram_schmidt_orthonormal`: Checks that $Q^T Q = I$ within numerical tolerance ($10^{-5}$).
- `test_least_squares_matches_sklearn`: Checks that custom least-squares weights and predictions match Scikit-Learn.
- `test_health_index_spectral`: Checks that the leading eigenvalue captures $> 80\%$ of total sensor variance.

---

### File 8: `run_demo.py`
**Goal**: Orchestrates all 5 stages from data loading to evaluation in order.

Run with:
```bash
python run_demo.py --explain
```

---

## 5. Summary of Results & Viva Key Takeaways

1. **Applied Linear Algebra Foundation**:
   - **Feature Selection / Rank**: 21 sensors $\to$ 7 constant sensors dropped ($\text{Nullity} = 7$) $\to$ 14 linearly independent sensors ($\text{Rank} = 14$ via RREF).
   - **Conditioning**: Orthonormal basis constructed via Gram-Schmidt ($Q^T Q = I$, orthogonality error $< 1.6 \times 10^{-8}$).
   - **Unsupervised Monitoring**: Covariance matrix eigen-decomposition yields a 1D Health Index capturing **86.83%** of engine degradation variance ($\lambda = 845.95$).
   - **Supervised Prediction**: Custom QR-based least squares achieves **Test RMSE = 20.99 cycles**, matching Scikit-Learn's benchmark **20.82 cycles**.
