import nbformat as nbf
import os

OUT = "/workspace/notebooks/phase-3-data-science-libraries"


def new_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    }
    return nb


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def save(nb, filename):
    path = os.path.join(OUT, filename)
    with open(path, "w") as f:
        nbf.write(nb, f)
    return path


# ---------------------------------------------------------------------------
# LECTURE 17 - NumPy Essentials
# ---------------------------------------------------------------------------
nb17 = new_notebook()
cells17 = []

cells17.append(md("# Lecture 17: NumPy Essentials: Arrays and Vectorisation"))
cells17.append(md(
    "## Learning Objectives\n\n"
    "- Create NumPy arrays using multiple constructor functions\n"
    "- Access and manipulate array elements via indexing, slicing, and boolean masking\n"
    "- Apply vectorised arithmetic and universal functions (ufuncs)\n"
    "- Understand and apply NumPy broadcasting rules\n"
    "- Use aggregation methods, np.where(), and np.clip() for data transformations"
))
cells17.append(md(
    "## Key Topics\n\n"
    "- np.array(), np.zeros(), np.ones(), np.arange(), np.linspace()\n"
    "- Array attributes: .shape, .dtype, .ndim\n"
    "- Indexing, slicing, and boolean masking\n"
    "- Vectorised arithmetic and universal functions (ufuncs)\n"
    "- Broadcasting rules\n"
    "- np.where(), np.clip()\n"
    "- Aggregation methods"
))

cells17.append(md(
    "### Creating Arrays with NumPy Constructors\n\n"
    "NumPy provides a rich set of functions to create arrays. The most fundamental is `np.array()`, which converts a Python list or tuple into an ndarray. For common use cases, specialised constructors like `np.zeros()`, `np.ones()`, `np.arange()`, and `np.linspace()` save time and make code more readable.\n\n"
    "`np.zeros()` creates an array filled with 0s, useful for initialising data structures or accumulators. `np.ones()` does the same with 1s. `np.arange()` generates evenly spaced values over a given range (like Python's `range()` but returns an array). `np.linspace()` generates a specified number of evenly spaced points between two bounds, which is ideal for mathematical plotting."
))
cells17.append(code(
    "import numpy as np\n\n"
    "# Creating arrays from lists\n"
    "arr1 = np.array([1, 2, 3, 4, 5])\n"
    "print('From list:', arr1)\n\n"
    "# Zeros and ones\n"
    "zeros = np.zeros((2, 3))\n"
    "ones = np.ones((3, 2))\n"
    "print('Zeros (2x3):\\n', zeros)\n"
    "print('Ones (3x2):\\n', ones)"
))
cells17.append(code(
    "# arange and linspace\n"
    "arange_arr = np.arange(0, 10, 2)     # start=0, stop=10, step=2\n"
    "linspace_arr = np.linspace(0, 1, 5)  # 5 points from 0 to 1 inclusive\n"
    "print('arange:', arange_arr)\n"
    "print('linspace:', linspace_arr)"
))
cells17.append(code(
    "# Array attributes: shape, dtype, ndim\n"
    "arr = np.array([[1, 2, 3], [4, 5, 6]])\n"
    "print('Array:\\n', arr)\n"
    "print('Shape:', arr.shape)\n"
    "print('Data type:', arr.dtype)\n"
    "print('Number of dims:', arr.ndim)"
))

cells17.append(md(
    "### Indexing, Slicing, and Boolean Masking\n\n"
    "NumPy supports all the familiar Python slicing syntax, but extended to multiple dimensions. You can select individual elements with `arr[i, j]`, slice rows with `arr[i:j, :]`, or step through dimensions with `arr[:, ::2]`. Boolean masking is one of NumPy's most powerful features: you pass a boolean array of the same shape, and NumPy returns only the elements where the mask is `True`. This makes conditional selection extremely concise and fast.\n\n"
    "Boolean masks are often created by applying comparison operators directly to arrays, such as `arr > 5`. The result can be used to filter, replace, or count elements meeting a condition."
))
cells17.append(code(
    "# Indexing and slicing\n"
    "arr = np.arange(12).reshape(3, 4)\n"
    "print('Original array:\\n', arr)\n"
    "print('Element [1, 2]:', arr[1, 2])\n"
    "print('First row:', arr[0, :])\n"
    "print('Last column:', arr[:, -1])\n"
    "print('Submatrix (rows 0-1, cols 1-3):\\n', arr[0:2, 1:3])"
))
cells17.append(code(
    "# Boolean masking for filtering\n"
    "data = np.array([12, 5, 8, 19, 3, 27, 14])\n"
    "mask = data > 10\n"
    "print('Data:', data)\n"
    "print('Mask:', mask)\n"
    "print('Values > 10:', data[mask])\n\n"
    "# Use mask to replace values\n"
    "data[data < 10] = 0\n"
    "print('After zeroing values < 10:', data)"
))

cells17.append(md(
    "### Vectorised Arithmetic and Universal Functions\n\n"
    "Vectorisation means applying an operation to every element of an array simultaneously, without writing an explicit loop. NumPy implements this through universal functions (ufuncs) such as `np.add`, `np.sqrt`, `np.exp`, and `np.sin`. These operate element-by-element and are implemented in compiled C code, making them orders of magnitude faster than Python loops.\n\n"
    "Arithmetic operators (`+`, `-`, `*`, `/`, `**`) are overloaded to behave as ufuncs. When you write `arr1 + arr2`, NumPy pairs up corresponding elements and computes the result in one pass. This not only makes code cleaner but also enables much better performance as dataset sizes grow."
))
cells17.append(code(
    "# Vectorised arithmetic\n"
    "a = np.array([1, 2, 3, 4])\n"
    "b = np.array([10, 20, 30, 40])\n"
    "print('a + b:', a + b)\n"
    "print('a * b:', a * b)\n"
    "print('a ** 2:', a ** 2)\n"
    "print('sqrt(a):', np.sqrt(a))\n"
    "print('exp(a):', np.exp(a))"
))
cells17.append(code(
    "# Vectorised vs loop performance comparison\n"
    "import time\n\n"
    "n = 10_000_000\n"
    "data = np.random.randn(n)\n\n"
    "# Python loop\n"
    "start = time.time()\n"
    "result_loop = [x**2 + 2*x + 1 for x in data]\n"
    "loop_time = time.time() - start\n\n"
    "# Vectorised NumPy\n"
    "start = time.time()\n"
    "result_np = data**2 + 2*data + 1\n"
    "np_time = time.time() - start\n\n"
    "print(f'Python loop: {loop_time:.3f}s')\n"
    "print(f'NumPy:       {np_time:.3f}s')\n"
    "print(f'Speedup:     {loop_time / np_time:.1f}x')"
))

cells17.append(md(
    "### Broadcasting Rules\n\n"
    "Broadcasting allows NumPy to perform arithmetic between arrays of different shapes. Instead of requiring identical shapes, NumPy \"stretches\" the smaller array across the larger one along missing or size-1 dimensions. The rule is simple: two dimensions are compatible when they are equal, or one of them is 1. If neither condition holds, broadcasting fails and NumPy raises a `ValueError`.\n\n"
    "For example, adding a shape `(3,)` array to a shape `(4, 3)` array works because the 1-D array is broadcast along the rows. Similarly, adding a scalar to any array broadcasts the scalar across every element. Understanding broadcasting is essential for writing clean, efficient array code."
))
cells17.append(code(
    "# Broadcasting in action\n"
    "matrix = np.array([[1, 2, 3],\n"
    "                   [4, 5, 6],\n"
    "                   [7, 8, 9],\n"
    "                   [10, 11, 12]])\n"
    "row_mean = np.mean(matrix, axis=0)\n"
    "print('Matrix shape:', matrix.shape)\n"
    "print('Row mean:', row_mean)\n\n"
    "# Broadcast row_mean across all rows\n"
    "centered = matrix - row_mean\n"
    "print('Centered matrix (broadcast subtract):\\n', centered)\n\n"
    "# Scalar broadcasting\n"
    "print('Add 10 to every element:\\n', matrix + 10)"
))
cells17.append(code(
    "# Broadcasting rules in action\n"
    "a = np.ones((4, 3))\n"
    "b = np.array([10, 20, 30])\n"
    "print('a shape:', a.shape, ', b shape:', b.shape)\n"
    "print('a + b (b broadcast):\\n', a + b)\n\n"
    "# Column vector broadcast\n"
    "c = np.array([[1], [2], [3], [4]])\n"
    "print('c shape:', c.shape)\n"
    "print('a + c (c broadcast):\\n', a + c)"
))

cells17.append(md(
    "### np.where(), np.clip(), and Aggregation Methods\n\n"
    "`np.where(condition, x, y)` returns elements chosen from `x` or `y` depending on the condition. It is the vectorised equivalent of an if-else statement. `np.clip(array, min, max)` caps all values to lie within `[min, max]`, which is useful for handling outliers.\n\n"
    "Aggregation methods like `.sum()`, `.mean()`, `.min()`, `.max()`, `.std()`, and `.cumsum()` reduce an array along one or all axes. Passing the `axis` parameter lets you control which dimension to reduce, giving you row-wise or column-wise statistics with a single call."
))
cells17.append(code(
    "# np.where() and np.clip()\n"
    "scores = np.array([55, 82, 91, 47, 68, 73])\n\n"
    "# np.where: pass/fail classification\n"
    "result = np.where(scores >= 60, 'Pass', 'Fail')\n"
    "print('Scores:', scores)\n"
    "print('Results:', result)\n\n"
    "# np.clip: cap outliers\n"
    "prices = np.array([1.5, 2.3, 100.0, 2.1, 0.5, 3.0])\n"
    "clipped = np.clip(prices, 1.0, 10.0)\n"
    "print('Original prices:', prices)\n"
    "print('Clipped prices:', clipped)"
))
cells17.append(code(
    "# Aggregation methods\n"
    "arr = np.arange(1, 13).reshape(3, 4)\n"
    "print('Array:\\n', arr)\n"
    "print('Sum all:', arr.sum())\n"
    "print('Mean all:', arr.mean())\n"
    "print('Sum axis=0 (column sum):', arr.sum(axis=0))\n"
    "print('Sum axis=1 (row sum):', arr.sum(axis=1))\n"
    "print('Cumulative sum:', np.cumsum(np.arange(1, 11)))"
))

cells17.append(md(
    "## Data Science Connection\n\n"
    "NumPy is the foundation of nearly every data science library in Python. Pandas, scikit-learn, SciPy, and even TensorFlow and PyTorch build on NumPy arrays internally. Vectorisation and broadcasting let you process millions of data points efficiently, and boolean masking is the engine behind most data filtering operations. Mastering NumPy is the first step toward writing high-performance data pipelines."
))
nb17.cells = cells17
save(nb17, "Lecture 17 - NumPy Essentials.ipynb")

# ---------------------------------------------------------------------------
# LECTURE 18 - NumPy Advanced
# ---------------------------------------------------------------------------
nb18 = new_notebook()
cells18 = []

cells18.append(md("# Lecture 18: NumPy Advanced: Linear Algebra and Randomness"))
cells18.append(md(
    "## Learning Objectives\n\n"
    "- Reshape, transpose, stack, and split NumPy arrays\n"
    "- Perform matrix multiplication with np.dot() and the @ operator\n"
    "- Compute matrix inverses and eigenvalues with np.linalg\n"
    "- Generate random numbers using np.random module\n"
    "- Use random seeds for reproducible experiments\n"
    "- Compare Python loop performance against vectorised NumPy"
))
cells18.append(md(
    "## Key Topics\n\n"
    "- Reshaping, transposing, stacking, splitting\n"
    "- np.dot(), @ operator\n"
    "- np.linalg.inv, np.linalg.eig\n"
    "- Random module: np.random.seed(), .randn(), .choice(), .uniform()\n"
    "- Setting random seeds for reproducibility\n"
    "- Performance comparison: Python loops vs vectorised NumPy"
))

cells18.append(md(
    "### Reshaping, Transposing, Stacking, and Splitting\n\n"
    "Manipulating array shapes is a common task in data preparation. `reshape()` changes an array's dimensions without altering its data (the total number of elements must stay the same). `T` or `transpose()` flips axes, turning rows into columns. `stack()` joins arrays along a new axis, while `concatenate()` joins along an existing axis. `split()` does the reverse, dividing an array into multiple sub-arrays.\n\n"
    "These operations are essential when preparing data for machine learning models, which often expect specific input shapes. For example, you might stack feature arrays column-wise or split a dataset into training and validation folds."
))
cells18.append(code(
    "# Reshaping and transposing\n"
    "arr = np.arange(12)\n"
    "print('Original:', arr)\n\n"
    "reshaped = arr.reshape(3, 4)\n"
    "print('Reshaped (3x4):\\n', reshaped)\n\n"
    "transposed = reshaped.T\n"
    "print('Transposed:\\n', transposed)"
))
cells18.append(code(
    "# Stacking and splitting\n"
    "a = np.array([1, 2, 3])\n"
    "b = np.array([4, 5, 6])\n\n"
    "print('Vertical stack:\\n', np.vstack([a, b]))\n"
    "print('Horizontal stack:\\n', np.hstack([a, b]))\n"
    "print('Column stack:\\n', np.column_stack([a, b]))\n\n"
    "# Splitting\n"
    "arr = np.arange(10)\n"
    "first, second, third = np.split(arr, [3, 7])\n"
    "print('Split result:', first, second, third)"
))

cells18.append(md(
    "### Matrix Multiplication, Inverses, and Eigenvalues\n\n"
    "Linear algebra is at the heart of many data science algorithms, from linear regression to PCA and neural networks. NumPy's `linalg` module provides all the operations you need. `np.dot(a, b)` and the `@` operator both compute matrix products. `np.linalg.inv()` computes the inverse of a square matrix, and `np.linalg.eig()` returns eigenvalues and eigenvectors.\n\n"
    "The `@` operator (Python 3.5+) is the most readable way to express matrix multiplication. For a linear regression solution, the normal equation `w = (X^T X)^{-1} X^T y` can be written in a single line of readable code."
))
cells18.append(code(
    "# Matrix multiplication with @ and np.dot\n"
    "X = np.array([[1, 2], [3, 4]])\n"
    "y = np.array([5, 6])\n\n"
    "print('X @ y:', X @ y)\n"
    "print('np.dot(X, y):', np.dot(X, y))"
))
cells18.append(code(
    "# Linear algebra: matrix inverse and linear regression\n"
    "np.random.seed(42)\n"
    "X = np.random.randn(100, 3)\n"
    "true_w = np.array([2.0, -1.5, 0.5])\n"
    "y = X @ true_w + np.random.randn(100) * 0.2\n\n"
    "# Normal equation: w = (X^T X)^{-1} X^T y\n"
    "XTX_inv = np.linalg.inv(X.T @ X)\n"
    "w_hat = XTX_inv @ X.T @ y\n\n"
    "print('True weights:', true_w)\n"
    "print('Estimated w:', np.round(w_hat, 3))\n\n"
    "# Eigenvalues\n"
    "cov_matrix = np.cov(X.T)\n"
    "eigvals, eigvecs = np.linalg.eig(cov_matrix)\n"
    "print('Eigenvalues:', np.round(eigvals, 3))"
))

cells18.append(md(
    "### The Random Module: Generating Random Numbers\n\n"
    "NumPy's `np.random` module provides functions to generate random numbers from various distributions. `np.random.randn()` draws from the standard normal distribution, `.uniform()` from a uniform distribution, and `.choice()` randomly samples from an array. You can also generate random integers with `.randint()`.\n\n"
    "Reproducibility is critical in data science. Setting `np.random.seed(seed_value)` initialises the random number generator to a deterministic state. Any code run after setting the seed will produce the same random numbers, making experiments reproducible across runs and across different machines."
))
cells18.append(code(
    "# Random number generation\n"
    "np.random.seed(42)\n\n"
    "# Normal distribution\n"
    "normal_sample = np.random.randn(1000)\n"
    "print('Normal sample mean:', np.mean(normal_sample).round(3))\n"
    "print('Normal sample std:', np.std(normal_sample).round(3))\n\n"
    "# Uniform distribution\n"
    "uniform_sample = np.random.uniform(0, 10, 1000)\n"
    "print('Uniform sample mean:', np.mean(uniform_sample).round(3))\n\n"
    "# Random choice\n"
    "choices = np.random.choice(['red', 'blue', 'green'], size=10, p=[0.5, 0.3, 0.2])\n"
    "print('Weighted choices:', choices)"
))
cells18.append(code(
    "# Monte Carlo: estimating pi\n"
    "np.random.seed(123)\n"
    "n_points = 1_000_000\n\n"
    "# Random points in unit square [-1, 1] x [-1, 1]\n"
    "x = np.random.uniform(-1, 1, n_points)\n"
    "y = np.random.uniform(-1, 1, n_points)\n\n"
    "# Count points inside unit circle\n"
    "inside = (x**2 + y**2) <= 1\n"
    "pi_estimate = 4 * inside.sum() / n_points\n\n"
    "print(f'Estimated pi: {pi_estimate:.5f}')\n"
    "print(f'Actual pi:    {np.pi:.5f}')"
))

cells18.append(md(
    "### Performance Comparison: Loops vs Vectorised NumPy\n\n"
    "Python loops are interpreted one iteration at a time, which incurs significant overhead for each operation. NumPy's vectorised operations delegate the iteration to highly optimised C and Fortran libraries (BLAS/LAPACK). The speed difference grows with data size; for large arrays, vectorised code can be 50-100x faster than equivalent loops.\n\n"
    "Beyond speed, vectorised code is shorter, more readable, and less error-prone. A rule of thumb: if you find yourself writing a `for` loop over array elements, there is probably a faster NumPy alternative."
))
cells18.append(code(
    "# Performance comparison: Python loops vs vectorised NumPy\n"
    "import time\n\n"
    "n = 5_000_000\n"
    "a = np.random.randn(n)\n"
    "b = np.random.randn(n)\n\n"
    "# Python loop\n"
    "start = time.time()\n"
    "result_loop = [a[i] * b[i] + a[i] for i in range(n)]\n"
    "loop_time = time.time() - start\n"
    "print(f'Python loop: {loop_time:.3f}s')\n\n"
    "# Vectorised\n"
    "start = time.time()\n"
    "result_np = a * b + a\n"
    "np_time = time.time() - start\n"
    "print(f'NumPy:       {np_time:.3f}s')\n"
    "print(f'Speedup:     {loop_time / np_time:.1f}x')"
))

cells18.append(md(
    "## Data Science Connection\n\n"
    "Advanced NumPy capabilities power the algorithms behind most machine learning libraries. Linear algebra operations drive linear regression, PCA, and neural network forward passes. The random module is essential for train-test splitting, bootstrap sampling, and stochastic optimisation. Understanding these tools gives you insight into how higher-level libraries work under the hood and allows you to implement custom algorithms efficiently."
))
nb18.cells = cells18
save(nb18, "Lecture 18 - NumPy Advanced.ipynb")

# ---------------------------------------------------------------------------
# LECTURE 19 - Pandas: Series and DataFrames
# ---------------------------------------------------------------------------
nb19 = new_notebook()
cells19 = []

cells19.append(md("# Lecture 19: Pandas: Series and DataFrames"))
cells19.append(md(
    "## Learning Objectives\n\n"
    "- Create pd.Series and pd.DataFrame from dictionaries, lists, and files\n"
    "- Explore DataFrames with .head(), .info(), .describe(), .shape\n"
    "- Select columns using bracket notation, .loc[], and .iloc[]\n"
    "- Filter rows with boolean conditions\n"
    "- Add, drop, and rename columns\n"
    "- Handle missing values with isna(), dropna(), fillna()\n"
    "- Apply functions to data with .apply()"
))
cells19.append(md(
    "## Key Topics\n\n"
    "- pd.Series and pd.DataFrame from dicts, lists, and files\n"
    "- df.head(), .info(), .describe(), .shape\n"
    "- Column selection: [] vs .loc[] vs .iloc[]\n"
    "- Boolean filtering\n"
    "- Adding, dropping, and renaming columns\n"
    "- isna(), dropna(), fillna()\n"
    "- The .apply() method"
))

cells19.append(md(
    "### Creating Series and DataFrames\n\n"
    "Pandas provides two core data structures: `Series` (1-dimensional labelled array) and `DataFrame` (2-dimensional table with labelled rows and columns). You can create them from Python dictionaries, lists, NumPy arrays, or by reading external files like CSV with `pd.read_csv()`. The dictionary keys become column names, and values (lists) become column data.\n\n"
    "Creating DataFrames from multiple sources is a common workflow. You might start with a dictionary for a small dataset, read a CSV for larger data, or combine NumPy arrays with column names. This flexibility makes Pandas the go-to tool for tabular data in Python."
))
cells19.append(code(
    "import pandas as pd\n"
    "import numpy as np\n\n"
    "# From dictionary\n"
    "data = {\n"
    "    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],\n"
    "    'Age': [25, 30, 35, 28],\n"
    "    'City': ['NYC', 'London', 'Paris', 'Tokyo']\n"
    "}\n"
    "df = pd.DataFrame(data)\n"
    "print('From dict:')\n"
    "print(df)"
))
cells19.append(code(
    "# From list of lists + columns\n"
    "rows = [\n"
    "    [1, 'Alice', 50000],\n"
    "    [2, 'Bob', 60000],\n"
    "    [3, 'Charlie', 55000]\n"
    "]\n"
    "df2 = pd.DataFrame(rows, columns=['ID', 'Name', 'Salary'])\n"
    "print('From lists:')\n"
    "print(df2)\n\n"
    "# Series from list\n"
    "s = pd.Series([10, 20, 30, 40], name='Scores')\n"
    "print('\\nSeries:')\n"
    "print(s)"
))
cells19.append(code(
    "# Exploring a DataFrame\n"
    "print('First 2 rows:')\n"
    "print(df.head(2))\n\n"
    "print('\\nInfo:')\n"
    "print(df.info())\n\n"
    "print('\\nDescribe (numeric only):')\n"
    "print(df.describe())\n\n"
    "print('\\nShape:', df.shape)\n"
    "print('Columns:', df.columns.tolist())"
))

cells19.append(md(
    "### Column Selection: [], .loc[], and .iloc[]\n\n"
    "Pandas offers three main ways to select columns and rows. Bracket notation `df['col']` returns a Series for a single column or a DataFrame for a list of columns. `.loc[]` is label-based: you pass row and column labels. `.iloc[]` is integer-position-based: you pass row and column indices.\n\n"
    "Understanding the distinction is crucial. `.loc[0, 'Age']` gets the value at index label `0` and column `'Age'`. `.iloc[0, 1]` gets the value at the first row and second column. Confusing them is a common source of bugs, especially after sorting or filtering when index labels may no longer be sequential."
))
cells19.append(code(
    "# Column selection\n"
    "print('Single column (Series):')\n"
    "print(df['Name'])\n\n"
    "print('\\nMultiple columns (DataFrame):')\n"
    "print(df[['Name', 'Age']])"
))
cells19.append(code(
    "# .loc vs .iloc\n"
    "print('loc[0, \"Name\"]:', df.loc[0, 'Name'])\n"
    "print('iloc[0, 1]:', df.iloc[0, 1])  # row 0, col 1 = Age\n\n"
    "print('\\nloc slice rows, all columns:')\n"
    "print(df.loc[0:2, :])\n\n"
    "print('\\niloc slice rows, specific columns:')\n"
    "print(df.iloc[0:2, [0, 2]])"
))

cells19.append(md(
    "### Boolean Filtering, Adding, Dropping, and Renaming Columns\n\n"
    "Boolean filtering works in Pandas much like NumPy: you create a boolean Series with a comparison and use it to select rows. Pandas also supports combining conditions with `&` (and), `|` (or), and `~` (not). This is the primary way to subset data based on conditions.\n\n"
    "Adding a column is as simple as assigning to a new key: `df['new_col'] = ...`. Dropping columns uses `df.drop(columns=['col1', 'col2'])` and dropping rows uses `df.drop(index=[...])`. Renaming columns with `df.rename(columns={'old': 'new'})` keeps your data well-organised and readable."
))
cells19.append(code(
    "# Boolean filtering\n"
    "df['Salary'] = [50000, 60000, 55000, 65000]\n"
    "filtered = df[df['Salary'] > 55000]\n"
    "print('Salary > 55000:')\n"
    "print(filtered)\n\n"
    "# Multiple conditions\n"
    "filtered2 = df[(df['Age'] > 25) & (df['City'] != 'Paris')]\n"
    "print('\\nAge > 25 and not Paris:')\n"
    "print(filtered2)"
))
cells19.append(code(
    "# Adding, dropping, renaming columns\n"
    "df['Bonus'] = df['Salary'] * 0.1\n"
    "print('With Bonus column:')\n"
    "print(df)\n\n"
    "df_dropped = df.drop(columns=['Bonus'])\n"
    "print('\\nAfter dropping Bonus:')\n"
    "print(df_dropped.columns.tolist())\n\n"
    "df_renamed = df.rename(columns={'Name': 'Employee', 'Salary': 'AnnualSalary'})\n"
    "print('\\nAfter renaming:')\n"
    "print(df_renamed.columns.tolist())"
))

cells19.append(md(
    "### Handling Missing Values and the .apply() Method\n\n"
    "Real-world data almost always has missing values. `isna()` returns a boolean mask indicating where values are missing. `dropna()` removes rows (or columns) with missing data. `fillna()` fills them with a specified value, the mean of the column, or using forward/backward fill. Choosing between dropping and filling depends on the context: dropping loses data, filling may introduce bias.\n\n"
    "The `.apply()` method lets you run a function on every element (or row/column) of a DataFrame. It is more flexible than vectorised operations because it can handle arbitrary Python logic. However, it is slower than vectorised alternatives, so use it when no built-in vectorised method exists."
))
cells19.append(code(
    "# Missing value handling\n"
    "df_missing = pd.DataFrame({\n"
    "    'A': [1, 2, np.nan, 4],\n"
    "    'B': [np.nan, 2, 3, 4],\n"
    "    'C': [1, 2, 3, 4]\n"
    "})\n"
    "print('Original:')\n"
    "print(df_missing)\n"
    "print('\\nisna():\\n', df_missing.isna())\n\n"
    "print('\\ndropna():')\n"
    "print(df_missing.dropna())\n\n"
    "print('\\nfillna with mean:')\n"
    "print(df_missing.fillna(df_missing.mean().round(1)))"
))
cells19.append(code(
    "# .apply() for feature engineering\n"
    "def age_category(age):\n"
    "    if age < 30:\n"
    "        return 'Young'\n"
    "    elif age < 40:\n"
    "        return 'Mid'\n"
    "    else:\n"
    "        return 'Senior'\n\n"
    "df['AgeGroup'] = df['Age'].apply(age_category)\n"
    "print('With AgeGroup:')\n"
    "print(df[['Name', 'Age', 'AgeGroup']])\n\n"
    "# .apply on axis=1 for row-wise operations\n"
    "df['Total_Comp'] = df.apply(\n"
    "    lambda row: row['Salary'] + row['Bonus'], axis=1\n"
    ")\n"
    "print('\\nTotal compensation:')\n"
    "print(df[['Name', 'Salary', 'Bonus', 'Total_Comp']])"
))

cells19.append(md(
    "## Data Science Connection\n\n"
    "Pandas DataFrames are the primary data structure for data cleaning, exploration, and preparation in Python. Every data science workflow starts with loading data into a DataFrame, inspecting it, handling missing values, and engineering features. The skills you learned here selecting columns, filtering rows, and applying functions are used in virtually every data science project."
))
nb19.cells = cells19
save(nb19, "Lecture 19 - Pandas - Series and DataFrames.ipynb")

# ---------------------------------------------------------------------------
# LECTURE 20 - Pandas: Grouping, Merging, Reshaping
# ---------------------------------------------------------------------------
nb20 = new_notebook()
cells20 = []

cells20.append(md("# Lecture 20: Pandas: Grouping, Merging, and Reshaping"))
cells20.append(md(
    "## Learning Objectives\n\n"
    "- Group data with groupby() and aggregate with multiple functions\n"
    "- Merge DataFrames using inner, left, right, and outer joins\n"
    "- Concatenate DataFrames with concat()\n"
    "- Reshape data with pivot_table() and melt()\n"
    "- Use stack() and unstack() for index/column pivoting\n"
    "- Work with multi-level indexes"
))
cells20.append(md(
    "## Key Topics\n\n"
    "- groupby() + .agg() with multiple functions\n"
    "- merge(): inner, left, right, outer joins\n"
    "- concat() for stacking\n"
    "- pivot_table() and melt()\n"
    "- stack() and unstack()\n"
    "- Working with multi-level indexes"
))

cells20.append(md(
    "### GroupBy: Split-Apply-Combine\n\n"
    "The `groupby()` method implements the split-apply-combine pattern. You split the data into groups based on one or more columns, apply a function to each group independently, and combine the results into a new DataFrame. This is the foundation of all grouped operations in Pandas.\n\n"
    "The `.agg()` method extends this by allowing multiple aggregation functions at once. You can pass a dictionary mapping column names to functions, or a list of functions to apply to each column. Common aggregations include `'sum'`, `'mean'`, `'count'`, `'min'`, `'max'`, and `'std'`."
))
cells20.append(code(
    "import pandas as pd\n"
    "import numpy as np\n\n"
    "# Sales data\n"
    "sales = pd.DataFrame({\n"
    "    'Region': ['North', 'South', 'North', 'South', 'North', 'South'],\n"
    "    'Product': ['A', 'A', 'B', 'B', 'A', 'B'],\n"
    "    'Revenue': [100, 200, 150, 250, 120, 180],\n"
    "    'Quantity': [10, 15, 12, 20, 11, 14]\n"
    "})\n"
    "print('Sales data:')\n"
    "print(sales)"
))
cells20.append(code(
    "# groupby with multiple aggregations\n"
    "grouped = sales.groupby('Region').agg({\n"
    "    'Revenue': ['sum', 'mean'],\n"
    "    'Quantity': ['sum', 'count']\n"
    "})\n"
    "print('Grouped by Region:')\n"
    "print(grouped)\n\n"
    "# Group by multiple columns\n"
    "grouped2 = sales.groupby(['Region', 'Product']).agg('sum')\n"
    "print('\\nGrouped by Region and Product:')\n"
    "print(grouped2)"
))
cells20.append(code(
    "# Using named aggregation\n"
    "result = sales.groupby('Region').agg(\n"
    "    total_revenue=('Revenue', 'sum'),\n"
    "    avg_quantity=('Quantity', 'mean'),\n"
    "    num_orders=('Revenue', 'count')\n"
    ").reset_index()\n"
    "print('Named aggregation:')\n"
    "print(result)"
))

cells20.append(md(
    "### Merging DataFrames: Joins\n\n"
    "Real-world data is rarely in a single table. Merging (or joining) combines DataFrames based on a common key column. `pd.merge()` supports four types of joins. An inner join keeps only rows with matching keys in both tables. A left join keeps all rows from the left table, filling NaN where the right table has no match. Right and outer joins behave analogously.\n\n"
    "Merging is fundamental to relational data analysis. For example, you might have a `customers` table and an `orders` table, and you need to merge them on `customer_id` to analyse customer behaviour. Choosing the wrong join type can lose or duplicate data, so understanding the semantics of each join is essential."
))
cells20.append(code(
    "# Merging DataFrames\n"
    "customers = pd.DataFrame({\n"
    "    'CustomerID': [1, 2, 3, 4],\n"
    "    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],\n"
    "    'City': ['NYC', 'London', 'Paris', 'Tokyo']\n"
    "})\n\n"
    "orders = pd.DataFrame({\n"
    "    'OrderID': [101, 102, 103, 104],\n"
    "    'CustomerID': [1, 2, 2, 5],\n"
    "    'Amount': [250, 180, 320, 90]\n"
    "})\n\n"
    "print('Customers:')\n"
    "print(customers)\n"
    "print('\\nOrders:')\n"
    "print(orders)"
))
cells20.append(code(
    "# Different join types\n"
    "inner = pd.merge(customers, orders, on='CustomerID', how='inner')\n"
    "print('Inner join:')\n"
    "print(inner)\n\n"
    "left = pd.merge(customers, orders, on='CustomerID', how='left')\n"
    "print('\\nLeft join:')\n"
    "print(left)\n\n"
    "outer = pd.merge(customers, orders, on='CustomerID', how='outer')\n"
    "print('\\nOuter join:')\n"
    "print(outer)"
))
cells20.append(code(
    "# concat for stacking\n"
    "q1 = pd.DataFrame({'Product': ['A', 'B'], 'Sales': [100, 150]})\n"
    "q2 = pd.DataFrame({'Product': ['A', 'B'], 'Sales': [120, 160]})\n\n"
    "stacked = pd.concat([q1, q2], ignore_index=True)\n"
    "print('Stacked vertically:')\n"
    "print(stacked)\n\n"
    "# concat with keys (creates MultiIndex)\n"
    "stacked_keys = pd.concat([q1, q2], keys=['Q1', 'Q2'])\n"
    "print('\\nStacked with keys:')\n"
    "print(stacked_keys)"
))

cells20.append(md(
    "### pivot_table() and melt()\n\n"
    "`pivot_table()` creates a spreadsheet-style summary table. You specify the values to aggregate, the index (rows), columns, and aggregation function. It is essentially a multidimensional version of `groupby()`. `melt()` does the reverse: it unpivots a wide table into a long format, which is often required for plotting libraries like seaborn.\n\n"
    "These two operations are complementary. `pivot_table()` makes data more compact and readable for human consumption, while `melt()` makes it suitable for machine learning and visualisation pipelines."
))
cells20.append(code(
    "# pivot_table\n"
    "sales2 = pd.DataFrame({\n"
    "    'Region': ['North', 'North', 'South', 'South'],\n"
    "    'Product': ['A', 'B', 'A', 'B'],\n"
    "    'Quarter': ['Q1', 'Q1', 'Q2', 'Q2'],\n"
    "    'Revenue': [100, 150, 200, 250]\n"
    "})\n\n"
    "pivot = pd.pivot_table(sales2,\n"
    "                       values='Revenue',\n"
    "                       index='Region',\n"
    "                       columns='Quarter',\n"
    "                       aggfunc='sum')\n"
    "print('Pivot table:')\n"
    "print(pivot)"
))
cells20.append(code(
    "# melt: unpivot\n"
    "wide = pd.DataFrame({\n"
    "    'Region': ['North', 'South'],\n"
    "    'Q1': [100, 200],\n"
    "    'Q2': [150, 250]\n"
    "})\n\n"
    "long = wide.melt(id_vars='Region',\n"
    "                 value_vars=['Q1', 'Q2'],\n"
    "                 var_name='Quarter',\n"
    "                 value_name='Revenue')\n"
    "print('Melted (long format):')\n"
    "print(long)"
))

cells20.append(md(
    "### stack(), unstack(), and Multi-Level Indexes\n\n"
    "Multi-level indexes (also called hierarchical indexes) let you work with higher-dimensional data in a 2D DataFrame. `stack()` pivots columns into row index levels, making the DataFrame longer. `unstack()` does the opposite, moving inner index levels to columns.\n\n"
    "These operations are particularly useful when you have grouped or pivoted data and need to rearrange it for a specific analysis or visualisation. They work seamlessly with the MultiIndex that `groupby()` and `pivot_table()` produce."
))
cells20.append(code(
    "# stack and unstack\n"
    "arrays = [['A', 'A', 'B', 'B'], ['X', 'Y', 'X', 'Y']]\n"
    "index = pd.MultiIndex.from_arrays(arrays, names=['Product', 'Store'])\n"
    "df_multi = pd.DataFrame({'Sales': [100, 150, 200, 130]}, index=index)\n"
    "print('MultiIndex DataFrame:')\n"
    "print(df_multi)\n\n"
    "unstacked = df_multi.unstack()\n"
    "print('\\nUnstacked:')\n"
    "print(unstacked)\n\n"
    "stacked = unstacked.stack()\n"
    "print('\\nStacked back:')\n"
    "print(stacked)"
))
cells20.append(code(
    "# Working with multi-level indexes\n"
    "sales3 = pd.DataFrame({\n"
    "    'Region': ['North', 'North', 'South', 'South'],\n"
    "    'Product': ['A', 'B', 'A', 'B'],\n"
    "    'Revenue': [100, 150, 200, 130]\n"
    "})\n\n"
    "grouped = sales3.groupby(['Region', 'Product']).sum()\n"
    "print('Grouped (MultiIndex):')\n"
    "print(grouped)\n"
    "print('\\nIndex levels:', grouped.index.names)\n\n"
    "# Access specific cross-section\n"
    "print('\\nNorth region only:')\n"
    "print(grouped.loc['North'])"
))

cells20.append(md(
    "## Data Science Connection\n\n"
    "Grouping, merging, and reshaping are the core data manipulation skills in data science. GroupBy operations power every 'aggregate by category' analysis. Merging is how you combine data from multiple sources. Reshaping with pivot tables and melt transforms data between formats suitable for analysis, visualisation, and machine learning pipelines. These skills transfer directly to SQL and Spark DataFrames."
))
nb20.cells = cells20
save(nb20, "Lecture 20 - Pandas - Grouping, Merging, and Reshaping.ipynb")

# ---------------------------------------------------------------------------
# LECTURE 21 - Pandas: Dates and Text
# ---------------------------------------------------------------------------
nb21 = new_notebook()
cells21 = []

cells21.append(md("# Lecture 21: Pandas: Working with Dates and Text"))
cells21.append(md(
    "## Learning Objectives\n\n"
    "- Parse dates with pd.to_datetime() and generate date ranges with pd.date_range()\n"
    "- Extract date components using the .dt accessor\n"
    "- Resample time series data with different aggregation functions\n"
    "- Shift and difference time series with .shift() and .diff()\n"
    "- Use the .str accessor for text cleaning and extraction\n"
    "- Work with categorical data types for memory efficiency"
))
cells21.append(md(
    "## Key Topics\n\n"
    "- pd.to_datetime() and pd.date_range()\n"
    "- .dt accessor: .year, .month, .dayofweek, etc.\n"
    "- resample() with aggregation\n"
    "- Shifting and differencing: .shift(), .diff()\n"
    "- .str accessor: .contains(), .extract(), .replace()\n"
    "- Categorical data type and memory savings"
))

cells21.append(md(
    "### Working with Dates and Times\n\n"
    "Date and time data requires special handling because of varying formats, time zones, and the need to extract components like year, month, or day of week. `pd.to_datetime()` converts strings, numbers, or Python datetime objects into Pandas Timestamp objects. `pd.date_range()` generates a fixed-frequency DatetimeIndex, which is useful for creating time series skeletons.\n\n"
    "Once a column is in datetime format, the `.dt` accessor unlocks dozens of properties and methods. You can extract `.year`, `.month`, `.day`, `.dayofweek`, `.quarter`, and many more. This makes it trivial to create features like 'is_weekend' or 'monthly averages' from raw date columns."
))
cells21.append(code(
    "import pandas as pd\n"
    "import numpy as np\n\n"
    "# Parsing dates\n"
    "dates = ['2024-01-15', '2024-02-20', '2024-03-25', 'invalid_date']\n"
    "parsed = pd.to_datetime(dates, errors='coerce')\n"
    "print('Parsed dates:', parsed.tolist())\n\n"
    "# Generating date ranges\n"
    "daily = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')\n"
    "print('Daily:', daily.tolist())\n\n"
    "business = pd.date_range(start='2024-01-01', periods=5, freq='B')\n"
    "print('Business days:', business.tolist())"
))
cells21.append(code(
    "# .dt accessor\n"
    "df_dates = pd.DataFrame({\n"
    "    'date': pd.date_range('2024-01-01', periods=10, freq='D'),\n"
    "    'value': np.random.randn(10)\n"
    "})\n"
    "df_dates['year'] = df_dates['date'].dt.year\n"
    "df_dates['month'] = df_dates['date'].dt.month\n"
    "df_dates['day'] = df_dates['date'].dt.day\n"
    "df_dates['dayofweek'] = df_dates['date'].dt.dayofweek\n"
    "df_dates['is_weekend'] = df_dates['date'].dt.dayofweek >= 5\n"
    "print('Date components:')\n"
    "print(df_dates)"
))

cells21.append(md(
    "### Resampling, Shifting, and Differencing\n\n"
    "Time series data often needs to be aggregated to a different frequency. `resample()` is similar to `groupby()` but for time-based grouping. You specify a frequency string like `'M'` (month end), `'W'` (weekly), `'H'` (hourly), and an aggregation function. This is essential for rolling up high-frequency data (e.g., daily sales to monthly totals).\n\n"
    "`shift()` moves data forward or backward in time, which is how you create lag features for forecasting models. `diff()` computes the difference between consecutive observations, which is useful for detrending a series or computing returns. Both are fundamental tools for time series feature engineering."
))
cells21.append(code(
    "# Resampling\n"
    "idx = pd.date_range('2024-01-01', periods=90, freq='D')\n"
    "ts = pd.Series(np.random.randn(90).cumsum(), index=idx, name='Price')\n\n"
    "monthly = ts.resample('M').agg(['mean', 'min', 'max'])\n"
    "print('Monthly resample:')\n"
    "print(monthly.head())"
))
cells21.append(code(
    "# Shifting and differencing\n"
    "df_ts = pd.DataFrame({\n"
    "    'value': [100, 102, 105, 103, 108, 110]\n"
    "}, index=pd.date_range('2024-01-01', periods=6, freq='D'))\n\n"
    "df_ts['lag_1'] = df_ts['value'].shift(1)\n"
    "df_ts['diff'] = df_ts['value'].diff()\n"
    "df_ts['pct_change'] = df_ts['value'].pct_change()\n"
    "print('Shift and diff:')\n"
    "print(df_ts)"
))

cells21.append(md(
    "### String Methods with .str Accessor\n\n"
    "Pandas provides a `.str` accessor that gives you vectorised string operations, mimicking Python's string methods. `.str.contains()` checks if a pattern exists in each element. `.str.extract()` pulls out substrings matching a regex pattern. `.str.replace()` substitutes patterns. These methods handle missing values gracefully (returning NaN instead of crashing).\n\n"
    "Text data in the real world is messy: inconsistent casing, extra whitespace, embedded symbols, and typos. The `.str` accessor lets you clean and normalise text columns efficiently across millions of rows, making it an indispensable tool for data preparation."
))
cells21.append(code(
    "# .str accessor for text cleaning\n"
    "df_text = pd.DataFrame({\n"
    "    'text': [\n"
    "        'Order #1234 - PENDING',\n"
    "        'Order #5678 - SHIPPED',\n"
    "        'Order #9012 - delivered',\n"
    "        None\n"
    "    ]\n"
    "})\n\n"
    "df_text['lower'] = df_text['text'].str.lower()\n"
    "df_text['has_shipped'] = df_text['text'].str.contains('shipped', case=False)\n"
    "df_text['order_id'] = df_text['text'].str.extract(r'#(\\d+)')\n"
    "df_text['status'] = df_text['text'].str.extract(r'-\\s*(\\w+)')\n"
    "df_text['status'] = df_text['status'].str.replace('pending', 'PENDING')\n"
    "print('Text cleaning:')\n"
    "print(df_text)"
))
cells21.append(code(
    "# More string operations\n"
    "reviews = pd.DataFrame({\n"
    "    'review': [\n"
    "        '  Great product!  ',\n"
    "        'Not bad... but could be better',\n"
    "        'Excellent! Highly recommended!'\n"
    "    ]\n"
    "})\n\n"
    "reviews['clean'] = reviews['review'].str.strip()\n"
    "reviews['word_count'] = reviews['clean'].str.split().str.len()\n"
    "reviews['has_excellent'] = reviews['clean'].str.contains('excellent', case=False)\n"
    "print('Review analysis:')\n"
    "print(reviews)"
))

cells21.append(md(
    "### Categorical Data Type\n\n\n"
    "Columns with a limited set of repeated values (like country codes, product categories, or yes/no) benefit from the `category` dtype. Instead of storing the full string for every row, Pandas stores a compact integer mapping internally. This can dramatically reduce memory usage, especially for columns with high cardinality.\n\n"
    "The `category` dtype also enables categorical operations like `.value_counts()` with `dropna=False`, and you can specify an order with `pd.CategoricalDtype(categories=[...], ordered=True)`. Ordered categories are respected by `.sort_values()` and `.min()` / `.max()`, which is useful for ordinal data like education levels or survey responses."
))
cells21.append(code(
    "# Categorical data type and memory savings\n"
    "n = 100_000\n"
    "df_cat = pd.DataFrame({\n"
    "    'category': np.random.choice(['Low', 'Medium', 'High', 'Critical'], n),\n"
    "    'value': np.random.randn(n)\n"
    "})\n\n"
    "memory_obj = df_cat['category'].memory_usage(deep=True)\n"
    "df_cat['category'] = df_cat['category'].astype('category')\n"
    "memory_cat = df_cat['category'].memory_usage(deep=True)\n\n"
    "print(f'Memory as object: {memory_obj / 1024:.1f} KB')\n"
    "print(f'Memory as category: {memory_cat / 1024:.1f} KB')\n"
    "print(f'Savings: {(1 - memory_cat / memory_obj) * 100:.1f}%')"
))
cells21.append(code(
    "# Ordered categorical\n"
    "df_order = pd.DataFrame({\n"
    "    'education': ['BSc', 'PhD', 'MSc', 'BSc', 'PhD']\n"
    "})\n\n"
    "cat_type = pd.CategoricalDtype(\n"
    "    categories=['BSc', 'MSc', 'PhD'], ordered=True\n"
    ")\n"
    "df_order['education'] = df_order['education'].astype(cat_type)\n\n"
    "print('Sorted by education level:')\n"
    "print(df_order.sort_values('education'))"
))

cells21.append(md(
    "## Data Science Connection\n\n"
    "Date/time handling and text processing are two of the most common challenges in real-world data. Time series analysis powers forecasting, anomaly detection, and trend analysis across finance, IoT, and operations. Text processing is essential for cleaning survey data, parsing logs, and preparing natural language data. The categorical dtype is a simple optimisation that can dramatically reduce memory and speed up processing."
))
nb21.cells = cells21
save(nb21, "Lecture 21 - Pandas - Working with Dates and Text.ipynb")

# ---------------------------------------------------------------------------
# LECTURE 22 - Matplotlib
# ---------------------------------------------------------------------------
nb22 = new_notebook()
cells22 = []

cells22.append(md("# Lecture 22: Data Visualisation with Matplotlib"))
cells22.append(md(
    "## Learning Objectives\n\n"
    "- Create line plots, bar charts, histograms, and scatter plots with pyplot\n"
    "- Use Figure and Axes objects for fine-grained control\n"
    "- Customise plots with titles, labels, legends, and axis limits\n"
    "- Apply colours, markers, line styles, and style sheets\n"
    "- Create subplot grids with plt.subplots()\n"
    "- Save figures to files with plt.savefig()"
))
cells22.append(md(
    "## Key Topics\n\n"
    "- The pyplot interface: plt.plot(), .bar(), .hist(), .scatter()\n"
    "- Figure and Axes objects: plt.figure(), plt.subplots()\n"
    "- Customising: title(), xlabel(), ylabel(), legend(), xlim()\n"
    "- Styling: colours, markers, line styles, plt.style.use()\n"
    "- Saving figures: plt.savefig()"
))

cells22.append(md(
    "### The Pyplot Interface: Basic Plot Types\n\n"
    "Matplotlib's `pyplot` module provides a MATLAB-like interface for creating plots. `plt.plot()` draws line plots, `plt.bar()` creates bar charts, `plt.hist()` generates histograms, and `plt.scatter()` produces scatter plots. Each function accepts data as the primary arguments and optional keyword arguments for customisation.\n\n"
    "Starting with `plt.figure()` creates a new figure, and subsequent plot commands draw on it. `plt.show()` renders the figure. This procedural interface is great for quick exploratory plots, though the object-oriented interface (using Figure and Axes directly) gives more control for complex layouts."
))
cells22.append(code(
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n\n"
    "# Line plot\n"
    "x = np.linspace(0, 10, 100)\n"
    "y = np.sin(x)\n\n"
    "plt.figure(figsize=(8, 4))\n"
    "plt.plot(x, y, label='sin(x)', color='blue', linewidth=2)\n"
    "plt.plot(x, np.cos(x), 'r--', label='cos(x)', linewidth=2)\n"
    "plt.title('Sine and Cosine Waves')\n"
    "plt.xlabel('x')\n"
    "plt.ylabel('y')\n"
    "plt.legend()\n"
    "plt.grid(True, alpha=0.3)\n"
    "plt.show()"
))
cells22.append(code(
    "# Bar chart, histogram, and scatter\n"
    "categories = ['A', 'B', 'C', 'D', 'E']\n"
    "values = [23, 45, 56, 78, 33]\n\n"
    "plt.figure(figsize=(12, 4))\n\n"
    "plt.subplot(1, 3, 1)\n"
    "plt.bar(categories, values, color='skyblue', edgecolor='navy')\n"
    "plt.title('Bar Chart')\n\n"
    "plt.subplot(1, 3, 2)\n"
    "data = np.random.randn(1000)\n"
    "plt.hist(data, bins=30, color='green', alpha=0.7, edgecolor='black')\n"
    "plt.title('Histogram')\n\n"
    "plt.subplot(1, 3, 3)\n"
    "x = np.random.randn(200)\n"
    "y = x + np.random.randn(200) * 0.5\n"
    "plt.scatter(x, y, alpha=0.5, c='purple', s=20)\n"
    "plt.title('Scatter Plot')\n\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

cells22.append(md(
    "### Figure and Axes Objects: The Object-Oriented Interface\n\n"
    "The object-oriented interface gives you explicit control over the figure and its axes. `plt.subplots()` creates a Figure and one or more Axes objects at once. You then call methods on the Axes objects (like `ax.plot()`, `ax.set_title()`) rather than using `plt.*` functions.\n\n"
    "This approach is essential for complex layouts. You can create a grid of subplots with a single call and access each Axes individually to customise titles, labels, and data. It also makes it easier to reuse plotting code inside functions."
))
cells22.append(code(
    "# Subplots grid for EDA\n"
    "np.random.seed(42)\n"
    "df = pd.DataFrame({\n"
    "    'A': np.random.randn(100),\n"
    "    'B': np.random.randn(100) * 2,\n"
    "    'C': np.random.randn(100) + 1\n"
    "})\n\n"
    "fig, axes = plt.subplots(2, 2, figsize=(10, 8))\n"
    "fig.suptitle('Exploratory Data Analysis', fontsize=14)\n\n"
    "axes[0, 0].hist(df['A'], bins=20, color='steelblue')\n"
    "axes[0, 0].set_title('Distribution of A')\n\n"
    "axes[0, 1].scatter(df['A'], df['B'], alpha=0.5)\n"
    "axes[0, 1].set_title('A vs B')\n\n"
    "axes[1, 0].plot(df['A'].cumsum(), color='green')\n"
    "axes[1, 0].set_title('Cumulative Sum of A')\n\n"
    "axes[1, 1].boxplot([df['A'], df['B'], df['C']], labels=['A', 'B', 'C'])\n"
    "axes[1, 1].set_title('Box Plots')\n\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

cells22.append(md(
    "### Customisation and Styling\n\n"
    "Matplotlib offers extensive customisation options. You can control colours (named, hex, RGB), markers (`.`, `o`, `s`, `^`), line styles (`-`, `--`, `-.`, `:`), and transparency with `alpha`. The `plt.style.use()` function lets you apply predefined style sheets like `'seaborn-v0_8'`, `'ggplot'`, or `'fivethirtyeight'` for instantly better-looking plots.\n\n"
    "Axis limits can be set with `xlim()` / `ylim()` or `ax.set_xlim()`. Legends are positioned with `loc` and can be placed outside the plot area. `plt.tight_layout()` automatically adjusts spacing to prevent overlapping elements. These customisations turn basic plots into publication-quality figures."
))
cells22.append(code(
    "# Styling with style sheets\n"
    "plt.style.use('seaborn-v0_8-darkgrid')\n\n"
    "# Custom styling: colours, markers, line styles\n"
    "x = np.linspace(0, 4 * np.pi, 100)\n\n"
    "plt.figure(figsize=(10, 6))\n"
    "plt.plot(x, np.sin(x), color='#E74C3C', marker='o', markersize=4,\n"
    "         linestyle='-', linewidth=2, label='sin(x)')\n"
    "plt.plot(x, np.sin(x + 1), color='#2ECC71', marker='s', markersize=4,\n"
    "         linestyle='--', linewidth=2, label='sin(x+1)')\n"
    "plt.plot(x, np.sin(x + 2), color='#3498DB', marker='^', markersize=4,\n"
    "         linestyle=':', linewidth=2, label='sin(x+2)')\n\n"
    "plt.title('Custom Styling Demo', fontsize=14, fontweight='bold')\n"
    "plt.xlabel('x')\n"
    "plt.ylabel('sin(x)')\n"
    "plt.legend(loc='upper right', frameon=True, shadow=True)\n"
    "plt.xlim(0, 4 * np.pi)\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells22.append(code(
    "# Saving figures\n"
    "fig, ax = plt.subplots(figsize=(8, 4))\n"
    "x = np.linspace(0, 10, 100)\n"
    "ax.plot(x, np.sin(x) * np.exp(-x/5), 'b-', linewidth=2)\n"
    "ax.set_title('Damped Oscillation')\n"
    "ax.set_xlabel('Time')\n"
    "ax.set_ylabel('Amplitude')\n"
    "ax.grid(True, alpha=0.3)\n"
    "plt.tight_layout()\n\n"
    "# Save to file\n"
    "plt.savefig('damped_oscillation.png', dpi=150, bbox_inches='tight')\n"
    "print('Figure saved as damped_oscillation.png')"
))

cells22.append(md(
    "## Data Science Connection\n\n"
    "Visualisation is essential at every stage of the data science workflow. Exploratory data analysis relies on quick histograms and scatter plots to understand distributions and relationships. Communicating results to stakeholders depends on clear, well-styled charts. Matplotlib is the foundational visualisation library in Python; seaborn, pandas' built-in plotting, and even higher-level libraries are built on top of it."
))
nb22.cells = cells22
save(nb22, "Lecture 22 - Data Visualisation with Matplotlib.ipynb")

# ---------------------------------------------------------------------------
# LECTURE 23 - Seaborn
# ---------------------------------------------------------------------------
nb23 = new_notebook()
cells23 = []

cells23.append(md("# Lecture 23: Statistical Visualisation with Seaborn"))
cells23.append(md(
    "## Learning Objectives\n\n"
    "- Configure seaborn themes and colour palettes\n"
    "- Create distribution plots: histplot, kdeplot, boxplot, violinplot\n"
    "- Visualise relationships with scatterplot, relplot, lmplot\n"
    "- Create categorical plots: countplot, barplot, catplot\n"
    "- Use pair plots and heatmaps for multivariate exploration\n"
    "- Build multi-panel plots with FacetGrid"
))
cells23.append(md(
    "## Key Topics\n\n"
    "- sns.set_theme() and sns.set_palette()\n"
    "- Distribution plots: histplot(), kdeplot(), boxplot(), violinplot()\n"
    "- Relationship plots: scatterplot(), relplot(), lmplot()\n"
    "- Categorical plots: countplot(), barplot(), catplot()\n"
    "- Pair plots and heatmaps (sns.pairplot(), sns.heatmap())\n"
    "- Facet grids for multi-panel plots"
))

cells23.append(md(
    "### Setting the Theme and Distribution Plots\n\n"
    "Seaborn extends Matplotlib with statistically meaningful visualisations and attractive default styles. `sns.set_theme()` applies a unified style that instantly improves the look of all plots. `sns.set_palette()` controls the colour palette, with options like `'husl'`, `'Set2'`, `'viridis'`, and `'coolwarm'`.\n\n"
    "Distribution plots reveal the shape, spread, and outliers of your data. `histplot()` creates histograms with optional KDE overlay. `kdeplot()` shows the estimated probability density. `boxplot()` displays quartiles and outliers, while `violinplot()` combines box plot and KDE for a richer picture."
))
cells23.append(code(
    "import seaborn as sns\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n"
    "import pandas as pd\n\n"
    "# Set theme and palette\n"
    "sns.set_theme(style='darkgrid')\n"
    "sns.set_palette('husl')\n\n"
    "# Generate data\n"
    "np.random.seed(42)\n"
    "data = pd.DataFrame({\n"
    "    'value': np.random.randn(500),\n"
    "    'group': np.random.choice(['A', 'B', 'C'], 500)\n"
    "})\n\n"
    "# Distribution plots\n"
    "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n\n"
    "sns.histplot(data['value'], bins=30, kde=True, ax=axes[0, 0])\n"
    "axes[0, 0].set_title('histplot with KDE')\n\n"
    "sns.kdeplot(data['value'], fill=True, ax=axes[0, 1])\n"
    "axes[0, 1].set_title('kdeplot')\n\n"
    "sns.boxplot(x='group', y='value', data=data, ax=axes[1, 0])\n"
    "axes[1, 0].set_title('boxplot')\n\n"
    "sns.violinplot(x='group', y='value', data=data, ax=axes[1, 1])\n"
    "axes[1, 1].set_title('violinplot')\n\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

cells23.append(md(
    "### Relationship and Categorical Plots\n\n"
    "Seaborn provides specialised functions for exploring relationships between variables. `scatterplot()` is the basic scatter plot. `relplot()` is a figure-level function that creates scatter or line plots with optional faceting by additional variables. `lmplot()` adds a regression line with confidence bands, making it easy to visualise linear trends.\n\n"
    "For categorical data, `countplot()` shows the count of observations in each category, similar to a bar chart of value counts. `barplot()` shows the mean (or other estimator) of a numeric variable for each category, with error bars. `catplot()` is the figure-level version that supports multiple categorical plot types and faceting."
))
cells23.append(code(
    "# Relationship plots\n"
    "df = pd.DataFrame({\n"
    "    'hours_studied': np.random.uniform(1, 20, 200),\n"
    "    'exam_score': np.random.uniform(1, 20, 200) * 4 + np.random.randn(200) * 10,\n"
    "    'passed': np.random.choice(['Yes', 'No'], 200, p=[0.7, 0.3])\n"
    "})\n\n"
    "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n\n"
    "sns.scatterplot(x='hours_studied', y='exam_score', data=df, ax=axes[0])\n"
    "axes[0].set_title('scatterplot')\n\n"
    "sns.scatterplot(x='hours_studied', y='exam_score', hue='passed',\n"
    "                data=df, alpha=0.6, ax=axes[1])\n"
    "axes[1].set_title('scatterplot with hue')\n\n"
    "sns.lmplot(x='hours_studied', y='exam_score', data=df, ax=axes[2])\n"
    "axes[2].set_title('lmplot with regression line')\n\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells23.append(code(
    "# Categorical plots\n"
    "tips = sns.load_dataset('tips')\n"
    "print('Tips dataset loaded:', tips.shape)\n\n"
    "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n\n"
    "sns.countplot(x='day', data=tips, ax=axes[0])\n"
    "axes[0].set_title('countplot - orders by day')\n\n"
    "sns.barplot(x='day', y='total_bill', data=tips, ax=axes[1])\n"
    "axes[1].set_title('barplot - avg bill by day')\n\n"
    "sns.catplot(x='day', y='total_bill', hue='sex', kind='box',\n"
    "            data=tips, ax=axes[2])\n"
    "axes[2].set_title('catplot - box by day/sex')\n\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

cells23.append(md(
    "### Pair Plots, Heatmaps, and FacetGrids\n\n"
    "`sns.pairplot()` creates a grid of scatter plots and histograms for every pair of columns in a DataFrame. It is the single most useful function for initial multivariate exploration. The diagonal shows distributions, and off-diagonal cells show pairwise relationships. Adding `hue` colour-codes the scatter plots by a categorical variable.\n\n"
    "`sns.heatmap()` visualises a matrix (typically a correlation matrix) with colour coding, making it easy to spot strong positive or negative relationships. `FacetGrid` is a powerful tool for creating multi-panel plots conditioned on one or more categorical variables, allowing you to compare distributions across subgroups."
))
cells23.append(code(
    "# Pair plot\n"
    "iris = sns.load_dataset('iris')\n"
    "print('Iris dataset loaded:', iris.shape)\n\n"
    "sns.pairplot(iris, hue='species', diag_kind='kde',\n"
    "             palette='Set2', height=2.5)\n"
    "plt.suptitle('Pair Plot of Iris Dataset', y=1.02)\n"
    "plt.show()"
))
cells23.append(code(
    "# Heatmap for correlation\n"
    "corr = iris.select_dtypes(include='number').corr()\n\n"
    "plt.figure(figsize=(8, 6))\n"
    "sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,\n"
    "            square=True, linewidths=1, fmt='.2f')\n"
    "plt.title('Correlation Heatmap - Iris Dataset')\n"
    "plt.show()"
))
cells23.append(code(
    "# FacetGrid for multi-panel plots\n"
    "g = sns.FacetGrid(tips, col='time', row='sex', margin_titles=True)\n"
    "g.map(sns.histplot, 'total_bill', bins=20)\n"
    "g.fig.suptitle('Total Bill Distribution by Time and Gender', y=1.02)\n"
    "plt.show()"
))

cells23.append(md(
    "## Data Science Connection\n\n"
    "Statistical visualisation is the bridge between raw data and actionable insights. Distribution plots reveal data quality issues (skewness, outliers, multimodality). Relationship plots suggest hypotheses about cause and effect. Pair plots and heatmaps provide a bird's-eye view of multivariate structure. Seaborn makes these analyses accessible with concise, high-level functions that produce publication-quality figures."
))
nb23.cells = cells23
save(nb23, "Lecture 23 - Statistical Visualisation with Seaborn.ipynb")

# ---------------------------------------------------------------------------
# LECTURE 24 - Mini-Project: EDA
# ---------------------------------------------------------------------------
nb24 = new_notebook()
cells24 = []

cells24.append(md("# Lecture 24: Mini-Project: Exploratory Data Analysis (EDA) on a Real Dataset"))
cells24.append(md(
    "## Learning Objectives\n\n"
    "- Load a real dataset and perform initial inspection\n"
    "- Clean data by handling missing values, duplicates, and type issues\n"
    "- Conduct univariate analysis with histograms, box plots, and summary statistics\n"
    "- Perform bivariate analysis with scatter plots and grouped comparisons\n"
    "- Explore multivariate patterns with pair plots and correlation matrices\n"
    "- Draw conclusions and communicate findings in a brief report"
))
cells24.append(md(
    "## Key Topics\n\n"
    "- Loading and inspecting a dataset\n"
    "- Cleaning: missing values, duplicates, type fixes\n"
    "- Univariate analysis: histograms, box plots, summary stats\n"
    "- Bivariate analysis: scatter plots, grouped bars, correlation matrix\n"
    "- Multivariate patterns with pair plots\n"
    "- Drawing conclusions and writing a brief report"
))

cells24.append(md(
    "### Loading and Inspecting the Dataset\n\n"
    "We will use the Iris dataset from `sklearn.datasets.load_iris()`, a classic dataset for classification and EDA. It contains 150 samples from three species of iris flowers, with four features: sepal length, sepal width, petal length, and petal width. The dataset is clean and well-documented, making it ideal for demonstrating the full EDA pipeline.\n\n"
    "The first step in any EDA is loading the data and performing an initial inspection. We check the shape, data types, missing values, and summary statistics. We also look at the first few rows to verify that the data loaded correctly and to get a sense of the values."
))
cells24.append(code(
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "from sklearn.datasets import load_iris\n\n"
    "# Load the Iris dataset\n"
    "iris = load_iris()\n"
    "df = pd.DataFrame(data=iris.data, columns=iris.feature_names)\n"
    "df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)\n\n"
    "# Initial inspection\n"
    "print('Shape:', df.shape)\n"
    "print('\\nFirst 5 rows:')\n"
    "print(df.head())\n"
    "print('\\nData types:')\n"
    "print(df.dtypes)\n"
    "print('\\nMissing values:')\n"
    "print(df.isna().sum())\n"
    "print('\\nSummary statistics:')\n"
    "print(df.describe())"
))

cells24.append(md(
    "### Data Cleaning\n\n"
    "Even though the Iris dataset is clean, we will demonstrate the standard cleaning steps. We check for missing values (there are none), duplicate rows, and ensure all columns have the correct data types. The species column is converted to categorical for efficiency.\n\n"
    "In a real dataset, cleaning often involves handling null values, fixing incorrect data types, removing duplicates, and dealing with outliers. These steps ensure that downstream analyses and models are built on reliable data."
))
cells24.append(code(
    "# Data cleaning steps\n"
    "print('Duplicate rows:', df.duplicated().sum())\n\n"
    "# Check for any missing values\n"
    "if df.isna().sum().sum() == 0:\n"
    "    print('No missing values found.')\n\n"
    "# Verify data types\n"
    "print('\\nSpecies categories:', df['species'].cat.categories.tolist())\n"
    "print('Species codes:', df['species'].cat.codes[:5])\n\n"
    "# Ensure numeric columns are float\n"
    "for col in iris.feature_names:\n"
    "    df[col] = df[col].astype(float)\n"
    "print('\\nAll numeric columns confirmed as float64')"
))

cells24.append(md(
    "### Univariate Analysis\n\n"
    "Univariate analysis examines each variable in isolation. We use histograms with KDE overlays to visualise distributions, and box plots to detect outliers and compare spread across species. Summary statistics (`.describe()`) give us the numerical values behind the visualisations.\n\n"
    "These plots reveal key insights: petal length and petal width are bimodal (separating setosa from the other two species), while sepal length and width show more overlap. Box plots confirm that setosa is clearly separable on petal features but less so on sepal features."
))
cells24.append(code(
    "# Univariate analysis: histograms\n"
    "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n"
    "features = iris.feature_names\n\n"
    "for ax, feature in zip(axes.flat, features):\n"
    "    sns.histplot(df[feature], kde=True, bins=20, ax=ax)\n"
    "    ax.set_title(f'Distribution of {feature}')\n\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells24.append(code(
    "# Univariate analysis: box plots grouped by species\n"
    "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n\n"
    "for ax, feature in zip(axes.flat, features):\n"
    "    sns.boxplot(x='species', y=feature, data=df, ax=ax,\n"
    "                palette='Set2')\n"
    "    ax.set_title(f'{feature} by Species')\n\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

cells24.append(md(
    "### Bivariate Analysis\n\n"
    "Bivariate analysis examines relationships between pairs of variables. Scatter plots coloured by species reveal which feature combinations best separate the classes. The correlation matrix quantifies linear relationships, and grouped bar plots show species-level feature means with error bars.\n\n"
    "The petal length vs. petal width scatter plot shows three well-separated clusters. The correlation matrix reveals very high correlation between petal length and petal width (0.96), suggesting redundancy. Sepal width is the least correlated with other features."
))
cells24.append(code(
    "# Bivariate analysis: scatter plots\n"
    "fig, axes = plt.subplots(1, 2, figsize=(14, 6))\n\n"
    "sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)',\n"
    "                hue='species', data=df, alpha=0.7, ax=axes[0])\n"
    "axes[0].set_title('Sepal: Length vs Width')\n\n"
    "sns.scatterplot(x='petal length (cm)', y='petal width (cm)',\n"
    "                hue='species', data=df, alpha=0.7, ax=axes[1])\n"
    "axes[1].set_title('Petal: Length vs Width')\n\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells24.append(code(
    "# Correlation matrix\n"
    "corr = df.select_dtypes(include='number').corr()\n\n"
    "plt.figure(figsize=(8, 6))\n"
    "sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,\n"
    "            square=True, linewidths=1, fmt='.2f')\n"
    "plt.title('Correlation Matrix - Iris Features')\n"
    "plt.show()\n\n"
    "print('Key finding: petal length and petal width are highly correlated (r=0.96)')"
))

cells24.append(md(
    "### Multivariate Patterns with Pair Plots\n\n"
    "Pair plots provide a comprehensive view of all pairwise relationships in one figure. The diagonal shows distributions for each feature, and off-diagonal cells show scatter plots for each pair. Adding `hue='species'` colour-codes the three iris species, making it easy to see which feature combinations separate the classes.\n\n"
    "The pair plot confirms that setosa is linearly separable from versicolor and virginica on petal features alone. Versicolor and virginica overlap somewhat on all features, though petal length and width still provide good separation."
))
cells24.append(code(
    "# Pair plot for multivariate exploration\n"
    "sns.pairplot(df, hue='species', palette='Set2', diag_kind='kde',\n"
    "             corner=True, height=2.5)\n"
    "plt.suptitle('Multivariate Pair Plot of Iris Features', y=1.02)\n"
    "plt.show()"
))
cells24.append(code(
    "# Grouped feature means\n"
    "means = df.groupby('species', observed=True)[features].mean()\n"
    "print('Feature means by species:')\n"
    "print(means.round(2))\n\n"
    "# Visualise\n"
    "means.T.plot(kind='bar', figsize=(10, 6), colormap='Set2', edgecolor='black')\n"
    "plt.title('Average Feature Values by Species')\n"
    "plt.ylabel('Centimeters')\n"
    "plt.xticks(rotation=45)\n"
    "plt.legend(title='Species')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

cells24.append(md(
    "### Findings Report\n\n"
    "Based on our exploratory data analysis of the Iris dataset, we draw the following conclusions. The dataset contains 150 observations across 3 species with no missing values or duplicates. The key findings are:\n\n"
    "1. **Setosa is easily separable**: Iris setosa has distinctly smaller petals (mean length 1.46 cm, width 0.25 cm) and larger sepals (mean width 3.42 cm) compared to the other two species. It forms an isolated cluster in petal feature space.\n"
    "2. **Versicolor vs virginica overlap**: These two species overlap in all feature dimensions, though they differ in central tendency. Petal length (mean: versicolor 4.26, virginica 5.55) provides the best single-feature separation.\n"
    "3. **Petal features are more discriminative**: Petal length and petal width show large between-species differences and small within-species variance, making them ideal for classification.\n"
    "4. **High feature correlation**: Petal length and width are strongly correlated (r=0.96), suggesting they carry redundant information.\n\n"
    "**Recommendation**: A classifier using petal length and petal width should achieve near-perfect accuracy on this dataset. Sepal features add limited discriminative power but may help distinguish versicolor from virginica."
))
cells24.append(code(
    "# Summary statistics for the report\n"
    "summary = df.groupby('species', observed=True)[features].agg(['mean', 'std'])\n"
    "print('=' * 70)\n"
    "print('EDA Summary Report - Iris Dataset')\n"
    "print('=' * 70)\n"
    "print(f'Dataset size: {df.shape[0]} rows, {df.shape[1]} columns')\n"
    "print(f'Species: {df[\"species\"].cat.categories.tolist()}')\n"
    "print(f'Samples per species:\\n{df[\"species\"].value_counts()}')\n"
    "print('\\nFeature statistics by species:')\n"
    "print(summary.round(2))\n"
    "print('\\n' + '=' * 70)\n"
    "print('Conclusion: The Iris dataset is clean, well-structured, and')\n"
    "print('exhibits clear class separation, especially on petal features.')\n"
    "print('Petal length and width alone are sufficient for accurate species classification.')"
))
cells24.append(code(
    "# Save the cleaned dataset for reference\n"
    "df.to_csv('iris_clean.csv', index=False)\n"
    "print('Cleaned Iris dataset saved as iris_clean.csv')\n\n"
    "print('\\nEDA complete! Key findings:')\n"
    "print('- No missing values or duplicates')\n"
    "print('- Setosa is easily separable from the other two species')\n"
    "print('- Petal features are most discriminative')\n"
    "print('- High correlation between petal length and petal width')"
))

cells24.append(md(
    "## Data Science Connection\n\n"
    "This mini-project demonstrates the standard EDA workflow used in every data science project. The pattern of loading, inspecting, cleaning, and visualising data is universal across domains. The Iris dataset is a benchmark for classification algorithms, and our EDA confirms that petal features alone can achieve near-perfect separation. The skills you practiced here interpreting distributions, identifying outliers, exploring relationships, and communicating findings are the foundation of data-driven decision-making."
))
nb24.cells = cells24
save(nb24, "Lecture 24 - Mini-Project EDA.ipynb")

print("All 8 notebooks generated successfully.")
