# Python Programming for Data Science — 32-Lecture Course Outline

---

## Phase 1: Foundations (Lectures 1–8)

### Lecture 1 — Welcome, Setup, and Your First Program
**Learning Objectives:**
- Install and run Python (Anaconda distribution) and VS Code / Jupyter Notebook
- Understand the difference between scripts, notebooks, and the REPL
- Write and execute a simple Python program

**Key Topics:**
- Why Python for data science?
- Installing Python via Anaconda
- Navigating Jupyter Notebook and Jupyter Lab
- The `print()` function and string literals
- Comments and basic program structure

**Data Science Connection:**
Running a first "Hello, Data Science" notebook — the environment every practitioner uses daily.

---

### Lecture 2 — Variables, Data Types, and Basic Operations
**Learning Objectives:**
- Declare variables and understand dynamic typing
- Work with `int`, `float`, `str`, and `bool`
- Perform arithmetic and logical operations

**Key Topics:**
- Variable assignment and naming conventions
- Numeric types: `int` vs `float`
- String basics: concatenation, repetition, f-strings
- Boolean values and comparison operators
- Type conversion (`int()`, `float()`, `str()`, `bool()`)

**Data Science Connection:**
Representing measurements (numeric data), labels (strings), and flags (booleans) — the raw materials of any dataset.

---

### Lecture 3 — Strings and String Methods
**Learning Objectives:**
- Manipulate strings using built-in methods
- Format strings for readable output
- Index and slice strings

**Key Topics:**
- String immutability
- Common methods: `.lower()`, `.upper()`, `.strip()`, `.split()`, `.join()`, `.replace()`
- Indexing (positive and negative)
- Slicing syntax `[start:stop:step]`
- f-string formatting with expressions

**Data Science Connection:**
Cleaning messy text columns — stripping whitespace, splitting CSV lines, normalising case.

---

### Lecture 4 — Lists: The Data Scientist's Workhorse
**Learning Objectives:**
- Create, index, and slice lists
- Use list methods for common operations
- Understand mutability and aliasing

**Key Topics:**
- List creation and heterogenous elements
- Indexing, slicing, and nested lists
- Methods: `.append()`, `.extend()`, `.insert()`, `.remove()`, `.pop()`, `.sort()`
- `in` operator and list membership
- Shallow copies vs deep copies with `copy()`

**Data Science Connection:**
Lists as raw sequences of observations — a stepping stone to NumPy arrays and Pandas Series.

---

### Lecture 5 — Tuples, Sets, and When to Use Them
**Learning Objectives:**
- Distinguish tuples, sets, and lists
- Leverage set operations for data tasks
- Use tuples for fixed records

**Key Topics:**
- Tuple creation, immutability, unpacking
- When to prefer a tuple over a list
- Set creation, uniqueness guarantee
- Set operations: union, intersection, difference, symmetric difference
- Frozen sets

**Data Science Connection:**
Finding unique categories in a dataset; deduplication; representing immutable feature vectors.

---

### Lecture 6 — Dictionaries and Mapping Data
**Learning Objectives:**
- Create and manipulate dictionaries
- Use dictionaries for structured data
- Employ dictionary comprehensions

**Key Topics:**
- Key-value pairs, hashability constraints
- Methods: `.keys()`, `.values()`, `.items()`, `.get()`, `.setdefault()`
- Iterating over dictionaries
- Dictionary comprehensions
- `defaultdict` and `Counter` from `collections`

**Data Science Connection:**
Representing a single record (row) as a dict; counting category frequencies with `Counter`; lookup tables for feature encoding.

---

### Lecture 7 — Control Flow: Conditionals and Loops
**Learning Objectives:**
- Write conditional logic with `if`/`elif`/`else`
- Use `for` and `while` loops effectively
- Control loops with `break`, `continue`, and `else`

**Key Topics:**
- Boolean expressions and truthiness
- `if`, `elif`, `else` chains
- `for` loops over lists, strings, dicts, and ranges
- The `range()` function
- `while` loops and infinite loop prevention
- `break`, `continue`, and the loop `else` clause

**Data Science Connection:**
Filtering rows, iterating over files, conditional feature engineering — the logic behind every data pipeline.

---

### Lecture 8 — Mini-Project: Building a Data Summary Tool
**Learning Objectives:**
- Apply all Phase 1 concepts in an integrated program
- Read a simple dataset (hard-coded list of dicts)
- Compute summary statistics manually

**Key Topics:**
- Putting it all together: lists of dicts as a mini DataFrame
- Computing mean, min, max, and count by category
- Filtering records with conditionals
- Formatting a clean text report

**Data Science Connection:**
Simulating the core of exploratory data analysis — summarising a dataset without any libraries.

---

## Phase 2: Core Programming (Lectures 9–16)

### Lecture 9 — Functions: Writing Reusable Code
**Learning Objectives:**
- Define and call functions with parameters
- Understand scope and return values
- Write docstrings and type hints

**Key Topics:**
- `def` statement and function signature
- Positional vs keyword arguments
- Default parameters and `*args` / `**kwargs`
- The `return` statement
- Local vs global scope
- Docstrings (PEP 257)
- Basic type hints (`: int`, `-> float`)

**Data Science Connection:**
Encapsulating a data-cleaning step or a feature-engineering formula into a reusable function.

---

### Lecture 10 — Error Handling and Defensive Programming
**Learning Objectives:**
- Anticipate and handle runtime errors
- Use `try`/`except`/`else`/`finally`
- Raise custom exceptions

**Key Topics:**
- Common exception types: `TypeError`, `ValueError`, `KeyError`, `FileNotFoundError`
- `try`/`except` blocks
- Catching specific exceptions
- `else` and `finally` clauses
- `raise` and custom exception classes
- Assertions for debugging

**Data Science Connection:**
Robust data pipelines that handle missing files, malformed rows, or type mismatches without crashing.

---

### Lecture 11 — File I/O: Reading and Writing Data
**Learning Objectives:**
- Read and write text and CSV files
- Use context managers (`with`)
- Handle file paths with `pathlib`

**Key Topics:**
- Opening and closing files
- `with` statement and context managers
- Reading: `.read()`, `.readline()`, `.readlines()`
- Writing: `.write()`, `.writelines()`
- CSV module: `csv.reader` and `csv.writer`
- `pathlib.Path` for cross-platform paths

**Data Science Connection:**
Loading raw data from disk — the gateway to every analysis pipeline.

---

### Lecture 12 — List Comprehensions and Generator Expressions
**Learning Objectives:**
- Write concise list comprehensions
- Use generator expressions for memory efficiency
- Combine comprehensions with conditionals

**Key Topics:**
- List comprehension syntax: `[expr for item in iterable]`
- Conditional comprehensions
- Nested comprehensions
- Generator expressions `(expr for item in iterable)`
- `map()`, `filter()`, and `lambda` functions
- When comprehensions beat explicit loops

**Data Science Connection:**
Transforming entire columns, filtering rows, and feature scaling — all in one expressive line.

---

### Lecture 13 — Iterators, Iterables, and the `itertools` Module
**Learning Objectives:**
- Understand the iterator protocol
- Leverage `itertools` for efficient iteration
- Chain, cycle, and group data

**Key Topics:**
- `iter()` and `next()`
- Custom iterators vs generators (`yield`)
- `itertools.chain`, `itertools.cycle`, `itertools.count`
- `itertools.groupby` for grouped operations
- `itertools.product`, `itertools.combinations`, `itertools.permutations`

**Data Science Connection:**
Efficiently streaming large datasets; generating feature interactions (combinations); grouping rows without Pandas.

---

### Lecture 14 — Modules, Packages, and the Standard Library
**Learning Objectives:**
- Import and use standard-library modules
- Create your own modules and packages
- Understand `if __name__ == "__main__"`

**Key Topics:**
- The `import` statement and variations
- The Python standard library tour: `math`, `random`, `statistics`, `os`, `sys`, `json`, `datetime`
- Creating a `.py` module
- Package structure with `__init__.py`
- The `if __name__ == "__main__"` guard

**Data Science Connection:**
Using `json` to read API responses; `datetime` for time-series timestamps; `statistics` for quick descriptive stats.

---

### Lecture 15 — Object-Oriented Programming for Data Science
**Learning Objectives:**
- Define classes and instantiate objects
- Use attributes, methods, and special methods
- Understand inheritance and composition

**Key Topics:**
- `class` definition and `__init__`
- Instance vs class vs static methods
- Special methods: `__str__`, `__repr__`, `__len__`, `__getitem__`
- Inheritance and `super()`
- Composition over inheritance
- `@property` decorator

**Data Science Connection:**
Modelling a custom data transformer, a feature store, or a model wrapper as a class.

---

### Lecture 16 — Mini-Project: CSV Data Cleaner CLI Tool
**Learning Objectives:**
- Build a command-line data-cleaning script
- Combine functions, file I/O, error handling, and OOP
- Produce a clean output file

**Key Topics:**
- Designing a `DataCleaner` class
- Methods: `load()`, `dropna()`, `fillna()`, `remove_duplicates()`, `save()`
- Command-line arguments with `sys.argv` or `argparse`
- Handling encoding errors and malformed rows
- Progress feedback

**Data Science Connection:**
A real-world data-cleaning pipeline — the most common first step in any data science project.

---

## Phase 3: Data Science Libraries (Lectures 17–24)

### Lecture 17 — NumPy Essentials: Arrays and Vectorisation
**Learning Objectives:**
- Create and manipulate NumPy arrays
- Perform vectorised operations
- Understand broadcasting

**Key Topics:**
- `np.array()`, `np.zeros()`, `np.ones()`, `np.arange()`, `np.linspace()`
- Array attributes: `.shape`, `.dtype`, `.ndim`
- Indexing, slicing, and boolean masking
- Vectorised arithmetic and universal functions (ufuncs)
- Broadcasting rules
- `np.where()`, `np.clip()`, aggregation methods

**Data Science Connection:**
Fast numerical computation — the engine behind every data science library.

---

### Lecture 18 — NumPy Advanced: Linear Algebra and Randomness
**Learning Objectives:**
- Perform matrix operations with NumPy
- Generate random numbers for simulations
- Use linear algebra for model foundations

**Key Topics:**
- Reshaping, transposing, stacking, splitting
- `np.dot()`, `@` operator, `np.linalg.inv`, `np.linalg.eig`
- Random module: `np.random.seed()`, `.randn()`, `.choice()`, `.uniform()`
- Setting random seeds for reproducibility
- Performance comparison: Python loops vs vectorised NumPy

**Data Science Connection:**
Simulation (Monte Carlo), statistical sampling, and the linear algebra underpinning regression and PCA.

---

### Lecture 19 — Pandas: Series and DataFrames
**Learning Objectives:**
- Create and inspect Series and DataFrames
- Select, filter, and modify data
- Handle missing values

**Key Topics:**
- `pd.Series` and `pd.DataFrame` from dicts, lists, and files
- `df.head()`, `.info()`, `.describe()`, `.shape`
- Column selection and `[]` vs `.loc[]` vs `.iloc[]`
- Boolean filtering
- Adding, dropping, and renaming columns
- `isna()`, `dropna()`, `fillna()`
- The `.apply()` method

**Data Science Connection:**
Loading and inspecting any real-world dataset — the bread and butter of data science.

---

### Lecture 20 — Pandas: Grouping, Merging, and Reshaping
**Learning Objectives:**
- Group data and compute aggregations
- Merge and join multiple DataFrames
- Reshape data with pivot tables and `melt`

**Key Topics:**
- `groupby()` + `.agg()` with multiple functions
- `merge()`: inner, left, right, outer joins
- `concat()` for stacking
- `pivot_table()` and `melt()`
- `stack()` and `unstack()`
- Working with multi-level indexes

**Data Science Connection:**
SQL-style operations in Python — aggregating sales, joining customer tables, and reshaping for visualisation.

---

### Lecture 21 — Pandas: Working with Dates and Text
**Learning Objectives:**
- Parse and manipulate datetime columns
- Perform time-series filtering and resampling
- Apply vectorised string operations

**Key Topics:**
- `pd.to_datetime()` and `pd.date_range()`
- `.dt` accessor: `.year`, `.month`, `.dayofweek`, etc.
- `resample()` with aggregation
- Shifting and differencing: `.shift()`, `.diff()`
- `.str` accessor: `.contains()`, `.extract()`, `.replace()`
- Categorical data type and memory savings

**Data Science Connection:**
Time-series analysis (stock prices, website traffic) and cleaning messy free-text columns.

---

### Lecture 22 — Data Visualisation with Matplotlib
**Learning Objectives:**
- Create line plots, bar charts, histograms, and scatter plots
- Customise figures with titles, labels, legends, and colours
- Build subplots

**Key Topics:**
- The `pyplot` interface: `plt.plot()`, `.bar()`, `.hist()`, `.scatter()`
- Figure and Axes objects
- `plt.figure()`, `plt.subplots()`
- Customising: `title()`, `xlabel()`, `ylabel()`, `legend()`, `xlim()`
- Styling: colours, markers, line styles, `plt.style.use()`
- Saving figures: `plt.savefig()`

**Data Science Connection:**
Communicating findings visually — the first plot in any EDA workflow.

---

### Lecture 23 — Statistical Visualisation with Seaborn
**Learning Objectives:**
- Create statistical plots with minimal code
- Visualise distributions, relationships, and categorical data
- Customise Seaborn themes

**Key Topics:**
- `sns.set_theme()` and `sns.set_palette()`
- Distribution plots: `histplot()`, `kdeplot()`, `boxplot()`, `violinplot()`
- Relationship plots: `scatterplot()`, `relplot()`, `lmplot()`
- Categorical plots: `countplot()`, `barplot()`, `catplot()`
- Pair plots and heatmaps (`sns.pairplot()`, `sns.heatmap()`)
- Facet grids for multi-panel plots

**Data Science Connection:**
Publication-ready statistical graphics — identifying patterns, outliers, and correlations.

---

### Lecture 24 — Mini-Project: Exploratory Data Analysis (EDA) on a Real Dataset
**Learning Objectives:**
- Conduct a full EDA workflow on a public dataset
- Combine Pandas, NumPy, Matplotlib, and Seaborn
- Produce a written summary of findings

**Key Topics:**
- Dataset: e.g., Titanic, Iris, or a chosen CSV dataset
- Loading and inspecting
- Cleaning: missing values, duplicates, type fixes
- Univariate analysis: histograms, box plots, summary stats
- Bivariate analysis: scatter plots, grouped bars, correlation matrix
- Multivariate patterns with pair plots
- Drawing conclusions and writing a brief report

**Data Science Connection:**
The canonical data science workflow — load, clean, explore, visualise, conclude.

---

## Phase 4: Advanced Topics & Machine Learning (Lectures 25–32)

### Lecture 25 — Introduction to Machine Learning with Scikit‑Learn
**Learning Objectives:**
- Understand the Scikit‑Learn API (fit / predict / transform)
- Prepare features and target variables
- Train a simple classification or regression model

**Key Topics:**
- Machine learning taxonomy: supervised vs unsupervised, classification vs regression
- Train/test split with `train_test_split()`
- Feature matrices (`X`) and target vectors (`y`)
- Fitting a model: `model.fit(X, y)`
- Making predictions: `model.predict(X_test)`
- First models: `LinearRegression`, `KNeighborsClassifier`, `LogisticRegression`

**Data Science Connection:**
Building a predictive model from a cleaned dataset — the transition from analysis to prediction.

---

### Lecture 26 — Model Evaluation and Cross-Validation
**Learning Objectives:**
- Evaluate models with appropriate metrics
- Use cross-validation for robust estimation
- Diagnose overfitting and underfitting

**Key Topics:**
- Confusion matrix, accuracy, precision, recall, F1-score
- `classification_report()` and `confusion_matrix()`
- Regression metrics: MSE, MAE, R-squared
- `cross_val_score()` and `cross_validate()`
- Learning curves and validation curves
- The bias-variance trade-off

**Data Science Connection:**
Knowing whether a model is actually good — rigorous evaluation before deployment.

---

### Lecture 27 — Feature Engineering and Preprocessing
**Learning Objectives:**
- Encode categorical variables
- Scale and normalise numeric features
- Create polynomial and interaction features

**Key Topics:**
- One-hot encoding: `OneHotEncoder`, `pd.get_dummies()`
- Label encoding vs ordinal encoding
- Standardisation: `StandardScaler`
- Normalisation: `MinMaxScaler`
- `ColumnTransformer` for mixed column types
- `Pipeline` for chaining preprocessing + model
- PolynomialFeatures and interaction terms

**Data Science Connection:**
Real-world data is messy — feature engineering is where domain knowledge adds the most value.

---

### Lecture 28 — Decision Trees, Random Forests, and Ensemble Methods
**Learning Objectives:**
- Train and interpret decision trees
- Understand bagging and random forests
- Use ensemble methods for better performance

**Key Topics:**
- `DecisionTreeClassifier` / `DecisionTreeRegressor`
- Visualising trees and feature importance
- `RandomForestClassifier` / `RandomForestRegressor`
- Hyperparameters: `n_estimators`, `max_depth`, `min_samples_split`
- Bagging (`BaggingClassifier`) vs boosting (concept only)
- `GradientBoostingClassifier` and `XGBoost` (intro)

**Data Science Connection:**
Tree-based models dominate tabular data competitions — Kaggle's go-to family of algorithms.

---

### Lecture 29 — Dimensionality Reduction and Clustering
**Learning Objectives:**
- Apply PCA for dimensionality reduction
- Use K-Means for unsupervised clustering
- Interpret cluster results

**Key Topics:**
- PCA: `PCA(n_components)`, explained variance ratio, scree plots
- 2D visualisation after PCA
- K-Means: `KMeans()`, `inertia`, elbow method, silhouette score
- `StandardScaler` before PCA / K-Means
- `DBSCAN` for density-based clustering (brief introduction)

**Data Science Connection:**
Visualising high-dimensional data and discovering hidden groupings in customer or document data.

---

### Lecture 30 — Working with Larger Datasets: Performance and Optimisation
**Learning Objectives:**
- Write efficient Pandas and NumPy code
- Profile and time code execution
- Use parallel processing and chunking

**Key Topics:**
- Profiling with `%timeit`, `cProfile`, and `.memory_usage()`
- Vectorisation vs iteration in Pandas
- `df.eval()` and `df.query()` for fast expressions
- Chunked reading with `pd.read_csv(chunksize=...)`
- `concurrent.futures` for parallel processing
- Dask (intro): parallel DataFrames for out-of-core data
- When to move to a database or Spark

**Data Science Connection:**
Real datasets don't fit in memory — scaling techniques every data scientist needs.

---

### Lecture 31 — Reproducible Workflows and Best Practices
**Learning Objectives:**
- Structure a data science project
- Use version control with Git
- Manage environments and dependencies
- Write clean, documented, tested code

**Key Topics:**
- Project directory structure (e.g., Cookiecutter Data Science)
- Git basics for data scientists: `init`, `add`, `commit`, `push`, branching
- Virtual environments: `conda env` and `venv`
- `requirements.txt` and `environment.yml`
- Writing unit tests with `pytest` for data pipelines
- Logging with the `logging` module
- Notebook best practices: cell order, hiding non-essential code, markdown narrative

**Data Science Connection:**
Professional data science is collaborative and reproducible — these practices separate amateurs from professionals.

---

### Lecture 32 — Capstone Project: End-to-End Data Science Pipeline
**Learning Objectives:**
- Complete a full data science project from raw data to insights
- Combine all skills from the course
- Present findings clearly

**Key Topics:**
- Choose a real dataset (e.g., Housing Prices, Customer Churn, Wine Quality)
- Phase 1 — Load and clean: handle missing values, outliers, type fixes
- Phase 2 — EDA: univariate and multivariate analysis, visualisations
- Phase 3 — Feature engineering: encoding, scaling, new feature creation
- Phase 4 — Modelling: train multiple models, tune hyperparameters
- Phase 5 — Evaluation: compare models, select best, interpret results
- Phase 6 — Communication: produce a final notebook with narrative, plots, and model-performance summary

**Data Science Connection:**
Portfolio-ready project demonstrating the complete data science lifecycle — the course's capstone and your career's starting point.

---

## Summary of Progression

| Phase | Lectures | Theme |
|-------|----------|-------|
| 1 — Foundations | 1–8 | Variables, data types, data structures, control flow, first mini-project |
| 2 — Core Programming | 9–16 | Functions, error handling, file I/O, comprehensions, OOP, CLI tool |
| 3 — Data Science Libraries | 17–24 | NumPy, Pandas, Matplotlib, Seaborn, full EDA project |
| 4 — Advanced & ML | 25–32 | Scikit‑Learn, evaluation, feature engineering, ensembles, clustering, performance, best practices, capstone |

By the end of Lecture 32, a student will have moved from writing their first `print()` statement to building a complete, end-to-end machine learning pipeline — ready to apply Python professionally in data science.
