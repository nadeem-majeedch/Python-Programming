# Python Programming for Data Science

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A comprehensive 32-lecture course progressing from absolute beginner to advanced practitioner in Python for data science. Each lecture is a fully self-contained Jupyter notebook with detailed explanations and runnable code examples.

## Repository Structure

```
├── Course outline.md                          # Full course syllabus
├── requirements.txt                           # Python dependencies
├── .gitignore                                 # Ignored files
├── notebooks/
│   ├── phase-1-foundations/                   # Lectures 01–08
│   ├── phase-2-core-programming/              # Lectures 09–16
│   ├── phase-3-data-science-libraries/        # Lectures 17–24
│   └── phase-4-advanced-ml/                   # Lectures 25–32
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/nadeem-majeedch/Python-Programming.git
cd Python-Programming

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

## Course Outline

### Phase 1: Foundations (Lectures 01–08)
> Variables, data types, data structures, control flow, and a summary mini-project.

| # | Lecture | Topics |
|---|---------|--------|
| 01 | [Welcome, Setup, and Your First Program](notebooks/phase-1-foundations/Lecture%201%20-%20Welcome%2C%20Setup%2C%20and%20Your%20First%20Program.ipynb) | Python setup, `print()`, comments, program structure |
| 02 | [Variables, Data Types, and Basic Operations](notebooks/phase-1-foundations/Lecture%202%20-%20Variables%2C%20Data%20Types%2C%20and%20Basic%20Operations.ipynb) | `int`, `float`, `str`, `bool`, type conversion |
| 03 | [Strings and String Methods](notebooks/phase-1-foundations/Lecture%203%20-%20Strings%20and%20String%20Methods.ipynb) | Indexing, slicing, `.split()`, `.join()`, f-strings |
| 04 | [Lists: The Data Scientist's Workhorse](notebooks/phase-1-foundations/Lecture%204%20-%20Lists%3A%20The%20Data%20Scientist's%20Workhorse.ipynb) | Mutable sequences, slicing, nested lists, `copy()` |
| 05 | [Tuples, Sets, and When to Use Them](notebooks/phase-1-foundations/Lecture%205%20-%20Tuples%2C%20Sets%2C%20and%20When%20to%20Use%20Them.ipynb) | Immutability, set operations, deduplication |
| 06 | [Dictionaries and Mapping Data](notebooks/phase-1-foundations/Lecture%206%20-%20Dictionaries%20and%20Mapping%20Data.ipynb) | Key-value stores, `Counter`, `defaultdict` |
| 07 | [Control Flow: Conditionals and Loops](notebooks/phase-1-foundations/Lecture%207%20-%20Control%20Flow%3A%20Conditionals%20and%20Loops.ipynb) | `if`/`elif`/`else`, `for`, `while`, `break`/`continue` |
| 08 | [Mini-Project: Building a Data Summary Tool](notebooks/phase-1-foundations/Lecture%208%20-%20Mini-Project%3A%20Building%20a%20Data%20Summary%20Tool.ipynb) | Lists of dicts as a mini DataFrame, summary statistics |

---

### Phase 2: Core Programming (Lectures 09–16)
> Functions, error handling, file I/O, comprehensions, iterators, modules, OOP, and a CLI tool.

| # | Lecture | Topics |
|---|---------|--------|
| 09 | [Functions: Writing Reusable Code](notebooks/phase-2-core-programming/Lecture%2009%20-%20Functions%20-%20Writing%20Reusable%20Code.ipynb) | `def`, arguments, scope, docstrings, type hints |
| 10 | [Error Handling and Defensive Programming](notebooks/phase-2-core-programming/Lecture%2010%20-%20Error%20Handling%20and%20Defensive%20Programming.ipynb) | `try`/`except`/`finally`, custom exceptions, assertions |
| 11 | [File I/O: Reading and Writing Data](notebooks/phase-2-core-programming/Lecture%2011%20-%20File%20I-O%20-%20Reading%20and%20Writing%20Data.ipynb) | `with` statement, CSV module, `pathlib` |
| 12 | [List Comprehensions and Generator Expressions](notebooks/phase-2-core-programming/Lecture%2012%20-%20List%20Comprehensions%20and%20Generator%20Expressions.ipynb) | Comprehensions, generators, `map()`/`filter()`/`lambda` |
| 13 | [Iterators, Iterables, and the itertools Module](notebooks/phase-2-core-programming/Lecture%2013%20-%20Iterators%2C%20Iterables%2C%20and%20the%20itertools%20Module.ipynb) | `yield`, `itertools.chain`, `groupby`, combinations |
| 14 | [Modules, Packages, and the Standard Library](notebooks/phase-2-core-programming/Lecture%2014%20-%20Modules%2C%20Packages%2C%20and%20the%20Standard%20Library.ipynb) | `json`, `datetime`, `math`, `statistics`, `random` |
| 15 | [Object-Oriented Programming for Data Science](notebooks/phase-2-core-programming/Lecture%2015%20-%20Object-Oriented%20Programming%20for%20Data%20Science.ipynb) | Classes, inheritance, `@property`, special methods |
| 16 | [Mini-Project: CSV Data Cleaner CLI Tool](notebooks/phase-2-core-programming/Lecture%2016%20-%20Mini-Project%20-%20CSV%20Data%20Cleaner%20CLI%20Tool.ipynb) | `argparse`, data-cleaning class, pipeline |

---

### Phase 3: Data Science Libraries (Lectures 17–24)
> NumPy, Pandas, Matplotlib, Seaborn, and a full EDA project.

| # | Lecture | Topics |
|---|---------|--------|
| 17 | [NumPy Essentials](notebooks/phase-3-data-science-libraries/Lecture%2017%20-%20NumPy%20Essentials.ipynb) | Arrays, broadcasting, boolean masking, ufuncs |
| 18 | [NumPy Advanced](notebooks/phase-3-data-science-libraries/Lecture%2018%20-%20NumPy%20Advanced.ipynb) | Linear algebra, random sampling, performance |
| 19 | [Pandas: Series and DataFrames](notebooks/phase-3-data-science-libraries/Lecture%2019%20-%20Pandas%20-%20Series%20and%20DataFrames.ipynb) | `.loc[]`/`.iloc[]`, boolean filtering, `.apply()` |
| 20 | [Pandas: Grouping, Merging, and Reshaping](notebooks/phase-3-data-science-libraries/Lecture%2020%20-%20Pandas%20-%20Grouping%2C%20Merging%2C%20and%20Reshaping.ipynb) | `groupby()`, `merge()`, `pivot_table()`, `melt()` |
| 21 | [Pandas: Working with Dates and Text](notebooks/phase-3-data-science-libraries/Lecture%2021%20-%20Pandas%20-%20Working%20with%20Dates%20and%20Text.ipynb) | `to_datetime()`, `resample()`, `.str` accessor |
| 22 | [Data Visualisation with Matplotlib](notebooks/phase-3-data-science-libraries/Lecture%2022%20-%20Data%20Visualisation%20with%20Matplotlib.ipynb) | `plt.subplots()`, custom styling, saving figures |
| 23 | [Statistical Visualisation with Seaborn](notebooks/phase-3-data-science-libraries/Lecture%2023%20-%20Statistical%20Visualisation%20with%20Seaborn.ipynb) | Distribution/relationship/categorical plots, heatmaps |
| 24 | [Mini-Project: EDA on a Real Dataset](notebooks/phase-3-data-science-libraries/Lecture%2024%20-%20Mini-Project%20EDA.ipynb) | Full EDA on Iris dataset: cleaning, analysis, visualisation |

---

### Phase 4: Advanced Topics & Machine Learning (Lectures 25–32)
> Scikit-Learn, model evaluation, feature engineering, ensembles, clustering, performance, best practices, capstone.

| # | Lecture | Topics |
|---|---------|--------|
| 25 | [Introduction to Machine Learning with Scikit-Learn](notebooks/phase-4-advanced-ml/Lecture%2025%20-%20Introduction%20to%20Machine%20Learning%20with%20Scikit-Learn.ipynb) | `train_test_split`, `KNeighborsClassifier`, `LinearRegression` |
| 26 | [Model Evaluation and Cross-Validation](notebooks/phase-4-advanced-ml/Lecture%2026%20-%20Model%20Evaluation%20and%20Cross-Validation.ipynb) | Confusion matrix, `cross_val_score`, learning curves |
| 27 | [Feature Engineering and Preprocessing](notebooks/phase-4-advanced-ml/Lecture%2027%20-%20Feature%20Engineering%20and%20Preprocessing.ipynb) | `OneHotEncoder`, `StandardScaler`, `Pipeline` |
| 28 | [Decision Trees, Random Forests, and Ensemble Methods](notebooks/phase-4-advanced-ml/Lecture%2028%20-%20Decision%20Trees%2C%20Random%20Forests%2C%20and%20Ensemble%20Methods.ipynb) | `DecisionTreeClassifier`, `RandomForestClassifier`, feature importance |
| 29 | [Dimensionality Reduction and Clustering](notebooks/phase-4-advanced-ml/Lecture%2029%20-%20Dimensionality%20Reduction%20and%20Clustering.ipynb) | PCA, K-Means, elbow method, DBSCAN |
| 30 | [Working with Larger Datasets: Performance and Optimisation](notebooks/phase-4-advanced-ml/Lecture%2030%20-%20Working%20with%20Larger%20Datasets%3A%20Performance%20and%20Optimisation.ipynb) | `%timeit`, chunked I/O, `concurrent.futures`, Dask intro |
| 31 | [Reproducible Workflows and Best Practices](notebooks/phase-4-advanced-ml/Lecture%2031%20-%20Reproducible%20Workflows%20and%20Best%20Practices.ipynb) | Git, virtual environments, `pytest`, `logging` |
| 32 | [Capstone Project: End-to-End Data Science Pipeline](notebooks/phase-4-advanced-ml/Lecture%2032%20-%20Capstone%20Project%3A%20End-to-End%20Data%20Science%20Pipeline.ipynb) | Full pipeline: load, clean, EDA, feature engineering, modelling, evaluation |

---

## Prerequisites

- Python 3.10 or higher
- Basic computer literacy (no programming experience required — the course starts from scratch)
- A willingness to type along with the examples

## How to Use This Course

1. **Start at Lecture 01** and work through sequentially — each lecture builds on the last.
2. **Run every code cell.** Type the examples yourself (don't just read them) to build muscle memory.
3. **Experiment.** After understanding an example, modify it to see what changes.
4. **Complete the mini-projects.** Lectures 08, 16, 24, and 32 are designed to consolidate everything learned in each phase.

## License

This course material is provided for educational purposes. Feel free to use, adapt, and share.

---

*Built with Python 3 and Jupyter Notebooks.*
### Engr. Dr. Muhammad Nadeem Majeed 
(Stay Blessed Always)
