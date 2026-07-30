#!/usr/bin/env python3
"""Generate all 8 Phase 2 (Core Programming) Jupyter notebooks."""

import nbformat as nbf
from pathlib import Path

OUTPUT_DIR = Path("/workspace/notebooks/phase-2-core-programming")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def new_nb():
    return nbf.v4.new_notebook()


def md(source):
    return nbf.v4.new_markdown_cell(source)


def code(source):
    return nbf.v4.new_code_cell(source)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def title_cell(lecture_number, title):
    return md(f"# Lecture {lecture_number} - {title}")


def objectives_cell(objectives):
    items = "\n".join(f"- {o}" for o in objectives)
    return md(f"## Learning Objectives\n\n{items}")


def key_topics_cell(topics):
    items = "\n".join(f"- {t}" for t in topics)
    return md(f"## Key Topics\n\n{items}")


def explanation_cell(text):
    return md(text)


def data_science_connection_cell(text):
    return md(f"## Data Science Connection\n\n{text}")


def section_header_cell(title):
    return md(f"### {title}")


# ===================================================================
# LECTURE 9 - Functions: Writing Reusable Code
# ===================================================================

def build_lecture_09():
    nb = new_nb()
    nb.cells = [
        title_cell(9, "Functions: Writing Reusable Code"),
        objectives_cell([
            "Understand how to define and call functions with the def statement",
            "Distinguish between positional and keyword arguments",
            "Use default parameters and *args/**kwargs for flexibility",
            "Leverage the return statement effectively",
            "Understand local vs global variable scope",
            "Write docstrings following PEP 257",
            "Add basic type hints to function signatures"
        ]),
        key_topics_cell([
            "def statement and function signature",
            "Positional vs keyword arguments",
            "Default parameters and *args/**kwargs",
            "The return statement",
            "Local vs global scope",
            "Docstrings (PEP 257)",
            "Basic type hints (: int, -> float)"
        ]),

        # -- def statement and function signature --
        explanation_cell("""\
## The `def` Statement and Function Signature

A function is a reusable block of code that performs a specific task. In Python, you define a function using the `def` keyword, followed by the function name, parentheses containing optional parameters, and a colon. The body of the function is indented beneath the definition line.

The **function signature** consists of the function name and its parameter list. It tells callers what inputs the function expects. A well-designed signature makes the function easy to understand and use. Functions help you avoid repeating code and make your programs more modular and testable."""),
        code("""\
# Defining a simple function with no parameters
def greet():
    print("Hello, welcome to Python for Data Science!")

# Calling the function
greet()
"""),
        code("""\
# Function with parameters
def add_numbers(a, b):
    result = a + b
    return result

sum_value = add_numbers(10, 25)
print(f"The sum is: {sum_value}")
"""),
        code("""\
# Function that processes a list of numbers
def compute_average(values):
    total = sum(values)
    count = len(values)
    return total / count

scores = [85, 92, 78, 94, 88]
avg = compute_average(scores)
print(f"Average score: {avg:.2f}")
"""),

        # -- Positional vs keyword arguments --
        explanation_cell("""\
## Positional vs Keyword Arguments

When calling a function, you can pass arguments **positionally** (in the order the parameters are defined) or by **keyword** (using the parameter name). Keyword arguments make your code more readable and allow you to skip optional parameters or pass them in any order.

Positional arguments must come before keyword arguments in a function call. Combining both styles is common, but you must follow this ordering rule. Understanding this distinction helps you write function calls that are both concise and self-documenting."""),
        code("""\
def describe_pet(animal_type, pet_name, age=None):
    print(f"I have a {animal_type} named {pet_name}.")
    if age:
        print(f"{pet_name} is {age} years old.")

# Positional arguments
describe_pet("dog", "Rex", 4)

# Keyword arguments (order does not matter)
describe_pet(age=2, animal_type="cat", pet_name="Whiskers")

# Mixed: positional first, then keyword
describe_pet("hamster", "Nibbles", age=1)
"""),

        # -- Default parameters and *args/**kwargs --
        explanation_cell("""\
## Default Parameters and `*args` / `**kwargs`

Default parameters let you assign a fallback value to a parameter so the caller can omit it. This is useful for optional configuration. However, be careful with mutable default values (like lists or dicts) — they are shared across all calls and can lead to surprising bugs.

When you don't know how many arguments will be passed, use `*args` to capture extra positional arguments as a tuple, and `**kwargs` to capture extra keyword arguments as a dictionary. This pattern is common in wrapper functions, decorators, and library code that needs to forward parameters to another function."""),
        code("""\
# Default parameters
def create_user(username, role="viewer", is_active=True):
    print(f"User: {username}, Role: {role}, Active: {is_active}")

create_user("alice")
create_user("bob", "admin")
create_user("carol", "editor", False)
"""),
        code("""\
# *args captures extra positional arguments
def sum_all(*args):
    print(f"Received {len(args)} numbers")
    return sum(args)

print(sum_all(1, 2, 3, 4, 5))

# **kwargs captures extra keyword arguments
def print_details(**kwargs):
    for key, value in kwargs.items():
        print(f"  {key}: {value}")

print_details(name="Alice", role="Data Scientist", years_experience=5)
"""),

        # -- The return statement --
        explanation_cell("""\
## The `return` Statement

The `return` statement exits a function and optionally passes a value back to the caller. A function without a `return` statement implicitly returns `None`. You can return multiple values as a tuple, which Python unpacks automatically on the caller side.

Returning early can simplify logic — for example, returning a default value when input validation fails. Well-placed return statements make functions easier to read and debug."""),
        code("""\
# Returning a single value
def square(x):
    return x * x

print(square(5))

# Returning multiple values as a tuple
def min_max(values):
    return min(values), max(values)

low, high = min_max([3, 7, 2, 9, 4])
print(f"Low: {low}, High: {high}")

# Early return for edge cases
def safe_divide(a, b):
    if b == 0:
        return float("inf")
    return a / b

print(safe_divide(10, 0))
print(safe_divide(10, 2))
"""),

        # -- Local vs global scope --
        explanation_cell("""\
## Local vs Global Scope

Variables defined inside a function are **local** to that function — they don't exist outside it. Variables defined at the top level of a script are **global**. Python resolves variable names using the LEGB rule: Local, Enclosing, Global, Built-in.

To modify a global variable inside a function, you must use the `global` keyword, but this is generally discouraged because it makes code harder to reason about. Instead, pass values as parameters and return results."""),
        code("""\
# Local vs global demonstration
message = "I am global"  # global variable

def greet_user():
    message = "I am local"  # local variable, different from global
    print("Inside function:", message)

greet_user()
print("Outside function:", message)
"""),
        code("""\
# Using 'global' (use sparingly)
counter = 0

def increment():
    global counter
    counter += 1

increment()
increment()
increment()
print(f"Counter after 3 calls: {counter}")
"""),

        # -- Docstrings --
        explanation_cell("""\
## Docstrings (PEP 257)

A **docstring** is a string literal that appears as the first statement in a function, class, or module. It documents what the component does. Docstrings are accessible at runtime via the `__doc__` attribute and the `help()` function.

PEP 257 recommends using triple double-quotes `\"\"\"` for docstrings. One-line docstrings are brief; multi-line docstrings start with a summary line, a blank line, then a more detailed description. Good docstrings are essential for team collaboration and library usability."""),
        code("""\
def calculate_bmi(weight_kg, height_m):
    \"\"\"Calculate Body Mass Index given weight in kg and height in meters.\"\"\"
    return weight_kg / (height_m ** 2)

print(calculate_bmi(70, 1.75))
print(calculate_bmi.__doc__)
"""),
        code("""\
def clean_text(text):
    \"\"\"
    Remove leading/trailing whitespace and convert text to lowercase.

    Useful as a preprocessing step before text analysis.

    Args:
        text (str): The input string to clean.

    Returns:
        str: The cleaned string with stripped whitespace and lowercased.
    \"\"\"
    return text.strip().lower()

sample = "  Hello DATA SCIENCE World!  "
print(repr(clean_text(sample)))
"""),

        # -- Type hints --
        explanation_cell("""\
## Basic Type Hints

Type hints (introduced in Python 3.5) let you annotate function parameters and return values with expected types. They are not enforced at runtime but are checked by tools like mypy and help IDEs provide better autocompletion and error detection.

Common type hints include `int`, `float`, `str`, `bool`, `List[int]`, `Dict[str, float]`, `Optional[int]`, and `-> ReturnType`. Type hints make your code self-documenting and catch bugs early in development."""),
        code("""\
from typing import List, Optional

def average(values: List[float]) -> float:
    \"\"\"Compute the mean of a list of numbers.\"\"\"
    return sum(values) / len(values)

def find_user(user_id: int, db: Optional[List[str]] = None) -> str:
    \"\"\"Look up a username by ID.\"\"\"
    if db is None:
        return "Guest"
    if 0 <= user_id < len(db):
        return db[user_id]
    return "Unknown"

print(average([10.5, 20.3, 30.1]))
print(find_user(1, ["Alice", "Bob", "Carol"]))
"""),

        # -- Data Science Connection --
        data_science_connection_cell("""\
Functions are the building blocks of data pipelines. Every data science project involves repetitive steps — loading data, cleaning columns, computing statistics, and transforming features. By wrapping each step in a well-documented function with clear type hints and flexible parameters (`*args`/`**kwargs`), you make your analysis reproducible, testable, and shareable with teammates.""")
    ]
    return nb


# ===================================================================
# LECTURE 10 - Error Handling and Defensive Programming
# ===================================================================

def build_lecture_10():
    nb = new_nb()
    nb.cells = [
        title_cell(10, "Error Handling and Defensive Programming"),
        objectives_cell([
            "Recognize and handle common Python exceptions",
            "Use try/except blocks to catch and handle errors gracefully",
            "Catch specific exceptions for targeted error handling",
            "Leverage else and finally clauses",
            "Raise exceptions and create custom exception classes",
            "Use assertions for debugging and preconditions"
        ]),
        key_topics_cell([
            "Common exception types: TypeError, ValueError, KeyError, FileNotFoundError",
            "try/except blocks",
            "Catching specific exceptions",
            "else and finally clauses",
            "raise and custom exception classes",
            "Assertions for debugging"
        ]),

        # -- Common exception types --
        explanation_cell("""\
## Common Exception Types

Python raises exceptions when it encounters an error during execution. Knowing the common built-in exceptions helps you diagnose and handle problems quickly:

- **TypeError**: Raised when an operation is applied to an object of inappropriate type.
- **ValueError**: Raised when a function receives an argument with the right type but an inappropriate value.
- **KeyError**: Raised when a dictionary key is not found.
- **FileNotFoundError**: Raised when trying to open a file that doesn't exist.

Recognizing these exceptions is the first step toward writing robust code that fails gracefully."""),
        code("""\
# TypeError examples
try:
    result = "5" + 10
except TypeError as e:
    print(f"TypeError: {e}")

# ValueError example
try:
    num = int("not_a_number")
except ValueError as e:
    print(f"ValueError: {e}")

# KeyError example
data = {"name": "Alice", "age": 30}
try:
    print(data["occupation"])
except KeyError as e:
    print(f"KeyError: Key '{e}' not found in dictionary")

# FileNotFoundError example
try:
    with open("nonexistent_file.csv") as f:
        content = f.read()
except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")
"""),

        # -- try/except blocks --
        explanation_cell("""\
## Try/Except Blocks

The `try/except` block is Python's primary mechanism for handling exceptions. Code that might raise an error goes inside the `try` block. If an exception occurs, execution jumps to the matching `except` block, preventing the program from crashing.

This is especially important in data science when processing messy real-world data — a single malformed row should not bring down the entire pipeline."""),
        code("""\
# Basic try/except
def safe_parse_int(value):
    try:
        return int(value)
    except:
        return None

values = ["42", "3.14", "abc", "100", "12.5xyz"]
parsed = [safe_parse_int(v) for v in values]
print(f"Parsed integers: {parsed}")
"""),

        # -- Catching specific exceptions --
        explanation_cell("""\
## Catching Specific Exceptions

Always catch the most specific exception types rather than using a bare `except:`. This prevents you from accidentally swallowing unexpected errors like `KeyboardInterrupt` or `MemoryError`. You can also chain multiple `except` clauses to handle different exception types in different ways.

Specific exception handling makes your code's intent clear and helps with debugging."""),
        code("""\
def divide_numbers(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("Cannot divide by zero!")
        return None
    except TypeError as e:
        print(f"Type error: {e}")
        return None
    else:
        print(f"Division successful: {result}")
        return result
    finally:
        print("Division attempt finished.")

print("Result:", divide_numbers(10, 2))
print("---")
print("Result:", divide_numbers(10, 0))
print("---")
print("Result:", divide_numbers(10, "x"))
"""),

        # -- else and finally --
        explanation_cell("""\
## `else` and `finally` Clauses

The `else` clause runs only if no exception was raised in the `try` block. It's useful for code that should execute only on success. The `finally` clause runs **unconditionally** — whether an exception occurred or not — making it ideal for cleanup actions like closing files or releasing resources.

Together, these clauses give you fine-grained control over the flow of error-prone code."""),
        code("""\
def read_config_file(filepath):
    try:
        f = open(filepath, "r")
        content = f.read()
    except FileNotFoundError:
        print("Config file not found. Using defaults.")
        return {"theme": "light", "font_size": 12}
    except PermissionError:
        print("Permission denied. Using defaults.")
        return {"theme": "light", "font_size": 12}
    else:
        print("Config file loaded successfully.")
        # parse simple key=value lines
        config = {}
        for line in content.strip().split("\\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                config[k.strip()] = v.strip()
        return config
    finally:
        try:
            f.close()
        except NameError:
            pass

config = read_config_file("config.txt")
print(config)
"""),

        # -- raise and custom exceptions --
        explanation_cell("""\
## `raise` and Custom Exception Classes

Sometimes you need to signal that something is wrong in your own code. You can `raise` an exception at any point. Python also lets you define **custom exception classes** by inheriting from `Exception` (or a subclass of it). This is invaluable for building domain-specific validation logic.

For example, in a data pipeline you might raise a `DataValidationError` when a column contains unexpected values, making the failure reason crystal clear."""),
        code("""\
# Defining a custom exception
class DataValidationError(Exception):
    \"\"\"Raised when data fails validation checks.\"\"\"
    pass

def validate_age(age):
    if not isinstance(age, (int, float)):
        raise DataValidationError(f"Age must be a number, got {type(age).__name__}")
    if age < 0 or age > 150:
        raise DataValidationError(f"Age {age} is out of valid range (0-150)")
    return True

# Test the validation
for value in [25, -5, "thirty", 101]:
    try:
        validate_age(value)
        print(f"Age {value}: valid")
    except DataValidationError as e:
        print(f"Validation failed for {value}: {e}")
"""),

        # -- Assertions --
        explanation_cell("""\
## Assertions for Debugging

An `assert` statement checks that a condition is `True`. If it is `False`, Python raises an `AssertionError`. Assertions are a lightweight debugging tool — they document invariants and catch bugs early during development.

Use assertions for conditions that **should always be true** if your code is correct. Do **not** use assertions for input validation in production code (use regular `if` checks and raise exceptions instead), because assertions can be disabled with the `-O` flag."""),
        code("""\
def normalize_scores(scores):
    \"\"\"Scale scores to 0-1 range.\"\"\"
    min_s, max_s = min(scores), max(scores)
    # Precondition: we have at least one score
    assert len(scores) > 0, "Scores list is empty"
    # Precondition: range is positive
    assert max_s > min_s, "All scores are identical (cannot normalize)"
    normalized = [(s - min_s) / (max_s - min_s) for s in scores]
    # Postcondition: all values in [0, 1]
    assert all(0 <= v <= 1 for v in normalized), "Normalization out of range"
    return normalized

print(normalize_scores([10, 20, 30, 40, 50]))
"""),

        data_science_connection_cell("""\
Real-world data is messy. Files are missing, columns contain invalid types, and values fall outside expected ranges. Defensive programming — using `try/except`, custom exceptions, and assertions — ensures your data pipeline can handle these problems without crashing. This builds trust and reliability in your analysis and is a hallmark of production-grade data science code.""")
    ]
    return nb


# ===================================================================
# LECTURE 11 - File I/O: Reading and Writing Data
# ===================================================================

def build_lecture_11():
    nb = new_nb()
    nb.cells = [
        title_cell(11, "File I/O: Reading and Writing Data"),
        objectives_cell([
            "Open and close files safely using the with statement",
            "Read file contents with .read(), .readline(), .readlines()",
            "Write data with .write() and .writelines()",
            "Use the csv module for structured data",
            "Navigate the filesystem with pathlib.Path"
        ]),
        key_topics_cell([
            "Opening and closing files",
            "with statement and context managers",
            "Reading: .read(), .readline(), .readlines()",
            "Writing: .write(), .writelines()",
            "CSV module: csv.reader and csv.writer",
            "pathlib.Path for cross-platform paths"
        ]),

        # -- Opening and closing files + with --
        explanation_cell("""\
## Opening and Closing Files with the `with` Statement

Before you can read or write a file, you must open it with the built-in `open()` function. The first argument is the file path, and the second is the mode: `'r'` for reading, `'w'` for writing (overwrites existing content), `'a'` for appending, and `'r+'` for reading and writing.

Always close a file after you're done. The safest way is to use the `with` statement, which acts as a **context manager** and automatically closes the file — even if an exception occurs inside the block. This is the idiomatic Python approach."""),
        code("""\
# Writing to a file (will create or overwrite)
with open("sample.txt", "w") as f:
    f.write("Hello, File I/O!\\n")
    f.write("This is line two.\\n")
    f.write("Line three is here.\\n")

print("File written successfully.")
"""),
        code("""\
# Reading the entire file
with open("sample.txt", "r") as f:
    content = f.read()
print("Using .read():")
print(content)
print("---")
"""),
        code("""\
# Reading line by line
with open("sample.txt", "r") as f:
    print("Using .readline():")
    line = f.readline()
    while line:
        print(f"  >> {line.strip()}")
        line = f.readline()

print("---")

with open("sample.txt", "r") as f:
    print("Using .readlines():")
    lines = f.readlines()
    print(f"Got {len(lines)} lines")
    for i, line in enumerate(lines, 1):
        print(f"  Line {i}: {line.strip()}")
"""),

        # -- CSV module --
        explanation_cell("""\
## The CSV Module: `csv.reader` and `csv.writer`

CSV (comma-separated values) is one of the most common data formats in data science. Python's built-in `csv` module makes reading and writing CSV files straightforward. `csv.reader` returns each row as a list of strings, while `csv.writer` accepts lists or tuples to write.

The module handles quoting, escaping, and different delimiters automatically. For more advanced use cases (e.g., mixed types, large files), the pandas library is preferred, but `csv` is perfect for lightweight data handling."""),
        code("""\
import csv

# Sample data
data = [
    ["Name", "Age", "City", "Salary"],
    ["Alice", 30, "New York", 75000],
    ["Bob", 25, "San Francisco", 82000],
    ["Charlie", 35, "Chicago", 68000],
    ["Diana", 28, "Seattle", 72000],
]

# Writing a CSV file
with open("employees.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)
print("employees.csv written successfully.")
"""),
        code("""\
# Reading a CSV file
with open("employees.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    print(f"Header: {header}")
    print("Rows:")
    for row in reader:
        print(f"  {row}")

# Compute average salary
with open("employees.csv", "r") as f:
    reader = csv.DictReader(f)
    salaries = [int(row["Salary"]) for row in reader]
    avg_salary = sum(salaries) / len(salaries)
    print(f"\\nAverage salary: ${avg_salary:,.2f}")
"""),

        # -- pathlib --
        explanation_cell("""\
## `pathlib.Path` for Cross-Platform Paths

The `pathlib` module, introduced in Python 3.4, provides an object-oriented way to handle filesystem paths. `Path` objects work across Windows, macOS, and Linux without manual string concatenation or separator concerns.

`Path` offers convenient methods like `.read_text()`, `.write_text()`, `.exists()`, `.suffix`, `.stem`, `.parent`, and `.glob()` for directory listings. It is the modern recommended way to work with file paths in Python."""),
        code("""\
from pathlib import Path

# Create a Path object
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
print(f"Directory exists: {data_dir.exists()}")

# Define a file path
file_path = data_dir / "notes.txt"
file_path.write_text("This file was created with pathlib!\\nPath operations are cross-platform.")

# Read back
content = file_path.read_text()
print(f"Read from {file_path}:")
print(content)

# Inspect the path
print(f"Parent: {file_path.parent}")
print(f"File name: {file_path.name}")
print(f"Stem (no extension): {file_path.stem}")
print(f"Suffix: {file_path.suffix}")
"""),

        data_science_connection_cell("""\
Every data science project starts with loading data from files. Whether you're reading CSV exports from a database, JSON responses from an API, or plain text logs, mastering file I/O is the foundation of your data pipeline. Using `pathlib` for cross-platform paths and the `csv` module for structured data prepares you for the more powerful file-handling tools in pandas.""")
    ]
    return nb


# ===================================================================
# LECTURE 12 - List Comprehensions and Generator Expressions
# ===================================================================

def build_lecture_12():
    nb = new_nb()
    nb.cells = [
        title_cell(12, "List Comprehensions and Generator Expressions"),
        objectives_cell([
            "Write concise list comprehensions with filtering and nesting",
            "Use generator expressions for memory-efficient iteration",
            "Apply map(), filter(), and lambda functions",
            "Know when comprehensions beat explicit loops"
        ]),
        key_topics_cell([
            "List comprehension syntax: [expr for item in iterable]",
            "Conditional comprehensions",
            "Nested comprehensions",
            "Generator expressions (expr for item in iterable)",
            "map(), filter(), and lambda functions",
            "When comprehensions beat explicit loops"
        ]),

        # -- List comprehension syntax --
        explanation_cell("""\
## List Comprehension Syntax

A **list comprehension** provides a compact way to create lists by applying an expression to each item in an iterable. The basic syntax is `[expression for item in iterable]`. This replaces the common pattern of creating an empty list and appending items in a `for` loop.

Comprehensions are more than syntactic sugar — they are typically faster than manual loops because the iteration is performed at the C level inside the Python interpreter."""),
        code("""\
# Traditional loop vs list comprehension
numbers = [1, 2, 3, 4, 5]

# Loop approach
squares_loop = []
for n in numbers:
    squares_loop.append(n ** 2)

# List comprehension approach
squares_comp = [n ** 2 for n in numbers]

print(f"Loop:     {squares_loop}")
print(f"Comprehension: {squares_comp}")
"""),
        code("""\
# Practical examples
temperatures_c = [0, 10, 20, 30, 40]
temperatures_f = [c * 9/5 + 32 for c in temperatures_c]
print(f"Celsius: {temperatures_c}")
print(f"Fahrenheit: {temperatures_f}")

words = ["hello", "world", "data", "science"]
uppercase_words = [w.upper() for w in words]
print(f"Uppercased: {uppercase_words}")
"""),

        # -- Conditional comprehensions --
        explanation_cell("""\
## Conditional Comprehensions

You can add an `if` clause at the end of a comprehension to filter items. You can also use a ternary expression (`x if cond else y`) at the front to transform items conditionally.

Conditional comprehensions let you combine filtering and transformation in a single readable line, which is especially handy when preprocessing datasets."""),
        code("""\
# Filtering with if
values = [10, 25, 3, 47, 18, 32, 5]
even_numbers = [v for v in values if v % 2 == 0]
print(f"Even numbers: {even_numbers}")

# Mapping with ternary
scores = [55, 82, 91, 47, 73, 60]
pass_fail = ["Pass" if s >= 60 else "Fail" for s in scores]
print(f"Results: {pass_fail}")

# Filter out None values
mixed = [10, None, 25, None, 30, 42, None]
cleaned = [v for v in mixed if v is not None]
print(f"Cleaned: {cleaned}")
"""),

        # -- Nested comprehensions --
        explanation_cell("""\
## Nested Comprehensions

You can nest comprehensions to work with multi-dimensional data. A nested comprehension with two `for` clauses is equivalent to a nested `for` loop, with the leftmost `for` being the outer loop.

Nested comprehensions are powerful for flattening matrices or generating combinations, but be careful — beyond two levels of nesting, readability suffers and a regular loop is preferable."""),
        code("""\
# Flatten a 2D matrix into a 1D list
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]
flat = [item for row in matrix for item in row]
print(f"Flattened: {flat}")

# Generate coordinate pairs
coords = [(x, y) for x in range(3) for y in range(3)]
print(f"Coordinates: {coords}")
"""),

        # -- Generator expressions --
        explanation_cell("""\
## Generator Expressions

A **generator expression** uses the same syntax as a list comprehension but with parentheses instead of square brackets: `(expr for item in iterable)`. It produces a generator object that yields items **lazily** — one at a time — without storing the entire sequence in memory.

This is critical when working with large datasets that don't fit in RAM. Generator expressions also compose well with functions like `sum()`, `min()`, and `max()` that accept iterables."""),
        code("""\
# List comprehension vs generator expression — memory comparison
import sys

# List comprehension (creates all values in memory)
squares_list = [x ** 2 for x in range(100000)]
print(f"List size: {sys.getsizeof(squares_list)} bytes")

# Generator expression (lazy evaluation)
squares_gen = (x ** 2 for x in range(100000))
print(f"Generator size: {sys.getsizeof(squares_gen)} bytes")

# Iterate through generator
for i, val in enumerate(squares_gen):
    if i < 5:
        print(f"  squares_gen[{i}] = {val}")
    else:
        break
"""),
        code("""\
# Practical: compute sum of squares without creating a list
total = sum(x ** 2 for x in range(1, 1001))
print(f"Sum of squares 1..1000: {total}")

# Read large file line-by-line lazily
with open("sample.txt", "r") as f:
    line_count = sum(1 for line in f)
print(f"Number of lines in sample.txt: {line_count}")
"""),

        # -- map, filter, lambda --
        explanation_cell("""\
## `map()`, `filter()`, and `lambda`

`map(function, iterable)` applies a function to every item and returns an iterator. `filter(function, iterable)` keeps only items for which the function returns `True`. A **lambda** is an anonymous inline function defined with the `lambda` keyword.

While list comprehensions and generator expressions cover most use cases in idiomatic Python, `map` and `filter` combined with `lambda` are still common, especially in code influenced by functional programming."""),
        code("""\
# map with lambda — convert temperatures
celsius = [0, 10, 20, 30, 40]
fahrenheit = list(map(lambda c: c * 9/5 + 32, celsius))
print(f"Fahrenheit: {fahrenheit}")

# filter with lambda — keep values above threshold
values = [12, 5, 28, 3, 19, 42, 7]
above_15 = list(filter(lambda x: x > 15, values))
print(f"Values > 15: {above_15}")

# Combine map and filter
result = list(map(lambda x: x * 2, filter(lambda x: x % 2 == 0, values)))
print(f"Even values doubled: {result}")
"""),

        # -- When comprehensions beat loops --
        explanation_cell("""\
## When Comprehensions Beat Explicit Loops

As a rule of thumb, use a list comprehension when you are building a list from an existing iterable with a simple transformation or filter. Use a generator expression when you only need to iterate once or when the dataset is large. Use an explicit loop when the logic is too complex to fit in a single expression (nested conditionals, side effects, multi-step processing).

Prefer clarity over brevity. If a comprehension makes the code harder to read, an explicit loop is better."""),
        code("""\
# Good comprehension — simple transformation
prices = [19.99, 34.50, 12.30, 45.00, 99.99]
with_tax = [round(p * 1.08, 2) for p in prices]
print(f"Prices with tax: {with_tax}")

# Explicit loop better — complex logic with side effects
log_entries = [
    "ERROR: disk full",
    "INFO: connection established",
    "WARNING: memory usage 85%",
    "ERROR: timeout",
    "INFO: request completed",
]
error_count = 0
for entry in log_entries:
    if entry.startswith("ERROR"):
        error_count += 1
        print(f"  [ALERT] {entry}")
print(f"\\nTotal errors: {error_count}")
"""),

    ]

    nb.cells.append(data_science_connection_cell("""\
Data science involves transforming large datasets — cleaning values, filtering rows, computing aggregations, and generating features. List comprehensions and generator expressions let you write these operations concisely and efficiently. A single comprehension can replace a multi-line loop, reducing bugs and improving readability. When combined with `map`/`filter` and `lambda`, you have a powerful functional toolkit for data transformation."""))

    return nb


# ===================================================================
# LECTURE 13 - Iterators, Iterables, and the itertools Module
# ===================================================================

def build_lecture_13():
    nb = new_nb()
    nb.cells = [
        title_cell(13, "Iterators, Iterables, and the itertools Module"),
        objectives_cell([
            "Understand the iterator protocol: iter() and next()",
            "Write generators with yield for memory-efficient streams",
            "Chain, cycle, and count with itertools",
            "Group data with itertools.groupby",
            "Generate combinations, permutations, and products"
        ]),
        key_topics_cell([
            "iter() and next()",
            "Custom iterators vs generators (yield)",
            "itertools.chain, itertools.cycle, itertools.count",
            "itertools.groupby for grouped operations",
            "itertools.product, itertools.combinations, itertools.permutations"
        ]),

        # -- iter() and next() --
        explanation_cell("""\
## `iter()` and `next()`

An **iterable** is any Python object that can return its elements one at a time — lists, tuples, strings, dictionaries, files, etc. An **iterator** is an object that remembers its position during iteration.

You obtain an iterator from an iterable using `iter()`, and you step through its values using `next()`. When the iterator is exhausted, `next()` raises `StopIteration`. This is exactly what happens behind the scenes when you write `for item in my_list`."""),
        code("""\
# Manual iteration with iter() and next()
fruits = ["apple", "banana", "cherry"]
iterator = iter(fruits)

print(next(iterator))
print(next(iterator))
print(next(iterator))

# Uncommenting the next line would raise StopIteration:
# print(next(iterator))
"""),
        code("""\
# Handling StopIteration gracefully
def manual_iterate(iterable):
    it = iter(iterable)
    while True:
        try:
            item = next(it)
            print(f"Got: {item}")
        except StopIteration:
            print("Iterator exhausted.")
            break

manual_iterate([10, 20, 30])
"""),

        # -- Custom iterators vs generators --
        explanation_cell("""\
## Custom Iterators vs Generators (`yield`)

An **iterator** is a class that implements `__iter__()` (returns self) and `__next__()` (returns the next item). A **generator** is a simpler way to create iterators using the `yield` keyword in a function. Each time `yield` is reached, the function's state is frozen, and the value is returned to the caller.

Generators are the most common way to write custom iterators in Python because they are less boilerplate than writing a full iterator class."""),
        code("""\
# Generator function with yield
def countdown(n):
    while n > 0:
        yield n
        n -= 1

for num in countdown(5):
    print(num)

# Convert to list
print(list(countdown(5)))
"""),
        code("""\
# Generator for streaming data (simulated)
def read_sensor_data(num_readings):
    \"\"\"Simulate streaming sensor readings.\"\"\"
    import random
    for i in range(num_readings):
        yield {
            "reading_id": i + 1,
            "temperature": round(random.uniform(20.0, 30.0), 2),
            "humidity": round(random.uniform(40.0, 70.0), 2),
        }

for reading in read_sensor_data(5):
    print(f"ID {reading['reading_id']}: {reading['temperature']}C, {reading['humidity']}%")
"""),

        # -- chain, cycle, count --
        explanation_cell("""\
## `itertools.chain`, `itertools.cycle`, `itertools.count`

The `itertools` module is a collection of tools for building efficient iterators:

- **`chain`**: Combine multiple iterables into one sequential iterator.
- **`cycle`**: Cycle through an iterable infinitely.
- **`count`**: Count from a start value infinitely with a configurable step.

These tools help you avoid creating intermediate lists and keep your code memory-efficient."""),
        code("""\
import itertools

# chain — concatenate iterables
numbers = [1, 2, 3]
letters = ["a", "b", "c"]
combined = list(itertools.chain(numbers, letters))
print(f"Chained: {combined}")

# count — infinite counter (use with break to avoid infinite loop)
for i, val in enumerate(itertools.count(10, 5)):
    if i >= 5:
        break
    print(f"Count: {val}")

# cycle — repeat indefinitely (use with break)
colors = ["red", "green", "blue"]
cyclic = itertools.cycle(colors)
for i, color in enumerate(cyclic):
    if i >= 7:
        break
    print(f"Cycle {i}: {color}")
"""),

        # -- groupby --
        explanation_cell("""\
## `itertools.groupby` for Grouped Operations

`groupby` groups consecutive elements in an iterable that share a common key. It returns an iterator of `(key, group)` pairs, where `group` is itself an iterator of items. **Important**: `groupby` only groups consecutive matching items, so you often need to sort by the key first.

This is useful for aggregating data that arrives in sorted order, like log entries or time-series data."""),
        code("""\
import itertools

# Simple grouping
data = ["apple", "apple", "banana", "banana", "banana", "cherry", "cherry"]
for key, group in itertools.groupby(data):
    print(f"Key: {key}, Count: {len(list(group))}")

print("---")

# Grouping records (must sort first for meaningful groups)
records = [
    {"city": "NYC", "sales": 100},
    {"city": "LA", "sales": 150},
    {"city": "NYC", "sales": 200},
    {"city": "CHI", "sales": 120},
    {"city": "LA", "sales": 180},
]
records.sort(key=lambda r: r["city"])
for city, group in itertools.groupby(records, key=lambda r: r["city"]):
    sales = [r["sales"] for r in group]
    print(f"{city}: total sales = {sum(sales)}, count = {len(sales)}")
"""),

        # -- product, combinations, permutations --
        explanation_cell("""\
## `product`, `combinations`, `permutations`

These itertools functions are essential for combinatorial operations:

- **`product`**: Cartesian product of input iterables (nested loop equivalent).
- **`combinations`**: All unique r-length tuples **without** regard to order.
- **`permutations`**: All unique r-length tuples **with** regard to order.

In data science, `combinations` is frequently used to generate pairs of features for interaction terms, and `product` is used for hyperparameter grid search."""),
        code("""\
import itertools

# product — Cartesian product (grid search)
param_grid = {
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 5, 7],
    "n_estimators": [50, 100],
}
keys = list(param_grid.keys())
combos = list(itertools.product(*param_grid.values()))
print(f"Grid search has {len(combos)} combinations")
for combo in combos[:3]:
    print(f"  {dict(zip(keys, combo))}")
print("  ...")
"""),
        code("""\
# combinations — feature pairs for interaction terms
features = ["age", "income", "score", "education"]
pairs = list(itertools.combinations(features, 2))
print(f"Feature pairs for interactions: {pairs}")

# permutations — ordered sequences
letters = ["A", "B", "C"]
perms = list(itertools.permutations(letters, 2))
print(f"Permutations: {perms}")
"""),

        data_science_connection_cell("""\
Iterators and generators are the engine behind streaming data pipelines. When you process a 10 GB CSV file, you cannot load it all into memory — generators let you process one row at a time. The `itertools` module provides high-performance building blocks for common data tasks: grouping records, generating feature combinations, and computing Cartesian products for hyperparameter grids.""")
    ]
    return nb


# ===================================================================
# LECTURE 14 - Modules, Packages, and the Standard Library
# ===================================================================

def build_lecture_14():
    nb = new_nb()
    nb.cells = [
        title_cell(14, "Modules, Packages, and the Standard Library"),
        objectives_cell([
            "Use the import statement and its variations",
            "Explore the Python standard library: math, random, statistics, os, sys, json, datetime",
            "Create your own .py module",
            "Understand package structure with __init__.py",
            "Use the if __name__ == '__main__' guard"
        ]),
        key_topics_cell([
            "The import statement and variations",
            "Standard library tour: math, random, statistics, os, sys, json, datetime",
            "Creating a .py module",
            "Package structure with __init__.py",
            "The if __name__ == '__main__' guard"
        ]),

        # -- import statement --
        explanation_cell("""\
## The Import Statement and Its Variations

Modules are files containing Python definitions and statements. To use code from another module, you `import` it. Python offers several import styles:

- `import module` — imports the entire module (access via `module.name`).
- `from module import name` — imports a specific name into the current namespace.
- `from module import name as alias` — imports with a custom alias.
- `import module as alias` — imports the module with an alias.

Choose the style that balances clarity and convenience. Avoid `from module import *` in production code because it pollutes the namespace and makes it unclear where names come from."""),
        code("""\
# Different import styles
import math
from random import randint
from statistics import mean as avg

print(f"Pi (math.pi): {math.pi}")
print(f"Random int 1-100: {randint(1, 100)}")
print(f"Average of [10, 20, 30]: {avg([10, 20, 30])}")
"""),

        # -- Standard library tour --
        explanation_cell("""\
## Standard Library Tour

Python's standard library is vast and well-maintained. Here are the modules most relevant to data science:

- **`math`**: Mathematical functions (sqrt, log, trigonometric).
- **`random`**: Random number generation, sampling, shuffling.
- **`statistics`**: Mean, median, mode, stdev.
- **`os`**: Operating system interface (environment variables, file paths).
- **`sys`**: System-specific parameters (command-line arguments, Python path).
- **`json`**: JSON encoding/decoding for data interchange.
- **`datetime`**: Date and time manipulation.

You rarely need to install third-party libraries for basic operations — the standard library has you covered."""),
        code("""\
import os
import sys
import json
from datetime import datetime, timedelta

# os — get current working directory
print(f"Current directory: {os.getcwd()}")

# sys — Python version
print(f"Python version: {sys.version}")

# json — parse a JSON string
api_response = '{"name": "Alice", "age": 30, "skills": ["Python", "SQL", "ML"]}'
data = json.loads(api_response)
print(f"Parsed JSON: {data}")

# datetime — working with dates
now = datetime.now()
print(f"Current time: {now}")
tomorrow = now + timedelta(days=1)
print(f"Tomorrow: {tomorrow}")
"""),
        code("""\
import math
import random
import statistics

# math — scientific computing
scores = [67, 88, 91, 54, 73, 82, 79]
normalized = [round((s - min(scores)) / (max(scores) - min(scores)), 3) for s in scores]
print(f"Normalized scores: {normalized}")

# random — sampling
sample = random.sample(range(1, 101), 5)
print(f"Random sample: {sample}")

# statistics — descriptive stats
print(f"Mean: {statistics.mean(scores):.2f}")
print(f"Median: {statistics.median(scores):.2f}")
print(f"Stdev: {statistics.stdev(scores):.2f}")
"""),

        # -- Creating a .py module --
        explanation_cell("""\
## Creating a `.py` Module

Any `.py` file is a module. You can define functions, classes, and variables in one file and import them into another. This is how you organize reusable code.

Let's create a small helper module called `data_helpers.py` and import it here. We'll also demonstrate the `if __name__ == '__main__'` guard, which ensures code only runs when the file is executed directly, not when imported as a module."""),
        code("""\
# Write a helper module to disk
module_code = '''
\"\"\"Data science helper utilities.\"\"\"

import csv
import statistics


def load_csv_column(filepath, col_index=0):
    \"\"\"Load a single column from a CSV file.\"\"\"
    values = []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            try:
                values.append(float(row[col_index]))
            except (ValueError, IndexError):
                continue
    return values


def describe_column(values):
    \"\"\"Return descriptive statistics for a list of numbers.\"\"\"
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


if __name__ == "__main__":
    # Test when run directly
    sample = [10, 20, 30, 40, 50]
    print(describe_column(sample))
'''

with open("data_helpers.py", "w") as f:
    f.write(module_code)

print("data_helpers.py created.")
"""),
        code("""\
# Import and use the module we just created
import data_helpers

# Create a sample CSV
import csv
with open("sample_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["value", "category"])
    for v in [15, 22, 31, 47, 53, 68, 74, 89]:
        writer.writerow([v, "A"])

# Use the helper
col = data_helpers.load_csv_column("sample_data.csv", 0)
print(f"Loaded values: {col}")
stats = data_helpers.describe_column(col)
for key, val in stats.items():
    print(f"  {key}: {val}")
"""),

        # -- Package structure --
        explanation_cell("""\
## Package Structure with `__init__.py`

A **package** is a directory containing an `__init__.py` file (can be empty) and one or more module files. The `__init__.py` file is executed when the package is imported, and it can be used to initialize package-level variables or import specific submodules.

Packages let you organize related modules into a hierarchy. For example, a data science project might have `data/` (loading), `transform/` (cleaning), and `models/` (training) sub-packages."""),
        code('''
# Create a small package structure
import os
from pathlib import Path

pkg_dir = Path("myds")
pkg_dir.mkdir(exist_ok=True)

# Create __init__.py
(pkg_dir / "__init__.py").write_text("""
\"\"\"myds - My Data Science Package.\"\"\"

from . import io_utils
from . import stats_utils
""")

# Create io_utils.py
(pkg_dir / "io_utils.py").write_text("""
def load_numbers(filepath):
    with open(filepath) as f:
        return [float(line.strip()) for line in f if line.strip()]
""")

# Create stats_utils.py
(pkg_dir / "stats_utils.py").write_text("""
def summarize(values):
    return {
        "count": len(values),
        "total": sum(values),
        "mean": sum(values) / len(values),
    }
""")

print("Package 'myds' created.")
'''),
        code("""\
# Import from our package
from myds import io_utils, stats_utils

# Write a test file
Path("test_numbers.txt").write_text("10\\n20\\n30\\n40\\n50\\n")

numbers = io_utils.load_numbers("test_numbers.txt")
print(f"Loaded: {numbers}")
summary = stats_utils.summarize(numbers)
print(f"Summary: {summary}")
"""),

        data_science_connection_cell("""\
Organizing code into modules and packages is essential as your data science projects grow. The Python standard library handles most day-to-day tasks — JSON parsing for API data, `datetime` for timestamps, `statistics` for quick summaries, and `os`/`sys` for environment interaction. Building your own package structure (like a `data_utils` package) keeps your analysis pipelines clean and reusable.""")
    ]
    return nb


# ===================================================================
# LECTURE 15 - Object-Oriented Programming for Data Science
# ===================================================================

def build_lecture_15():
    nb = new_nb()
    nb.cells = [
        title_cell(15, "Object-Oriented Programming for Data Science"),
        objectives_cell([
            "Define classes and use __init__ to initialize objects",
            "Distinguish between instance, class, and static methods",
            "Implement special methods: __str__, __repr__, __len__, __getitem__",
            "Use inheritance and super()",
            "Understand composition over inheritance",
            "Apply @property for computed attributes"
        ]),
        key_topics_cell([
            "class definition and __init__",
            "Instance vs class vs static methods",
            "Special methods: __str__, __repr__, __len__, __getitem__",
            "Inheritance and super()",
            "Composition over inheritance",
            "@property decorator"
        ]),

        # -- class definition and __init__ --
        explanation_cell("""\
## Class Definition and `__init__`

A **class** is a blueprint for creating objects. The `__init__` method is the constructor — it's called when you create an instance. Inside `__init__`, you define instance attributes that store data unique to each object.

In data science, classes are useful for encapsulating state and behavior together. For example, a `DataFrame` holds data and provides methods like `.mean()` and `.dropna()`. You'll create classes that model datasets, transformers, and models."""),
        code("""\
class Dataset:
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.row_count = len(data)

    def info(self):
        print(f"Dataset '{self.name}': {self.row_count} rows")

# Create instances
ds1 = Dataset("Customers", ["Alice", "Bob", "Charlie"])
ds2 = Dataset("Products", ["Widget", "Gadget"])

ds1.info()
ds2.info()
print(f"ds1 has {ds1.row_count} rows, ds2 has {ds2.row_count} rows")
"""),

        # -- Instance vs class vs static methods --
        explanation_cell("""\
## Instance vs Class vs Static Methods

- **Instance methods** take `self` and operate on instance data (most common).
- **Class methods** take `cls` and operate on the class itself. Defined with `@classmethod`. Useful for alternative constructors.
- **Static methods** take neither `self` nor `cls`. Defined with `@staticmethod`. They behave like regular functions but live in the class namespace.

Understanding the difference helps you design clean APIs for your data science classes."""),
        code("""\
class DataAnalyzer:
    category = "General"  # class attribute

    def __init__(self, values):
        self.values = values

    def range(self):
        return max(self.values) - min(self.values)

    @classmethod
    def from_csv_column(cls, filepath, col_index=0):
        import csv
        values = []
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                try:
                    values.append(float(row[col_index]))
                except (ValueError, IndexError):
                    continue
        return cls(values)

    @staticmethod
    def is_valid(value):
        return value is not None and value >= 0

# Instance method
analyzer = DataAnalyzer([10, 25, 18, 33, 47])
print(f"Range: {analyzer.range()}")

# Class method (alternative constructor)
col_analyzer = DataAnalyzer.from_csv_column("sample_data.csv")
print(f"From CSV — Range: {col_analyzer.range():.2f}")

# Static method (utility)
print(f"Valid? {DataAnalyzer.is_valid(42)}  {DataAnalyzer.is_valid(-1)}  {DataAnalyzer.is_valid(None)}")
"""),

        # -- Special methods --
        explanation_cell("""\
## Special Methods: `__str__`, `__repr__`, `__len__`, `__getitem__`

Special methods (dunder methods) let your objects behave like built-in types.

- `__str__`: User-friendly string (used by `print()`).
- `__repr__`: Developer-friendly string (used by `repr()`, debugging).
- `__len__`: Supports `len(obj)`.
- `__getitem__`: Supports `obj[key]` and iteration.

Implementing these makes your classes intuitive to use."""),
        code("""\
class TimeSeries:
    def __init__(self, timestamps, values):
        self.timestamps = timestamps
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return (self.timestamps[index], self.values[index])

    def __str__(self):
        return f"TimeSeries({len(self)} points: {min(self.values):.1f} to {max(self.values):.1f})"

    def __repr__(self):
        return f"TimeSeries(timestamps={self.timestamps!r}, values={self.values!r})"

ts = TimeSeries(["2024-01-01", "2024-01-02", "2024-01-03"], [22.5, 23.1, 21.8])
print(f"Length: {len(ts)}")
print(f"Item 0: {ts[0]}")
print(f"String: {ts}")
print(f"Repr: {repr(ts)}")
"""),

        # -- Inheritance --
        explanation_cell("""\
## Inheritance and `super()`

**Inheritance** lets you define a child class that reuses, extends, or overrides behavior from a parent class. Use `super()` to call the parent's methods.

In data science, inheritance is useful for building model families. For example, you might have a `BaseModel` class with a `.train()` and `.predict()` interface, and then `LinearRegression`, `DecisionTree`, and `RandomForest` subclasses that each implement these methods differently."""),
        code("""\
class BaseModel:
    def __init__(self, name):
        self.name = name
        self._trained = False

    def fit(self, X, y):
        self._trained = True
        print(f"{self.name}: Model trained on {len(X)} samples.")

    def predict(self, X):
        if not self._trained:
            raise RuntimeError("Model not trained yet!")
        return [0] * len(X)

class LinearRegression(BaseModel):
    def __init__(self):
        super().__init__("LinearRegression")
        self.coefficients = None

    def fit(self, X, y):
        super().fit(X, y)
        import random
        self.coefficients = [random.uniform(-1, 1) for _ in range(len(X[0]))]
        print(f"  Coefficients learned: {[round(c, 3) for c in self.coefficients]}")

class DecisionTree(BaseModel):
    def __init__(self, max_depth=5):
        super().__init__("DecisionTree")
        self.max_depth = max_depth

    def fit(self, X, y):
        super().fit(X, y)
        print(f"  Tree built with max_depth={self.max_depth}")

models = [LinearRegression(), DecisionTree(max_depth=3)]
X = [[1, 2], [3, 4], [5, 6]]
y = [3, 7, 11]
for model in models:
    model.fit(X, y)
    print(f"  Predictions: {model.predict(X)}\\n")
"""),

        # -- Composition --
        explanation_cell("""\
## Composition Over Inheritance

**Composition** means building a class by including instances of other classes as attributes. The principle "favor composition over inheritance" reminds you that putting objects together is often more flexible than rigid class hierarchies.

For example, a `Pipeline` class that holds a list of transformer objects (each with `.fit()` and `.transform()` methods) uses composition. You can add, remove, or reorder transformers without changing any class hierarchy."""),
        code("""\
class Scaler:
    def fit(self, X):
        self.min_ = min(X)
        self.max_ = max(X)
        return self

    def transform(self, X):
        return [(x - self.min_) / (self.max_ - self.min_) for x in X]

class LogTransformer:
    def transform(self, X):
        import math
        return [math.log(x) for x in X]

# Composition: Pipeline holds transformers
class Pipeline:
    def __init__(self):
        self.steps = []

    def add_step(self, transformer):
        self.steps.append(transformer)
        return self

    def fit_transform(self, X):
        result = X
        for step in self.steps:
            if hasattr(step, "fit"):
                step.fit(result)
            result = step.transform(result)
        return result

pipeline = Pipeline()
pipeline.add_step(Scaler()).add_step(LogTransformer())
result = pipeline.fit_transform([1, 2, 5, 10, 20])
print(f"Pipeline result: {[round(r, 3) for r in result]}")
"""),

        # -- @property --
        explanation_cell("""\
## The `@property` Decorator

The `@property` decorator lets you define methods that can be accessed like attributes (without parentheses). This is useful for **computed attributes** — values derived from other instance data.

Properties provide a clean interface while keeping the underlying implementation flexible. You can change the computation later without breaking code that uses `.attribute` instead of `.attribute()`."""),
        code("""\
class DataFrameSummary:
    def __init__(self, data):
        self._data = data

    @property
    def count(self):
        return len(self._data)

    @property
    def mean(self):
        return sum(self._data) / len(self._data)

    @property
    def range(self):
        return max(self._data) - min(self._data)

    @property
    def summary(self):
        return {
            "count": self.count,
            "mean": round(self.mean, 2),
            "range": round(self.range, 2),
            "min": min(self._data),
            "max": max(self._data),
        }

df = DataFrameSummary([12, 25, 18, 37, 42, 31, 28])
print(f"Mean: {df.mean}")         # Accessed like an attribute
print(f"Count: {df.count}")       # No parentheses needed
print(f"Full summary: {df.summary}")
"""),

        data_science_connection_cell("""\
Object-oriented programming lets you model real-world data science workflows as objects. A `DataFrame` holds tabular data and exposes methods to filter, group, and aggregate. A `Pipeline` chains data transformations. A `Model` base class with subclasses cleanly separates different algorithms. Combined with `@property` for computed attributes and composition for flexible workflows, OOP makes your data science code modular, reusable, and production-ready.""")
    ]
    return nb


# ===================================================================
# LECTURE 16 - Mini-Project: CSV Data Cleaner CLI Tool
# ===================================================================

def build_lecture_16():
    nb = new_nb()
    nb.cells = [
        title_cell(16, "Mini-Project: CSV Data Cleaner CLI Tool"),
        objectives_cell([
            "Design and implement a DataCleaner class",
            "Load CSV data from a file path",
            "Handle missing values with dropna() and fillna()",
            "Remove duplicate rows with remove_duplicates()",
            "Save clean data with save()",
            "Parse command-line arguments with argparse",
            "Handle encoding errors and malformed rows gracefully",
            "Provide progress feedback during processing"
        ]),
        key_topics_cell([
            "Designing a DataCleaner class",
            "Methods: load(), dropna(), fillna(), remove_duplicates(), save()",
            "Command-line arguments with argparse",
            "Handling encoding errors and malformed rows",
            "Progress feedback"
        ]),

        # -- Designing DataCleaner --
        explanation_cell("""\
## Designing a `DataCleaner` Class

The `DataCleaner` class is the heart of this mini-project. It encapsulates all the logic for loading a CSV file, inspecting it for common data quality issues, cleaning it, and saving the result. The design follows the pattern of sklearn-style transformers: you create an instance, call methods in sequence, and each method returns `self` so calls can be chained.

Key design decisions:
- Store the raw data internally as a list of dictionaries (list of rows).
- Keep track of the original and current row counts so you can report what was removed.
- Each cleaning method prints progress feedback so the user knows what happened."""),
        code("""\
import csv

class DataCleaner:
    def __init__(self):
        self.raw_data = None
        self.clean_data = None
        self.original_row_count = 0
        self.current_row_count = 0

    def load(self, filepath, encoding="utf-8"):
        \"\"\"Load CSV data from a file.\"\"\"
        self.raw_data = []
        skipped = 0
        try:
            with open(filepath, "r", encoding=encoding) as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise ValueError("Empty or invalid CSV file")
                self.header = reader.fieldnames
                for row in reader:
                    self.raw_data.append(row)
                self.original_row_count = len(self.raw_data)
                self.clean_data = list(self.raw_data)
                self.current_row_count = self.original_row_count
                print(f"Loaded {self.original_row_count} rows from '{filepath}'")
                print(f"Columns: {', '.join(self.header)}")
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found.")
            raise
        except UnicodeDecodeError:
            print(f"Unicode error with {encoding}. Trying 'latin-1'...")
            return self.load(filepath, encoding="latin-1")
        return self

    def info(self):
        \"\"\"Print summary statistics about the current data.\"\"\"
        print(f"Rows: {self.current_row_count}/{self.original_row_count}")
        print(f"Columns: {self.header}")
        if self.clean_data:
            print(f"Missing values: {sum(1 for r in self.clean_data for v in r.values() if v == '')}")

# Quick test (we'll create a sample CSV first)
print("DataCleaner class designed with chaining in mind.")
print("Continuing to full implementation...")
"""),

        # -- dropna, fillna, remove_duplicates --
        explanation_cell("""\
## Methods: `dropna()`, `fillna()`, `remove_duplicates()`

- **`dropna()`**: Removes rows where **any** field is empty or missing. This is the simplest approach and is appropriate when missing data is rare.
- **`fillna(value)`**: Replaces empty fields with a specified value (e.g., `"Unknown"` for strings or `0` for numbers). This preserves row count.
- **`remove_duplicates()`**: Drops rows that are exact duplicates, keeping the first occurrence. Duplicate rows can skew statistical summaries and model training.

Each method prints a progress message so the user can track what was cleaned and how many rows were affected."""),
        code("""\
def dropna(self):
    \"\"\"Remove rows with any missing (empty) values.\"\"\"
    before = self.current_row_count
    self.clean_data = [
        row for row in self.clean_data
        if all(v.strip() != "" for v in row.values())
    ]
    self.current_row_count = len(self.clean_data)
    removed = before - self.current_row_count
    print(f"dropna: removed {removed} rows with missing values. {self.current_row_count} remaining.")
    return self

def fillna(self, fill_value="MISSING"):
    \"\"\"Replace empty values with a specified fill value.\"\"\"
    count = 0
    for row in self.clean_data:
        for key in row:
            if row[key].strip() == "":
                row[key] = fill_value
                count += 1
    print(f"fillna: filled {count} empty cells with '{fill_value}'.")
    return self

def remove_duplicates(self):
    \"\"\"Remove duplicate rows (exact match on all fields).\"\"\"
    before = self.current_row_count
    seen = set()
    unique = []
    for row in self.clean_data:
        # Create a tuple of values as a hashable key
        key = tuple(row.values())
        if key not in seen:
            seen.add(key)
            unique.append(row)
    self.clean_data = unique
    self.current_row_count = len(self.clean_data)
    removed = before - self.current_row_count
    print(f"remove_duplicates: removed {removed} duplicate rows. {self.current_row_count} remaining.")
    return self

# Bind methods to the class
DataCleaner.dropna = dropna
DataCleaner.fillna = fillna
DataCleaner.remove_duplicates = remove_duplicates

print("Methods dropna(), fillna(), and remove_duplicates() added to DataCleaner.")
"""),

        # -- save --
        explanation_cell("""\
## `save()` Method and Full Implementation

The `save()` method writes the cleaned data to a new CSV file. It uses `csv.DictWriter` with the stored header. After saving, it prints a summary of what was accomplished — comparing original vs final row count and listing the operations performed.

This completes the core `DataCleaner` class. Now let's put all the pieces together into a single cohesive implementation."""),
        code("""\
def save(self, output_path):
    \"\"\"Save the cleaned data to a CSV file.\"\"\"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=self.header)
        writer.writeheader()
        writer.writerows(self.clean_data)
    print(f"Saved {self.current_row_count} rows to '{output_path}'")
    print(f"Data quality summary: removed {self.original_row_count - self.current_row_count} problematic rows "
          f"({((self.original_row_count - self.current_row_count)/max(self.original_row_count,1)*100):.1f}% reduction)")
    return self

DataCleaner.save = save

# Now demonstrate the full implementation
print("DataCleaner is complete. Let's test it on a sample CSV.")
"""),

        # -- argparse --
        explanation_cell("""\
## Command-Line Arguments with `argparse`

The `argparse` module makes it easy to build user-friendly command-line interfaces. For our CSV Cleaner CLI, we'll accept:

- `input`: Path to the input CSV file (positional, required).
- `-o` / `--output`: Output path (default: `cleaned_output.csv`).
- `--dropna` / `--fillna` / `--remove-duplicates`: Flags to enable each cleaning operation.
- `--fill-value`: Value to use for filling missing cells (default: "MISSING").

argparse automatically generates help text, handles validation, and provides useful error messages."""),
        code("""\
import argparse

def setup_parser():
    '''Configure the command-line argument parser.'''
    parser = argparse.ArgumentParser(
        description="Clean a CSV file by removing missing values, duplicates, and more.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples:
  python csv_cleaner.py data.csv -o clean.csv --dropna --remove-duplicates
  python csv_cleaner.py data.csv --fillna --fill-value 0
  python csv_cleaner.py data.csv --dropna --fillna --remove-duplicates
'''
    )
    parser.add_argument("input", help="Path to the input CSV file to clean")
    parser.add_argument("-o", "--output", default="cleaned_output.csv",
                        help="Output CSV file path (default: cleaned_output.csv)")
    parser.add_argument("--dropna", action="store_true",
                        help="Remove rows with missing values")
    parser.add_argument("--fillna", action="store_true",
                        help="Fill missing values with --fill-value")
    parser.add_argument("--fill-value", default="MISSING",
                        help="Value to use when filling missing cells (default: MISSING)")
    parser.add_argument("--remove-duplicates", action="store_true",
                        help="Remove duplicate rows")
    return parser

# Demonstrate the parser
parser = setup_parser()
print("argparse configured with the following options:")
parser.print_help()
"""),

        # -- Full pipeline demo --
        explanation_cell("""\
## Full Pipeline Demonstration

Let's create a realistic (but small) CSV file with common data quality issues — missing values, duplicates, and encoding artifacts — and run the full `DataCleaner` pipeline on it.

We'll then verify the output by loading the cleaned file."""),
        code("""\
# Create a sample CSV with quality issues
sample_csv = '''Name,Age,City,Salary
Alice,30,New York,75000
Bob,25,San Francisco,82000
Charlie,,Chicago,68000
Diana,28,Seattle,72000
Alice,30,New York,75000
Eve,35,,
Frank,40,Boston,95000
Diana,28,Seattle,72000
Grace,,,
'''

with open("dirty_data.csv", "w") as f:
    f.write(sample_csv)

print("Created 'dirty_data.csv' with duplicates, missing values, and empty rows.")
"""),
        code("""\
# Run the full cleaning pipeline
cleaner = DataCleaner()
cleaner.load("dirty_data.csv")
cleaner.info()
print("---")
cleaner.dropna()
cleaner.remove_duplicates()
cleaner.save("clean_data.csv")
print("---")

# Verify the cleaned file
print("\\nCleaned data contents:")
with open("clean_data.csv", "r") as f:
    print(f.read())
"""),

        # -- Handling encoding errors and malformed rows --
        explanation_cell("""\
## Handling Encoding Errors and Malformed Rows

Real-world CSV files often have encoding issues (e.g., UTF-8 vs Latin-1) or malformed rows (wrong number of columns). A robust data cleaner must handle these gracefully.

Our `load()` method already tries `utf-8` first, and falls back to `latin-1` on `UnicodeDecodeError`. For malformed rows, we can add a try/except around the CSV parsing. Here's an enhanced version of the loading logic that demonstrates these defensive techniques."""),
        code("""\
def safe_load(filepath):
    \"\"\"Robust CSV loading with encoding fallback and malformed-row handling.\"\"\"
    data = []
    skipped = 0

    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            with open(filepath, "r", encoding=encoding) as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    print("Warning: File appears empty (no header found).")
                    return data
                for i, row in enumerate(reader):
                    try:
                        # Skip rows with wrong number of columns
                        if len(row) != len(reader.fieldnames):
                            skipped += 1
                            continue
                        data.append(row)
                    except Exception as e:
                        skipped += 1
                        print(f"  Skipped row {i+2}: {e}")
            print(f"Loaded with encoding '{encoding}': {len(data)} rows, {skipped} skipped")
            return data
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with '{encoding}': {e}")
            continue

    print("Failed to read file with any encoding.")
    return data

# Test on a file that would normally cause issues
print("safe_load is ready for production use.")
print(f"Test on dirty_data.csv: {len(safe_load('dirty_data.csv'))} rows loaded")
"""),

        data_science_connection_cell("""\
This mini-project integrates everything you've learned in Phase 2: functions, error handling, file I/O, comprehensions, iterators, modules, and OOP. The `DataCleaner` class is a real tool you'll find yourself using in every data science project. It demonstrates how defensive programming — handling encoding issues, malformed rows, and missing values — turns a fragile script into a robust, reusable utility. Adding an `argparse`-based CLI makes it accessible to non-programmers and easily integrable into shell pipelines.""")
    ]
    return nb


# ===================================================================
# Main: Build and write all notebooks
# ===================================================================

def write_notebook(nb, filename):
    path = OUTPUT_DIR / filename
    with open(path, "w") as f:
        nbf.write(nb, f)
    print(f"  Created: {path}")


def main():
    notebooks = [
        ("Lecture 09 - Functions - Writing Reusable Code.ipynb", build_lecture_09()),
        ("Lecture 10 - Error Handling and Defensive Programming.ipynb", build_lecture_10()),
        ("Lecture 11 - File I-O - Reading and Writing Data.ipynb", build_lecture_11()),
        ("Lecture 12 - List Comprehensions and Generator Expressions.ipynb", build_lecture_12()),
        ("Lecture 13 - Iterators, Iterables, and the itertools Module.ipynb", build_lecture_13()),
        ("Lecture 14 - Modules, Packages, and the Standard Library.ipynb", build_lecture_14()),
        ("Lecture 15 - Object-Oriented Programming for Data Science.ipynb", build_lecture_15()),
        ("Lecture 16 - Mini-Project - CSV Data Cleaner CLI Tool.ipynb", build_lecture_16()),
    ]

    print(f"Generating {len(notebooks)} notebooks in {OUTPUT_DIR}...")
    for filename, nb in notebooks:
        write_notebook(nb, filename)
    print("Done.")


if __name__ == "__main__":
    main()
