#!/usr/bin/env python3
"""Generate all 8 Phase 1 Jupyter notebooks using nbformat."""

import nbformat as nbf
import os

OUTPUT_DIR = "/workspace/notebooks/phase-1-foundations"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_notebook(lecture_num, title, cells):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    }
    nb.cells = cells
    fname = f"Lecture {lecture_num} - {title}.ipynb"
    path = os.path.join(OUTPUT_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"  Created: {fname}")
    return path


def md(source):
    return nbf.v4.new_markdown_cell(source)


def code(source):
    return nbf.v4.new_code_cell(source)


def learning_objectives(items):
    bullets = "\n".join(f"- {item}" for item in items)
    return md(f"## Learning Objectives\n\n{bullets}")


def key_topics(items):
    bullets = "\n".join(f"- {item}" for item in items)
    return md(f"## Key Topics\n\n{bullets}")


def data_science_connection(text):
    return md(f"## Data Science Connection\n\n{text}")


# ──────────────────────────────────────────────
# Lecture 1
# ──────────────────────────────────────────────
def lecture_1():
    title = "Welcome, Setup, and Your First Program"
    cells = [
        md(f"# Lecture 1: {title}"),
        learning_objectives([
            "Understand why Python is the leading language for data science",
            "Install Python via the Anaconda distribution",
            "Navigate Jupyter Notebook and Jupyter Lab interfaces",
            "Write and run your first Python program using print()",
            "Use comments and understand basic program structure"
        ]),
        key_topics([
            "Why Python for data science?",
            "Installing Python via Anaconda",
            "Navigating Jupyter Notebook and Jupyter Lab",
            "The print() function and string literals",
            "Comments and basic program structure"
        ]),
        md(
            "## Why Python for Data Science?\n\n"
            "Python has become the de facto language for data science due to its simplicity, readability, "
            "and an incredibly rich ecosystem of libraries. Unlike languages like R or MATLAB, Python is a "
            "general-purpose language that you can use for everything from web development to machine learning. "
            "This means you only need to learn one language to build end-to-end data products.\n\n"
            "The key libraries that make Python so powerful for data science are: **NumPy** for numerical "
            "computing, **pandas** for data manipulation, **Matplotlib** and **seaborn** for visualization, "
            "and **scikit-learn** for machine learning. Each builds on the last, and by the end of this course "
            "you will be comfortable using all of them. For now, we start with pure Python — the foundation "
            "everything else rests on."
        ),
        md(
            "## Installing Python via Anaconda\n\n"
            "Anaconda is a free distribution of Python that bundles together Python itself, the most popular "
            "data science libraries, and powerful tools like Jupyter Notebook. It is the recommended way to "
            "get started because it eliminates the hassle of installing each library individually.\n\n"
            "To install, visit https://www.anaconda.com/products/individual and download the installer for "
            "your operating system. Run the installer and follow the prompts. Once finished, you can launch "
            "Jupyter Notebook from the Anaconda Navigator GUI or by typing `jupyter notebook` in your terminal. "
            "You can verify Python is installed by opening a terminal and typing `python --version`."
        ),
        md(
            "## Navigating Jupyter Notebook and Jupyter Lab\n\n"
            "Jupyter Notebook runs in your web browser and provides an interactive environment where you can "
            "mix code, text, and visualizations in a single document called a *notebook*. A notebook is made "
            "up of *cells* — each cell is either a code cell (where you write Python) or a markdown cell "
            "(where you write explanatory text). You run a cell by clicking the Run button or pressing "
            "Shift+Enter.\n\n"
            "Jupyter Lab is the next-generation interface that offers a more IDE-like experience with a file "
            "browser, multiple tabs, and a terminal built right into the browser window. Both tools use the "
            "same `.ipynb` file format, so you can switch between them freely. We recommend starting with "
            "Jupyter Notebook for simplicity, then graduating to Jupyter Lab as you become more comfortable."
        ),
        md(
            "## The print() Function and String Literals\n\n"
            "The `print()` function is the simplest way to display output in Python. You pass it one or more "
            "values, and it prints them to the screen. In data science, you will use `print()` constantly — "
            "to inspect data, debug your code, and report results. A *string literal* is a sequence of "
            "characters enclosed in quotes — either single quotes `'...'` or double quotes `\"...\"`.\n\n"
            "Python also supports triple-quoted strings using `'''...'''` or `\"\"\"...\"\"\"` which can span "
            "multiple lines. These are useful for printing formatted output or writing docstrings (which we "
            "will cover later). You can also embed basic arithmetic directly inside `print()` and Python will "
            "evaluate the expression before displaying it."
        ),
        code(
            "# Print a simple greeting\n"
            'print("Hello, Data Science World!")'
        ),
        code(
            "# Print multiple items with a separator\n"
            'print("The answer is", 42)\n'
            "# Print without newline by using the end parameter\n"
            'print("Processing...", end=" ")\n'
            'print("Done!")'
        ),
        code(
            "# Multi-line strings using triple quotes\n"
            'poem = """Roses are red,\n'
            'Violets are blue,\n'
            'Data science is fun,\n'
            "And so are you!\"\"\"\n"
            "print(poem)\n\n"
            "# Basic arithmetic inside print\n"
            'print("Sum:", 10 + 5)\n'
            'print("Product:", 3 * 7)\n'
            'print("Division:", 100 / 4)'
        ),
        md(
            "## Comments and Basic Program Structure\n\n"
            "Comments are notes you leave in your code for yourself and other humans. The Python interpreter "
            "ignores everything that follows a `#` on the same line. Good comments explain *why* you are doing "
            "something, not *what* the code does (the code itself should make that clear). In data science, "
            "comments are invaluable for documenting your thought process, citing data sources, and explaining "
            "nontrivial transformations.\n\n"
            "A well-structured Python program typically starts with import statements, then defines functions "
            "or variables, and finally contains the main logic. Even in a notebook, it is good practice to "
            "organize cells in a logical order — data loading comes first, then cleaning, then analysis, "
            "then visualization. Comments help future-you (and your collaborators) understand the narrative "
            "of your analysis."
        ),
        code(
            "# This is a single-line comment\n"
            'print("Comments make code easier to understand")\n'
            "# The next line calculates the average of three test scores\n"
            "avg = (85 + 92 + 78) / 3\n"
            'print("Average score:", avg)\n\n'
            "# TODO: Load the real dataset here in the next cell"
        ),
        code(
            "# Program structure: imports -> variables -> logic\n"
            "import math  # standard library module\n\n"
            "# Constants and configuration\n"
            "SAMPLE_SIZE = 100\n"
            'DATASET_NAME = "survey_results"\n\n'
            "# Core logic\n"
            "radius = 5\n"
            "area = math.pi * radius ** 2\n"
            'print(f"Area of a circle with radius {radius}: {area:.2f}")'
        ),
        data_science_connection(
            "The `print()` function is your window into what your code is doing. Every data scientist uses it "
            "constantly — to preview the first few rows of a dataset (`print(df.head())`), to check the shape "
            "of transformed data, or to print a final model accuracy score. Mastering `print()` and comments "
            "now will pay dividends in every subsequent lecture."
        ),
    ]
    return create_notebook(1, title, cells)


# ──────────────────────────────────────────────
# Lecture 2
# ──────────────────────────────────────────────
def lecture_2():
    title = "Variables, Data Types, and Basic Operations"
    cells = [
        md(f"# Lecture 2: {title}"),
        learning_objectives([
            "Assign variables and follow Python naming conventions",
            "Distinguish between int and float numeric types",
            "Manipulate strings with concatenation, repetition, and f-strings",
            "Use Boolean values and comparison operators",
            "Convert between types using int(), float(), str(), and bool()"
        ]),
        key_topics([
            "Variable assignment and naming conventions",
            "Numeric types: int vs float",
            "String basics: concatenation, repetition, f-strings",
            "Boolean values and comparison operators",
            "Type conversion (int(), float(), str(), bool())"
        ]),
        md(
            "## Variable Assignment and Naming Conventions\n\n"
            "A variable is a name that refers to a value stored in memory. In Python, you create a variable "
            "with the assignment operator `=`. Python is dynamically typed, meaning a variable can hold any "
            "type of data, and the type can change over the life of the variable. This flexibility is great "
            "for rapid prototyping, but it also means you need to be careful about what type a variable holds "
            "at any given moment.\n\n"
            "Python naming conventions (defined in PEP 8) recommend `snake_case` for variable names: lowercase "
            "words separated by underscores. Names should be descriptive but concise. Avoid single-letter names "
            "except for trivial loop counters (`i`, `j`). Also avoid reserved keywords like `for`, `if`, "
            "`while`, and built-in names like `list`, `dict`, `sum` — these will cause confusing errors."
        ),
        code(
            "# Valid variable assignments\n"
            "age = 28\n"
            "temperature_celsius = 22.5\n"
            'first_name = "Ada"\n'
            "is_employed = True\n\n"
            "# Print each variable to see its value\n"
            'print("Age:", age)\n'
            'print("Temperature:", temperature_celsius)\n'
            'print("Name:", first_name)\n'
            'print("Employed:", is_employed)\n\n'
            "# Use type() to check the data type\n"
            'print("type of age:", type(age))\n'
            'print("type of temperature:", type(temperature_celsius))'
        ),
        code(
            "# Bad variable names to avoid\n"
            "# l = [1, 2, 3]       # hard to read, shadows letter l\n"
            "# O = 100             # looks like zero\n"
            "# list = [1, 2, 3]    # shadows the built-in list() function\n\n"
            "# Good variable names\n"
            "customer_count = 143\n"
            "total_revenue = 45999.95\n"
            'is_subscribed = False\n\n'
            'print(f"Customers: {customer_count}, Revenue: ${total_revenue}")'
        ),
        md(
            "## Numeric Types: int vs float\n\n"
            "Python has two primary numeric types: `int` (integer) and `float` (floating-point number). "
            "Integers are whole numbers with no decimal point, like `42` or `-7`. Floats are numbers with "
            "a decimal point, like `3.14` or `-0.001`. Unless you have a specific reason to use an integer, "
            "division in Python 3 always returns a float.\n\n"
            "Be aware that floats have limited precision due to how computers represent them in binary. For "
            "example, `0.1 + 0.2` gives `0.30000000000000004`, not exactly `0.3`. This matters in data "
            "science when comparing floating-point numbers — never use `==` on floats; use a small tolerance "
            "instead. We will revisit this when we get to NumPy."
        ),
        code(
            "# Integer operations\n"
            "a = 10\n"
            "b = 3\n"
            'print("a + b =", a + b)\n'
            'print("a - b =", a - b)\n'
            'print("a * b =", a * b)\n'
            'print("a / b =", a / b)   # division always returns float\n'
            'print("a // b =", a // b) # floor division\n'
            'print("a % b =", a % b)   # modulo (remainder)\n'
            'print("a ** b =", a ** b) # exponentiation'
        ),
        code(
            "# Float precision gotcha\n"
            'print("0.1 + 0.2 =", 0.1 + 0.2)\n'
            'print("Is 0.1 + 0.2 == 0.3?", 0.1 + 0.2 == 0.3)\n\n'
            "# Better way to compare floats: use a tolerance\n"
            "tolerance = 1e-9\n"
            "result = 0.1 + 0.2\n"
            'print("Close enough?", abs(result - 0.3) < tolerance)\n\n'
            "# Type mixing: int + float => float\n"
            'print("int + float:", 5 + 2.5)'
        ),
        md(
            "## String Basics: Concatenation, Repetition, f-Strings\n\n"
            "Strings can be joined using the `+` operator (concatenation) or repeated using the `*` operator. "
            "While `+` works, it becomes unwieldy when combining many values. A much cleaner approach is the "
            "**f-string** — a string literal prefixed with `f` or `F` that lets you embed expressions directly "
            "inside curly braces `{}`. Introduced in Python 3.6, f-strings are the modern, preferred way to "
            "build strings in Python.\n\n"
            "In data science, you will use f-strings constantly — to format print statements, build file paths, "
            "create dynamic plot titles, and generate formatted report output. They are faster and more "
            "readable than older approaches like `%` formatting or `.format()`."
        ),
        code(
            "# String concatenation with +\n"
            'greeting = "Hello" + " " + "World"\n'
            'print("Concatenation:", greeting)\n\n'
            "# String repetition with *\n"
            'separator = "=" * 30\n'
            'print("Repetition:")\n'
            "print(separator)\n"
            'print("Chapter 1")\n'
            "print(separator)"
        ),
        code(
            "# f-strings: embed variables and expressions\n"
            'name = "Alice"\n'
            "score = 95.5\n"
            'print(f"Student: {name}, Score: {score}")\n\n'
            "# f-strings can evaluate expressions inline\n"
            "total = 100\n"
            "count = 3\n"
            'print(f"Average: {total / count:.2f}")  # :.2f means 2 decimal places\n\n'
            "# f-strings in data science context\n"
            "accuracy = 0.9342\n"
            "model_name = 'Random Forest'\n"
            'print(f"{model_name} accuracy: {accuracy:.2%}")  # :.2% formats as percentage'
        ),
        md(
            "## Boolean Values and Comparison Operators\n\n"
            "Booleans represent truth values: `True` and `False` (note the capital letters). They are the "
            "result of comparison operators like `==` (equal), `!=` (not equal), `<`, `>`, `<=`, `>=`. "
            "Booleans are the foundation of decision-making in code — they power `if` statements, `while` "
            "loops, and filter conditions in data analysis.\n\n"
            "In data science, comparison operators are essential for filtering rows in a dataset. For example, "
            "`df[df['age'] > 30]` keeps only rows where the age column is greater than 30. Understanding how "
            "comparisons work at the fundamental level is critical before moving to pandas."
        ),
        code(
            "# Comparison operators\n"
            'print("5 == 5:", 5 == 5)\n'
            'print("5 != 3:", 5 != 3)\n'
            'print("10 > 7:", 10 > 7)\n'
            'print("3 <= 3:", 3 <= 3)\n'
            "print('\"Python\" == \"python\":', \"Python\" == \"python\")\n"
            "print('\"apple\" < \"banana\":', \"apple\" < \"banana\")  # lexicographic comparison\n"
        ),
        code(
            "# Boolean logic in data filtering context\n"
            "age = 25\n"
            "salary = 60000\n"
            'has_experience = True\n\n'
            "# Combine comparisons with 'and' and 'or'\n"
            "eligible = (age > 21) and (salary >= 50000)\n"
            'print("Eligible for loan:", eligible)\n\n'
            "hired = has_experience or (age < 30)\n"
            'print("Candidate hired:", hired)\n\n'
            "# Use not to invert a boolean\n"
            'print("Not True:", not True)\n'
            'print("Not False:", not False)'
        ),
        md(
            "## Type Conversion (int(), float(), str(), bool())\n\n"
            "Sometimes you need to explicitly convert a value from one type to another. Python provides "
            "built-in functions for this: `int()` converts to integer, `float()` to float, `str()` to "
            "string, and `bool()` to boolean. This is especially common in data science when reading data "
            "from files — numbers may arrive as strings and need converting before you can compute with them.\n\n"
            "Not all conversions are valid. For example, `int('hello')` will raise a `ValueError`. Always "
            "validate or handle exceptions when converting user-supplied or file-supplied data. The `bool()` "
            "function has a special behavior: it returns `False` for `0`, `0.0`, `''` (empty string), `None`, "
            "and empty containers; everything else is `True`."
        ),
        code(
            "# Converting between types\n"
            'price_str = "49.99"\n'
            "quantity_str = '3'\n\n"
            "# Convert string to numeric types\n"
            "price = float(price_str)\n"
            "quantity = int(quantity_str)\n"
            "total = price * quantity\n"
            'print(f"Total: ${total:.2f}")\n\n'
            "# Convert number back to string for display\n"
            "total_str = str(total)\n"
            'print("Type of total_str:", type(total_str))'
        ),
        code(
            "# bool() conversion rules\n"
            'print("bool(0):", bool(0))\n'
            'print("bool(1):", bool(1))\n'
            'print("bool(\\"\\"):", bool(""))\n'
            'print("bool(\\"Python\\"):", bool("Python"))\n'
            'print("bool([]):", bool([]))\n'
            'print("bool([1, 2]):", bool([1, 2]))\n'
            'print("bool(None):", bool(None))'
        ),
        data_science_connection(
            "Every dataset you work with will contain a mix of data types: numeric columns (int/float), "
            "text columns (strings), and categorical/Boolean columns. When you load a CSV into pandas, it "
            "automatically infers types, but real-world data is messy — you will frequently need to cast "
            "columns to the correct type before analysis. The type conversion skills you learned here are "
            "exactly what you will use to clean and prepare data for modeling."
        ),
    ]
    return create_notebook(2, title, cells)


# ──────────────────────────────────────────────
# Lecture 3
# ──────────────────────────────────────────────
def lecture_3():
    title = "Strings and String Methods"
    cells = [
        md(f"# Lecture 3: {title}"),
        learning_objectives([
            "Understand why strings are immutable and what that means in practice",
            "Apply common string methods: .lower(), .upper(), .strip(), .split(), .join(), .replace()",
            "Access individual characters with positive and negative indexing",
            "Extract substrings using slicing syntax [start:stop:step]",
            "Format output with f-string expressions and format specifiers"
        ]),
        key_topics([
            "String immutability",
            "Common methods: .lower(), .upper(), .strip(), .split(), .join(), .replace()",
            "Indexing (positive and negative)",
            "Slicing syntax [start:stop:step]",
            "f-string formatting with expressions"
        ]),
        md(
            "## String Immutability\n\n"
            "Strings in Python are **immutable** — once a string is created, you cannot change individual "
            "characters in place. If you try `s[0] = 'X'`, Python raises a `TypeError`. Instead, string "
            "*methods* return a **new** string with the modification applied, leaving the original unchanged. "
            "This is a common source of bugs for beginners who forget to assign the result back to a variable.\n\n"
            "Immutability has important benefits: strings can be used as keys in dictionaries, they are "
            "hashable, and they are safe to share across threads. In data science, you will often process "
            "text data (customer names, product descriptions, log files) by creating cleaned-up copies "
            "rather than modifying the originals — immutability encourages this safe pattern."
        ),
        code(
            "# Strings are immutable — this will fail:\n"
            "# s = \"hello\"\n"
            "# s[0] = \"H\"  # TypeError: 'str' object does not support item assignment\n\n"
            "# Correct: string methods return a new string\n"
            's = "hello"\n'
            "s_capitalized = s.capitalize()\n"
            'print("Original:", s)\n'
            'print("New:", s_capitalized)\n\n'
            "# Each method call creates a fresh string\n"
            'name = "  ada lovelace  "\n'
            "clean = name.strip().title()\n"
            'print(f"Original: [{name}]")\n'
            'print(f"Cleaned: [{clean}]")'
        ),
        code(
            "# Immutability error demonstration\n"
            '# Trying to modify a string in place:\n'
            'tweet = "i love data science"\n'
            '# tweet[0] = "I"  # uncomment to see the error\n\n'
            "# Correct approach: create a new string\n"
            "tweet_fixed = tweet[0].upper() + tweet[1:]\n"
            'print("Original:", tweet)\n'
            'print("Fixed:", tweet_fixed)'
        ),
        md(
            "## Common String Methods: .lower(), .upper(), .strip(), .split(), .join(), .replace()\n\n"
            "Python strings come with a rich set of built-in methods. The most commonly used in data "
            "cleaning are: `.lower()` and `.upper()` for case normalization; `.strip()` (and its variants "
            "`.lstrip()` / `.rstrip()`) for removing whitespace from the edges; `.split()` for breaking "
            "a string into a list; `.join()` for assembling a list back into a string; and `.replace()` "
            "for substituting characters or substrings.\n\n"
            "A powerful technique is **method chaining** — calling multiple methods on the same string in "
            "a single expression. Since each method returns a new string, you can chain them like "
            "`s.strip().lower().replace(' ', '_')`. This is idiomatic Python and widely used in data "
            "science pipelines to clean text data efficiently."
        ),
        code(
            "# Case conversion and whitespace removal\n"
            'raw = "  Hello WORLD!  "\n'
            'print("lower():", raw.lower())\n'
            'print("upper():", raw.upper())\n'
            'print("strip():", raw.strip())\n'
            'print("title():", raw.strip().title())\n\n'
            "# Method chaining for data cleaning\n"
            'dirty = "   DATA_SCIENCE   "\n'
            "clean = dirty.strip().lower().replace('_', ' ')\n"
            'print("Cleaned:", clean)'
        ),
        code(
            "# split() and join() — critical for CSV/text processing\n"
            'csv_line = "Alice,25,Engineer,75000"\n'
            "fields = csv_line.split(',')\n"
            'print("split result:", fields)\n\n'
            "# join() is called on the delimiter, takes a list of strings\n"
            "tags = ['python', 'data-science', 'tutorial']\n"
            "tag_string = ' | '.join(tags)\n"
            'print("Joined tags:", tag_string)\n\n'
            "# Practical: normalize user input\n"
            'user_input = "  New York  "\n'
            "normalized = user_input.strip().lower().replace(' ', '_')\n"
            'print(f"Normalized city key: {normalized}")'
        ),
        code(
            "# replace() for cleaning text\n"
            'review = "The product was good, but shipping was SLOW!!!"\n'
            "clean_review = review.replace('!', '').replace('SLOW', 'slow')\n"
            'print("Original:", review)\n'
            'print("Cleaned:", clean_review)\n\n'
            "# Remove multiple whitespace using split+join\n"
            'messy = "Python    is   awesome"\n'
            'clean = " ".join(messy.split())\n'
            'print("Whitespace cleaned:", clean)'
        ),
        md(
            "## Indexing (Positive and Negative)\n\n"
            "Every character in a string has an index. Positive indices start at 0 for the first character "
            "and increase to the right. Negative indices start at -1 for the last character and decrease "
            "to the left. This dual indexing system lets you access characters from either end without "
            "needing to know the string length.\n\n"
            "In data science, you will use indexing to extract specific characters from identifiers, pick "
            "the last letter of a code, or check the first character of a column value. For example, "
            "`user_id[-4:]` might give you the last 4 digits of a user ID — a quick way to extract a "
            "batch code or year."
        ),
        code(
            "# Positive indexing (starts at 0)\n"
            'word = "PYTHON"\n'
            'print("Word:", word)\n'
            'print("Index 0:", word[0])\n'
            'print("Index 1:", word[1])\n'
            'print("Index 5:", word[5])\n\n'
            "# Negative indexing (starts at -1 for the last character)\n"
            'print("Index -1:", word[-1])\n'
            'print("Index -2:", word[-2])\n'
            'print("Index -6:", word[-6])\n\n'
            "# Practical: extract file extension\n"
            'filename = "report_2024.csv"\n'
            'extension = filename[-4:]\n'  # this uses slicing, coming next
            'print(f"File: {filename}")'
        ),
        code(
            "# Accessing characters in a data science context\n"
            'product_code = "DS-PY-101-B"\n'
            'print("Product code:", product_code)\n'
            'print("First char:", product_code[0])\n'
            'print("Last char:", product_code[-1])\n'
            'print("Third char:", product_code[2])\n\n'
            "# Get the department code (first two characters)\n"
            "dept = product_code[:2]\n"
            'print("Department:", dept)\n\n'
            "# IndexError when out of bounds\n"
            '# print(product_code[20])  # uncomment to see IndexError'
        ),
        md(
            "## Slicing Syntax [start:stop:step]\n\n"
            "Slicing allows you to extract any contiguous (or non-contiguous) portion of a string. The "
            "syntax is `string[start:stop:step]`, where `start` is inclusive, `stop` is exclusive, and "
            "`step` is the increment. All three are optional. Omitting `start` defaults to the beginning, "
            "omitting `stop` defaults to the end, and omitting `step` defaults to 1.\n\n"
            "Slicing is one of the most elegant features of Python. A negative step reverses direction, "
            "letting you reverse a string with `[::-1]`. In data science, slicing is essential for "
            "extracting date parts from ISO-formatted strings, trimming file extensions, or generating "
            "substrings for feature engineering."
        ),
        code(
            "# Basic slicing examples\n"
            'text = "Data Science"\n'
            'print("Full string:", text)\n'
            'print("text[0:4]:", text[0:4])\n'
            'print("text[5:12]:", text[5:12])\n'
            'print("text[:4]:", text[:4])\n'
            'print("text[5:]:", text[5:])\n'
            'print("text[:]:", text[:])  # full copy\n\n'
            "# Using step\n"
            'print("text[::2]:", text[::2])\n'
            'print("text[::-1]:", text[::-1])  # reverse'
        ),
        code(
            "# Practical slicing for data cleaning\n"
            'date_str = "2024-03-15"\n'
            "year = date_str[:4]\n"
            "month = date_str[5:7]\n"
            "day = date_str[8:10]\n"
            'print(f"Date: {year}-{month}-{day}")\n\n'
            "# Extract domain from email\n"
            'email = "alice@example.com"\n'
            "at_pos = email.index('@')  # find the @ symbol\n"
            "username = email[:at_pos]\n"
            "domain = email[at_pos + 1:]\n"
            'print(f"Username: {username}")\n'
            'print(f"Domain: {domain}")\n\n'
            "# Negative step to reverse\n"
            'palindrome = "racecar"\n'
            'print("Is palindrome?", palindrome == palindrome[::-1])'
        ),
        md(
            "## f-string Formatting with Expressions\n\n"
            "f-strings are not just for embedding simple variables — they can contain any valid Python "
            "expression, including arithmetic, function calls, and method calls. You can also use format "
            "specifiers after a colon to control how values are displayed: `:.2f` for two decimal places, "
            "`:.2%` for percentage, `:>10` for right-alignment within a width of 10, and many more.\n\n"
            "In data science, f-strings are used everywhere: printing model metrics with controlled "
            "precision, formatting currency values, creating dynamic plot titles, and building report "
            "strings. They combine readability with power, making your output both informative and "
            "professional-looking."
        ),
        code(
            "# Expressions inside f-strings\n"
            "prices = [9.99, 15.49, 23.00]\n"
            'print(f"Min: ${min(prices):.2f}, Max: ${max(prices):.2f}")\n\n'
            "# Method calls inside f-strings\n"
            'city = "  paris  "\n'
            'print(f"City: {city.strip().title()}")\n\n'
            "# Conditional expressions\n"
            "score = 85\n"
            'print(f"Result: {\'Pass\' if score >= 70 else \'Fail\'}")'
        ),
        code(
            "# Format specifiers for alignment and padding\n"
            'for name, price in [("Apple", 0.5), ("Banana", 0.25), ("Cherry", 0.75)]:\n'
            "    print(f\"{name:>8}  ${price:.2f}\")\n\n"
            "# Percentage formatting\n"
            "accuracy = 0.8765\n"
            'print(f"Model accuracy: {accuracy:.2%}")\n\n'
            "# Large numbers with comma separator\n"
            "population = 8_100_000_000\n"
            'print(f"World population: {population:,}")'
        ),
        data_science_connection(
            "Text data is everywhere in data science: customer reviews, tweets, survey responses, log files, "
            "and product descriptions. The string methods and slicing techniques you just learned are the "
            "exact tools you will use to clean and prepare this text for analysis. When you later work with "
            "pandas, you will apply these same methods to entire columns at once using `.str` accessor — "
            "for example, `df['name'].str.strip().str.lower()`. Master strings now and text preprocessing "
            "in pandas will feel intuitive."
        ),
    ]
    return create_notebook(3, title, cells)


# ──────────────────────────────────────────────
# Lecture 4
# ──────────────────────────────────────────────
def lecture_4():
    title = "Lists: The Data Scientist's Workhorse"
    cells = [
        md(f"# Lecture 4: {title}"),
        learning_objectives([
            "Create lists with heterogeneous elements",
            "Access elements using indexing, slicing, and work with nested lists",
            "Modify lists with .append(), .extend(), .insert(), .remove(), .pop()",
            "Sort lists with .sort() and sorted()",
            "Test membership with the in operator",
            "Understand the difference between shallow copies and deep copies"
        ]),
        key_topics([
            "List creation and heterogeneous elements",
            "Indexing, slicing, and nested lists",
            "Methods: .append(), .extend(), .insert(), .remove(), .pop(), .sort()",
            "in operator and list membership",
            "Shallow copies vs deep copies with copy()"
        ]),
        md(
            "## List Creation and Heterogeneous Elements\n\n"
            "A list is an ordered, mutable collection of elements enclosed in square brackets `[]`. Unlike "
            "arrays in many other languages, Python lists can hold elements of **different types** — you can "
            "mix integers, strings, floats, and even other lists in a single list. This flexibility makes "
            "lists ideal for storing heterogeneous data like a row from a spreadsheet.\n\n"
            "Lists are the closest native Python data structure to a data frame row or a CSV record. In "
            "data science, you will often use lists to hold column names, store raw data before converting "
            "it to a NumPy array or pandas DataFrame, or accumulate results during iteration."
        ),
        code(
            "# Creating lists with mixed types\n"
            "empty_list = []\n"
            "numbers = [1, 2, 3, 4, 5]\n"
            'mixed = [42, "hello", 3.14, True]\n'
            'print("Empty:", empty_list)\n'
            'print("Numbers:", numbers)\n'
            'print("Mixed:", mixed)\n\n'
            "# List from a range (common in data processing)\n"
            "indices = list(range(10))\n"
            'print("Indices:", indices)'
        ),
        code(
            "# Lists as data records\n"
            '# Think of each list as one row of a dataset\n'
            'record_1 = ["Alice", 28, "Engineer", 75000]\n'
            'record_2 = ["Bob", 35, "Data Scientist", 95000]\n'
            'record_3 = ["Charlie", 22, "Intern", 45000]\n\n'
            "all_records = [record_1, record_2, record_3]\n"
            'print("All records:", all_records)\n\n'
            "# Check the length\n"
            'print("Number of records:", len(all_records))\n'
            'print("Fields in first record:", len(record_1))'
        ),
        md(
            "## Indexing, Slicing, and Nested Lists\n\n"
            "Lists support the same indexing and slicing syntax as strings: `my_list[0]` gives the first "
            "element, `my_list[-1]` gives the last, and `my_list[1:4]` gives elements at indices 1, 2, and 3. "
            "But unlike strings, lists are mutable — you can assign to an index: `my_list[0] = new_value`. "
            "This mutability is what makes lists suitable for building and modifying collections of data.\n\n"
            "Nested lists (lists within lists) are useful for representing 2D data like spreadsheets or "
            "matrices. Accessing elements in nested lists uses multiple indices: `matrix[row][col]`. In data "
            "science, you might use nested lists to hold a small dataset before converting it to a NumPy "
            "array for numerical computation."
        ),
        code(
            "# Indexing and slicing\n"
            'fruits = ["apple", "banana", "cherry", "date", "elderberry"]\n'
            'print("First:", fruits[0])\n'
            'print("Last:", fruits[-1])\n'
            'print("First three:", fruits[:3])\n'
            'print("Every other:", fruits[::2])\n\n'
            "# Lists are mutable — assign to an index\n"
            'fruits[1] = "blueberry"\n'
            'print("After change:", fruits)'
        ),
        code(
            "# Nested lists (2D data)\n"
            'matrix = [\n'
            "    [1, 2, 3],\n"
            "    [4, 5, 6],\n"
            "    [7, 8, 9]\n"
            "]\n"
            'print("Matrix:", matrix)\n'
            'print("Row 0:", matrix[0])\n'
            'print("Element [1][2]:", matrix[1][2])\n\n'
            "# Nested list as a mini spreadsheet\n"
            'data = [\n'
            '    ["Name", "Age", "Salary"],\n'
            '    ["Alice", 28, 75000],\n'
            '    ["Bob", 35, 95000]\n'
            "]\n"
            'print("Column names:", data[0])\n'
            'print("First data row:", data[1])\n'
            'print("Bob\'s salary:", data[2][2])'
        ),
        md(
            "## List Methods: .append(), .extend(), .insert(), .remove(), .pop(), .sort()\n\n"
            "Lists have a rich set of methods for modification. The most important for data science are: "
            "`.append()` to add a single element to the end; `.extend()` to add all elements from another "
            "iterable; `.insert()` to add at a specific position; `.remove()` to delete the first occurrence "
            "of a value; `.pop()` to remove and return an element by index (default is the last element); "
            "and `.sort()` to sort in place.\n\n"
            "Note that methods like `.append()` and `.sort()` modify the list **in place** and return "
            "`None`. This is a common pitfall: `my_list = my_list.append(x)` will set `my_list` to `None`. "
            "Always call these methods without assignment."
        ),
        code(
            "# append() and extend()\n"
            "stack = []\n"
            "stack.append(10)\n"
            "stack.append(20)\n"
            "stack.append(30)\n"
            'print("After appends:", stack)\n\n'
            "more = [40, 50]\n"
            "stack.extend(more)\n"
            'print("After extend:", stack)\n\n'
            "# insert() and remove()\n"
            "stack.insert(0, 5)\n"
            'print("After insert at 0:", stack)\n'
            "stack.remove(30)\n"
            'print("After remove(30):", stack)'
        ),
        code(
            "# pop() and sort()\n"
            "tasks = [3, 1, 4, 1, 5, 9, 2, 6]\n"
            'print("Original:", tasks)\n\n'
            "# Sort in place (modifies the list)\n"
            "tasks.sort()\n"
            'print("Sorted:", tasks)\n\n'
            "# Sort in reverse\n"
            "tasks.sort(reverse=True)\n"
            'print("Reverse sorted:", tasks)\n\n'
            "# Pop from the end (like a stack)\n"
            "last = tasks.pop()\n"
            'print("Popped:", last)\n'
            'print("Remaining:", tasks)\n\n'
            "# sorted() returns a NEW list (does not modify original)\n"
            "original = [3, 1, 4]\n"
            "sorted_copy = sorted(original)\n"
            'print("Original unchanged:", original)\n'
            'print("Sorted copy:", sorted_copy)'
        ),
        md(
            "## The `in` Operator and List Membership\n\n"
            "The `in` operator checks whether a value exists in a list, returning `True` or `False`. It is "
            "a clean and readable way to test membership: `if 'apple' in fruits:`. The `not in` variant "
            "checks for absence. Under the hood, Python scans the list from beginning to end (O(n) time "
            "complexity), so for very large lists this is slower than using a set.\n\n"
            "In data science, `in` is often used in filtering — for example, keeping only rows where a "
            "category value is in a whitelist: `if category in ['A', 'B', 'C']:`. It is also useful for "
            "data validation, such as checking whether a required column name exists before processing."
        ),
        code(
            "# The in operator\n"
            'colors = ["red", "green", "blue"]\n'
            'print("red in colors:", "red" in colors)\n'
            'print("yellow in colors:", "yellow" in colors)\n'
            'print("yellow not in colors:", "yellow" not in colors)\n\n'
            "# Filtering with in\n"
            'allowed_categories = ["A", "B", "C"]\n'
            'record = "B"\n'
            "if record in allowed_categories:\n"
            '    print(f"Category {record} is allowed")\n\n'
            "# in with nested structures\n"
            "dataset = [\n"
            '    {"id": 1, "status": "active"},\n'
            '    {"id": 2, "status": "inactive"}\n'
            "]\n"
            "# Check if any record has a certain value\n"
            'statuses = [r["status"] for r in dataset]\n'
            'print("active present:", "active" in statuses)'
        ),
        md(
            "## Shallow Copies vs Deep Copies with copy()\n\n"
            "When you assign a list to a new variable with `new_list = old_list`, you are NOT creating a "
            "copy — both variables point to the same object in memory. Modifying one will affect the other. "
            "To create an actual copy, you can use `list.copy()` or `old_list[:]`, which creates a **shallow "
            "copy** — a new list object whose elements are references to the same objects as the original.\n\n"
            "For nested lists, a shallow copy is often insufficient because modifying a nested element (like "
            "a sublist) will still affect both the original and the copy. To fully decouple nested structures, "
            "you need a **deep copy** via `copy.deepcopy()`. In data science, understanding copies is "
            "critical when preprocessing data — you want to avoid accidentally mutating your original data."
        ),
        code(
            "# Assignment vs copy\n"
            "original = [1, 2, 3]\n"
            "assigned = original  # both point to the same list\n"
            "assigned.append(4)\n"
            'print("Original after assign-append:", original)\n\n'
            "# Shallow copy with .copy()\n"
            "original = [1, 2, 3]\n"
            "shallow = original.copy()\n"
            "shallow.append(4)\n"
            'print("Original after copy-append:", original)\n'
            'print("Shallow copy:", shallow)'
        ),
        code(
            "# Shallow copy pitfall with nested lists\n"
            "original = [[1, 2], [3, 4]]\n"
            "shallow = original.copy()\n"
            "shallow[0].append(99)\n"
            'print("Original:", original)\n'
            'print("Shallow:", shallow)\n\n'
            "# Deep copy to the rescue\n"
            "import copy\n"
            "original = [[1, 2], [3, 4]]\n"
            "deep = copy.deepcopy(original)\n"
            "deep[0].append(99)\n"
            'print("Original after deep copy mod:", original)\n'
            'print("Deep copy:", deep)'
        ),
        data_science_connection(
            "Lists are the building block of data science in Python. When you call `list(df['column'])` in "
            "pandas, you get a list. When you use `.tolist()` on a NumPy array, you get a list. When you "
            "read lines from a file, you get a list of strings. Everything you learn about lists — indexing, "
            "slicing, methods, copies — transfers directly to working with data. In the next lecture, we will "
            "meet tuples and sets, which complement lists for specific use cases."
        ),
    ]
    return create_notebook(4, title, cells)


# ──────────────────────────────────────────────
# Lecture 5
# ──────────────────────────────────────────────
def lecture_5():
    title = "Tuples, Sets, and When to Use Them"
    cells = [
        md(f"# Lecture 5: {title}"),
        learning_objectives([
            "Create tuples and understand why immutability matters",
            "Unpack tuples into individual variables",
            "Know when to prefer a tuple over a list",
            "Create sets and leverage their uniqueness guarantee",
            "Perform set operations: union, intersection, difference, symmetric difference",
            "Understand frozen sets and their use cases"
        ]),
        key_topics([
            "Tuple creation, immutability, unpacking",
            "When to prefer a tuple over a list",
            "Set creation, uniqueness guarantee",
            "Set operations: union, intersection, difference, symmetric difference",
            "Frozen sets"
        ]),
        md(
            "## Tuple Creation, Immutability, and Unpacking\n\n"
            "A tuple is an ordered, immutable collection of elements, written with parentheses `()`. Once "
            "created, you cannot add, remove, or change elements. This immutability makes tuples **hashable** "
            "(if all their elements are also hashable), which means they can be used as dictionary keys and "
            "members of sets — something lists cannot do.\n\n"
            "**Tuple unpacking** is a powerful feature: you can assign each element of a tuple to a separate "
            "variable in one line: `name, age, city = person`. This is extensively used in data science for "
            "iterating over collections of structured data, returning multiple values from functions, and "
            "swapping variables elegantly."
        ),
        code(
            "# Creating tuples\n"
            "empty = ()\n"
            "single = (42,)  # trailing comma is REQUIRED for single-element tuple\n"
            "coordinates = (40.7128, -74.0060)\n"
            'person = ("Alice", 28, "Engineer")\n'
            'print("Coordinates:", coordinates)\n'
            'print("Person:", person)\n\n'
            "# Tuples are immutable\n"
            '# coordinates[0] = 0  # TypeError\n\n'
            "# Tuples can be used as dictionary keys\n"
            'location_map = {(40.71, -74.01): "NYC", (34.05, -118.24): "LA"}\n'
            'print("Location map:", location_map)'
        ),
        code(
            "# Tuple unpacking\n"
            'record = ("Bob", 35, 95000)\n'
            "name, age, salary = record\n"
            'print(f"Name: {name}, Age: {age}, Salary: ${salary}")\n\n'
            "# Swapping variables with tuples\n"
            "a, b = 10, 20\n"
            'print(f"Before swap: a={a}, b={b}")\n'
            "a, b = b, a\n"
            'print(f"After swap: a={a}, b={b}")\n\n'
            "# Unpacking in a for loop (common in data science)\n"
            'employees = [\n'
            '    ("Alice", 28, 75000),\n'
            '    ("Bob", 35, 95000),\n'
            '    ("Charlie", 22, 45000)\n'
            "]\n"
            "for name, age, salary in employees:\n"
            '    print(f"{name} is {age} years old, earns ${salary}")'
        ),
        md(
            "## When to Prefer a Tuple Over a List\n\n"
            "Use a tuple when: (1) the data should not change (immutability provides safety), (2) you need "
            "to use the collection as a dictionary key, (3) you want to return multiple values from a "
            "function, or (4) you want to communicate to other developers that the sequence is fixed. Tuples "
            "are also slightly more memory-efficient than lists.\n\n"
            "In data science, tuples are commonly used for: coordinates (lat, lon), RGB color values, "
            "function return values, dictionary keys for composite identifiers (e.g., `(user_id, date)`), "
            "and fixed configuration parameters that should not be changed accidentally."
        ),
        code(
            "# Tuple as a dictionary key (composite key)\n"
            "sales = {}\n"
            'sales[("NY", "2024-01")] = 15000\n'
            'sales[("CA", "2024-01")] = 22000\n'
            'sales[("NY", "2024-02")] = 18000\n'
            'print("NY Jan sales:", sales[("NY", "2024-01")])\n\n'
            "# Tuple for fixed configuration\n"
            'COLORS = [\n'
            '    ("red", "#FF0000"),\n'
            '    ("green", "#00FF00"),\n'
            '    ("blue", "#0000FF")\n'
            "]\n"
            "for name, hex_code in COLORS:\n"
            '    print(f"{name}: {hex_code}")'
        ),
        code(
            "# Tuple vs list: immutability as a safety feature\n"
            "# If you pass a list to a function, it can be modified\n"
            "def add_item_buggy(items):\n"
            "    items.append('secret')\n"
            '    return items\n\n'
            "my_list = ['a', 'b']\n"
            "result = add_item_buggy(my_list)\n"
            'print("Original modified:", my_list)\n\n'
            "# With a tuple, the caller knows the data is safe\n"
            "def process_coordinates(coords):\n"
            "    # coords[0] = 0  # Would raise TypeError\n"
            '    return coords\n\n'
            'my_tuple = (10, 20)\n'
            "result = process_coordinates(my_tuple)\n"
            'print("Tuple unchanged:", my_tuple)'
        ),
        md(
            "## Set Creation and the Uniqueness Guarantee\n\n"
            "A set is an unordered collection of **unique** elements, created with curly braces `{}` or the "
            "`set()` constructor. Duplicates are automatically eliminated. Sets are also **unhashable** "
            "themselves (you cannot put a set inside another set), but all elements within a set must be "
            "hashable (immutable).\n\n"
            "The uniqueness guarantee is extremely useful in data science for deduplication tasks. For example, "
            "finding the unique categories in a column, identifying unique customers, or removing duplicate "
            "entries from a list. Membership testing with `in` is O(1) on average for sets, compared to O(n) "
            "for lists — a huge performance win on large datasets."
        ),
        code(
            "# Set creation and deduplication\n"
            'numbers = [1, 2, 2, 3, 3, 3, 4, 5, 5]\n'
            "unique_numbers = set(numbers)\n"
            'print("Original list:", numbers)\n'
            'print("Unique set:", unique_numbers)\n\n'
            "# Direct set literal\n"
            'categories = {"A", "B", "C", "A", "B"}\n'
            'print("Categories:", categories)\n\n'
            "# Fast membership testing\n"
            'print("Is B a category?", "B" in categories)\n'
            'print("Is Z a category?", "Z" in categories)'
        ),
        code(
            "# Deduplication in a data science context\n"
            'user_ids = [101, 102, 103, 101, 104, 102, 105, 101]\n'
            "unique_users = set(user_ids)\n"
            'print("User IDs:", user_ids)\n'
            'print("Unique users:", unique_users)\n'
            'print("Number of unique users:", len(unique_users))\n\n'
            "# Remove duplicates while preserving order (Python 3.7+)\n"
            "seen = set()\n"
            "ordered_unique = []\n"
            "for uid in user_ids:\n"
            "    if uid not in seen:\n"
            "        ordered_unique.append(uid)\n"
            "        seen.add(uid)\n"
            'print("Ordered unique:", ordered_unique)'
        ),
        md(
            "## Set Operations: Union, Intersection, Difference, Symmetric Difference\n\n"
            "Sets support mathematical set operations that map directly to common data analysis tasks. The "
            "operators are: `|` for union (all elements from both sets), `&` for intersection (elements in "
            "both), `-` for difference (elements in the first but not the second), and `^` for symmetric "
            "difference (elements in either set but not both).\n\n"
            "In data science, these operations are invaluable for comparing groups. For example: finding "
            "customers who purchased in both months (intersection), finding users who signed up but never "
            "purchased (difference), or combining two lists of tags (union). These operations are highly "
            "optimized and much faster than equivalent list comprehensions."
        ),
        code(
            "# Set operations\n"
            'power_users = {"Alice", "Bob", "Charlie", "Diana"}\n'
            'active_users = {"Bob", "Diana", "Eve", "Frank"}\n\n'
            'print("Power users:", power_users)\n'
            'print("Active users:", active_users)\n\n'
            "# Union: all users\n"
            'all_users = power_users | active_users\n'
            'print("Union (all):", all_users)\n\n'
            "# Intersection: users in both groups\n"
            'both = power_users & active_users\n'
            'print("Intersection (both):", both)\n\n'
            "# Difference: power users NOT active\n"
            'power_only = power_users - active_users\n'
            'print("Difference (power only):", power_only)\n\n'
            "# Symmetric difference: in exactly one group\n"
            'exactly_one = power_users ^ active_users\n'
            'print("Symmetric diff:", exactly_one)'
        ),
        code(
            "# Set operations in data analysis\n"
            'last_week = {"prod_a", "prod_b", "prod_c", "prod_d"}\n'
            'this_week = {"prod_c", "prod_d", "prod_e", "prod_f"}\n\n'
            "# Products launched both weeks\n"
            'repeated = last_week & this_week\n'
            'print("Repeated products:", repeated)\n\n'
            "# New products this week (not sold last week)\n"
            'new_products = this_week - last_week\n'
            'print("New products:", new_products)\n\n'
            "# Discontinued products (sold last week but not this week)\n"
            'discontinued = last_week - this_week\n'
            'print("Discontinued:", discontinued)'
        ),
        md(
            "## Frozen Sets\n\n"
            "A `frozenset` is an immutable version of a set. It supports all set operations but cannot be "
            "modified after creation — no `.add()`, `.remove()`, or `.discard()`. Because of its immutability, "
            "a `frozenset` is **hashable** and can be used as a dictionary key or as an element of another set.\n\n"
            "Frozen sets are useful when you need to use a set as a dictionary key (e.g., grouping data by "
            "a set of tags) or when you want a constant set of values that should not be accidentally modified. "
            "They are also used in caching and memoization where the key needs to be hashable."
        ),
        code(
            "# Frozenset creation\n"
            'frozen = frozenset([1, 2, 3, 3, 4])\n'
            'print("Frozenset:", frozen)\n'
            'print("Type:", type(frozen))\n\n'
            "# Frozensets are hashable — can be dictionary keys\n"
            "group_a = frozenset({'feature_a', 'feature_b'})\n"
            "group_b = frozenset({'feature_c', 'feature_d'})\n"
            "feature_groups = {\n"
            '    group_a: "Group A contains features for login",\n'
            '    group_b: "Group B contains features for checkout"\n'
            "}\n"
            "for group, description in feature_groups.items():\n"
            '    print(f"{group}: {description}")'
        ),
        code(
            "# Frozenset with set operations\n"
            "base_tags = frozenset(['python', 'data-science', 'tutorial'])\n"
            "extra_tags = frozenset(['python', 'machine-learning', 'beginner'])\n\n"
            "# All operations work on frozensets\n"
            'print("Union:", base_tags | extra_tags)\n'
            'print("Intersection:", base_tags & extra_tags)\n'
            'print("Difference:", base_tags - extra_tags)\n\n'
            "# Frozenset inside a set (not possible with regular sets)\n"
            "s = frozenset([1, 2])\n"
            "t = frozenset([3, 4])\n"
            "set_of_frozensets = {s, t}\n"
            'print("Set of frozensets:", set_of_frozensets)'
        ),
        data_science_connection(
            "Tuples and sets are specialized tools that every data scientist reaches for regularly. "
            "Tuples serve as lightweight, immutable records — perfect for coordinates, function returns, "
            "and dictionary keys in aggregation tasks. Sets power deduplication, membership tests, and "
            "group comparisons, all of which are everyday operations in exploratory data analysis. "
            "Understanding when to reach for each structure will make your code faster, safer, and more "
            "expressive."
        ),
    ]
    return create_notebook(5, title, cells)


# ──────────────────────────────────────────────
# Lecture 6
# ──────────────────────────────────────────────
def lecture_6():
    title = "Dictionaries and Mapping Data"
    cells = [
        md(f"# Lecture 6: {title}"),
        learning_objectives([
            "Understand key-value pairs and hashability constraints",
            "Use dictionary methods: .keys(), .values(), .items(), .get(), .setdefault()",
            "Iterate over dictionaries efficiently",
            "Write dictionary comprehensions",
            "Use defaultdict and Counter from the collections module"
        ]),
        key_topics([
            "Key-value pairs, hashability constraints",
            "Methods: .keys(), .values(), .items(), .get(), .setdefault()",
            "Iterating over dictionaries",
            "Dictionary comprehensions",
            "defaultdict and Counter from collections"
        ]),
        md(
            "## Key-Value Pairs and Hashability Constraints\n\n"
            "A dictionary (or `dict`) is an unordered collection of key-value pairs, where each key maps to "
            "a value. Dictionaries are created with curly braces `{}` using the syntax `{key: value, ...}`. "
            "Keys must be **hashable** (immutable types like strings, numbers, and tuples), while values can "
            "be any Python object — including lists, other dictionaries, or even functions.\n\n"
            "Dictionaries are the closest native Python data structure to a database record or a JSON object. "
            "In data science, you will use dictionaries constantly to represent individual data records, "
            "store configuration parameters, build lookup tables, and accumulate group-wise statistics. "
            "Their O(1) average lookup time makes them extremely efficient for mapping operations."
        ),
        code(
            "# Creating dictionaries\n"
            "empty = {}\n"
            "person = {\n"
            '    "name": "Alice",\n'
            "    'age': 28,\n"
            "    'city': 'New York',\n"
            "    'skills': ['Python', 'SQL', 'Statistics']\n"
            "}\n"
            'print("Person:", person)\n'
            'print("Name:", person["name"])\n\n'
            "# Keys must be hashable — tuples work, lists do not\n"
            "locations = {\n"
            '    (40.71, -74.01): "NYC",\n'
            '    (34.05, -118.24): "LA"\n'
            "}\n"
            'print("Location:", locations[(40.71, -74.01)])'
        ),
        code(
            "# Accessing and modifying dictionaries\n"
            'student = {"id": 101, "name": "Bob", "grade": 85}\n'
            'print("Original:", student)\n\n'
            "# Access with square brackets\n"
            'print("Grade:", student["grade"])\n\n'
            "# Modify existing key\n"
            'student["grade"] = 90\n'
            'print("After grade update:", student)\n\n'
            "# Add new key-value pair\n"
            'student["passed"] = True\n'
            'print("After adding passed:", student)\n\n'
            "# KeyError if key does not exist\n"
            '# print(student["gpa"])  # KeyError'
        ),
        md(
            "## Dictionary Methods: .keys(), .values(), .items(), .get(), .setdefault()\n\n"
            "Python dictionaries provide methods that make common operations convenient: `.keys()` returns "
            "a view of all keys, `.values()` returns a view of all values, and `.items()` returns a view "
            "of all key-value pairs as tuples. These views are dynamic — they reflect changes to the "
            "dictionary in real time.\n\n"
            "The `.get()` method is a safer alternative to bracket access: it returns `None` (or a default "
            "you provide) if the key is missing, instead of raising a `KeyError`. The `.setdefault()` method "
            "goes a step further: it returns the value if the key exists, and if not, inserts the key with "
            "a default value and returns that. Both are invaluable for building counting and aggregation "
            "logic."
        ),
        code(
            "# .keys(), .values(), .items()\n"
            'inventory = {"apples": 10, "bananas": 5, "cherries": 20}\n'
            'print("Keys:", list(inventory.keys()))\n'
            'print("Values:", list(inventory.values()))\n'
            'print("Items:", list(inventory.items()))\n\n'
            "# Iterating over items\n"
            "for item, quantity in inventory.items():\n"
            '    print(f"{item}: {quantity}")'
        ),
        code(
            "# .get() for safe access\n"
            'scores = {"Alice": 95, "Bob": 87}\n'
            'print("Alice:", scores.get("Alice"))\n'
            'print("Charlie:", scores.get("Charlie"))\n'
            'print("Charlie with default:", scores.get("Charlie", 0))\n\n'
            "# .setdefault() — get or insert\n"
            'visits = {}\n'
            'users = ["Alice", "Bob", "Alice", "Charlie", "Bob", "Alice"]\n'
            "for user in users:\n"
            '    visits.setdefault(user, 0)\n'
            "    visits[user] += 1\n"
            'print("Visit counts:", visits)'
        ),
        md(
            "## Iterating Over Dictionaries\n\n"
            "You can iterate over a dictionary directly in a `for` loop, which yields keys by default. "
            "For most cases, you will want to iterate over `.items()` to get both the key and value in "
            "each iteration. You can also iterate over `.keys()` or `.values()` when you only need one side.\n\n"
            "Dictionary iteration is fundamental to data aggregation tasks. For example, you might iterate "
            "over a list of raw records and build a dictionary that groups values by a category key. This "
            'pattern \u2014 "iterate and accumulate into a dict" \u2014 appears in virtually every data science '
            "workflow."
        ),
        code(
            "# Different ways to iterate\n"
            'grades = {"Alice": [85, 90, 92], "Bob": [78, 81, 85], "Charlie": [92, 95, 98]}\n\n'
            "# Iterate over keys (default)\n"
            'print("Students:")\n'
            "for student in grades:\n"
            "    print(f\"  - {student}\")\n\n"
            "# Iterate over items\n"
            'print("\\nAverages:")\n'
            "for student, score_list in grades.items():\n"
            "    avg = sum(score_list) / len(score_list)\n"
            "    print(f\"  {student}: {avg:.1f}\")"
        ),
        code(
            "# Building an aggregation dictionary\n"
            'sales_data = [\n'
            '    ("NY", 100), ("CA", 200), ("NY", 150),\n'
            '    ("TX", 300), ("CA", 250), ("NY", 50)\n'
            "]\n\n"
            "# Aggregate sales by state\n"
            "state_sales = {}\n"
            "for state, amount in sales_data:\n"
            "    if state in state_sales:\n"
            "        state_sales[state] += amount\n"
            "    else:\n"
            "        state_sales[state] = amount\n"
            'print("Sales by state:", state_sales)\n\n'
            "# Same thing using .get()\n"
            "state_sales2 = {}\n"
            "for state, amount in sales_data:\n"
            "    state_sales2[state] = state_sales2.get(state, 0) + amount\n"
            'print("Sales by state (v2):", state_sales2)'
        ),
        md(
            "## Dictionary Comprehensions\n\n"
            "Just like list comprehensions, Python supports dictionary comprehensions with the syntax "
            "`{key_expr: value_expr for item in iterable}`. They provide a concise way to build dictionaries "
            "from existing iterables. You can also add filtering conditions with `if` clauses.\n\n"
            "Dictionary comprehensions are widely used in data science for: transforming one mapping into "
            "another (e.g., converting column names to lowercase), creating lookup tables from lists, "
            "and filtering dictionaries by key or value conditions."
        ),
        code(
            "# Basic dictionary comprehension\n"
            "squares = {x: x**2 for x in range(1, 6)}\n"
            'print("Squares:", squares)\n\n'
            "# From two lists (zipping)\n"
            'cities = ["NYC", "LA", "Chicago"]\n'
            "populations = [8336817, 3898747, 2746388]\n"
            "city_pop = {city: pop for city, pop in zip(cities, populations)}\n"
            'print("City populations:", city_pop)\n\n'
            "# Filtering with a condition\n"
            "even_squares = {x: x**2 for x in range(10) if x % 2 == 0}\n"
            'print("Even squares:", even_squares)'
        ),
        code(
            "# Dictionary comprehension in data science context\n"
            "# Transform column names to lowercase\n"
            'raw_columns = ["Name", "Age ", " Salary", "City"]\n'
            "clean_columns = {col: col.strip().lower() for col in raw_columns}\n"
            'print("Cleaned mapping:", clean_columns)\n\n'
            "# Invert a dictionary (swap keys and values)\n"
            'original = {"a": 1, "b": 2, "c": 3}\n'
            "inverted = {v: k for k, v in original.items()}\n"
            'print("Inverted:", inverted)\n\n'
            "# Filter by value\n"
            'scores = {"Alice": 95, "Bob": 72, "Charlie": 88, "Diana": 60}\n'
            "passing = {name: score for name, score in scores.items() if score >= 70}\n"
            'print("Passing students:", passing)'
        ),
        md(
            "## defaultdict and Counter from collections\n\n"
            "The `collections` module provides specialized dictionary classes. `defaultdict` is a dictionary "
            "that supplies a default value for missing keys automatically, eliminating the need for explicit "
            "checks. You provide a factory function (like `int`, `list`, `float`) when creating it.\n\n"
            "`Counter` is a dictionary subclass designed for counting hashable objects. It takes an iterable "
            "and automatically counts occurrences. It also provides helpful methods like `.most_common()` "
            "to get the most frequent items. Both `defaultdict` and `Counter` are indispensable tools in "
            "the data scientist's toolkit."
        ),
        code(
            "# defaultdict for clean aggregation\n"
            "from collections import defaultdict, Counter\n\n"
            "# Without defaultdict (verbose)\n"
            "word_counts = {}\n"
            'text = "the cat in the hat the cat sat".split()\n'
            "for word in text:\n"
            "    if word not in word_counts:\n"
            "        word_counts[word] = 0\n"
            "    word_counts[word] += 1\n"
            'print("Manual count:", word_counts)\n\n'
            "# With defaultdict (clean)\n"
            "word_counts2 = defaultdict(int)\n"
            "for word in text:\n"
            "    word_counts2[word] += 1\n"
            'print("DefaultDict count:", dict(word_counts2))'
        ),
        code(
            "# defaultdict with list factory\n"
            "from collections import defaultdict\n\n"
            "# Group items by category\n"
            'products = [\n'
            '    ("fruit", "apple"), ("fruit", "banana"),\n'
            '    ("veg", "carrot"), ("fruit", "cherry"),\n'
            '    ("veg", "broccoli"), ("dairy", "milk")\n'
            "]\n\n"
            "by_category = defaultdict(list)\n"
            "for category, product in products:\n"
            "    by_category[category].append(product)\n"
            'print("Grouped products:", dict(by_category))\n\n'
            "# Access a missing key returns empty list\n"
            'print("Meat category:", by_category["meat"])'
        ),
        code(
            "# Counter for frequency analysis\n"
            "from collections import Counter\n\n"
            '# Survey responses\n'
            'responses = [\n'
            '    "Python", "R", "Python", "Python", "Julia",\n'
            '    "R", "Python", "JavaScript", "Julia", "Python", "R"\n'
            "]\n\n"
            "lang_counts = Counter(responses)\n"
            'print("Language counts:", lang_counts)\n\n'
            "# Most common\n"
            'print("Most common:", lang_counts.most_common(2))\n\n'
            "# Counter operations\n"
            'more_responses = ["Python", "Rust", "Rust", "Python"]\n'
            "lang_counts.update(more_responses)\n"
            'print("After update:", lang_counts)\n\n'
            "# Count characters in a string\n"
            'text = "hello world"\n'
            "char_count = Counter(text)\n"
            'print("Character frequencies:", char_count)'
        ),
        data_science_connection(
            "Dictionaries are the backbone of data representation in Python. When you load a JSON API "
            "response, it becomes a nested dict. When you use pandas, each row can be represented as a "
            "dict (and `df.to_dict()` gives you exactly that). `Counter` is essentially a one-line "
            "frequency table — the same operation that would take multiple lines in SQL. Mastering "
            "dictionaries is a critical step toward understanding how data is stored, accessed, and "
            "manipulated in Python."
        ),
    ]
    return create_notebook(6, title, cells)


# ──────────────────────────────────────────────
# Lecture 7
# ──────────────────────────────────────────────
def lecture_7():
    title = "Control Flow: Conditionals and Loops"
    cells = [
        md(f"# Lecture 7: {title}"),
        learning_objectives([
            "Evaluate Boolean expressions and understand truthiness",
            "Write if/elif/else chains for decision-making",
            "Use for loops over lists, strings, dicts, and ranges",
            "Generate sequences with the range() function",
            "Use while loops and prevent infinite loops",
            "Control loop flow with break, continue, and the loop else clause"
        ]),
        key_topics([
            "Boolean expressions and truthiness",
            "if/elif/else chains",
            "for loops over lists, strings, dicts, and ranges",
            "The range() function",
            "while loops and infinite loop prevention",
            "break, continue, and the loop else clause"
        ]),
        md(
            "## Boolean Expressions and Truthiness\n\n"
            "Every value in Python has an inherent truth value, known as **truthiness**. The following are "
            "considered `False`: `None`, `False`, zero numeric values (`0`, `0.0`, `0j`), empty sequences "
            "(`''`, `[]`, `()`), and empty mappings (`{}`). Everything else is `True`. This means you can "
            "use any value directly in a conditional: `if my_list:` is equivalent to `if len(my_list) > 0:`.\n\n"
            "Boolean expressions can be combined with `and`, `or`, and `not`. Python uses **short-circuit "
            "evaluation**: in `a and b`, if `a` is `False`, `b` is never evaluated. In `a or b`, if `a` is "
            "`True`, `b` is never evaluated. This is both an optimization and a useful programming pattern."
        ),
        code(
            "# Truthiness of different values\n"
            "values = [None, False, 0, 0.0, '', [], {}, 'hello', [1, 2], {'a': 1}, 42]\n"
            "for val in values:\n"
            "    print(f\"bool({val!r:8}) = {bool(val)}\")\n\n"
            "# Using truthiness in conditionals\n"
            'name = ""\n'
            "if name:\n"
            '    print(f"Hello, {name}")\n'
            "else:\n"
            '    print("Name is empty!")'
        ),
        code(
            "# Short-circuit evaluation\n"
            "# In 'and', if first is False, second is not evaluated\n"
            'def risky_operation():\n'
            '    print("Risky operation executed!")\n'
            '    return True\n\n'
            "result = False and risky_operation()\n"
            'print("Result of False and risky():", result)\n\n'
            "# Practical: safe attribute access\n"
            'data = {"name": "Alice"}\n'
            "if 'age' in data and data['age'] > 18:\n"
            '    print("Adult")\n'
            "else:\n"
            '    print("Age unknown or under 18")'
        ),
        md(
            "## if/elif/else Chains\n\n"
            "Conditional execution is the backbone of decision-making in code. The `if` statement executes "
            "a block if its condition is `True`. You can chain additional conditions with `elif` (short for "
            "'else if'), and provide a fallback with `else`. Only one branch in the chain executes — the "
            "first one whose condition is `True`.\n\n"
            "In data science, conditionals are used everywhere: filtering data, assigning categories, "
            "handling missing values, and implementing business logic. For example, you might classify "
            "a customer as 'High', 'Medium', or 'Low' value based on their spending, or flag outliers "
            "based on z-score thresholds."
        ),
        code(
            "# Basic if/elif/else\n"
            "score = 85\n\n"
            "if score >= 90:\n"
            "    grade = 'A'\n"
            "elif score >= 80:\n"
            "    grade = 'B'\n"
            "elif score >= 70:\n"
            "    grade = 'C'\n"
            "elif score >= 60:\n"
            "    grade = 'D'\n"
            "else:\n"
            "    grade = 'F'\n"
            'print(f"Score: {score}, Grade: {grade}")'
        ),
        code(
            "# Conditional logic for data filtering\n"
            'transactions = [\n'
            '    {"amount": 250, "type": "purchase"},\n'
            '    {"amount": 5000, "type": "purchase"},\n'
            '    {"amount": 150, "type": "refund"},\n'
            '    {"amount": 12000, "type": "purchase"},\n'
            '    {"amount": 75, "type": "purchase"}\n'
            "]\n\n"
            "for txn in transactions:\n"
            "    if txn['amount'] > 10000:\n"
            "        txn['flag'] = 'large_txn'\n"
            "    elif txn['type'] == 'refund':\n"
            "        txn['flag'] = 'refund'\n"
            "    elif txn['amount'] < 100:\n"
            "        txn['flag'] = 'small_txn'\n"
            "    else:\n"
            "        txn['flag'] = 'normal'\n\n"
            "for txn in transactions:\n"
            "    print(txn)"
        ),
        md(
            "## for Loops Over Lists, Strings, Dicts, and Ranges\n\n"
            "The `for` loop is the primary tool for iteration in Python. It works with any **iterable** — "
            "lists, strings, dictionaries, sets, tuples, files, and more. The loop variable takes each "
            "element in turn, and the loop body executes once per element.\n\n"
            "Iterating over different data structures follows slightly different patterns: over a list, you "
            "get each element; over a string, you get each character; over a dictionary, by default you get "
            "each key (use `.items()` to get key-value pairs). The `range()` function generates sequences "
            "of numbers and is commonly used when you need to iterate a specific number of times or when you "
            "need index-based access."
        ),
        code(
            "# for loops over different data structures\n"
            "# Over a list\n"
            'fruits = ["apple", "banana", "cherry"]\n'
            'print("Fruits:")\n'
            "for fruit in fruits:\n"
            "    print(f\"  - {fruit}\")\n\n"
            "# Over a string (character by character)\n"
            'print("\\nCharacters:")\n'
            "for char in \"Python\":\n"
            "    print(f\"  '{char}'\")\n\n"
            "# Over a dictionary\n"
            'ages = {"Alice": 28, "Bob": 35, "Charlie": 22}\n'
            'print("\\nAges:")\n'
            "for name, age in ages.items():\n"
            "    print(f\"  {name} is {age}\")"
        ),
        code(
            "# for loop with range()\n"
            'print("Range 0 to 4:")\n'
            "for i in range(5):\n"
            "    print(f\"  {i}\")\n\n"
            'print("\\nRange 2 to 7:")\n'
            "for i in range(2, 8):\n"
            "    print(f\"  {i}\")\n\n"
            'print("\\nRange 10 to 1 step -2:")\n'
            "for i in range(10, 0, -2):\n"
            "    print(f\"  {i}\")\n\n"
            "# Practical: enumerate with index\n"
            'colors = ["red", "green", "blue"]\n'
            'print("\\nEnumerated colors:")\n'
            "for idx, color in enumerate(colors):\n"
            "    print(f\"  {idx}: {color}\")"
        ),
        md(
            "## The range() Function\n\n"
            "`range(start, stop, step)` generates an immutable sequence of numbers. It is lazy — it does not "
            "create the entire list in memory, making it memory-efficient even for very large ranges. "
            "`range(stop)` starts at 0 and goes up to (but not including) `stop`. `range(start, stop)` starts "
            "at `start`. `range(start, stop, step)` increments by `step` (which can be negative).\n\n"
            "In data science, `range()` is commonly used for: creating index sequences for iteration, "
            "generating bins for histograms, creating evenly spaced values for plotting, and batch processing "
            "large datasets in chunks."
        ),
        code(
            "# range() variations\n"
            "# Basic range from 0 to stop-1\n"
            'print("list(range(5)):", list(range(5)))\n\n'
            "# Range with start and stop\n"
            'print("list(range(2, 8)):", list(range(2, 8)))\n\n'
            "# Range with step\n"
            'print("list(range(0, 20, 3)):", list(range(0, 20, 3)))\n\n'
            "# Negative step (reverse)\n"
            'print("list(range(10, 0, -2)):", list(range(10, 0, -2)))'
        ),
        code(
            "# Practical use of range() in data science\n"
            "# Simulating batch processing\n"
            "total_rows = 100\n"
            "batch_size = 30\n\n"
            "for batch_start in range(0, total_rows, batch_size):\n"
            "    batch_end = min(batch_start + batch_size, total_rows)\n"
            "    print(f\"Processing rows {batch_start} to {batch_end}\")\n\n"
            "# Creating evenly spaced points for plotting\n"
            "num_points = 10\n"
            "spacing = 0.5\n"
            "points = [i * spacing for i in range(num_points)]\n"
            'print("Data points:", points)'
        ),
        md(
            "## while Loops and Infinite Loop Prevention\n\n"
            "A `while` loop repeats as long as its condition is `True`. It is used when you do not know "
            "in advance how many iterations you need — for example, reading from a file until EOF, or "
            "waiting for a condition to change. Because the loop checks the condition only at the start "
            "of each iteration, it is possible to write an **infinite loop** if the condition never becomes "
            "`False`.\n\n"
            "To prevent infinite loops, always ensure that something changes inside the loop body that "
            "will eventually make the condition `False`. Common strategies include: incrementing a counter, "
            "removing elements from a collection, or reading input until a sentinel value is received. A "
            "safety break after a maximum number of iterations is also a good practice."
        ),
        code(
            "# Basic while loop with counter\n"
            "count = 0\n"
            "while count < 5:\n"
            "    print(f\"Iteration {count}\")\n"
            "    count += 1\n"
            'print("Loop finished!")'
        ),
        code(
            "# While loop with sentinel value\n"
            '# Simulating reading data until a termination signal\n'
            'data_stream = [10, 20, 30, -1, 40, 50]\n'
            "idx = 0\n"
            "results = []\n\n"
            "while idx < len(data_stream) and data_stream[idx] != -1:\n"
            "    results.append(data_stream[idx] * 2)\n"
            "    idx += 1\n"
            'print("Processed values:", results)\n'
            'print(f"Stopped at index {idx} (value was {data_stream[idx]})")\n\n'
            "# Safety max-iterations guard\n"
            "max_iter = 1000\n"
            "iteration = 0\n"
            "while iteration < max_iter:\n"
            "    # Simulated processing\n"
            "    if iteration >= 3:\n"
            "        break  # normal exit\n"
            "    iteration += 1"
        ),
        md(
            "## break, continue, and the Loop else Clause\n\n"
            "`break` immediately exits the loop, skipping any remaining iterations. It is commonly used "
            "to stop searching once a target is found. `continue` skips the rest of the current iteration "
            "and moves to the next one. It is useful for filtering — you can skip invalid items and process "
            "only valid ones.\n\n"
            "The `else` clause on a loop is a Python feature that is often misunderstood. The `else` block "
            "executes **only if the loop completed normally** (i.e., without hitting a `break`). It does "
            "**not** execute if the loop was exited via `break`. This is useful for search scenarios: if "
            "you loop through items looking for something and `break` when found, the `else` block runs if "
            "the item was never found."
        ),
        code(
            "# break: exit loop early\n"
            'numbers = [3, 7, 2, 9, 4, 6]\n'
            "target = 9\n"
            "for num in numbers:\n"
            "    if num == target:\n"
            "        print(f\"Found {target} at index {numbers.index(target)}\")\n"
            "        break\n"
            "else:\n"
            "    print(f\"{target} not found\")\n\n"
            "# Without break, else would execute\n"
            "for num in [1, 2, 3]:\n"
            "    print(f\"Checking {num}\")\n"
            "else:\n"
            '    print("Loop completed without break")'
        ),
        code(
            "# continue: skip invalid data\n"
            'raw_data = [10, -5, 20, None, 30, -1, 40, 0]\n'
            "valid = []\n"
            "for value in raw_data:\n"
            "    if value is None or value < 0:\n"
            "        continue  # skip invalid\n"
            "    valid.append(value * 2)\n"
            'print("Original:", raw_data)\n'
            'print("Processed valid:", valid)\n\n'
            "# break + else for searching\n"
            'def find_value(data, target):\n'
            "    for item in data:\n"
            "        if item == target:\n"
            '            print(f"Found {target}")\n'
            "            break\n"
            "    else:\n"
            '        print(f"{target} not found")\n\n'
            "find_value([1, 2, 3, 4, 5], 3)\n"
            "find_value([1, 2, 3, 4, 5], 99)"
        ),
        data_science_connection(
            "Control flow is what makes your data processing dynamic and adaptable. Every data cleaning "
            "script uses conditionals to handle missing values and loops to process each row. The "
            "`break`/`continue`/`else` patterns you learned here appear in real-world code for tasks "
            "like early stopping during model training (break when validation loss stops improving), "
            "skipping corrupted files during data ingestion (continue), and validating that all "
            "required columns exist before processing (loop-else)."
        ),
    ]
    return create_notebook(7, title, cells)


# ──────────────────────────────────────────────
# Lecture 8
# ──────────────────────────────────────────────
def lecture_8():
    title = "Mini-Project: Building a Data Summary Tool"
    cells = [
        md(f"# Lecture 8: {title}"),
        learning_objectives([
            "Combine all Phase 1 concepts into a single working program",
            "Use a list of dicts as a mini DataFrame",
            "Compute mean, min, max, and count grouped by category",
            "Filter records using conditionals",
            "Format a clean text report for output"
        ]),
        key_topics([
            "Putting it all together: lists of dicts as a mini DataFrame",
            "Computing mean, min, max, and count by category",
            "Filtering records with conditionals",
            "Formatting a clean text report"
        ]),
        md(
            "## Putting It All Together: Lists of Dicts as a Mini DataFrame\n\n"
            "Throughout Phase 1, you have learned about variables, data types, strings, lists, tuples, "
            "sets, dictionaries, conditionals, and loops. Now it is time to combine them all into a "
            "single, practical program. In this mini-project, you will build a **Data Summary Tool** that "
            "analyzes a small dataset stored as a list of dictionaries — exactly the same structure that "
            "pandas uses internally to represent a DataFrame.\n\n"
            "Each dictionary in the list represents one record (like a row in a spreadsheet), with keys "
            "as column names and values as the data. This structure is intuitive, flexible, and widely "
            "used in real-world Python for representing structured data before loading it into specialized "
            "libraries. By the end of this lecture, you will have written a program that computes "
            "summary statistics, filters data, and generates a formatted report."
        ),
        md(
            "## Building the Dataset\n\n"
            "We will create a dataset of employee records. Each record will be a dictionary with fields "
            "like `name`, `department`, `salary`, and `years_of_experience`. Storing data this way makes "
            "it easy to understand, modify, and extend. You can add more fields or more records without "
            "changing the code that processes them.\n\n"
            "This dataset will serve as the input to our summary tool. In real data science work, this "
            "data would come from a CSV file or a database query, but the principles of analysis are "
            "exactly the same."
        ),
        code(
            "# Step 1: Build the dataset (list of dicts)\n"
            "employees = [\n"
            '    {"name": "Alice", "dept": "Engineering", "salary": 95000, "years": 5},\n'
            '    {"name": "Bob", "dept": "Marketing", "salary": 72000, "years": 3},\n'
            '    {"name": "Charlie", "dept": "Engineering", "salary": 110000, "years": 8},\n'
            '    {"name": "Diana", "dept": "Sales", "salary": 85000, "years": 4},\n'
            '    {"name": "Eve", "dept": "Marketing", "salary": 68000, "years": 2},\n'
            '    {"name": "Frank", "dept": "Engineering", "salary": 125000, "years": 12},\n'
            '    {"name": "Grace", "dept": "Sales", "salary": 92000, "years": 6},\n'
            '    {"name": "Hank", "dept": "Marketing", "salary": 77000, "years": 3},\n'
            "]\n\n"
            'print(f"Dataset loaded: {len(employees)} records")\n'
            'print(f"Departments: {sorted(set(e[\'dept\'] for e in employees))}")\n'
            'print("\\nFirst 3 records:")\n'
            "for emp in employees[:3]:\n"
            "    print(f\"  {emp}\")"
        ),
        md(
            "## Computing Mean, Min, Max, and Count by Category\n\n"
            "One of the most common data science tasks is computing summary statistics **grouped by "
            "category**. For example, you might want the average salary per department, or the minimum "
            "years of experience in each department. We will use a dictionary where each key is a "
            "department name and each value is a list of the relevant values, then compute statistics "
            "on each list.\n\n"
            "This pattern — group, aggregate, report — is the foundation of tools like SQL's `GROUP BY` "
            "and pandas' `.groupby()`. Implementing it manually from scratch will give you a deep "
            "understanding of what those tools do under the hood."
        ),
        code(
            "# Step 2: Compute summary statistics by department\n"
            "from collections import defaultdict\n\n"
            "# Group salaries by department\n"
            "dept_salaries = defaultdict(list)\n"
            "for emp in employees:\n"
            "    dept_salaries[emp['dept']].append(emp['salary'])\n\n"
            'print("Department Salary Summary")\n'
            'print("=" * 50)\n\n'
            "for dept, salaries in sorted(dept_salaries.items()):\n"
            "    count = len(salaries)\n"
            "    total = sum(salaries)\n"
            "    mean = total / count\n"
            "    min_sal = min(salaries)\n"
            "    max_sal = max(salaries)\n"
            "    print(f\"{dept:15s}  Count: {count}  Mean: ${mean:,.0f}  \"\n"
            "          f\"Min: ${min_sal:,}  Max: ${max_sal:,}\")"
        ),
        code(
            "# Step 3: General-purpose group stats function\n"
            "def group_stats(data, group_key, value_key):\n"
            "    \"\"\"Group data by group_key and compute stats on value_key.\"\"\"\n"
            "    grouped = defaultdict(list)\n"
            "    for record in data:\n"
            "        grouped[record[group_key]].append(record[value_key])\n\n"
            "    results = {}\n"
            "    for group, values in grouped.items():\n"
            "        results[group] = {\n"
            "            'count': len(values),\n"
            "            'mean': sum(values) / len(values),\n"
            "            'min': min(values),\n"
            "            'max': max(values),\n"
            "            'total': sum(values),\n"
            "        }\n"
            "    return results\n\n\n"
            "# Use the function\n"
            "stats = group_stats(employees, 'dept', 'salary')\n"
            'print("Engineering stats:", stats["Engineering"])\n'
            'print("Marketing stats:", stats["Marketing"])'
        ),
        md(
            "## Filtering Records with Conditionals\n\n"
            "Another fundamental data operation is **filtering** — selecting only the records that meet "
            "certain criteria. You have already learned about `if` statements and comparison operators. "
            "Now you will apply them to filter our employee dataset. For example, you might want only "
            "employees with a salary above a threshold, or only those in a specific department.\n\n"
            "Filtering is done with a list comprehension or a for loop with an `if` condition. This is "
            "exactly what pandas does internally when you write `df[df['salary'] > 90000]`. Building "
            "the filter manually reinforces how the operation works at a fundamental level."
        ),
        code(
            "# Step 4: Filter records with conditionals\n"
            "# Filter: employees with salary > 90000\n"
            'high_earners = [emp for emp in employees if emp["salary"] > 90000]\n'
            'print(f"High earners (salary > $90k): {len(high_earners)} employees")\n'
            "for emp in high_earners:\n"
            "    print(f\"  {emp['name']:10s}  {emp['dept']:15s}  ${emp['salary']:,}\")\n\n"
            "# Filter: employees with 5+ years experience\n"
            'senior = [emp for emp in employees if emp["years"] >= 5]\n'
            'print(f"\\nSenior employees (5+ years): {len(senior)}")\n'
            "for emp in senior:\n"
            "    print(f\"  {emp['name']:10s}  {emp['dept']:15s}  {emp['years']} years\")"
        ),
        code(
            "# Step 5: Multi-condition filtering\n"
            "# Filter: Engineering dept with salary > 100k\n"
            "eng_high = [\n"
            "    emp for emp in employees\n"
            "    if emp['dept'] == 'Engineering' and emp['salary'] > 100000\n"
            "]\n"
            'print("Engineering high earners:")\n'
            "for emp in eng_high:\n"
            "    print(f\"  {emp['name']} - ${emp['salary']:,}\")\n\n"
            "# Filter: Marketing or Sales with < 5 years experience\n"
            "junior_business = [\n"
            "    emp for emp in employees\n"
            "    if emp['dept'] in ('Marketing', 'Sales') and emp['years'] < 5\n"
            "]\n"
            'print("\\nJunior Marketing/Sales employees:")\n'
            "for emp in junior_business:\n"
            "    print(f\"  {emp['name']} - {emp['dept']} - {emp['years']} years\")"
        ),
        md(
            "## Formatting a Clean Text Report\n\n"
            "The final step is to present our findings in a well-formatted report. A good data science "
            "report is readable, structured, and contains all the key information without unnecessary "
            "detail. We will use f-strings with alignment specifiers, separators, and clear section "
            "headings to build a professional-looking output.\n\n"
            "This skill — presenting results clearly — is just as important as the analysis itself. "
            "In practice, you would write this report to a file or include it in an email. The formatting "
            "techniques you use here (alignment, separators, grouping) apply equally to generating "
            "PDF reports, HTML tables, or even dashboard labels."
        ),
        code(
            "# Step 6: Generate a complete formatted report\n"
            "def generate_report(data):\n"
            "    lines = []\n"
            '    lines.append("=" * 60)\n'
            '    lines.append("EMPLOYEE DATA SUMMARY REPORT")\n'
            '    lines.append("=" * 60)\n\n'
            "    # Section 1: Overview\n"
            '    lines.append(f"\\nTotal employees: {len(data)}")\n'
            '    departments = sorted(set(e["dept"] for e in data))\n'
            '    lines.append(f"Departments: {", ".join(departments)}")\n'
            "    total_salary = sum(e['salary'] for e in data)\n"
            "    lines.append(f'Total salary budget: ${total_salary:,}')\n"
            "    avg_salary = total_salary / len(data)\n"
            "    lines.append(f'Overall average salary: ${avg_salary:,.0f}')\n\n"
            "    # Section 2: By-department breakdown\n"
            '    lines.append("-" * 60)\n'
            '    lines.append("BREAKDOWN BY DEPARTMENT")\n'
            '    lines.append("-" * 60)\n\n'
            '    stats = group_stats(data, "dept", "salary")\n'
            "    for dept in sorted(stats.keys()):\n"
            "        s = stats[dept]\n"
            "        lines.append(\n"
            '            f"{dept:15s} | Count: {s[\'count\']}  "\n'
            '            f"Avg: ${s[\'mean\']:>8,.0f}  "\n'
            '            f"Min: ${s[\'min\']:>6,}  Max: ${s[\'max\']:>6,}"\n'
            "        )\n\n"
            "    # Section 3: High earners\n"
            '    lines.append("-" * 60)\n'
            '    lines.append("HIGH EARNERS (Salary > $90,000)")\n'
            '    lines.append("-" * 60)\n'
            '    high = [e for e in data if e["salary"] > 90000]\n'
            "    for e in sorted(high, key=lambda x: x['salary'], reverse=True):\n"
            "        lines.append(\n"
            '            f"  {e[\'name\']:12s}  {e[\'dept\']:15s}  ${e[\'salary\']:,}"\n'
            "        )\n\n"
            '    lines.append("=" * 60)\n'
            '    lines.append("END OF REPORT")\n'
            '    lines.append("=" * 60)\n'
            "    return '\\n'.join(lines)\n\n\n"
            "# Generate and print the report\n"
            "report = generate_report(employees)\n"
            "print(report)"
        ),
        code(
            "# Step 7: Save report to a file\n"
            "report = generate_report(employees)\n"
            "with open('employee_summary_report.txt', 'w') as f:\n"
            "    f.write(report)\n"
            'print("Report saved to employee_summary_report.txt")\n\n'
            "# Bonus: Filter by years of experience and compute stats\n"
            "def filter_and_report(data, min_years):\n"
            "    filtered = [e for e in data if e['years'] >= min_years]\n"
            "    print(f'\\nEmployees with {min_years}+ years: {len(filtered)}')\n"
            "    if filtered:\n"
            "        avg_sal = sum(e['salary'] for e in filtered) / len(filtered)\n"
            "        print(f'Average salary: ${avg_sal:,.0f}')\n"
            "        depts = {e['dept'] for e in filtered}\n"
            "        print(f'Departments represented: {depts}')\n\n\n"
            "filter_and_report(employees, 5)\n"
            "filter_and_report(employees, 10)"
        ),
        data_science_connection(
            "This mini-project is a microcosm of what professional data scientists do every day: load "
            "data, clean it, compute summaries, filter based on conditions, and present results. The "
            "`list of dicts` structure you used here is exactly how pandas represents data internally "
            "(as a collection of Series objects backed by NumPy arrays). The grouping and aggregation "
            "logic you built manually is what `df.groupby('dept')['salary'].agg(['mean', 'min', 'max'])` "
            "does in a single line. By understanding the fundamentals, you will appreciate and use "
            "higher-level tools more effectively. Congratulations — you have completed Phase 1 of your "
            "Python for Data Science journey!"
        ),
    ]
    return create_notebook(8, title, cells)


# ──────────────────────────────────────────────
# Main: generate all notebooks
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating Phase 1 notebooks...\n")
    lecture_1()
    lecture_2()
    lecture_3()
    lecture_4()
    lecture_5()
    lecture_6()
    lecture_7()
    lecture_8()
    print("\nAll 8 notebooks created successfully!")
