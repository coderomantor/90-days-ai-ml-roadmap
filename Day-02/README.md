# Day 02: NumPy 2D Arrays and Pandas Introduction

## Goal

Understand how tabular data is represented in Python using NumPy 2D arrays and Pandas DataFrames.

Day 01 focused on NumPy basics. Day 02 connects those array skills to real dataset-style thinking, which is essential before machine learning model training.

## Topics Covered

- NumPy 2D arrays
- Rows, columns, and shape
- `axis=0` and `axis=1`
- Pandas DataFrames
- Selecting one or more columns
- Creating calculated columns
- Basic data exploration
- Simple feature engineering

## Key Concepts

### NumPy Array vs Pandas DataFrame

A NumPy array is mainly used for fast numerical calculations. A Pandas DataFrame is used for labeled table-like data with rows and columns.

In AI/ML:

- Rows usually represent examples or records.
- Columns usually represent features.
- Clean tabular data is often prepared before training a model.

### Axis Meaning

- `axis=0` means calculate down the rows for each column.
- `axis=1` means calculate across the columns for each row.

Example:

```python
df[["Math", "Python", "Statistics"]].sum(axis=1)
```

This calculates each student's total marks across the subject columns.

### Feature Engineering

Feature engineering means creating useful new information from existing raw data.

In this project:

- `Total` is created from subject marks.
- `Average` is created from subject marks.
- `Status` is created from the average score.

## Mini Project: Student Performance DataFrame

The project creates a small student dataset and calculates:

- Total marks
- Average marks
- Pass or needs improvement status
- Subject-wise class averages
- Top student by average marks
- Basic DataFrame exploration output

## Project Structure

```text
Day-02/
├── README.md
├── resources.md
├── project/
│   ├── student_performance_dataframe.py
│   └── requirements.txt
└── screenshots/
```

## How To Run

From the `Day-02/project/` folder:

```bash
pip install -r requirements.txt
python student_performance_dataframe.py
```

## Output Screenshot

Terminal output screenshot: [screenshots/day-02-terminal-output.png](screenshots/day-02-terminal-output.png)

## Important Pandas Syntax

Select one column:

```python
df["Average"]
```

Select multiple columns:

```python
df[["Student", "Average", "Status"]]
```

Sort by average marks:

```python
df.sort_values("Average", ascending=False)
```

Load a CSV file:

```python
pd.read_csv("students.csv")
```

## Knowledge Check Result

I reviewed my Day 2 understanding before moving forward.

Status: **Ready for Day 3**

Strong areas:

- Difference between NumPy arrays and Pandas DataFrames
- Meaning of `axis=0` and `axis=1`
- Creating `Total` and `Average` columns
- Using `np.where()` for vectorized status labels
- Understanding why clean data matters before ML training

Correction:

- To select multiple columns in Pandas, use double brackets:

```python
df[["Student", "Average", "Status"]]
```

## Interview Questions

1. What is the difference between a NumPy array and a Pandas DataFrame?
2. What does `axis=0` mean?
3. What does `axis=1` mean?
4. How do you create a new column in Pandas?
5. What is feature engineering?
6. Why should data be inspected before model training?
7. What does `np.where()` do?

## Reflection

Today I learned how raw numerical data becomes structured tabular data. NumPy is useful for numerical arrays, while Pandas makes data easier to label, inspect, and manipulate.

The hardest parts were understanding `axis` and `np.where()`, but I now understand that `axis=1` works across columns for each row, while `axis=0` works down rows for each column.

This prepares me for Day 3, where I will start learning more about Pandas DataFrames and data cleaning.
