"""Student Performance DataFrame using NumPy and Pandas.

Day 02 project for the 90 Days AI/ML Engineer Roadmap.
The goal is to practice 2D arrays, DataFrames, and simple data exploration.
"""

import numpy as np
import pandas as pd


PASSING_AVERAGE = 80
SUBJECT_COLUMNS = ["Math", "Python", "Statistics"]


def build_student_dataframe():
    """Create a small student marks DataFrame from a NumPy 2D array."""
    student_names = ["Ali", "Sara", "Ahmed", "Fatima", "Zain"]

    # Rows represent students. Columns represent subjects.
    marks = np.array(
        [
            [78, 85, 90],
            [88, 92, 80],
            [70, 75, 82],
            [95, 89, 91],
            [82, 80, 77],
        ]
    )

    df = pd.DataFrame(marks, columns=SUBJECT_COLUMNS)
    df.insert(0, "Student", student_names)

    return df


def add_performance_columns(df):
    """Add total, average, and status columns."""
    df = df.copy()

    df["Total"] = df[SUBJECT_COLUMNS].sum(axis=1)
    df["Average"] = df[SUBJECT_COLUMNS].mean(axis=1)
    df["Status"] = np.where(
        df["Average"] >= PASSING_AVERAGE,
        "Pass",
        "Needs Improvement",
    )

    return df


def print_data_exploration(df):
    """Print beginner-friendly DataFrame exploration output."""
    print("First Rows")
    print("-" * 60)
    print(df.head().to_string(index=False))

    print("\nDataset Shape")
    print("-" * 60)
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\nStatistical Summary")
    print("-" * 60)
    print(df[SUBJECT_COLUMNS + ["Total", "Average"]].describe().to_string())

    print("\nSelected Columns")
    print("-" * 60)
    print(df[["Student", "Average", "Status"]].to_string(index=False))


def print_class_summary(df):
    """Print simple class-level insights."""
    top_student = df.loc[df["Average"].idxmax()]
    sorted_students = df.sort_values("Average", ascending=False)

    print("\nSubject Averages")
    print("-" * 60)

    for subject in SUBJECT_COLUMNS:
        print(f"{subject}: {df[subject].mean():.2f}")

    print("\nTop Student")
    print("-" * 60)
    print(f"{top_student['Student']} with an average of {top_student['Average']:.2f}")

    print("\nStudents Sorted By Average")
    print("-" * 60)
    print(sorted_students[["Student", "Average", "Status"]].to_string(index=False))


def main():
    df = build_student_dataframe()
    df = add_performance_columns(df)

    print("Student Performance DataFrame")
    print("=" * 60)
    print(df.to_string(index=False))
    print()

    print_data_exploration(df)
    print_class_summary(df)


if __name__ == "__main__":
    main()
