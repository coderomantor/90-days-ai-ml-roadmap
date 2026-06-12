"""Data Preprocessing Pipeline using Pandas.

Day 09 project for the 90 Days AI/ML Engineer Roadmap.
The goal is to practice feature scaling, normalization, and standardization.
"""

from pathlib import Path

import pandas as pd


OUTPUT_PATH = Path(__file__).resolve().parent / "preprocessed_employee_dataset.csv"

NUMERIC_FEATURES = [
    "Age",
    "Salary",
    "Experience",
    "PerformanceScore",
    "TrainingHours",
]


def build_employee_dataframe():
    """Create a small employee dataset for preprocessing practice."""
    employee_data = {
        "Employee": [
            "Ali",
            "Sara",
            "Ahmed",
            "Ayesha",
            "Usman",
            "Fatima",
            "Zain",
            "Hira",
        ],
        "Department": [
            "IT",
            "HR",
            "Finance",
            "IT",
            "Marketing",
            "Finance",
            "IT",
            "HR",
        ],
        "Age": [24, 28, 29, 31, 27, 34, 30, 26],
        "Salary": [50000, 62000, 57000, 58000, 54000, 65000, 60000, 59000],
        "Experience": [2, 4, 5, 6, 3, 7, 4, 3],
        "PerformanceScore": [78, 86, 82, 88, 80, 91, 85, 83],
        "TrainingHours": [12, 18, 16, 20, 14, 24, 19, 15],
    }

    return pd.DataFrame(employee_data)


def min_max_normalize(series):
    """Scale a numeric column to a 0 to 1 range."""
    minimum_value = series.min()
    maximum_value = series.max()

    return (series - minimum_value) / (maximum_value - minimum_value)


def standardize(series):
    """Scale a numeric column using mean and standard deviation."""
    return (series - series.mean()) / series.std()


def add_scaled_features(df):
    """Create normalized and standardized versions of numeric features."""
    preprocessed_df = df.copy()

    for column in NUMERIC_FEATURES:
        preprocessed_df[f"{column}_Normalized"] = min_max_normalize(
            preprocessed_df[column]
        ).round(3)
        preprocessed_df[f"{column}_Standardized"] = standardize(
            preprocessed_df[column]
        ).round(3)

    return preprocessed_df


def print_section(title):
    """Print a consistent terminal section header."""
    print(f"\n{title}")
    print("-" * 70)


def print_scaling_summary(df):
    """Print a short summary of scaled feature ranges."""
    print_section("Scaling Summary")

    for column in NUMERIC_FEATURES:
        normalized_column = f"{column}_Normalized"
        standardized_column = f"{column}_Standardized"

        print(f"{column}:")
        print(
            "  Normalized range: "
            f"{df[normalized_column].min():.3f} to {df[normalized_column].max():.3f}"
        )
        print(
            "  Standardized mean/std: "
            f"{df[standardized_column].mean():.3f} / "
            f"{df[standardized_column].std():.3f}"
        )


def main():
    employee_df = build_employee_dataframe()

    print("Data Preprocessing Pipeline")
    print("=" * 70)

    print_section("Original Dataset")
    print(employee_df.to_string(index=False))

    preprocessed_df = add_scaled_features(employee_df)

    print_section("Preprocessed Dataset Preview")
    preview_columns = [
        "Employee",
        "Age",
        "Age_Normalized",
        "Age_Standardized",
        "Salary",
        "Salary_Normalized",
        "Salary_Standardized",
        "Experience",
        "Experience_Normalized",
        "Experience_Standardized",
    ]
    print(preprocessed_df[preview_columns].to_string(index=False))

    print_scaling_summary(preprocessed_df)

    preprocessed_df.to_csv(OUTPUT_PATH, index=False)

    print_section("Final Notes")
    print("1. Normalization converted numeric features to a 0 to 1 range.")
    print("2. Standardization centered numeric features around the mean.")
    print("3. Scaling helps models compare features with different units fairly.")
    print(f"4. Preprocessed dataset saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
