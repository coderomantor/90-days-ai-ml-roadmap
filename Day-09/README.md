# Day 09: Feature Scaling, Normalization, and Standardization

## Goal

Learn how to scale numeric features so they are easier for machine learning models to compare.

Day 08 focused on feature engineering and encoding. Day 09 builds on that by preparing numeric columns through normalization and standardization, which are common preprocessing steps before model training.

## Topics Covered

- Feature scaling
- Normalization
- Standardization
- Numeric preprocessing
- Data preprocessing pipeline
- Saving preprocessed datasets

## Learning Objectives

- Create a small employee dataset using Pandas
- Identify numeric columns for preprocessing
- Apply min-max normalization
- Apply standardization using mean and standard deviation
- Compare original and scaled values
- Save a preprocessed dataset as a CSV file
- Understand why scaling matters before machine learning

## Concepts Learned

### Feature Scaling

Feature scaling means changing numeric columns so their values are easier to compare.

This matters because some machine learning models are affected by the size of numbers.

Example:

- `Salary` may be around `50000`
- `Experience` may be around `2` to `7`

Without scaling, the model may treat salary as more important just because the numbers are larger.

### Normalization

Normalization usually scales values to a range between `0` and `1`.

```python
normalized = (value - minimum) / (maximum - minimum)
```

Normalization is useful when values need to be placed inside a fixed range.

### Standardization

Standardization scales values based on the mean and standard deviation.

```python
standardized = (value - mean) / standard_deviation
```

Standardized values are centered around `0`, which helps many machine learning algorithms work more reliably.

### Preprocessing Pipeline

A preprocessing pipeline is a repeatable set of steps used to prepare data before machine learning.

In this project, the pipeline is:

1. Create the dataset
2. Select numeric features
3. Apply normalization
4. Apply standardization
5. Save the preprocessed dataset

## Mini Project: Data Preprocessing Pipeline

The project creates an employee dataset and applies feature scaling to numeric columns.

The project includes:

- Employee DataFrame creation
- Numeric feature selection
- Min-max normalization
- Standardization
- Scaled feature comparison
- Scaling summary
- CSV export of the preprocessed dataset

## Project Structure

```text
Day-09/
├── README.md
├── resources.md
├── project/
│   ├── data_preprocessing_pipeline.py
│   └── requirements.txt
└── screenshots/
```

## How To Run

From the `Day-09/project/` folder:

```bash
pip install -r requirements.txt
python data_preprocessing_pipeline.py
```

## Output

The project prints:

- Original employee dataset
- Preprocessed dataset preview
- Normalized feature ranges
- Standardized feature mean and standard deviation
- Final notes explaining why scaling matters

The project also saves:

- `preprocessed_employee_dataset.csv`

After running the project, save a terminal screenshot in the `screenshots/` folder.

## Skills Demonstrated

- Creating Pandas DataFrames
- Selecting numeric features
- Applying min-max normalization
- Applying standardization
- Creating reusable preprocessing functions
- Comparing original and scaled values
- Saving preprocessed data as a CSV file
- Writing readable beginner-friendly preprocessing code

## Real-World Applications

Feature scaling is used in many machine learning workflows.

Examples:

- Preparing data for regression models
- Scaling customer, sales, employee, or finance datasets
- Helping distance-based models compare features fairly
- Preparing numeric features before training
- Building repeatable preprocessing pipelines

## Key Takeaway

Scaling does not change the meaning of the data. It changes the numeric range so machine learning models can compare features more fairly.

## Interview Questions

1. What is feature scaling?
2. Why do we scale numeric features before machine learning?
3. What is normalization?
4. What is standardization?
5. What is the difference between normalization and standardization?
6. When is min-max scaling useful?
7. Why can large numeric values affect some machine learning models?
8. What is a preprocessing pipeline?

## Reflection

Today I learned that preparing numeric data is an important part of machine learning. Even if a dataset is clean and encoded, numeric columns may still have very different ranges.

The most important lesson from Day 09 is that preprocessing helps models learn from features more fairly instead of being influenced only by large number ranges.

## Next Step

Continue to Day 10 and learn how to split a dataset into training and testing data.
