"""Print a beginner-friendly overview of the selected student dataset.

Run from the project folder with:
    python explore_data.py

This script only reads the CSV. It does not train a model or change the data.
"""

from pathlib import Path

import pandas as pd


# Keep the source CSV in data/raw so the original download stays untouched.
DATA_PATH = Path(__file__).parent / "data" / "raw" / "student_lifestyle_dataset.csv"

# These are the columns that describe hours in one day. We use them to check
# whether the dataset's time-budget pattern is consistent across its records.
DAILY_HOUR_COLUMNS = [
    "Study_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
]

TARGET_COLUMN = "Stress_Level"


def main() -> None:
    """Load the CSV and print basic data-quality and distribution summaries."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Place student_lifestyle_dataset.csv in data/raw/ first."
        )

    # Read the downloaded file without modifying it.
    data = pd.read_csv(DATA_PATH)

    print("DATASET OVERVIEW")
    print(f"File: {DATA_PATH.relative_to(Path(__file__).parent)}")
    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}")
    print("Column names:")
    for column in data.columns:
        print(f"  - {column}")

    print("\nMISSING VALUES")
    print(data.isna().sum().to_string())

    print("\nDUPLICATE ROWS")
    print(int(data.duplicated().sum()))

    print("\nNUMERIC COLUMN SUMMARY")
    print(data.describe().round(2).to_string())

    if TARGET_COLUMN in data.columns:
        print(f"\n{TARGET_COLUMN.upper()} COUNTS")
        print(data[TARGET_COLUMN].value_counts(dropna=False).to_string())
        print(f"\n{TARGET_COLUMN.upper()} PROPORTIONS")
        proportions = data[TARGET_COLUMN].value_counts(normalize=True, dropna=False)
        print((100 * proportions).round(1).astype(str).add("%").to_string())
    else:
        print(f"\nWARNING: target column {TARGET_COLUMN!r} was not found.")

    # A total of exactly 24 hours in every row may mean the daily activities
    # were constructed as a fixed time budget. Report the observation so it
    # can be considered during interpretation; this check does not prove why.
    if all(column in data.columns for column in DAILY_HOUR_COLUMNS):
        daily_total = data[DAILY_HOUR_COLUMNS].sum(axis=1)
        # isclose avoids counting tiny floating-point rounding differences as
        # a real difference (for example, 23.999999999999996 instead of 24).
        approximately_24 = (daily_total - 24).abs().lt(1e-9)
        print("\nDAILY-HOUR TOTAL CHECK")
        print(f"Minimum total: {daily_total.min():.2f}")
        print(f"Maximum total: {daily_total.max():.2f}")
        print(
            "Rows totaling approximately 24 hours: "
            f"{int(approximately_24.sum())} / {len(data)}"
        )

    if "Student_ID" in data.columns:
        print("\nIDENTIFIER CHECK")
        print(f"Unique Student_ID values: {data['Student_ID'].nunique()} / {len(data)}")
        print("Student_ID is an identifier; it should not be used as a model feature.")


if __name__ == "__main__":
    main()
