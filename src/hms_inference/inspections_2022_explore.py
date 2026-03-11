from pathlib import Path
import pandas as pd


def load_raw_inspections_2022(project_root: Path) -> pd.DataFrame:
    csv_path = (
        project_root
        / "data"
        / "UrBAN"
        / "data"
        / "annotations"
        / "inspections_2022.csv"
    )
    return pd.read_csv(csv_path)


if __name__ == "__main__":
    project_root = Path.cwd()
    df = load_raw_inspections_2022(project_root)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nShape:")
    print(df.shape)

    print("\nFirst 10 rows:")
    print(df.head(10).to_string())

    # Show unique category values
    if "Category" in df.columns:
        print("\nUnique Category values:")
        cats = sorted(df["Category"].dropna().astype(str).unique().tolist())
        for c in cats:
            print("-", c)

        print("\nCategory counts:")
        print(df["Category"].value_counts(dropna=False))

    # Show likely hive/date columns if present
    for col in [
        "Tag number",
        "Hive number",
        "Date",
        "Datetime",
        "Time",
        "Action detail",
        "Category",
        "Is alive",
        "Queen status",
    ]:
        if col in df.columns:
            print(f"\nSample values from column: {col}")
            print(df[col].dropna().astype(str).head(10).tolist())
