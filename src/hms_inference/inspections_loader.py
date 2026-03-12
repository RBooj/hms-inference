from pathlib import Path
import pandas as pd

RELEVANT_CATEGORIES_2022 = {"hive grading", "hive status", "frames of bees", "varroa"}
VALID_HIVE_GRADING_2022 = {
    "weak": "weak",
    "medium": "medium",
    "strong": "strong",
}
VARROA_HIGH_THRESHOLD = 10


def load_inspections_2022(project_root: Path) -> pd.DataFrame:
    csv_path = (
        project_root
        / "data"
        / "UrBAN"
        / "data"
        / "annotations"
        / "inspections_2022.csv"
    )

    df = pd.read_csv(csv_path)

    # normalize hive id
    df["hive_id"] = pd.to_numeric(df["Tag number"], errors="coerce").astype("Int64")

    # parse date
    df["inspection_date"] = pd.to_datetime(df["Date"], utc=True)

    # parse queen status
    df["queen_present"] = df["Queen status"].map(
        {"queenright": True, "queenless": False}
    )

    # operate only on needed columns
    df = df[
        ["hive_id", "inspection_date", "Category", "Action detail", "queen_present"]
    ].copy()

    # only keep values from Category column that are needed
    df_relevant = df[df["Category"].isin(RELEVANT_CATEGORIES_2022)].copy()

    # Pivot by hive_id and inspection_date
    pivot = df_relevant.pivot_table(
        index=["hive_id", "inspection_date"],
        columns="Category",
        values="Action detail",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None

    # rename columns
    pivot = pivot.rename(
        columns={
            "hive status": "hive_status_raw",
            "hive grading": "hive_grading_raw",
            "frames of bees": "frames_of_bees",
            "varroa": "varroa_raw",
        }
    )

    # incorporate "deadout" recordings into grading scheme
    pivot["hive_grading_raw"] = (
        pivot["hive_grading_raw"].astype("string").str.strip().str.lower()
    )
    pivot["hive_status_raw"] = (
        pivot["hive_status_raw"].astype("string").str.strip().str.lower()
    )
    pivot["hive_state"] = pivot["hive_grading_raw"].map(VALID_HIVE_GRADING_2022)

    # normalize hive_state to desired labels only
    pivot["hive_state"] = (
        pivot["hive_state"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map(VALID_HIVE_GRADING_2022)
    )

    # parse queen status from original data column
    queen_table = df.groupby(["hive_id", "inspection_date"], as_index=False)[
        "queen_present"
    ].first()
    pivot = pivot.merge(queen_table, on=["hive_id", "inspection_date"], how="outer")
    pivot.loc[pivot["hive_status_raw"] == "deadout", "queen_present"] = False
    pivot.loc[pivot["hive_status_raw"] == "deadout", "hive_state"] = "deadout"
    pivot = pivot.drop(columns=["hive_grading_raw", "hive_status_raw"], errors="ignore")

    # parse frames of bees and varroa count
    pivot["frames_of_bees"] = pd.to_numeric(pivot["frames_of_bees"], errors="coerce")
    pivot["varroa_count"] = pd.to_numeric(pivot["varroa_raw"], errors="coerce")
    pivot = pivot.drop(columns=["varroa_raw"], errors="ignore")

    # derive "varroa_high" from varroa counts.
    pivot["varroa_high"] = (
            pivot["varroa_count"] >= VARROA_HIGH_THRESHOLD
            ).where(pivot["varroa_count"].notna())
    pivot = pivot.drop(columns=["varroa_count"], errors="ignore")

    # drop rows where all relevent outputs are missing
    pivot = pivot.dropna(
        subset=["queen_present", "hive_state", "frames_of_bees", "varroa_high"],
        how="all",
    )
    pivot = pivot.sort_values(["hive_id", "inspection_date"]).reset_index(drop=True)
    pivot["queen_present"] = pivot["queen_present"].astype("boolean")


    return pivot


def load_inspections_2021(project_root: Path) -> pd.DataFrame:
    csv_path = (
        project_root
        / "data"
        / "UrBAN"
        / "data"
        / "annotations"
        / "inspections_2021.csv"
    )
    df = pd.read_csv(csv_path)

    # normalize hive id
    df["hive_id"] = pd.to_numeric(df["Tag number"], errors="coerce").astype("Int64")

    # parse date - 2021 data only has date, not time.
    # Assume 12 noon as time
    date = pd.to_datetime(df["Date"], errors="coerce")
    df["inspection_date"] = date + pd.Timedelta(hours=12)

    # parse queen presence check
    # define "QR" = queenright "QNS" = queen not seen
    queen_map = {"QR": True, "QNS": False}
    df["queen_present"] = df["Queen status"].map(queen_map)

    # parse frames of bees - NaN = 0 frames
    for col in ["Fob 1st", "Fob 2nd", "Fob 3rd"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["fob_total"] = df["Fob 1st"] + df["Fob 2nd"] + df["Fob 3rd"]

    # Construct output DataFrame
    out = df[["hive_id", "inspection_date", "queen_present", "fob_total"]].dropna(
        subset=["hive_id", "inspection_date"]
    )
    out = out.sort_values(["hive_id", "inspection_date"]).reset_index(drop=True)

    return out


if __name__ == "__main__":
    project_root = Path.cwd()
    inspections_2021 = load_inspections_2021(project_root)
    print("\n2021 Annotations --------------------------------------------------")
    print(inspections_2021.head(10))
    print(
        "Hives: ",
        inspections_2021["hive_id"].nunique(),
        "rows: ",
        len(inspections_2021),
    )
    print(
        "Queen Present true reported: \n",
        inspections_2021["queen_present"].value_counts(dropna=False),
    )

    inspections_2022 = load_inspections_2022(project_root)

    print("\n2022 Annotations --------------------------------------------------")
    print("\nRows:", len(inspections_2022))
    print("Hives:", inspections_2022["hive_id"].nunique())

    print("\nHead:")
    print(inspections_2022.head(10).to_string())

    print("\nStrength distribution:")
    print(inspections_2022["hive_state"].value_counts(dropna=False))

    print("\nQueen distribution:")
    print(inspections_2022["queen_present"].value_counts(dropna=False))

    print("\nFrames of bees stats:")
    print(inspections_2022["frames_of_bees"].describe())

    print("\nVarroa stats:")
    print(inspections_2022["varroa_high"].describe())

    # save to csv for inspection
    inspections_2022.to_csv("inspections_2022_parsed_data.csv")
