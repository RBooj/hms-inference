from pathlib import Path
import pandas as pd


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
    insp = load_inspections_2021(project_root)
    print(insp.head(10))
    print("Hives: ", insp["hive_id"].nunique(), "rows: ", len(insp))
    print(
        "Queen Present true reported: \n",
        insp["queen_present"].value_counts(dropna=False),
    )
