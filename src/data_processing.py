# src/data_processing.py
import pandas as pd

REQUIRED_COLS = {"WEEK", "COUNTRY", "EVENTS", "FATALITIES"}

def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["WEEK"] = pd.to_datetime(df["WEEK"])
    return df

def aggregate_to_country_week(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse multiple admin/event-type rows into ONE row per COUNTRY+WEEK.
    This must happen before any shift()/lag logic — see Phase 4 mistake note."""
    return (
        df.groupby(["WEEK", "COUNTRY"], as_index=False)
          .agg(EVENTS=("EVENTS", "sum"), FATALITIES=("FATALITIES", "sum"))
    )

def build_complete_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every country has a row for every week in the dataset's
    global date range. Missing weeks -> 0 events/fatalities (documented
    assumption: absence in ACLED weekly aggregates means no recorded
    conflict activity that week, not 'unknown')."""
    all_weeks = sorted(df["WEEK"].unique())
    all_countries = df["COUNTRY"].unique()
    
    full_index = pd.MultiIndex.from_product(
        [all_weeks, all_countries], names=["WEEK", "COUNTRY"]
    )
    panel = (
        df.set_index(["WEEK", "COUNTRY"])
          .reindex(full_index, fill_value=0)
          .reset_index()
    )
    return panel.sort_values(["COUNTRY", "WEEK"]).reset_index(drop=True)

def clean_and_grid(path: str) -> pd.DataFrame:
    raw = load_raw(path)
    agg = aggregate_to_country_week(raw)
    panel = build_complete_panel(agg)
    return panel



#PRE-CHECK (A standard deviation (std) of 0.0 means your data grid is perfectly balanced.)
# df = clean_and_grid("data/raw-data/MEdata_07-03-26.csv")
# assert df.duplicated(subset=["WEEK", "COUNTRY"]).sum() == 0
# print(df.groupby("COUNTRY")["WEEK"].count().describe())  # every country same count?


