# src/features.py
import pandas as pd

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["COUNTRY", "WEEK"]).copy()
    g = df.groupby("COUNTRY")["FATALITIES"]
    df["fatalities_lag1"] = g.shift(1)
    df["fatalities_lag2"] = g.shift(2)
    df["fatalities_lag3"] = g.shift(3)
    df["events_lag1"] = df.groupby("COUNTRY")["EVENTS"].shift(1)
    return df

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling stats over t-4..t-1 (historical-only window), NOT t-3..t
    (current week). Both are technically leakage-free for predicting t+1,
    but the historical-only window is simpler to explain and audit."""
    df = df.sort_values(["COUNTRY", "WEEK"]).copy()
    shifted = df.groupby("COUNTRY")["FATALITIES"].shift(1)
    df["fatalities_roll_mean"] = shifted.rolling(4).mean().reset_index(level=0, drop=True) \
        if False else df.groupby("COUNTRY")["FATALITIES"].shift(1).rolling(4).mean()
    df["fatalities_roll_std"] = df.groupby("COUNTRY")["FATALITIES"].shift(1).rolling(4).std()
    return df

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["COUNTRY", "WEEK"]).copy()
    df["next_week_fatalities"] = df.groupby("COUNTRY")["FATALITIES"].shift(-1)
    return df.dropna(subset=["next_week_fatalities"])