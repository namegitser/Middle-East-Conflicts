import pandas as pd

def chronological_split(df: pd.DataFrame, train_frac: float = 0.8):
    unique_weeks = sorted(df["WEEK"].unique())
    cutoff = unique_weeks[int(len(unique_weeks) * train_frac)]
    train_df = df[df["WEEK"] < cutoff]
    test_df = df[df["WEEK"] >= cutoff]
    return train_df, test_df, cutoff