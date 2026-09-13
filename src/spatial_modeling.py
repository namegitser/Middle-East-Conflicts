import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import warnings
warnings.filterwarnings('ignore')
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



from src.data_processing import load_raw
from src.features import add_lag_features, add_rolling_features, add_target

def aggregate_to_admin1_week(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["WEEK", "COUNTRY", "ADMIN1"], as_index=False)
          .agg(
              EVENTS=("EVENTS", "sum"),
              FATALITIES=("FATALITIES", "sum"),
              CENTROID_LATITUDE=("CENTROID_LATITUDE", "mean"),
              CENTROID_LONGITUDE=("CENTROID_LONGITUDE", "mean"),
          )
    )

def build_admin1_panel(df: pd.DataFrame) -> pd.DataFrame:
    all_weeks = sorted(df["WEEK"].unique())
    locations = df[["COUNTRY", "ADMIN1"]].drop_duplicates()
    
    full_index = pd.MultiIndex.from_product(
        [all_weeks, locations["COUNTRY"].unique()], names=["WEEK", "COUNTRY"]
    )
    panel_base = pd.DataFrame(index=full_index).reset_index()
    panel = panel_base.merge(locations, on="COUNTRY", how="left")
    
    merged = pd.merge(panel, df, on=["WEEK", "COUNTRY", "ADMIN1"], how="left").fillna({
        "EVENTS": 0, "FATALITIES": 0
    })
    
    merged["CENTROID_LATITUDE"] = merged.groupby("ADMIN1")["CENTROID_LATITUDE"].ffill().bfill()
    merged["CENTROID_LONGITUDE"] = merged.groupby("ADMIN1")["CENTROID_LONGITUDE"].ffill().bfill()
    
    return merged.sort_values(["COUNTRY", "ADMIN1", "WEEK"]).reset_index(drop=True)

def add_spatial_lag(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """Computes the average fatalities of the nearest neighboring ADMIN1 regions at week t-1."""
    # Get unique admin centroids
    admin_coords = df[["ADMIN1", "CENTROID_LATITUDE", "CENTROID_LONGITUDE"]].drop_duplicates().dropna()
    
    # Convert lat/lon to radians for Haversine distance
    coords_rad = np.radians(admin_coords[["CENTROID_LATITUDE", "CENTROID_LONGITUDE"]].values)
    tree = BallTree(coords_rad, metric='haversine')
    
    # Query nearest neighbors (k+1 because the nearest neighbor is always itself)
    k = min(n_neighbors + 1, len(admin_coords))
    distances, indices = tree.query(coords_rad, k=k)
    
    admin_list = admin_coords["ADMIN1"].values
    neighbor_map = {}
    for i, admin in enumerate(admin_list):
        # Exclude index 0 (itself) and take the nearest k-1 neighbors
        neighbor_indices = indices[i, 1:]
        neighbor_map[admin] = admin_list[neighbor_indices]
    
    # Create a lookup table of fatalities per (WEEK, ADMIN1)
    fatality_lookup = df.set_index(["WEEK", "ADMIN1"])["FATALITIES"].to_dict()
    
    spatial_lags = []
    for row in df.itertuples():
        w = row.WEEK
        # Look at the previous week (t-1) for neighbors
        prev_w = w - pd.Timedelta(weeks=1)
        neighbors = neighbor_map.get(row.ADMIN1, [])
        
        neighbor_fats = [fatality_lookup.get((prev_w, n), 0.0) for n in neighbors]
        spatial_lags.append(np.mean(neighbor_fats) if neighbor_fats else 0.0)
        
    df["spatial_lag_fatalities"] = spatial_lags
    return df

if __name__ == "__main__":
    print("1. Loading raw data...")
    raw = load_raw("data/raw-data/MEdata_07-03-26.csv")
    
    print("2. Aggregating and gridding to ADMIN1 resolution...")
    admin1_df = aggregate_to_admin1_week(raw)
    admin1_panel = build_admin1_panel(admin1_df)
    
    # Temporary key swap to run lag/rolling functions per province
    admin1_panel["REAL_COUNTRY"] = admin1_panel["COUNTRY"]
    admin1_panel["COUNTRY"] = admin1_panel["COUNTRY"] + "_" + admin1_panel["ADMIN1"]
    
    print("3. Engineering localized features...")
    admin1_panel = add_lag_features(admin1_panel)
    admin1_panel = add_rolling_features(admin1_panel)
    
    print("4. Computing spatial-lag (neighbor spillover) features...")
    admin1_panel = add_spatial_lag(admin1_panel, n_neighbors=5)
    admin1_panel = add_target(admin1_panel)
    
    print(f"Panel shape: {admin1_panel.shape}. Spatial features attached successfully.")
    print(admin1_panel[["WEEK", "REAL_COUNTRY", "ADMIN1", "FATALITIES", "fatalities_lag1", "spatial_lag_fatalities"]].head())