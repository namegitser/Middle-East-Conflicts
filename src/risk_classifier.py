import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

from data_processing import clean_and_grid
from features import add_lag_features, add_rolling_features, add_target
from models import  FEATURES, TARGET
from split import chronological_split

def add_risk_label(df: pd.DataFrame, percentile: float = 0.90) -> pd.DataFrame:
    """High-risk = next week's fatalities exceed this COUNTRY'S OWN historical
    expanding 90th percentile (shifted to prevent leakage)."""
    df = df.sort_values(["COUNTRY", "WEEK"]).copy()
    thresholds = df.groupby("COUNTRY")["FATALITIES"].transform(
        lambda s: s.expanding().quantile(percentile).shift(1)
    )
    # Fill initial NaNs in expanding window with 0 or global fallback
    thresholds = thresholds.fillna(0)
    df["high_risk_next_week"] = (df["next_week_fatalities"] > thresholds).astype(int)
    return df

if __name__ == "__main__":
    print("1. Loading and gridding data...")
    df = clean_and_grid("data/raw-data/MEdata_07-03-26.csv")
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_target(df)
    df = add_risk_label(df, percentile=0.90)

    print("2. Chronological split...")
    train_df, test_df, cutoff = chronological_split(df, train_frac=0.8)

    X_train = train_df[FEATURES].fillna(0)
    y_train_risk = train_df["high_risk_next_week"]
    
    X_test = test_df[FEATURES].fillna(0)
    y_test_risk = test_df["high_risk_next_week"]

    print("3. Training Random Forest Classifier (balanced weights)...")
    clf = RandomForestClassifier(
        n_estimators=300, 
        class_weight="balanced", 
        random_state=42, 
        n_jobs=-1
    )
    clf.fit(X_train, y_train_risk)

    print("4. Evaluating Risk Classifier...")
    risk_probs = clf.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test_risk, risk_probs)
    risk_preds = clf.predict(X_test)

    print(f"\n=== RISK CLASSIFIER ROC-AUC: {auc_score:.4f} ===\n")
    print(classification_report(y_test_risk, risk_preds, target_names=["Normal Week", "High-Risk Escalation"]))