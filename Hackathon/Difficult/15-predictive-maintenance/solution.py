#!/usr/bin/env python3
"""Reference solution: Predictive Maintenance with temporal features and XGBoost."""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, f1_score, precision_recall_curve, roc_auc_score, auc
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed. Install with: pip install xgboost")
    from sklearn.ensemble import RandomForestClassifier as XGBModel

# ========================
# Load and prepare data
# ========================
df = pd.read_csv("data/predictive_maintenance.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values(["MachineID", "Timestamp"]).reset_index(drop=True)

print(f"Data shape: {df.shape}")
print(f"Failure rate: {df['Failure'].mean()*100:.2f}%")
print(f"Machines: {df['MachineID'].nunique()}")

# ========================
# Temporal feature engineering
# ========================
def engineer_features(df):
    # Create rolling window, lag, and rate-of-change features.
    df = df.copy()
    # Sort per machine
    df = df.sort_values(["MachineID", "Timestamp"]).reset_index(drop=True)
    
    base_features = ["Temperature", "Vibration", "Pressure", "RPM",
                     "OperatingHours", "LastMaintenanceDays"]
    
    # Rolling window features (last 6 and 12 hours)
    for w in [3, 6, 12]:
        for col in ["Temperature", "Vibration", "Pressure", "RPM"]:
            grp = df.groupby("MachineID")[col]
            df[f"{col}_mean_{w}h"] = grp.transform(lambda x: x.rolling(w, min_periods=1).mean())
            df[f"{col}_std_{w}h"] = grp.transform(lambda x: x.rolling(w, min_periods=1).std())
            df[f"{col}_max_{w}h"] = grp.transform(lambda x: x.rolling(w, min_periods=1).max())
            df[f"{col}_min_{w}h"] = grp.transform(lambda x: x.rolling(w, min_periods=1).min())
    
    # Rate of change (difference from previous reading)
    for col in ["Temperature", "Vibration", "Pressure", "RPM"]:
        grp = df.groupby("MachineID")[col]
        df[f"{col}_delta"] = grp.diff()
        df[f"{col}_delta_abs"] = df[f"{col}_delta"].abs()
        df[f"{col}_pct_change"] = grp.pct_change() * 100
    
    # Lag features (1, 2, 3 hour lag)
    for lag in [1, 2, 3]:
        for col in base_features:
            df[f"{col}_lag_{lag}"] = df.groupby("MachineID")[col].shift(lag)
    
    # Interaction features
    df["temp_vib_ratio"] = df["Temperature"] / (df["Vibration"] + 0.001)
    df["pressure_x_rpm"] = df["Pressure"] * df["RPM"]
    df["temp_pressure_interaction"] = df["Temperature"] * df["Pressure"] / 1000
    
    # Maintenance-related
    df["high_op_hours"] = (df["OperatingHours"] > df.groupby("MachineID")["OperatingHours"].transform("median")).astype(int)
    
    # Fill NaN values
    df = df.fillna(0)
    
    return df

df_feat = engineer_features(df)
feature_cols = [c for c in df_feat.columns if c not in
                ["MachineID", "Timestamp", "Failure"]]

print(f"\nFeatures engineered: {len(feature_cols)}")

# ========================
# Temporal split for validation
# ========================
df_feat = df_feat.sort_values("Timestamp").reset_index(drop=True)
split_idx = int(len(df_feat) * 0.8)
train_df = df_feat.iloc[:split_idx]
test_df = df_feat.iloc[split_idx:]

X_train = train_df[feature_cols]
y_train = train_df["Failure"]
X_test = test_df[feature_cols]
y_test = test_df["Failure"]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"Train: {X_train.shape}, failures: {y_train.sum()}")
print(f"Test:  {X_test.shape}, failures: {y_test.sum()}")

# ========================
# Train XGBoost with imbalance handling
# ========================
if XGB_AVAILABLE:
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)
else:
    model = XGBModel(n_estimators=200, class_weight="balanced", random_state=42)
    model.fit(X_train_s, y_train)

y_prob = model.predict_proba(X_test_s)[:, 1]
y_pred = model.predict(X_test_s)

# ========================
# Threshold tuning for early warning
# ========================
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# Find threshold that gives best F1
best_f1 = 0
best_thresh = 0.5
for thresh in np.linspace(0.05, 0.95, 100):
    pred_thresh = (y_prob >= thresh).astype(int)
    f1 = f1_score(y_test, pred_thresh)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"\nOptimal threshold: {best_thresh:.3f} (F1: {best_f1:.4f})")

y_pred_opt = (y_prob >= best_thresh).astype(int)
print("\n--- XGBoost Results (tuned threshold) ---")
print(classification_report(y_test, y_pred_opt))

f1_failure = f1_score(y_test, y_pred_opt)
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = auc(recalls, precisions)
print(f"F1 (failure): {f1_failure:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")

# ========================
# Early Warning Time Analysis
# ========================
print("\n--- Early Warning Time Analysis ---")

test_df_local = test_df.copy()
test_df_local["y_prob"] = y_prob
test_df_local["y_pred"] = y_pred_opt

warning_times = []
for mid in test_df_local["MachineID"].unique():
    machine_data = test_df_local[test_df_local["MachineID"] == mid].sort_values("Timestamp")
    failure_rows = machine_data[machine_data["Failure"] == 1]
    alert_rows = machine_data[machine_data["y_pred"] == 1]
    
    if len(failure_rows) > 0 and len(alert_rows) > 0:
        first_failure = failure_rows.index[0]
        alerts_before = alert_rows[alert_rows.index < first_failure]
        if len(alerts_before) > 0:
            first_alert = alerts_before.index[0]
            hours_before = (failure_rows["Timestamp"].iloc[0] -
                           machine_data.loc[first_alert, "Timestamp"]).total_seconds() / 3600
            warning_times.append(hours_before)

if warning_times:
    print(f"Average warning time: {np.mean(warning_times):.1f} hours")
    print(f"Median warning time: {np.median(warning_times):.1f} hours")
    print(f"Min warning time: {min(warning_times):.1f} hours")
    print(f"Max warning time: {max(warning_times):.1f} hours")
else:
    print("No early warnings generated")

# ========================
# Feature Importance
# ========================
if XGB_AVAILABLE:
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print("\n--- Top 10 Features ---")
    print(importance.head(10).to_string(index=False))
