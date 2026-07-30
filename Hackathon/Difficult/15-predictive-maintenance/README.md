# Problem 15: Predictive Maintenance

## Domain
Time Series + Classification

## Problem Statement
A manufacturing plant operates 100 industrial machines that are critical to production. Unexpected equipment failures cause costly downtime averaging $50,000 per hour. Your task is to build a predictive maintenance system that uses real-time sensor readings to forecast failures before they happen. The plant manager needs at least 12 hours of warning to schedule maintenance without disrupting production.

The dataset contains sensor readings collected hourly from each machine over several months. Only ~5% of readings are followed by a failure within 24 hours, creating a severe class imbalance. You must engineer temporal features that capture degradation trends and alert the team early enough to act.

## Objectives
1. Predict equipment failure within the next 24 hours from current sensor readings
2. Achieve F1 > 0.70 on the minority (failure) class
3. Maintain an average early warning time of >12 hours before actual failure
4. Handle the severe class imbalance (95:5) appropriately

## Dataset
- 5000 sensor readings from 100 machines
- Columns: MachineID, Timestamp, Temperature, Vibration, Pressure, RPM, OperatingHours, LastMaintenanceDays, Failure (binary target)
- CSV at `data/predictive_maintenance.csv`

## Success Criteria
- **F1-score**: >0.70 on the failure class
- **Early Warning Time**: Average prediction >12 hours before actual failure
- **Precision-Recall AUC**: >0.75
- **Robustness**: Consistent performance across different machine types and operating conditions

## Starter Code
`starter_code.py` loads data and trains a basic RandomForest. You must extend with temporal feature engineering and proper imbalance handling.

## Constraints
- You cannot use future data to predict past failures (no data leakage)
- Feature engineering must respect temporal ordering
- Cross-validation must use time-based (not random) splits
