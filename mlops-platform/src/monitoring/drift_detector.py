import json
import os

import pandas as pd

try:
    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import DataDriftPreset
    from evidently.legacy.metrics import DatasetDriftMetric
except ImportError:  # Fallback for older Evidently versions
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently.metrics import DatasetDriftMetric

DRIFT_THRESHOLD = 0.3  # share of drifted features that triggers retrain

def run_drift_report(reference_path: str, current_data: pd.DataFrame, report_path: str) -> dict:
    reference = pd.read_csv(reference_path)
    
    report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric(),
    ])
    report.run(reference_data=reference, current_data=current_data)
    
    result = report.as_dict()
    drift_score  = result['metrics'][1]['result']['share_of_drifted_columns']
    dataset_drift = result['metrics'][1]['result']['dataset_drift']
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report.save_html(report_path)
    
    return {
        "drift_score": drift_score,
        "dataset_drift_detected": dataset_drift,
        "should_retrain": drift_score >= DRIFT_THRESHOLD
    }

if __name__ == "__main__":
    # Quick test: simulate drifted data
    import numpy as np
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    reference_path = os.path.join(base_dir, "Data", "reference_data.csv")
    report_path = os.path.join(base_dir, "reports", "drift_report.html")

    ref = pd.read_csv(reference_path)
    drifted = ref.copy()
    drifted['MonthlyCharges'] *= np.random.uniform(1.3, 1.8, len(drifted))
    result = run_drift_report(reference_path, drifted, report_path)
    print(json.dumps(result, indent=2))