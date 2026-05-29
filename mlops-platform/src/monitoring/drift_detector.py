import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric
import json, os

DRIFT_THRESHOLD = 0.3   # share of drifted features that triggers retrain

def run_drift_report(reference_path: str, current_data: pd.DataFrame) -> dict:
    reference = pd.read_csv(reference_path)
    
    report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric(),
    ])
    report.run(reference_data=reference, current_data=current_data)
    
    result = report.as_dict()
    drift_score  = result['metrics'][1]['result']['share_of_drifted_columns']
    dataset_drift = result['metrics'][1]['result']['dataset_drift']
    
    os.makedirs("reports", exist_ok=True)
    report.save_html("reports/drift_report.html")
    
    return {
        "drift_score": drift_score,
        "dataset_drift_detected": dataset_drift,
        "should_retrain": drift_score >= DRIFT_THRESHOLD
    }

if __name__ == "__main__":
    # Quick test: simulate drifted data
    import numpy as np
    ref = pd.read_csv("data/reference_data.csv")
    drifted = ref.copy()
    drifted['MonthlyCharges'] *= np.random.uniform(1.3, 1.8, len(drifted))
    result = run_drift_report("data/reference_data.csv", drifted)
    print(json.dumps(result, indent=2))