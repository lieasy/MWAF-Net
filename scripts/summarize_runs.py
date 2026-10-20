import argparse
import json
from pathlib import Path

import numpy as np


def metrics_from_confusion(confusion):
    cm = np.asarray(confusion, dtype=np.float64)
    if (cm.ndim != 2 or cm.shape[0] != cm.shape[1] or cm.shape[0] < 2
            or not np.isfinite(cm).all() or (cm < 0).any()
            or not np.equal(cm, np.floor(cm)).all() or cm.sum() <= 0):
        raise ValueError("confusion_counts must be a nonnegative square count matrix")
    diagonal = np.diag(cm)
    recall = np.divide(diagonal, cm.sum(1), out=np.zeros(len(cm)), where=cm.sum(1) > 0)
    precision = np.divide(diagonal, cm.sum(0), out=np.zeros(len(cm)), where=cm.sum(0) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(len(cm)), where=(precision + recall) > 0)
    result = {
        "accuracy_percent": float(diagonal.sum() / cm.sum() * 100),
        "uar_percent": float(recall.mean() * 100),
        "class_recall_percent": (recall * 100).tolist(),
        "class_precision_percent": (precision * 100).tolist(),
        "class_f1_percent": (f1 * 100).tolist(),
    }
    if len(cm) == 2:
        result.update(precision_percent=float(precision[1] * 100),
                      f1_percent=float(f1[1] * 100))
    return result


def summarize_runs(records, expected_runs=5):
    if expected_runs < 2 or len(records) != expected_runs:
        raise ValueError(f"Expected {expected_runs} independent run summaries, got {len(records)}")
    reference = records[0]
    for key in ("task", "class_names", "prediction_unit"):
        if key not in reference:
            raise ValueError(f"Missing metadata: {key}")
        if any(row.get(key) != reference[key] for row in records):
            raise ValueError(f"All runs must have identical {key}")
    if reference["prediction_unit"] != "segment":
        raise ValueError("This release summarises the manuscript's segment-level metrics")
    per_run = []
    row_normalized = []
    for row in records:
        cm = np.asarray(row["confusion_counts"], dtype=np.float64)
        calculated = metrics_from_confusion(cm)
        if len(cm) != len(reference["class_names"]):
            raise ValueError("Class labels do not match the confusion matrix")
        if len(cm) == 2 and row.get("positive_class_index", 1) != 1:
            raise ValueError("Binary summaries must use class index 1 as positive")
        # Reject inconsistent inputs instead of silently repairing a historical table.
        for key, value in calculated.items():
            if key in row and not np.allclose(row[key], value, rtol=0, atol=1e-7):
                raise ValueError(f"Reported {key} disagrees with this run's confusion_counts")
        per_run.append(calculated)
        row_normalized.append(np.divide(cm, cm.sum(1, keepdims=True),
                              out=np.zeros_like(cm), where=cm.sum(1, keepdims=True) > 0) * 100)
    summaries = {}
    for key in per_run[0]:
        values = np.asarray([row[key] for row in per_run], dtype=np.float64)
        summaries[key] = {
            "mean": values.mean(axis=0).tolist(),
            "sample_sd": values.std(axis=0, ddof=1).tolist(),
            "per_run": values.tolist(),
        }
    matrices = np.stack(row_normalized)
    return {
        "task": reference["task"], "class_names": reference["class_names"],
        "prediction_unit": "segment", "runs": expected_runs, "sd_ddof": 1,
        "aggregation": "equal run weights; no concatenation of overlapping test sets",
        "metrics": summaries,
        "row_normalized_confusion_percent": {
            "mean": matrices.mean(0).tolist(),
            "sample_sd": matrices.std(0, ddof=1).tolist(),
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", type=Path, nargs="+", help="Per-run evaluate.py JSON files")
    p.add_argument("--expected_runs", type=int, default=5)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if len({path.resolve() for path in a.inputs}) != len(a.inputs):
        raise ValueError("Do not provide the same run file more than once")
    if a.output is not None and a.output.exists():
        raise FileExistsError(a.output)
    records = [json.loads(path.read_text(encoding="utf-8")) for path in a.inputs]
    result = summarize_runs(records, a.expected_runs)
    result["source_files"] = [str(path) for path in a.inputs]
    text = json.dumps(result, indent=2)
    if a.output is not None:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
