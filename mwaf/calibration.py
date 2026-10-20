import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.model_selection import StratifiedGroupKFold


def _validate_probabilities(probabilities, labels):
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if p.ndim != 2 or p.shape[1] < 2 or y.shape != (len(p),) or not len(p):
        raise ValueError("Expected nonempty [samples, classes] probabilities and labels")
    if (not np.isfinite(p).all() or (p < 0).any() or (p.sum(1) <= 0).any()
            or (y < 0).any() or (y >= p.shape[1]).any()):
        raise ValueError("Invalid probabilities or class labels")
    return p / p.sum(1, keepdims=True), y


def class_balanced_nll(probabilities, labels):
    p, y = _validate_probabilities(probabilities, labels)
    counts = np.bincount(y, minlength=p.shape[1])
    if (counts == 0).any():
        raise ValueError("Class-balanced NLL requires every class")
    losses = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
    return float(np.mean(np.bincount(y, weights=losses, minlength=len(counts)) / counts))


def aggregate_recordings(probabilities, labels, recording_ids):
    p, y = _validate_probabilities(probabilities, labels)
    recording_ids = np.asarray(recording_ids).astype(str)
    if recording_ids.shape != y.shape:
        raise ValueError("One recording ID is required per segment")
    recordings, inverse = np.unique(recording_ids, return_inverse=True)
    summed = np.zeros((len(recordings), p.shape[1]), dtype=np.float64)
    np.add.at(summed, inverse, p)
    counts = np.bincount(inverse)
    labels_out = np.empty(len(recordings), dtype=np.int64)
    for i in range(len(recordings)):
        labels_here = np.unique(y[inverse == i])
        if len(labels_here) != 1:
            raise ValueError("A recording cannot have multiple target labels")
        labels_out[i] = labels_here[0]
    return summed / counts[:, None], labels_out, recordings


def selection_nll(probabilities, labels, recording_ids=None, unit="segment"):
    if unit == "recording":
        if recording_ids is None:
            raise ValueError("Recording-level selection requires recording IDs")
        probabilities, labels, _ = aggregate_recordings(probabilities, labels, recording_ids)
    elif unit != "segment":
        raise ValueError("Selection unit must be segment or recording")
    return class_balanced_nll(probabilities, labels)


def apply_bias(probabilities, bias):
    p = np.asarray(probabilities, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    if p.ndim != 2 or bias.shape != (p.shape[1],) or not np.isfinite(bias).all():
        raise ValueError("Bias must contain one finite value per class")
    return softmax(np.log(np.clip(p, 1e-12, 1)) + bias[None, :], axis=1)


def _fit_bias(probabilities, labels, recording_ids, unit, penalty, bound):
    nclasses = probabilities.shape[1]
    if unit == "recording":
        _, record_labels, recordings = aggregate_recordings(probabilities, labels, recording_ids)
        _, inverse = np.unique(recording_ids, return_inverse=True)
        counts = np.bincount(inverse)

    def objective(bias):
        # Apply the same segment transformation used at test time, then (only
        # for Murmur model selection) average transformed probabilities by file.
        calibrated = apply_bias(probabilities, bias)
        if unit == "recording":
            averaged = np.column_stack([
                np.bincount(inverse, weights=calibrated[:, c], minlength=len(recordings)) / counts
                for c in range(nclasses)
            ])
            nll = class_balanced_nll(averaged, record_labels)
        else:
            nll = class_balanced_nll(calibrated, labels)
        return nll + penalty * float(np.mean(bias ** 2))

    fitted = minimize(objective, np.zeros(nclasses), method="L-BFGS-B",
                      bounds=[(-bound, bound)] * nclasses,
                      options={"maxiter": 300, "ftol": 1e-11})
    if not fitted.success or not np.isfinite(fitted.fun):
        raise RuntimeError(f"Class-bias optimisation failed: {fitted.message}")
    return fitted.x


def fit_validation_bias(probabilities, labels, patient_ids, recording_ids, *,
                        source_split, unit="segment", folds=3,
                        penalties=(1.0, 0.3, 0.1, 0.03), bound=1.5, seed=42):
    if source_split != "val":
        raise ValueError("Bias fitting and selection are permitted on val only")
    p, y = _validate_probabilities(probabilities, labels)
    groups = np.asarray(patient_ids).astype(str)
    rids = np.asarray(recording_ids).astype(str)
    if groups.shape != y.shape or rids.shape != y.shape:
        raise ValueError("Patient and recording IDs must align with validation segments")
    if folds < 2 or bound <= 0 or any(x <= 0 for x in penalties):
        raise ValueError("Invalid folds, penalty, or bias bounds")
    for recording in np.unique(rids):
        keep = rids == recording
        if len(np.unique(groups[keep])) != 1 or len(np.unique(y[keep])) != 1:
            raise ValueError("Each recording must belong to one group and class")
    zero = np.zeros(p.shape[1])
    report = dict(source_split="val", unit=unit, requested_folds=folds,
                  group_unit="linked patient / duplicate-audio group",
                  bound=float(bound), penalties=[float(x) for x in penalties],
                  objective="class-balanced NLL + penalty * mean(bias**2)",
                  bias_application="per segment before any validation recording mean",
                  rule="best mean NLL + SE of best; prefer no bias, then largest penalty",
                  seed=int(seed))
    # Never manufacture segment-wise folds if independent groups are too few.
    if any(len(np.unique(groups[y == c])) < folds for c in range(p.shape[1])):
        report.update(status="no_bias_insufficient_class_groups", selected_penalty=None,
                      bias=zero.tolist(), candidates=[])
        return zero, report
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(p, y, groups))
    for fit, held in splits:
        if set(groups[fit]) & set(groups[held]):
            raise AssertionError("Patient leakage in calibration folds")
        if len(np.unique(y[fit])) != p.shape[1] or len(np.unique(y[held])) != p.shape[1]:
            report.update(status="no_bias_missing_fold_class", selected_penalty=None,
                          bias=zero.tolist(), candidates=[])
            return zero, report
    candidates = []
    for penalty in [None, *sorted(set(float(v) for v in penalties), reverse=True)]:
        scores = []
        for fit, held in splits:
            bias = zero if penalty is None else _fit_bias(p[fit], y[fit], rids[fit], unit, penalty, bound)
            scores.append(selection_nll(apply_bias(p[held], bias), y[held], rids[held], unit))
        candidates.append(dict(penalty=penalty, fold_nll=scores, mean_nll=float(np.mean(scores)),
                               se_nll=float(np.std(scores, ddof=1) / np.sqrt(folds))))
    best = min(candidates, key=lambda row: row["mean_nll"])
    threshold = best["mean_nll"] + best["se_nll"]
    chosen = next(row for row in candidates if row["mean_nll"] <= threshold + 1e-12)
    penalty = chosen["penalty"]
    bias = zero if penalty is None else _fit_bias(p, y, rids, unit, penalty, bound)
    report.update(status="selected", selected_penalty=penalty, bias=bias.tolist(),
                  candidates=candidates, one_se_threshold=float(threshold),
                  folds=[dict(fit_groups=len(np.unique(groups[a])), heldout_groups=len(np.unique(groups[b])))
                         for a, b in splits])
    return bias, report
