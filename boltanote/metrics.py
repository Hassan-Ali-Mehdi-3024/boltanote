"""Metrics, judged the way a blind user would judge the app.

The model scores 15 classes; the app speaks one of 7 note values or stays silent ("none"). So we
add front + back probabilities into 8 groups and score those. The app only speaks when the top
value's probability is at least a threshold (picked on the validation photos). Two numbers then
matter most:
  - wrong-value rate: how often it confidently says the wrong amount (the costly mistake)
  - coverage: how often it speaks at all on photos that do show a note (otherwise you move the
    phone and it tries again a third of a second later)
"""

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from .models import GROUP_OF, GROUPS

NONE = GROUPS.index("none")


def to_groups(probs):
    probs = np.asarray(probs)
    out = np.zeros((len(probs), len(GROUPS)), dtype=probs.dtype)
    for c, g in enumerate(GROUP_OF):
        out[:, g] += probs[:, c]
    return out


def speaking_threshold(gprobs, groups, max_wrong=0.002):
    """Lowest threshold at which the wrong-value rate on these photos is <= max_wrong."""
    gprobs, groups = np.asarray(gprobs), np.asarray(groups)
    for t in np.round(np.arange(0.40, 0.991, 0.01), 2):
        if speak_stats(gprobs, groups, t)["wrong_value_rate"] <= max_wrong:
            return float(t)
    return 0.99


def speak_stats(gprobs, groups, t):
    pred, conf = gprobs.argmax(1), gprobs.max(1)
    speaks = (pred != NONE) & (conf >= t)
    has_note = groups != NONE
    wrong = speaks & (pred != groups)
    return {
        "threshold": float(t),
        "wrong_value_rate": float(wrong.mean()),
        "coverage_on_notes": float(speaks[has_note].mean()) if has_note.any() else None,
        "accuracy_when_speaking": float((pred[speaks] == groups[speaks]).mean()) if speaks.any() else None,
        "false_alarm_on_no_note": float(speaks[~has_note].mean()) if (~has_note).any() else None,
    }


def summarise(probs, groups, threshold, labels=None):
    """probs: [N, 15]; groups: [N] group index; labels: [N] class index if known (front/back)."""
    g, groups = to_groups(probs), np.asarray(groups)
    pred = g.argmax(1)
    present = sorted(set(groups.tolist()))
    out = {
        "n": int(len(groups)),
        "value_accuracy": float((pred == groups).mean()),
        "value_macro_f1": float(f1_score(groups, pred, labels=present, average="macro", zero_division=0)),
        "per_value_recall": {GROUPS[k]: float((pred[groups == k] == k).mean()) for k in present},
        "confusion_matrix_values": confusion_matrix(groups, pred, labels=range(len(GROUPS))).tolist(),
        "speaking": speak_stats(g, groups, threshold),
    }
    if labels is not None and (np.asarray(labels) >= 0).all():
        out["class_accuracy_15"] = float((np.asarray(probs).argmax(1) == np.asarray(labels)).mean())
    return out
