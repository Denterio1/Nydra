"""Fair (stratified) sampling for large image datasets."""
from __future__ import annotations

import pandas as pd

UNLABELED = "(unlabeled)"


def class_allocation(counts, budget):
    """counts: {label: size}. Returns {label: n_to_take}, sum <= budget.

    Equal share per class, capped by class size, leftovers go to bigger classes.
    If there are more classes than budget, take 1 image from evenly spaced classes.
    """
    n = len(counts)
    if n == 0 or budget <= 0:
        return {}
    if n > budget:
        labels = sorted(counts, key=str)
        return {labels[int(i * n / budget)]: 1 for i in range(budget)}
    items = sorted(counts.items(), key=lambda kv: (kv[1], str(kv[0])))
    alloc, left = {}, int(budget)
    for i, (lab, size) in enumerate(items):
        take = min(int(size), left // (n - i))
        alloc[lab] = take
        left -= take
    return alloc


def stratified_sample(df, max_images, label_col="label", seed=0):
    """Return at most max_images rows with a fair share per class (reproducible)."""
    if len(df) <= max_images:
        return df.reset_index(drop=True)
    if label_col in df.columns:
        lbl = df[label_col].fillna(UNLABELED).astype(str)
    else:
        lbl = pd.Series(UNLABELED, index=df.index)
    work = df.assign(_lbl=lbl)
    alloc = class_allocation(work["_lbl"].value_counts().to_dict(), max_images)
    parts = [g.sample(alloc[k], random_state=seed)
             for k, g in work.groupby("_lbl", sort=True) if alloc.get(k, 0) > 0]
    return pd.concat(parts).drop(columns="_lbl").reset_index(drop=True)
