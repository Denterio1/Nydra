import pandas as pd
from src.sampling import class_allocation, stratified_sample


def mk(sizes):
    rows = [{"file_path": f"{k}/{i}.jpg", "label": k} for k, n in sizes.items() for i in range(n)]
    return pd.DataFrame(rows)


def test_under_cap_returns_everything():
    df = mk({"a": 5, "b": 5})
    assert len(stratified_sample(df, 100)) == 10


def test_tiny_class_is_kept_whole():
    df = mk({"big": 10000, "mid": 500, "tiny": 40})
    out = stratified_sample(df, 150)
    vc = out["label"].value_counts()
    assert len(out) == 150
    assert vc["tiny"] == 40
    assert vc["big"] == vc["mid"] == 55


def test_budget_respected_many_classes():
    df = mk({f"c{i}": 30 for i in range(50)})
    out = stratified_sample(df, 150)
    assert len(out) == 150
    assert out["label"].nunique() == 50


def test_more_classes_than_budget():
    df = mk({f"c{i}": 3 for i in range(400)})
    out = stratified_sample(df, 100)
    assert len(out) == 100
    assert out["label"].nunique() == 100


def test_single_class():
    out = stratified_sample(mk({"only": 1000}), 150)
    assert len(out) == 150


def test_no_label_column():
    df = pd.DataFrame({"file_path": [f"{i}.jpg" for i in range(500)]})
    assert len(stratified_sample(df, 120)) == 120


def test_nan_labels_grouped():
    df = mk({"a": 400})
    df.loc[:199, "label"] = None
    out = stratified_sample(df, 100)
    assert len(out) == 100


def test_reproducible():
    df = mk({"a": 500, "b": 500})
    a = stratified_sample(df, 100)["file_path"].tolist()
    b = stratified_sample(df, 100)["file_path"].tolist()
    assert a == b


def test_allocation_empty_and_zero():
    assert class_allocation({}, 10) == {}
    assert class_allocation({"a": 5}, 0) == {}
