import numpy as np
import pandas as pd
import pytest

from subgroup_audit.audit import EnumerationAuditor, audit, enumerate_subgroups, subgroup_table


def test_enumeration_counts():
    g = pd.DataFrame({"a": ["x", "y", "x", "y"], "b": ["1", "1", "2", "2"]})
    descs, members = enumerate_subgroups(g, max_order=2)
    assert len(descs) == 2 + 2 + 4
    assert members.shape == (4, 8)
    assert members[:, [d["order"] == 2 for d in descs]].sum(axis=1).tolist() == [1, 1, 1, 1]


def test_xor_gerrymandering_detected(xor_data):
    y, p = xor_data["label"].to_numpy(), xor_data["prediction"].to_numpy()
    res = audit(y, p, xor_data[["sex", "race"]], "fpr", max_order=2)
    assert res["overall_rate"] == pytest.approx(0.25)
    # every single-attribute group is exactly fair...
    assert all(abs(r["gap"]) < 1e-12 for r in res["single_attribute"])
    # ...but every intersection is off by 15 points
    assert abs(res["worst_intersection"]["gap"]) == pytest.approx(0.15)
    # size weighting: alpha = 400 negatives / 3200 rows
    assert res["max_weighted_gap"] == pytest.approx(400 / 3200 * 0.15)


def test_weighting_ranks_large_groups_first():
    # small group with a huge gap vs. large group with a moderate gap
    y = np.zeros(1000, int)
    g = pd.DataFrame({"grp": ["small"] * 20 + ["large"] * 400 + ["rest"] * 580})
    p = np.r_[np.ones(20), np.r_[np.ones(200), np.zeros(200)], np.zeros(580) + (np.arange(580) < 58)]
    table, _ = subgroup_table(y, p, g, "fpr", max_order=1, min_size=30)
    order = table["subgroup"].tolist()
    assert order.index("grp=large") < order.index("grp=small")
    assert table.set_index("subgroup").loc["grp=small", "gap"] > table.set_index("subgroup").loc["grp=large", "gap"]
    assert not table.set_index("subgroup").loc["grp=small", "reliable"]


def test_enumeration_auditor_matches_table(xor_data):
    y, p = xor_data["label"].to_numpy(), xor_data["prediction"].to_numpy()
    g, signed, desc = EnumerationAuditor(xor_data[["sex", "race"]], 2).best_group(y, p, "fpr")
    assert abs(signed) == pytest.approx(400 / 3200 * 0.15)
    assert "&" in desc
