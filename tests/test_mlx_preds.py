"""MLX preds <-> d3train eval --preds format, and the agreement of two runs (no model, no MLX needed)."""
import json
from pathlib import Path

import pytest

from pusula import mlx_preds

BENCH = Path(__file__).resolve().parents[1] / "benchmark" / "tr_decision_bench_v1.jsonl"


@pytest.fixture()
def data(tmp_path):
    lines = BENCH.read_text(encoding="utf-8").splitlines()[:2]
    p = tmp_path / "bench.jsonl"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


def _mlx_rows(items, wrong=()):
    """One confident MLX row per benchmark question; items in `wrong` predict another option."""
    rows = {}
    for it in items:
        k, label = len(it["keys"]), it["keys"].index(it["gold"])
        pred = (label + 1) % k if it["item"] in wrong else label
        probs = [0.1 / (k - 1)] * k
        probs[pred] = 0.9
        rows[it["item"]] = {"i": it["item"], "name": it["q"], "label": label, "probs": probs}
    return rows


def test_export_round_trips_through_the_d3train_format(data, tmp_path):
    items = mlx_preds.benchmark_items(data)
    rows = _mlx_rows(items, wrong={1})
    out = mlx_preds.export_rows(rows, items)
    assert out[0]["gold"] == items[0]["gold"] and out[0]["pred"] == items[0]["gold"]
    assert out[1]["pred"] != out[1]["gold"] and out[1]["keys"] == items[1]["keys"]
    f = tmp_path / "gpu.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in out) + '{"item": 9', encoding="utf-8")  # partial last line
    back = mlx_preds.load_any(str(f), items)
    assert {i: {k: v for k, v in r.items() if k != "pred"} for i, r in back.items()} == rows
    assert back[1]["pred"] == items[1]["keys"].index(out[1]["pred"])


def test_export_refuses_rows_that_do_not_match_the_benchmark(data):
    items = mlx_preds.benchmark_items(data)
    rows = _mlx_rows(items)
    rows[0] = {**rows[0], "label": (rows[0]["label"] + 1) % len(items[0]["keys"])}
    with pytest.raises(SystemExit):
        mlx_preds.export_rows(rows, items)


def test_agreement_counts_changed_answers(data):
    items = mlx_preds.benchmark_items(data)
    a, b = _mlx_rows(items), _mlx_rows(items, wrong={0})
    g = mlx_preds.agreement(a, b)
    assert g["all"]["n"] == len(items) and g["all"]["same"] == len(items) - 1
    assert g["per"][items[0]["q"]]["max_dp"] == pytest.approx(0.9 - 0.1 / (len(items[0]["keys"]) - 1), abs=1e-4)


def test_d3train_rows_are_read_by_option_name_and_a_gold_mismatch_stops(data, tmp_path):
    items = mlx_preds.benchmark_items(data)
    rows = _mlx_rows(items, wrong={0})
    out = mlx_preds.export_rows(rows, items)
    r = out[0]
    order = list(range(len(r["keys"])))[::-1]  # same options, another order
    shuffled = {**r, "keys": [r["keys"][j] for j in order], "p": [r["p"][j] for j in order]}
    f = tmp_path / "shuffled.jsonl"
    f.write_text(json.dumps(shuffled) + "\n", encoding="utf-8")
    assert mlx_preds.load_any(str(f), items)[0] == {**rows[0], "pred": (rows[0]["label"] + 1) % len(r["keys"])}
    other = next(k for k in r["keys"] if k != r["gold"])
    f.write_text(json.dumps({**r, "gold": other}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        mlx_preds.load_any(str(f), items)


def test_mcnemar_exact_p():
    assert mlx_preds.mcnemar_p(0, 0) == 1.0
    assert mlx_preds.mcnemar_p(5, 0) == pytest.approx(2 / 32)
    assert mlx_preds.mcnemar_p(3, 3) == 1.0
    assert mlx_preds.mcnemar_p(154, 83) < 1e-5


def test_d3train_rows_keep_the_recorded_answer_when_probabilities_tie(data, tmp_path):
    from pusula import mlx_d3

    items = mlx_preds.benchmark_items(data)
    it = next(x for x in items if len(x["keys"]) >= 2)
    keys = it["keys"]
    p = [0.0] * len(keys)
    p[0] = p[1] = 0.5   # rounded to 5 digits the two options tie; the run had chosen the second one
    row = {"item": it["item"], "id": it["id"], "q": it["q"], "gold": it["gold"], "pred": keys[1], "conf": 0.5,
           "keys": keys, "p": p}
    f = tmp_path / "gpu.jsonl"
    f.write_text(json.dumps(row) + "\n", encoding="utf-8")
    r = mlx_preds.load_any(str(f), items)[it["item"]]
    assert r["pred"] == 1 and mlx_d3._pred(r) == 1
    ok = int(keys[1] == it["gold"])
    assert mlx_d3.score([r])["micro_acc"] == ok
    mlx = {"i": it["item"], "name": it["q"], "label": r["label"], "probs": [0.0] * len(keys)}
    mlx["probs"][1] = 1.0
    assert mlx_preds.agreement({it["item"]: r}, {it["item"]: mlx})["all"]["same"] == 1


def test_d3train_rows_with_an_unknown_answer_stop(data, tmp_path):
    items = mlx_preds.benchmark_items(data)
    it = items[0]
    row = {"item": 0, "id": it["id"], "q": it["q"], "gold": it["gold"], "pred": "no such option", "conf": 1.0,
           "keys": it["keys"], "p": [1.0 / len(it["keys"])] * len(it["keys"])}
    f = tmp_path / "gpu.jsonl"
    f.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        mlx_preds.load_any(str(f), items)
