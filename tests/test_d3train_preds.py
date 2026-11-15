"""eval --preds: one row per question, in benchmark item order, consistent with the report (15 Nov 2026)."""
import pytest

torch = pytest.importorskip("torch")

from test_d3model import FakeDecider  # noqa: E402

from pusula.d3train import build_items, evaluate  # noqa: E402
from pusula.schema import Example, Question  # noqa: E402


class _Decider(FakeDecider):
    """Like D3Decider.logits: [B, max(counts)], padded options get -1e4."""

    def logits(self, ids, counts):
        rows = []
        for x, c in zip(ids, counts):
            g = torch.Generator().manual_seed(sum(v * (i + 1) for i, v in enumerate(x)) % (2**31))
            rows.append(torch.cat([torch.randn(c, generator=g), torch.full((max(counts) - c,), -1e4)]))
        return torch.stack(rows)


def _examples():
    out = []
    for k in range(7):
        # 5, 3 and 2 options: one micro-batch holds several option counts
        qs = {"ekip": Question(type="choice", instructions="Hangi ekip?", criteria={k: None for k in "abcde"}),
              "seviye": Question(type="score", instructions="Ne kadar acil?", criteria=["düşük", "orta", "yüksek"]),
              "acil": Question(type="noul", instructions="Acil mi?", criteria={"true": "Evet"})}
        out.append(Example(id=f"m{k}", state=f"Mesaj {k} " + "x" * (k * 7), questions=qs,
                           labels={"gold": {"seviye": str(k % 3), "ekip": "abcde"[k % 5], "acil": ("false", "true")[k % 2]}}))
    return out


def test_preds_follow_item_order_and_match_the_report():
    exs = _examples()
    items = build_items(exs, _Decider(), "gold", None)
    preds = []
    m = evaluate(_Decider(), items, 4000, preds=preds)
    assert len({it.count for it in items}) == 3
    # item order = messages in file order, each message's label keys in order (the benchmark's item index)
    assert [(r["item"], r["id"], r["q"]) for r in preds] == [
        (i, ex.id, q) for i, (ex, q) in enumerate((ex, q) for ex in exs for q in ex.labels["gold"])]
    for r in preds:
        ex = exs[int(r["id"][1:])]
        assert r["keys"] == ex.questions[r["q"]].option_keys() and len(r["p"]) == len(r["keys"])
        assert r["gold"] == ex.labels["gold"][r["q"]]
        assert r["pred"] == r["keys"][r["p"].index(max(r["p"]))]
        assert r["conf"] == pytest.approx(max(r["p"]), abs=1e-4) and sum(r["p"]) == pytest.approx(1, abs=1e-3)
    assert m["n"] == len(preds) == 21
    assert m["micro_acc"] == round(sum(r["pred"] == r["gold"] for r in preds) / 21, 4)
