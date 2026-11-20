"""Per-question prediction files of the MLX runs and of `d3train eval --preds` (GPU), side by side.

    python -m pusula.mlx_preds export local/mlx/4bit-pusula.preds.jsonl --data benchmark/tr_decision_bench_v1.jsonl \
        --out local/mlx/4bit-pusula.d3train-preds.jsonl
    python -m pusula.mlx_preds compare gpu-pusula.preds.jsonl local/mlx/4bit-pusula.preds.jsonl \
        --data benchmark/tr_decision_bench_v1.jsonl --a-label "GPU BF16" --b-label "MLX 4-bit" --confusion aciliyet

`export` writes an MLX preds file in the format of `d3train eval --preds`:
{"item", "id", "q", "gold", "pred", "conf", "keys", "p"}, where `keys` are the option keys of the question
(`Question.option_keys()`, the order the label index and the probabilities of the MLX file use) and `item` is the
benchmark order (messages in file order, then the `labels.gold` keys of each message). Every row is checked against
the benchmark: the question name and the gold key must match.

`compare` reads either format, keeps the questions that both files contain, and prints the paired table of
`mlx_d3.paired` plus how often the two runs give the same answer. This module does not load a model and does not
change `mlx_d3`, so it is safe to use while an MLX run is going.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from . import mlx_d3


def benchmark_items(data: str) -> list[dict]:
    """The benchmark questions in `item` order: {"item", "id", "q", "gold", "keys"} (as d3train.build_items)."""
    from .jsonl import read_examples

    out = []
    for ex in read_examples(data):
        for name, gold in ex.labels.get("gold", {}).items():
            out.append({"item": len(out), "id": ex.id, "q": name, "gold": gold,
                        "keys": ex.questions[name].option_keys()})
    return out


def _check(row: dict, it: dict, label: int) -> None:
    if row["name"] != it["q"] or not 0 <= label < len(it["keys"]) or it["keys"][label] != it["gold"]:
        raise SystemExit(f"item {it['item']}: the preds row ({row['name']}, label {label}) does not match the "
                         f"benchmark ({it['q']}, gold {it['gold']!r}, options {it['keys']})")


def export_rows(rows: dict[int, dict], items: list[dict]) -> list[dict]:
    """MLX rows {i, name, label, probs} -> d3train --preds rows, checked against the benchmark."""
    out = []
    for i in sorted(rows):
        r, it = rows[i], items[i]
        _check(r, it, r["label"])
        if len(r["probs"]) != len(it["keys"]):
            raise SystemExit(f"item {i}: {len(r['probs'])} probabilities for {len(it['keys'])} options")
        best = max(range(len(r["probs"])), key=r["probs"].__getitem__)
        out.append({"item": i, "id": it["id"], "q": it["q"], "gold": it["gold"], "pred": it["keys"][best],
                    "conf": r["probs"][best], "keys": it["keys"], "p": r["probs"]})
    return out


def load_any(path: str, items: Optional[list[dict]] = None) -> dict[int, dict]:
    """A preds file of either format as MLX rows {i, name, label, probs}. A d3train row is checked against the
    benchmark when `items` is given (same id, question and option keys)."""
    rows = {}
    text = Path(path).read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        text = text[: text.rfind("\n") + 1]  # a run that is still writing: its last line may be partial
    for line in text.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "item" in r:  # d3train eval --preds
            keys, gold = [str(k) for k in r["keys"]], str(r["gold"])
            prob = dict(zip(keys, r["p"]))  # probabilities are read by option name, never by position
            if len(prob) != len(keys) or len(keys) != len(r["p"]) or gold not in prob:
                raise SystemExit(f"{path}: item {r['item']}: keys {keys}, gold {gold!r} and p do not fit together")
            if items is not None:
                it = items[r["item"]]
                if (r["id"], r["q"], str(it["gold"])) != (it["id"], it["q"], gold) or set(keys) != set(it["keys"]):
                    raise SystemExit(f"{path}: item {r['item']} is ({r['id']}, {r['q']}, gold {gold!r}, {keys}), the "
                                     f"benchmark has ({it['id']}, {it['q']}, gold {it['gold']!r}, {it['keys']})")
                keys = it["keys"]  # benchmark (schema) order, so both files use the same label index
            row = {"i": r["item"], "name": r["q"], "label": keys.index(gold), "probs": [prob[k] for k in keys]}
        else:
            row = r
            if items is not None:
                _check(row, items[row["i"]], row["label"])
        rows[row["i"]] = row
    return rows


def agreement(a: dict[int, dict], b: dict[int, dict]) -> dict:
    """How often two runs on the same questions give the same answer, and how far apart their probabilities are
    (largest absolute difference over the options of a question), overall and per question type."""
    per: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for i in sorted(a):
        pa, pb = a[i]["probs"], b[i]["probs"]
        same = int(max(range(len(pa)), key=pa.__getitem__) == max(range(len(pb)), key=pb.__getitem__))
        per[a[i]["name"]].append((same, max(abs(x - y) for x, y in zip(pa, pb))))

    def block(v: list[tuple[int, float]]) -> dict:
        d = sorted(x[1] for x in v)
        return {"n": len(v), "same": sum(x[0] for x in v), "mean_dp": round(sum(d) / len(d), 4),
                "p95_dp": round(d[int(0.95 * (len(d) - 1))], 4), "max_dp": round(d[-1], 4)}

    return {"all": block([x for v in per.values() for x in v]), "per": {k: block(v) for k, v in sorted(per.items())}}


def mcnemar_p(fixed: int, broken: int) -> float:
    """Exact two-sided McNemar test: the probability of a split at least this uneven of the discordant questions
    (A wrong and B right against A right and B wrong) if both were equally likely."""
    from math import comb

    n, k = fixed + broken, min(fixed, broken)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def mcnemar_markdown(c: dict) -> str:
    rows = ["| question | n | A right, B wrong | A wrong, B right | McNemar p |", "|---|---|---|---|---|"]
    for name, v in [*c["per"].items(), ("**all**", c["all"])]:
        rows.append(f"| {name} | {v['n']} | {v['broken']} | {v['fixed']} | {mcnemar_p(v['fixed'], v['broken']):.3g} |")
    return "\n".join(rows) + "\n"


def agreement_markdown(g: dict) -> str:
    rows = ["| question | n | same answer | changed | mean max-dp | p95 max-dp | max max-dp |", "|---|---|---|---|---|---|---|"]
    for name, v in [*g["per"].items(), ("**all**", g["all"])]:
        rows.append(f"| {name} | {v['n']} | {100 * v['same'] / v['n']:.2f} | {v['n'] - v['same']} | "
                    f"{v['mean_dp']:.4f} | {v['p95_dp']:.4f} | {v['max_dp']:.4f} |")
    return "\n".join(rows) + "\n"


def run_export(a) -> None:
    rows = export_rows(mlx_d3.load_preds(a.preds), benchmark_items(a.data))
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} rows, every row checked against {a.data}: {a.out}")


def run_compare(a) -> None:
    items = benchmark_items(a.data)
    ra, rb = load_any(a.a, items), load_any(a.b, items)
    common = ra.keys() & rb.keys()
    if not common:
        raise SystemExit("the two files have no question in common")
    print(f"A: {len(ra)} questions, B: {len(rb)} questions, compared on the {len(common)} in both\n")
    ra = {i: ra[i] for i in common}
    rb = {i: rb[i] for i in common}
    c = mlx_d3.paired(ra, rb)
    text = mlx_d3.paired_markdown(c, a.a_label, a.b_label)
    g = agreement(ra, rb)
    text += "\nCorrectness, question by question (exact McNemar test on the questions only one run gets right):\n\n"
    text += mcnemar_markdown(c)
    text += "\nSame answer in both runs (max-dp: largest absolute probability difference over the options):\n\n"
    text += agreement_markdown(g)
    for name in a.confusion:
        text += f"\n`{name}` (rows: gold, columns: predicted)\n\n| gold | {a.a_label} | {a.b_label} |\n|---|---|---|\n"
        ca, cb = mlx_d3.confusion(ra, name), mlx_d3.confusion(rb, name)
        for gold in range(len(ca)):
            text += f"| {gold} | {', '.join(map(str, ca[gold]))} | {', '.join(map(str, cb[gold]))} |\n"
    print(text)
    if a.json:
        Path(a.json).write_text(json.dumps({"paired": c, "agreement": g, "common": len(common)}, indent=2),
                                encoding="utf-8")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m pusula.mlx_preds", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="an MLX preds file in the d3train eval --preds format")
    e.add_argument("preds", help=".preds.jsonl of an MLX run")
    e.add_argument("--data", required=True, help="the benchmark file the run used")
    e.add_argument("--out", required=True)
    k = sub.add_parser("compare", help="paired comparison of two preds files (MLX or d3train format)")
    k.add_argument("a", help="run A (reference)")
    k.add_argument("b", help="run B")
    k.add_argument("--data", required=True, help="the benchmark file both runs used")
    k.add_argument("--a-label", default="A")
    k.add_argument("--b-label", default="B")
    k.add_argument("--confusion", nargs="*", default=[], help="question types to print a confusion matrix for")
    k.add_argument("--json", default=None, help="also write the numbers to this file")
    a = p.parse_args(argv)
    run_export(a) if a.cmd == "export" else run_compare(a)


if __name__ == "__main__":
    main()
