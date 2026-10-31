"""CPU checkpoint replay, including dropout, subsampling and epoch boundaries."""
import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from pusula import train as T  # noqa: E402
from pusula.model import PusulaModel  # noqa: E402


@pytest.fixture
def pilot(tmp_path, tiny_engine, example, choice_q, tokenizer, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    initial = tmp_path / "initial"
    tiny_engine.model.meta["base_model"] = "tiny"
    tiny_engine.model.save(initial, tokenizer, 128)
    example.questions = {"q": choice_q}
    example.labels = {"final": {"q": choice_q.option_keys()[0]}}
    data = "\n".join(example.model_dump_json() for _ in range(4)) + "\n"
    for split in ("train", "val"):
        (tmp_path / f"{split}.jsonl").write_text(data)
    return ["--data", str(tmp_path), "--init", str(initial), "--epochs", "2", "--max-length", "128",
            "--token-budget", "1", "--max-train-options", "2", "--save-every", "2", "--log-every", "2"]


@pytest.mark.parametrize("stop", [1, 4, 8])
def test_resume_matches_full(pilot, tmp_path, stop):
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    T.main([*pilot, "--out", str(full)])
    T.main([*pilot, "--out", str(resumed), "--max-steps", str(stop)])
    with (resumed / "train_log.jsonl").open("a") as log:
        log.write('{"step": 999}\n')  # Work logged after the last durable checkpoint is replayed.
    T.main([*pilot, "--out", str(resumed), "--resume"])
    for name in ("ckpt", "best"):
        a, _, _ = PusulaModel.load(full / name)
        b, _, _ = PusulaModel.load(resumed / name)
        for key, weight in a.state_dict().items():
            assert torch.allclose(weight, b.state_dict()[key], atol=1e-6, rtol=0)
    logs = [(path / "train_log.jsonl").read_text() for path in (full, resumed)]
    assert logs[0] == logs[1]
    assert len([row for row in logs[0].splitlines() if "step" in json.loads(row)]) == 4
    assert (full / "val_metrics.json").read_text() == (resumed / "val_metrics.json").read_text()


def test_init_loads_scorer(pilot, tmp_path, tiny_engine, monkeypatch):
    forward = T.forward_batch

    def check(model, *args):
        assert torch.equal(model.scorer.weight, tiny_engine.model.scorer.weight)
        assert model.meta["base_model"] == "tiny"
        monkeypatch.setattr(T, "forward_batch", forward)
        return forward(model, *args)

    monkeypatch.setattr(T, "forward_batch", check)
    T.main([*pilot, "--out", str(tmp_path / "init"), "--max-steps", "1"])


def test_legacy_namespace_and_init_length(pilot, tmp_path, tiny_engine, monkeypatch):
    """curate.flag_noise passes a hand-built Namespace without the checkpoint options; --max-length wins over --init."""
    import argparse
    import copy

    monkeypatch.setattr(T.PusulaModel, "from_base", classmethod(lambda cls, *a, **k: copy.deepcopy(tiny_engine.model)))
    ns = argparse.Namespace(data=pilot[1], out=str(tmp_path / "legacy"), base_model="tiny", max_length=96, epochs=1,
                            lr=3e-5, head_lr=1e-4, weight_decay=0.01, warmup=0.06, alpha=0.5, token_budget=1,
                            max_train_options=2, log_every=2, seed=7)
    T.train(ns)
    assert json.loads((tmp_path / "legacy" / "best" / "pusula.json").read_text())["max_length"] == 96
    T.main([*pilot, "--out", str(tmp_path / "init96"), "--max-length", "96", "--epochs", "1"])
    assert json.loads((tmp_path / "init96" / "best" / "pusula.json").read_text())["max_length"] == 96
