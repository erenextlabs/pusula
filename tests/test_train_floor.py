"""Student --min-risky: the floor stops a run before any model loads, a mix above it goes on (12 Nov 2026, Nano 2.1)."""
import pytest

pytest.importorskip("torch")

from pusula import train as T  # noqa: E402
from pusula.schema import Example  # noqa: E402
from pusula.taxonomy import load_taxonomy  # noqa: E402


class Loaded(Exception):
    pass


@pytest.fixture
def data(tmp_path):
    q = load_taxonomy().universal["guvenlik"]
    rows = [Example(id=f"m{i}", state=f"Mesaj {i}", questions={"guvenlik": q},
                    labels={"final": {"guvenlik": "prompt_enjeksiyonu" if i == 0 else "guvenli"}}) for i in range(10)]
    text = "\n".join(r.model_dump_json() for r in rows) + "\n"
    for split in ("train", "val"):
        (tmp_path / f"{split}.jsonl").write_text(text)
    return tmp_path


@pytest.fixture
def no_model(monkeypatch):
    def boom(*a, **k):
        raise Loaded

    monkeypatch.setattr(T.PusulaModel, "load", boom)
    monkeypatch.setattr(T.PusulaModel, "from_base", boom)
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", boom)


def run(data, floors):
    T.main(["--data", str(data), "--out", str(data / "out"), "--min-risky", floors])


def test_floor_stops_before_model(data, no_model):
    with pytest.raises(SystemExit, match="guvenlik: riskli pay %10"):
        run(data, "guvenlik=10.5")
    assert not (data / "out").exists()


def test_floor_met_goes_on(data, no_model):
    with pytest.raises(Loaded):  # the check passed and the run went on to load a model
        run(data, "guvenlik=10")


def test_missing_name_and_bad_text_stop(data, no_model):
    with pytest.raises(SystemExit, match="dolandiricilik: eğitimde yok"):
        run(data, "dolandiricilik=1")
    with pytest.raises(SystemExit, match="tanınmadı"):
        run(data, "guvenlik=abc")


def test_no_floor_is_unchanged(data, no_model):
    with pytest.raises(Loaded):
        run(data, "")
