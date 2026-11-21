"""MLX backend: batching, scoring, adapter keys and (on Apple silicon) the forward pass of a tiny random d3."""
import json

import pytest

from pusula import mlx_d3


def test_batches_cover_every_item_once_and_respect_the_budget():
    lengths = [5, 100, 7, 6, 98, 50, 51, 5]
    out = mlx_d3.batches(lengths, token_budget=200)
    flat = sorted(i for b in out for i in b)
    assert flat == list(range(len(lengths)))
    for b in out:
        assert max(lengths[i] for i in b) * len(b) <= 200 or len(b) == 1


def test_score_matches_the_d3train_report_fields():
    rows = [{"name": "a", "label": 0, "probs": [0.9, 0.1]}, {"name": "a", "label": 1, "probs": [0.8, 0.2]},
            {"name": "b", "label": 2, "probs": [0.1, 0.2, 0.7]}]
    r = mlx_d3.score(rows)
    assert r["n"] == 3 and r["acc"] == {"a": 0.5, "b": 1.0}
    assert r["micro_acc"] == round(2 / 3, 4) and r["macro_acc"] == 0.75
    assert 0 <= r["ece"] <= 1


def _rows(spec):
    """spec: [(item, name, label, predicted, options)] -> preds rows with a confident prediction."""
    out = {}
    for i, name, label, pred, k in spec:
        probs = [0.02 / (k - 1)] * k
        probs[pred] = 0.98
        out[i] = {"i": i, "name": name, "label": label, "probs": probs}
    return out


def test_combine_own_base_takes_only_the_base_questions_from_the_base_run():
    adapter = _rows([(0, "duygu", 1, 1, 3), (1, "ozel_soru", 0, 1, 2)])
    base = _rows([(0, "duygu", 1, 0, 3), (1, "ozel_soru", 0, 0, 2)])
    rows = mlx_d3.combine_own_base(adapter, base, ("ozel_soru", "eksik_bilgi"))   # v2.0 PUSULA_OWN_BASE=1
    assert rows[0] is adapter[0] and rows[1] is base[1]
    assert mlx_d3.score(list(rows.values()))["micro_acc"] == 1.0


def test_combine_own_base_default_follows_v21_templates_mode():
    adapter = _rows([(0, "ozel_soru", 0, 0, 2), (1, "eksik_bilgi", 1, 0, 2)])
    base = _rows([(0, "ozel_soru", 0, 1, 2), (1, "eksik_bilgi", 1, 1, 2)])
    rows = mlx_d3.combine_own_base(adapter, base)   # own questions keep the adapter, only eksik_bilgi goes to base d3
    assert rows[0] is adapter[0] and rows[1] is base[1]


def test_combine_and_paired_refuse_runs_on_different_questions():
    a = _rows([(0, "duygu", 1, 1, 3)])
    with pytest.raises(SystemExit):
        mlx_d3.combine_own_base(a, _rows([(1, "duygu", 1, 1, 3)]))
    with pytest.raises(SystemExit):
        mlx_d3.paired(a, _rows([(0, "duygu", 2, 1, 3)]))


def test_paired_counts_fixed_and_broken_answers():
    a = _rows([(0, "x", 0, 0, 2), (1, "x", 0, 1, 2), (2, "y", 1, 1, 2), (3, "y", 1, 0, 2)])
    b = _rows([(0, "x", 0, 1, 2), (1, "x", 0, 0, 2), (2, "y", 1, 1, 2), (3, "y", 1, 1, 2)])
    c = mlx_d3.paired(a, b)
    assert (c["all"]["fixed"], c["all"]["broken"]) == (2, 1)
    assert c["all"]["a"] == 0.5 and c["all"]["b"] == 0.75 and c["all"]["diff"] == 0.25
    assert c["per"]["y"] == {**c["per"]["y"], "fixed": 1, "broken": 0, "a": 0.5, "b": 1.0}
    assert mlx_d3.confusion(a, "y") == [[0, 0], [1, 1]]
    assert "| **micro / macro** | 4 |" in mlx_d3.paired_markdown(c, "A", "B")


def test_adapter_keys_map_to_text_layer_paths():
    m = mlx_d3._ADAPTER_KEY.match("base_model.model.language_model.layers.3.self_attn.q_proj.lora_B.weight")
    assert m and m.group(1) == "layers.3.self_attn.q_proj" and m.group(2) == "B"
    assert mlx_d3._ADAPTER_KEY.match("base_model.model.visual.blocks.0.attn.qkv.lora_A.weight") is None


def test_report_paths_hide_the_home_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert mlx_d3.shown_path(str(tmp_path / "models" / "d3-mlx-4bit")) == "~/models/d3-mlx-4bit"
    assert mlx_d3.shown_path("~/models/x") == "~/models/x"
    assert mlx_d3.shown_path("local/mlx/a.preds.jsonl") == "local/mlx/a.preds.jsonl"
    assert mlx_d3.shown_path(None) is None


def test_softmax_is_a_distribution():
    p = mlx_d3._softmax([1000.0, 1000.0, -1e4])
    assert abs(sum(p) - 1) < 1e-9 and abs(p[0] - 0.5) < 1e-9


# ---- forward pass (Apple silicon only) ------------------------------------------------------------------------

TINY = {
    "model_type": "qwen3_5",
    "text_config": {
        "model_type": "qwen3_5_text", "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "rms_norm_eps": 1e-6,
        "vocab_size": 97, "max_position_embeddings": 512, "full_attention_interval": 2,
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": 16, "linear_num_key_heads": 2,
        "linear_num_value_heads": 4, "linear_value_head_dim": 16, "tie_word_embeddings": True,
        "attn_output_gate": True, "partial_rotary_factor": 0.25,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                            "mrope_section": [2, 1, 1]},
    },
}


@pytest.fixture
def tiny():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    mx.random.seed(0)
    model = mlx_d3.build_model(json.loads(json.dumps(TINY)))
    mx.eval(model.parameters())

    class Tok:
        pad_token_id = 0

    def make(causal=False, switch=None, readout=None):
        switch = switch if switch is not None else {"on": False}
        r = readout if readout is not None else mx.random.normal((8, 64))
        return mlx_d3.MLXDecider(model, r, None, Tok(), list("ABCDEFGH"), switch, causal=causal)

    return mx, model, make


def test_left_padded_batch_matches_single_prompts(tiny):
    mx, _, make = tiny
    d = make()
    seqs = [[5, 6, 7, 8, 9, 10, 11], [12, 13, 14], [20, 21, 22, 23, 24]]
    batched = d.logits(seqs, [4, 4, 4])
    for row, s in enumerate(seqs):
        single = d.logits([s], [4])
        assert abs(batched[row] - single[0]).max() < 2e-3


def test_full_attention_is_noncausal(tiny):
    mx, _, make = tiny
    seq = [5, 6, 7, 8, 9, 10, 11, 12]
    readout = mx.random.normal((8, 64))   # one readout: only the mask differs between the two runs
    noncausal, causal = make(readout=readout).logits([seq], [4]), make(causal=True, readout=readout).logits([seq], [4])
    assert abs(noncausal - causal).max() > 1e-4


def test_base_temperature_divides_answers_without_an_adapter(tiny):
    mx, model, _ = tiny

    class Tok:
        pad_token_id = 0

    readout = mx.random.normal((8, 64))
    seq = [5, 6, 7, 8, 9]
    plain = mlx_d3.MLXDecider(model, readout, None, Tok(), list("ABCDEFGH"), {"on": False}).logits([seq], [4])
    warm = mlx_d3.MLXDecider(model, readout, None, Tok(), list("ABCDEFGH"), {"on": False},
                             base_temperature=2.0).logits([seq], [4])
    assert abs(warm - plain / 2).max() < 1e-4
    adapted = mlx_d3.MLXDecider(model, readout, readout, Tok(), list("ABCDEFGH"), {"on": False},
                                base_temperature=2.0, has_adapter=True)
    assert abs(adapted.logits([seq], [4]) - plain).max() < 1e-4             # adapter answer: no temperature
    assert abs(adapted.logits([seq], [4], base=True) - plain / 2).max() < 1e-4   # base answer: divided


def test_lora_switch_and_scale(tiny, tmp_path):
    mx, model, make = tiny
    import numpy as np
    from safetensors.numpy import save_file

    rng = np.random.default_rng(0)
    key = "base_model.model.language_model.layers.1.self_attn.q_proj"
    q_out = model.language_model.model.layers[1].self_attn.q_proj.weight.shape[0]
    save_file({f"{key}.lora_A.weight": rng.normal(size=(2, 64)).astype(np.float32),
               f"{key}.lora_B.weight": rng.normal(size=(q_out, 2)).astype(np.float32)},
              str(tmp_path / "adapter_model.safetensors"))
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4}))
    seq = [5, 6, 7, 8, 9]
    readout = mx.random.normal((8, 64))
    before = make(readout=readout).logits([seq], [4])
    switch = {"on": False}
    assert mlx_d3.apply_adapter(model, tmp_path, switch) == 1
    off = make(switch=switch, readout=readout).logits([seq], [4])
    switch["on"] = True
    on = make(switch=switch, readout=readout).logits([seq], [4])
    assert abs(off - before).max() < 1e-5
    assert abs(on - before).max() > 1e-4


# ---- model format, run trace, conversion (no MLX needed) ----------------------------------------------------

def _model_dir(path, **cfg):
    path.mkdir(parents=True, exist_ok=True)
    base = {"format_id": "d3-code-readout-v1", "attention_mode": "noncausal_full_attention", "codes": list("ABCD")}
    (path / "decision_config.json").write_text(json.dumps({**base, **cfg}), encoding="utf-8")
    return path


def test_load_refuses_d3_ultra_formats_before_loading_weights(tmp_path):
    with pytest.raises(ValueError, match="prefix_bidir"):
        mlx_d3.MLXDecider.load(str(_model_dir(tmp_path / "ultra", attention_mode="prefix_bidir",
                                              prompt_layout="header-v1")))
    with pytest.raises(ValueError, match="header-v1"):
        mlx_d3.MLXDecider.load(str(_model_dir(tmp_path / "layout", prompt_layout="header-v1")))


def test_load_refuses_an_adapter_of_another_base(tmp_path):
    model = _model_dir(tmp_path / "d3")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "pusula_d3.json").write_text(json.dumps({"attention_mode": "prefix_bidir", "prompt_layout": "header-v1"}))
    with pytest.raises(ValueError, match="only fits its own base model"):
        mlx_d3.MLXDecider.load(str(model), adapter=str(adapter))
    (adapter / "pusula_d3.json").write_text(json.dumps({"format_id": "d3-code-readout-v1"}))   # trained on d3
    fmt = mlx_d3.MLXDecider.check_format(model, adapter)
    assert fmt.codes == list("ABCD") and fmt.temperature == 1.0


class _Item:
    def __init__(self, name, label):
        self.name, self.label = name, label


def _resume(preds, trace, items, index):
    with mlx_d3.locked_append(preds) as f:
        return mlx_d3.resume_rows(f, preds, trace, items, index)


def test_resume_needs_the_same_run_trace(tmp_path):
    preds = tmp_path / "run.preds.jsonl"
    items = [_Item("duygu", 0), _Item("duygu", 1), _Item("aciliyet", 2)]
    trace = {"mlx": "~/m", "bits": 4, "data_sha256": "x", "base": False, "limit": None, "seed": None}
    assert _resume(preds, trace, items, [0, 1, 2]) == {}
    assert json.loads(mlx_d3.trace_path(preds).read_text()) == trace
    row = {"i": 0, "name": "duygu", "label": 0, "probs": [1.0, 0.0]}
    preds.write_text(json.dumps(row) + "\n" + '{"i": 1, "na', encoding="utf-8")   # killed mid-write
    assert _resume(preds, trace, items, [0, 1, 2]) == {0: row}
    assert preds.read_text(encoding="utf-8") == json.dumps(row) + "\n"
    with pytest.raises(SystemExit, match="base"):
        _resume(preds, {**trace, "base": True}, items, [0, 1, 2])
    with pytest.raises(SystemExit, match="not a question of this run"):
        _resume(preds, trace, items, [1, 2])                                   # another --limit subset
    with pytest.raises(SystemExit, match="not a question of this run"):
        _resume(preds, trace, [_Item("aciliyet", 0)] + items[1:], [0, 1, 2])   # another data file
    preds.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="twice"):
        _resume(preds, trace, items, [0, 1, 2])
    mlx_d3.trace_path(preds).unlink()
    with pytest.raises(SystemExit, match="no run trace"):
        _resume(preds, trace, items, [0, 1, 2])


def test_a_second_run_on_the_same_preds_file_stops(tmp_path):
    preds = tmp_path / "run.preds.jsonl"
    with mlx_d3.locked_append(preds):
        with pytest.raises(SystemExit, match="in use"):
            with mlx_d3.locked_append(preds):
                pass


def test_combine_sends_templates_recognized_by_content_to_base_d3():
    adapter = _rows([(0, "duygu", 1, 1, 3), (1, "eksik_bilgi_v2", 0, 1, 2)])
    adapter[1]["base_template"] = True    # eval matched the question to the eksik_bilgi template by its content
    base = _rows([(0, "duygu", 1, 0, 3), (1, "eksik_bilgi_v2", 0, 0, 2)])
    rows = mlx_d3.combine_own_base(adapter, base)
    assert rows[0] is adapter[0] and rows[1] is base[1]


def test_base_template_flags_match_by_name_and_content():
    from pusula.schema import Example
    from pusula.taxonomy import load_taxonomy

    universal = load_taxonomy().universal
    q = universal["eksik_bilgi"]
    other = next(v for k, v in universal.items() if k != "eksik_bilgi")
    keys = q.option_keys()
    ex = Example(id="x", state="durum", questions={"eksik_bilgi": q, "renamed": q, "other": other},
                 labels={"gold": {"eksik_bilgi": keys[0], "renamed": keys[0], "other": other.option_keys()[0]}})
    assert mlx_d3.base_template_flags([ex], "gold") == [True, True, False]


def test_base_templates_follow_the_gpu_engine():
    pytest.importorskip("torch")
    from pusula.inference import D3Engine

    assert mlx_d3.BASE_TEMPLATES == tuple(D3Engine.BASE_TEMPLATES)


def test_convert_refuses_an_existing_output_and_a_source_without_licenses(tmp_path):
    src = _model_dir(tmp_path / "d3")
    for name in ("tokenizer.json", "tokenizer_config.json", "readout.safetensors", "LICENSE"):
        (src / name).write_text("x")
    with pytest.raises(SystemExit, match="NOTICE"):
        mlx_d3.convert(str(src), str(tmp_path / "out"), 4)
    (tmp_path / "out").mkdir()
    with pytest.raises(SystemExit, match="already exists"):
        mlx_d3.convert(str(src), str(tmp_path / "out"), 4)


def test_missing_files_lists_what_an_interrupted_conversion_lacks(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text("{}")
    assert "model*.safetensors" in mlx_d3.missing_files(d) and "LICENSE" in mlx_d3.missing_files(d)
    for name in mlx_d3.REQUIRED_FILES:
        (d / name).write_text("x")
    (d / "model.safetensors").write_text("x")
    assert mlx_d3.missing_files(d) == []
