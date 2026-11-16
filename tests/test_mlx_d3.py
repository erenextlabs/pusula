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
    _, _, make = tiny
    seq = [5, 6, 7, 8, 9, 10, 11, 12]
    noncausal, causal = make().logits([seq], [4]), make(causal=True).logits([seq], [4])
    assert abs(noncausal - causal).max() > 1e-4


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
