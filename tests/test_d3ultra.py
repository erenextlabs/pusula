"""d3-ultra (prefix_bidir, header-v1) prompt and attention contract; no model weights needed."""
import json

import pytest

from pusula import d3model
from pusula.d3model import (
    HEADER_LAYOUT,
    NONCAUSAL,
    PREFIX_BIDIR,
    D3Decider,
    PromptIds,
    StateBlocks,
    adapter_format,
    block_attention,
    encode_prompts,
    load_format,
    prompt_for,
    state_end_char,
    state_token_count,
)
from pusula.schema import Question

CODES = [chr(ord("A") + i) for i in range(26)]
CHOICE = Question(type="choice", instructions="Hangi ekip?", criteria={"iade": "Ürün iadesi", "diger": None})


class Tok:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"

    def __call__(self, texts, add_special_tokens=False, return_offsets_mapping=False):
        out = {"input_ids": [[ord(c) % 1000 for c in t] for t in texts]}
        if return_offsets_mapping:
            out["offset_mapping"] = [[(i, i + 1) for i in range(len(t))] for t in texts]
        return out


class Decider:
    tokenizer = Tok()
    codes = CODES
    layout = HEADER_LAYOUT


def test_header_lists_the_question_before_the_state():
    text = prompt_for(Decider, "Kargo gelmedi", CHOICE)
    assert ("Questions in this request:\n1. Hangi ekip?\n   - iade: Ürün iadesi\n   - diger\n\nState:\nKargo gelmedi\n\n"
            "Question:\nHangi ekip?\n\nOptions:\nA: iade: Ürün iadesi\nB: diger\n\n") in text


def test_header_follows_the_shown_order_and_noul_lists_no_options():
    text = prompt_for(Decider, "x", CHOICE, order=[1, 0])
    assert "   - diger\n   - iade: Ürün iadesi\n\nState:" in text and "A: diger\nB: iade: Ürün iadesi" in text
    noul = Question(type="noul", instructions="Acil mi?", criteria={"true": "Evet"})
    assert "1. Acil mi?\n\nState:" in prompt_for(Decider, "x", noul)
    score = Question(type="score", instructions="Düzey?", criteria=["az", "çok"])
    assert "1. Düzey?\n   - az\n   - çok\n\nState:" in prompt_for(Decider, "x", score)


def test_plain_d3_prompt_is_unchanged():
    class Plain:
        tokenizer = Tok()
        codes = CODES

    assert "Questions in this request" not in prompt_for(Plain, "x", CHOICE)


def test_state_end_finds_the_state_even_with_a_marker_inside():
    state = "a\n\nQuestion:\nb"
    text = "<user>State:\n" + state + "\n\nQuestion:\nreal"
    assert text[: state_end_char(text, state)].endswith(state)
    empty = "<user>State:\n(empty)\n\nQuestion:\nq"
    assert empty[: state_end_char(empty, "")].endswith("(empty)")


def test_token_straddling_the_state_end_goes_to_the_question():
    assert state_token_count([(0, 3), (3, 6), (6, 9)], 7) == 2
    with pytest.raises(ValueError):
        state_token_count([(0, 3)], 1)


def test_encode_carries_state_counts_and_refuses_a_wrong_layout():
    prompt = prompt_for(Decider, "Kargo", CHOICE)
    row = encode_prompts(Tok(), [prompt], HEADER_LAYOUT, True)[0]
    assert isinstance(row, PromptIds) and 0 < row.state < len(row)
    assert prompt[: row.state].endswith("State:\nKargo")
    with pytest.raises(ValueError):  # rendered for plain d3, sent to a header-v1 model
        encode_prompts(Tok(), [d3model.render(Tok(), "Kargo", CHOICE, CODES)], HEADER_LAYOUT, True)


def _config(tmp_path, **kw):
    cfg = {"format_id": "d3-code-readout-v1", "codes": CODES, "attention_mode": NONCAUSAL, **kw}
    (tmp_path / "decision_config.json").write_text(json.dumps(cfg))
    return tmp_path


def test_load_format_reads_d3_and_d3_ultra(tmp_path):
    fmt = load_format(_config(tmp_path, temperature=1.0))
    assert (fmt.attention_mode, fmt.layout, fmt.temperature) == (NONCAUSAL, None, 1.0)
    table = {"max_questions": [1, 4, 16], "temperature": [1.046, 1.0, 1.0, 1.0]}
    fmt = load_format(_config(tmp_path, attention_mode=PREFIX_BIDIR, prompt_layout=HEADER_LAYOUT,
                              temperature=1.0, temperature_by_questions=table))
    assert (fmt.attention_mode, fmt.layout, fmt.temperature) == (PREFIX_BIDIR, HEADER_LAYOUT, 1.046)


@pytest.mark.parametrize("bad", [{"attention_mode": "causal"}, {"prompt_layout": "other"}, {"pooling": "mean"}])
def test_load_format_refuses_what_it_cannot_run(tmp_path, bad):
    with pytest.raises(ValueError):
        load_format(_config(tmp_path, **bad))


def test_adapter_format_of_old_and_new_adapters(tmp_path):
    assert adapter_format(tmp_path) == (NONCAUSAL, None)  # saved before d3-ultra: trained on d3
    (tmp_path / "pusula_d3.json").write_text(json.dumps({"attention_mode": PREFIX_BIDIR, "prompt_layout": HEADER_LAYOUT}))
    assert adapter_format(tmp_path) == (PREFIX_BIDIR, HEADER_LAYOUT)


def test_adapter_for_another_base_is_refused(tmp_path):
    pytest.importorskip("torch")
    d3 = tmp_path / "d3"
    d3.mkdir()
    _config(d3, attention_mode=PREFIX_BIDIR, prompt_layout=HEADER_LAYOUT)
    adapter = tmp_path / "v20"
    adapter.mkdir()
    with pytest.raises(ValueError, match="only fits its own base"):
        D3Decider.load(str(d3), adapter=str(adapter), device="cpu", load_in_4bit=False)


def test_block_attention_equals_the_dense_block_mask():
    torch = pytest.importorskip("torch")
    torch.manual_seed(3)
    rows, width, heads, dim = 3, 9, 2, 8
    q, k, v = (torch.randn(rows, heads, width, dim) for _ in range(3))
    blocks = StateBlocks.of(width, [9, 7, 5], [4, 3, 1])
    out, _ = block_attention(None, q, k, v, blocks, scaling=None)
    out = out.transpose(1, 2)
    for row, (first, end) in enumerate(blocks.bounds):
        mask = torch.zeros(width, width, dtype=torch.bool)
        mask[first:end, first:end] = True      # the state reads only the state
        mask[end:, first:] = True              # the question reads the state and itself
        dense = torch.nn.functional.scaled_dot_product_attention(q[row:row + 1], k[row:row + 1], v[row:row + 1],
                                                                 attn_mask=mask[None, None])
        assert torch.allclose(out[row:row + 1, :, first:], dense[:, :, first:], atol=1e-5)
        assert (out[row, :, :first] == 0).all()   # left padding is never read


def test_block_attention_expands_grouped_keys():
    torch = pytest.importorskip("torch")
    q, k, v = torch.randn(1, 4, 6, 8), torch.randn(1, 2, 6, 8), torch.randn(1, 2, 6, 8)
    out, _ = block_attention(None, q, k, v, StateBlocks.of(6, [6], [3]))
    assert out.shape == (1, 6, 4, 8)


def test_block_attention_needs_state_blocks_and_no_dropout():
    torch = pytest.importorskip("torch")
    x = torch.randn(1, 2, 4, 8)
    with pytest.raises(TypeError):
        block_attention(None, x, x, x, torch.ones(1, 1, 1, 4).bool())
    with pytest.raises(ValueError):
        block_attention(None, x, x, x, StateBlocks.of(4, [4], [2]), dropout=0.1)


def test_state_blocks_of_left_padded_rows():
    assert StateBlocks.of(10, [10, 6], [4, 2]).bounds == [(0, 4), (4, 6)]
    for lengths, states in (([5], [0]), ([5], [5]), ([11], [2])):
        with pytest.raises(ValueError):
            StateBlocks.of(10, lengths, states)


class _Out:
    def __init__(self, x):
        self.last_hidden_state = x


class _Backbone:
    def __call__(self, input_ids, attention_mask, use_cache):
        import torch

        return _Out(torch.ones((*input_ids.shape, 2)))


def _blocked(**kw):
    import torch

    return D3Decider(_Backbone(), torch.tensor([[1.0, 0.0], [0.0, 2.0]]), Tok(), ["A", "B"], torch.device("cpu"),
                     layout=HEADER_LAYOUT, attention_mode=PREFIX_BIDIR, blocks=[None], **kw)


def test_logits_of_a_blocked_model_need_state_counts():
    pytest.importorskip("torch")
    dec = _blocked()
    with pytest.raises(ValueError):
        dec.logits([[1, 2, 3]], [2])
    row = PromptIds([1, 2, 3])
    row.state = 2
    assert dec.logits([row], [2]).tolist() == [[1.0, 2.0]]
    assert dec.blocks[0].bounds == [(0, 2)]


def test_base_temperature_applies_to_base_answers_only():
    pytest.importorskip("torch")
    row = PromptIds([1, 2, 3])
    row.state = 2
    # no adapter: the base model answers, divided by its temperature
    assert _blocked(base_temperature=2.0).logits([row], [2]).tolist() == [[0.5, 1.0]]
    # an adapter is loaded: its own answer has no base temperature
    assert _blocked(base_temperature=2.0, has_adapter=True).logits([row], [2]).tolist() == [[1.0, 2.0]]


def test_blocked_scores_stay_fp32_under_bf16_autocast():
    torch = pytest.importorskip("torch")
    row = PromptIds([1, 2, 3])
    row.state = 2
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert _blocked().logits([row], [2]).dtype == torch.float32
