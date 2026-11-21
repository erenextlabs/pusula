"""Pusula on Apple silicon: Decision 3.0 (vllm-sr/d3) and the Pusula adapter with MLX.

    python -m pusula.mlx_d3 convert --d3 ~/d3 --out ~/d3-mlx-4bit --bits 4
    python -m pusula.mlx_d3 eval --mlx ~/d3-mlx-4bit --adapter ~/pusula \
        --data benchmark/tr_decision_bench_v1.jsonl --out local/mlx/pusula_mlx_4bit.json

`convert` quantizes the base d3 text backbone once. The Pusula LoRA adapter (PEFT format, as published) is applied
at run time in full precision on top of the quantized weights, as on the GPU 4-bit path (NF4 base + LoRA), so the
base Decision 3.0 answer (adapter off, base readout) stays available for own_base routing.

The forward pass follows d3model: the softmax-attention layers are non-causal, the Gated DeltaNet layers stay
causal, and the final-norm hidden state at the last prompt position is multiplied by the 255-row code readout.
Requires `pip install -e ".[mlx]"` on an Apple silicon Mac.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

from .d3model import NONCAUSAL, adapter_format, load_format

# files copied next to the MLX weights: tokenizer, d3 prompt contract and readout, licenses
COPY_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
              "decision_config.json", "readout.safetensors", "LICENSE", "NOTICE")
# a converted directory is complete only with all of these (and at least one model*.safetensors)
REQUIRED_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "decision_config.json",
                  "readout.safetensors", "LICENSE", "NOTICE")
# own_base routing of v2.1 (PUSULA_OWN_BASE=templates): the questions inference.D3Engine.BASE_TEMPLATES sends to base
# d3, recognized by name or by content. Kept here because pusula.inference imports torch; a test checks both match.
BASE_TEMPLATES = ("eksik_bilgi",)
_ADAPTER_KEY = re.compile(r"^(?:base_model\.model\.)?(?:model\.)?language_model\.(layers\.\d+\..+)\.lora_([AB])\.weight$")


# ---- conversion ------------------------------------------------------------------------------------

def _text_weights(d3_dir: Path) -> dict:
    """The language-model weights of a d3 checkpoint under the keys mlx-lm's qwen3_5 sanitize expects
    (``model.language_model.*``); the vision tower, the MTP head and any LM head are left out (d3 reads codes
    through its own readout, never through the vocabulary)."""
    import mlx.core as mx

    out = {}
    for shard in sorted(d3_dir.glob("model*.safetensors")):
        for key, value in mx.load(str(shard)).items():
            m = re.match(r"^(?:model\.)?language_model\.(.+)$", key)
            if m is None or m.group(1).startswith(("mtp.", "lm_head")):
                continue
            out[f"model.language_model.{m.group(1)}"] = value
    if not out:
        raise ValueError(f"no language-model weights found in {d3_dir}")
    return out


def _text_config(d3_dir: Path) -> dict:
    from mlx_lm.utils import load_config

    config = load_config(d3_dir)
    text = dict(config.get("text_config", config))
    # no LM head is loaded or used; with tied embeddings mlx-lm builds none (d3 reads its codes through the readout)
    text["tie_word_embeddings"] = True
    config["text_config"] = text
    return config


def build_model(config: dict):
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    return Model(ModelArgs.from_dict(config))


def missing_files(mlx_dir: str | Path) -> list[str]:
    """The files a complete `convert` output has and `mlx_dir` lacks (empty: the directory is complete). With a
    sharded checkpoint every shard named in model.safetensors.index.json must be there."""
    path = Path(mlx_dir).expanduser()
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    index = path / "model.safetensors.index.json"
    if index.is_file():
        shards = sorted(set(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values()))
        missing += [name for name in shards if not (path / name).is_file()] or ([] if shards else [index.name])
    elif not any(path.glob("model*.safetensors")):
        missing.append("model*.safetensors")
    return missing


def convert(d3: str, out: str, bits: int, group_size: int = 64) -> Path:
    """Convert into `<out>.partial`, check that every required file is there, then rename it to `out`: an interrupted
    conversion never leaves a directory that looks complete."""
    src, dst = Path(d3).expanduser(), Path(out).expanduser()
    if dst.exists():
        raise SystemExit(f"{dst} already exists: remove it first")
    lacking = [name for name in REQUIRED_FILES[1:] if not (src / name).is_file()]
    if lacking:
        raise SystemExit(f"{src} lacks {', '.join(lacking)}")
    load_format(src)  # a d3 format this module can read
    tmp = dst.with_name(dst.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    _convert_into(src, tmp, bits, group_size)
    missing = missing_files(tmp)
    if missing:
        raise SystemExit(f"conversion in {tmp} is incomplete, missing {', '.join(missing)}")
    tmp.rename(dst)
    return dst


def _convert_into(src: Path, dst: Path, bits: int, group_size: int) -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.utils import quantize_model, save_config, save_model

    # Freed source buffers go back to the OS between shards instead of staying in MLX's cache.
    mx.set_cache_limit(0)
    config = _text_config(src)
    model = build_model(config)
    # No outside reference to the lazy bf16 source arrays: once a saved shard is evaluated its sources can
    # be freed, so the peak stays near one shard instead of the whole ~50 GB checkpoint.
    model.load_weights(list(model.sanitize(_text_weights(src)).items()), strict=True)
    if bits < 16:
        model, config = quantize_model(model, config, group_size, bits)
    else:
        model.update(dict((k, v.astype(mx.bfloat16)) for k, v in tree_flatten(model.parameters())))
    save_model(dst, model, donate_model=True)
    config["pusula_mlx"] = {"source": str(config.get("_name_or_path", "vllm-sr/d3")), "bits": bits,
                            "group_size": group_size, "format_id": "d3-code-readout-v1"}
    save_config(config, dst / "config.json")
    for name in COPY_FILES:
        if (src / name).is_file():
            shutil.copy(src / name, dst / name)


# ---- model -----------------------------------------------------------------------------------------

def _lora_class():
    import mlx.core as mx
    import mlx.nn as nn

    class SwitchableLoRA(nn.Module):
        """A (quantized) linear layer plus a LoRA update that a shared switch turns on and off."""

        def __init__(self, base, a, b, scale: float, switch: dict):
            super().__init__()
            self.base = base
            self.lora_a = a.T.astype(mx.float32)  # [in, r]
            self.lora_b = b.T.astype(mx.float32)  # [r, out]
            self.scale = scale
            self._switch = switch

        def __call__(self, x):
            y = self.base(x)
            if not self._switch["on"]:
                return y
            z = (x.astype(mx.float32) @ self.lora_a) @ self.lora_b
            return (y.astype(mx.float32) + self.scale * z).astype(y.dtype)

    return SwitchableLoRA


def apply_adapter(model, adapter_dir: Path, switch: dict) -> int:
    """Wrap every LoRA target of a PEFT adapter; returns the number of wrapped layers."""
    import mlx.core as mx

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if cfg.get("use_dora") or cfg.get("use_rslora") or cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise ValueError("only plain LoRA adapters are supported (no DoRA, rsLoRA or per-layer ranks)")
    scale = cfg["lora_alpha"] / cfg["r"]
    pairs: dict[str, dict[str, object]] = defaultdict(dict)
    for key, value in mx.load(str(adapter_dir / "adapter_model.safetensors")).items():
        m = _ADAPTER_KEY.match(key)
        if m is None:
            raise ValueError(f"unexpected adapter tensor {key}")
        pairs[m.group(1)][m.group(2)] = value
    lora = _lora_class()
    text = model.language_model.model
    for path, ab in pairs.items():
        parent_path, leaf = path.rsplit(".", 1)
        parent = text
        for part in parent_path.split("."):
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        base = getattr(parent, leaf)
        a, b = ab["A"], ab["B"]
        setattr(parent, leaf, lora(base, a, b, scale, switch))
    mx.eval(model.parameters())
    return len(pairs)


class MLXDecider:
    """d3 on MLX with the same interface as d3model.D3Decider (tokenizer, codes, encode, logits) for evaluation."""

    def __init__(self, model, readout, base_readout, tokenizer, codes: list[str], switch: dict, causal: bool = False,
                 base_temperature: float = 1.0, has_adapter: bool = False):
        self.model = model
        self.readout = readout            # [255, hidden] fp32
        self.base_readout = base_readout  # with an adapter: the base model's readout (logits(base=True))
        self.tokenizer = tokenizer
        self.codes = codes
        self._switch = switch
        self.causal = causal              # debugging only: d3 is non-causal in its full-attention layers
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.base_temperature = base_temperature  # as d3model.D3Decider: divides base answers (adapter off or absent)
        self.has_adapter = has_adapter

    @staticmethod
    def check_format(path: Path, adapter: Optional[Path] = None):
        """The model format of decision_config.json (d3model.load_format) when this module runs it as the GPU path does;
        otherwise a ValueError, so a d3-ultra model or an adapter of another base never runs with the wrong prompt or mask."""
        fmt = load_format(path)
        if fmt.attention_mode != NONCAUSAL or fmt.layout is not None:
            raise ValueError(f"{path}: attention mode {fmt.attention_mode!r} with prompt layout {fmt.layout!r} is not "
                             f"supported on MLX yet (only {NONCAUSAL!r} without a prompt layout, as vllm-sr/d3)")
        if adapter is not None and adapter_format(adapter) != (fmt.attention_mode, fmt.layout):
            raise ValueError(f"adapter {adapter} was trained for {adapter_format(adapter)}, the base model at {path} is "
                             f"{(fmt.attention_mode, fmt.layout)}: an adapter only fits its own base model")
        return fmt

    @classmethod
    def load(cls, mlx_dir: str, adapter: Optional[str] = None, causal: bool = False) -> "MLXDecider":
        path = Path(mlx_dir).expanduser()
        adapter_dir = Path(adapter).expanduser() if adapter else None
        fmt = cls.check_format(path, adapter_dir)

        import mlx.core as mx
        from transformers import AutoTokenizer

        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        model = build_model(config)
        if "quantization" in config:
            import mlx.nn as nn
            q = config["quantization"]
            weights = mx.load(str(path / "model.safetensors")) if (path / "model.safetensors").is_file() else {}
            for shard in sorted(path.glob("model-*.safetensors")):
                weights.update(mx.load(str(shard)))
            nn.quantize(model, group_size=q["group_size"], bits=q["bits"],
                        class_predicate=lambda p, m: hasattr(m, "to_quantized") and f"{p}.scales" in weights)
        else:
            weights = {}
            for shard in sorted(path.glob("model*.safetensors")):
                weights.update(mx.load(str(shard)))
        model.load_weights(list(weights.items()), strict=True)
        model.eval()
        mx.eval(model.parameters())
        switch = {"on": False}
        base_readout = mx.load(str(path / "readout.safetensors"))["weight"].astype(mx.float32)
        readout = base_readout
        if adapter_dir is not None:
            apply_adapter(model, adapter_dir, switch)
            readout = mx.load(str(adapter_dir / "readout.safetensors"))["weight"].astype(mx.float32)
            switch["on"] = True
        tokenizer = AutoTokenizer.from_pretrained(str(path))
        return cls(model, readout, base_readout if adapter_dir else None, tokenizer, fmt.codes, switch, causal,
                   base_temperature=fmt.temperature, has_adapter=adapter_dir is not None)

    def encode(self, prompts: Sequence[str]) -> list[list[int]]:
        return self.tokenizer(list(prompts), add_special_tokens=False)["input_ids"]

    def hidden(self, sequences: Sequence[Sequence[int]]):
        """Final-norm hidden state at the last position [B, hidden] for left-padded prompts."""
        import mlx.core as mx

        width = max(len(s) for s in sequences)
        padded = any(len(s) != width for s in sequences)
        ids = [[self.pad_id] * (width - len(s)) + list(s) for s in sequences]
        x = mx.array(ids, dtype=mx.int32)
        present = mx.array([[False] * (width - len(s)) + [True] * len(s) for s in sequences])
        if self.causal:
            fa_mask = mx.tril(mx.ones((width, width), dtype=mx.bool_))[None, None] & present[:, None, None, :]
        else:
            fa_mask = present[:, None, None, :] if padded else None
        ssm_mask = present if padded else None
        text = self.model.language_model.model
        h = text.embed_tokens(x)
        for layer in text.layers:
            h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=None)
        return text.norm(h[:, -1:, :])[:, 0]

    def logits(self, sequences: Sequence[Sequence[int]], counts: Sequence[int], base: bool = False):
        """Code logits [B, max(counts)] as a float32 numpy array; padded options get -1e4."""
        import mlx.core as mx
        import numpy as np

        use_base = base and self.base_readout is not None
        on = self._switch["on"]
        self._switch["on"] = on and not use_base
        try:
            k = max(counts)
            h = self.hidden(sequences).astype(mx.float32)
            out = h @ (self.base_readout if use_base else self.readout)[:k].T
            if use_base or not self.has_adapter:
                out = out / self.base_temperature
            mx.eval(out)
        finally:
            self._switch["on"] = on
        logits = np.array(out, dtype=np.float32)
        logits[np.arange(k)[None] >= np.asarray(counts)[:, None]] = -1e4
        return logits


# ---- evaluation ------------------------------------------------------------------------------------

def batches(lengths: Sequence[int], token_budget: int, max_padding: float = 0.25) -> list[list[int]]:
    """Index batches over length-sorted items: padded tokens within the budget, padding at most `max_padding`."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    out, start = [], 0
    while start < len(order):
        end, real = start + 1, lengths[order[start]]
        while end < len(order):
            n, width = end - start + 1, lengths[order[end]]
            if width * n > token_budget or width * n > (real + width) * (1 + max_padding):
                break
            real += width
            end += 1
        out.append(order[start:end])
        start = end
    return out


def score(rows: list[dict], bins: int = 15) -> dict:
    """The report of d3train.evaluate from per-item predictions: macro and micro accuracy, ECE, per question."""
    per_q: dict[str, list[int]] = defaultdict(list)
    conf, ok_all = [], []
    for r in rows:
        ok = int(_pred(r) == r["label"])
        per_q[r["name"]].append(ok)
        conf.append(max(r["probs"]))
        ok_all.append(ok)
    acc = {k: round(sum(v) / len(v), 4) for k, v in sorted(per_q.items())}
    ece = 0.0
    for b in range(bins):
        sel = [i for i, c in enumerate(conf) if b / bins < c <= (b + 1) / bins]
        if sel:
            ece += len(sel) / len(conf) * abs(sum(ok_all[i] for i in sel) / len(sel) - sum(conf[i] for i in sel) / len(sel))
    return {"n": len(ok_all), "macro_acc": round(sum(acc.values()) / max(len(acc), 1), 4),
            "micro_acc": round(sum(ok_all) / max(len(ok_all), 1), 4), "ece": round(ece, 4), "acc": acc}


def load_preds(path) -> dict[int, dict]:
    """A .preds.jsonl file of `eval` as {item index: row}."""
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["i"]] = r
    return rows


def _same_items(a: dict[int, dict], b: dict[int, dict]) -> None:
    if a.keys() != b.keys():
        raise SystemExit(f"the two runs cover different questions ({len(a)} and {len(b)} items, "
                         f"{len(a.keys() ^ b.keys())} not in both)")
    bad = [i for i in a if (a[i]["name"], a[i]["label"]) != (b[i]["name"], b[i]["label"])]
    if bad:
        raise SystemExit(f"{len(bad)} items differ in question name or gold label, e.g. item {bad[0]}")


def combine_own_base(adapter_rows: dict[int, dict], base_rows: dict[int, dict],
                     base_questions: Sequence[str] = ("eksik_bilgi",)) -> dict[int, dict]:
    """The own_base setup from two runs: the questions in `base_questions`, and every question `eval` marked as a
    BASE_TEMPLATES template by content (`base_template` in its row, the rule of inference.D3Engine._use_base), are
    answered by base d3 (adapter off), every other question by the adapter. Valid because the adapter switched off is exactly base d3 (checked with
    `eval --base`, see benchmark/README_MLX.md). The default follows PUSULA_OWN_BASE=templates, the v2.1 default
    (inference.D3Engine._use_base): only the eksik_bilgi template goes to base d3. The v2.0 default ("1") also sends
    own questions there; on the benchmark that is `ozel_soru`, so it is ("ozel_soru", "eksik_bilgi")."""
    _same_items(adapter_rows, base_rows)
    return {i: (base_rows[i] if r["name"] in base_questions or r.get("base_template") else r)
            for i, r in adapter_rows.items()}


def _pred(r: dict) -> int:
    """The answer of a row: the recorded one when the file has it (d3train --preds rows, read by mlx_preds.load_any:
    their rounded probabilities can tie), else the most likely option."""
    if "pred" in r:
        return r["pred"]
    p = r["probs"]
    return max(range(len(p)), key=p.__getitem__)


def paired(a_rows: dict[int, dict], b_rows: dict[int, dict]) -> dict:
    """Run B against run A on the same questions: accuracy, B - A with a paired standard error, fixed and broken
    answers (A wrong and B right, A right and B wrong), per question type and overall."""
    _same_items(a_rows, b_rows)

    def block(items: list[int]) -> dict:
        n = len(items)
        oa = [int(_pred(a_rows[i]) == a_rows[i]["label"]) for i in items]
        ob = [int(_pred(b_rows[i]) == b_rows[i]["label"]) for i in items]
        d = [y - x for x, y in zip(oa, ob)]
        mean = sum(d) / n
        se = math.sqrt(sum((v - mean) ** 2 for v in d) / (n - 1) / n) if n > 1 else 0.0
        fixed, broken = d.count(1), d.count(-1)
        return {"n": n, "a": round(sum(oa) / n, 4), "b": round(sum(ob) / n, 4), "diff": round(mean, 4),
                "se": round(se, 4), "fixed": fixed, "broken": broken,
                "z": round(mean / se, 2) if se else None}

    by_name: dict[str, list[int]] = defaultdict(list)
    for i in sorted(a_rows):
        by_name[a_rows[i]["name"]].append(i)
    per = {name: block(items) for name, items in sorted(by_name.items())}
    return {"all": block(sorted(a_rows)), "per": per,
            "macro_a": round(sum(v["a"] for v in per.values()) / len(per), 4),
            "macro_b": round(sum(v["b"] for v in per.values()) / len(per), 4),
            "ece_a": score(list(a_rows.values()))["ece"], "ece_b": score(list(b_rows.values()))["ece"]}


def confusion(rows: dict[int, dict], name: str) -> list[list[int]]:
    """Rows: gold label, columns: predicted label, for one question type."""
    sel = [r for r in rows.values() if r["name"] == name]
    k = max(len(r["probs"]) for r in sel) if sel else 0
    m = [[0] * k for _ in range(k)]
    for r in sel:
        m[r["label"]][_pred(r)] += 1
    return m


def _pct(x: float) -> str:
    return f"{100 * x:.2f}"


def paired_markdown(c: dict, a_label: str, b_label: str) -> str:
    rows = [f"| question | n | {a_label} | {b_label} | difference | fixed | broken |", "|---|---|---|---|---|---|---|"]
    for name, v in c["per"].items():
        rows.append(f"| {name} | {v['n']} | {_pct(v['a'])} | {_pct(v['b'])} | "
                    f"{100 * v['diff']:+.1f} ± {100 * v['se']:.1f} | {v['fixed']} | {v['broken']} |")
    t = c["all"]
    rows.append(f"| **micro / macro** | {t['n']} | **{_pct(t['a'])} / {_pct(c['macro_a'])}** | "
                f"**{_pct(t['b'])} / {_pct(c['macro_b'])}** | **{100 * t['diff']:+.2f} ± {100 * t['se']:.2f}** | "
                f"{t['fixed']} | {t['broken']} |")
    rows.append(f"| ECE | | {c['ece_a']:.3f} | {c['ece_b']:.3f} | | | |")
    return "\n".join(rows) + f"\n\nPaired difference over all questions: z = {t['z']}.\n"


def shown_path(path: Optional[str]) -> Optional[str]:
    """A path as written in a report: under the home folder it starts with `~`, so reports carry no user name."""
    if not path:
        return path
    p, home = Path(path).expanduser(), Path.home()
    try:
        return str(Path("~") / p.relative_to(home))
    except ValueError:
        return str(path)


def run_combine(a) -> None:
    rows = combine_own_base(load_preds(a.adapter_preds), load_preds(a.base_preds), a.base_questions)
    out = Path(a.out)
    preds = out.with_suffix(".preds.jsonl")
    preds.write_text("".join(json.dumps(rows[i]) + "\n" for i in sorted(rows)), encoding="utf-8")
    report = {"model": "pusula-d3-mlx-own-base", "adapter_preds": shown_path(a.adapter_preds),
              "base_preds": shown_path(a.base_preds),
              "base_questions": list(a.base_questions), **score([rows[i] for i in sorted(rows)])}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def run_compare(a) -> None:
    ra, rb = load_preds(a.a), load_preds(a.b)
    c = paired(ra, rb)
    text = paired_markdown(c, a.a_label, a.b_label)
    for name in a.confusion:
        text += f"\n`{name}` (rows: gold, columns: predicted)\n\n| gold | {a.a_label} | {a.b_label} |\n|---|---|---|\n"
        ca, cb = confusion(ra, name), confusion(rb, name)
        for g in range(len(ca)):
            text += f"| {g} | {', '.join(map(str, ca[g]))} | {', '.join(map(str, cb[g]))} |\n"
    print(text)
    if a.json:
        Path(a.json).write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")


def _softmax(row) -> list[float]:
    m = max(row)
    e = [math.exp(v - m) for v in row]
    s = sum(e)
    return [v / s for v in e]


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_trace(a, source: str) -> dict:
    """What fixes the rows of an `eval` run: a preds file is resumed only by a run with the same trace."""
    def full(path: Optional[str]) -> Optional[str]:   # relative paths from two folders must not look the same
        return shown_path(str(Path(path).expanduser().resolve())) if path else None

    mlx = Path(a.mlx).expanduser()
    meta = json.loads((mlx / "config.json").read_text(encoding="utf-8")).get("pusula_mlx", {})
    return {"mlx": full(a.mlx), "bits": meta.get("bits"), "group_size": meta.get("group_size"),
            "adapter": full(a.adapter), "data_sha256": _sha256(a.data), "label_source": source,
            "base": bool(a.base), "limit": a.limit or None, "seed": a.seed if a.limit else None, "causal": bool(a.causal)}


def trace_path(preds: Path) -> Path:
    return preds.with_name(preds.name.removesuffix(".jsonl") + ".run.json")


def resume_rows(f, preds: Path, trace: dict, items: list, index: list[int]) -> dict[int, dict]:
    """The rows already in the open, locked preds file `f`, checked against the run: same trace, every row once,
    every row a question of this run. A partial last line (a run killed mid-write) is cut off."""
    f.seek(0)
    text = f.read()
    tpath = trace_path(preds)
    if text.strip():
        if not tpath.is_file():
            raise SystemExit(f"{preds} has rows but no run trace {tpath.name}: it cannot be resumed safely; "
                             "remove it or use another --out")
        old = json.loads(tpath.read_text(encoding="utf-8"))
        if old != trace:
            diff = sorted(k for k in trace.keys() | old.keys() if old.get(k) != trace.get(k))
            raise SystemExit(f"{preds} belongs to another run (differs in {', '.join(diff)}): remove it or use another --out")
    if text and not text.endswith("\n"):
        text = text[: text.rfind("\n") + 1]
        f.truncate(len(text.encode("utf-8")))
    wanted = set(index)
    done: dict[int, dict] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        i = r["i"]
        if i in done:
            raise SystemExit(f"{preds}: item {i} appears twice")
        if i not in wanted or (r["name"], r["label"]) != (items[i].name, items[i].label):
            raise SystemExit(f"{preds}: item {i} ({r['name']}) is not a question of this run")
        done[i] = r
    if not done:
        tmp = tpath.with_name(tpath.name + ".tmp")
        tmp.write_text(json.dumps(trace, indent=2), encoding="utf-8")
        tmp.replace(tpath)
    return done


@contextlib.contextmanager
def locked_append(path: Path):
    """`path` opened for reading and appending, with an exclusive lock: a second run on the same file stops at once."""
    import fcntl

    with path.open("a+", encoding="utf-8") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"{path} is in use by another run") from None
        yield f


def base_template_flags(examples, source: str) -> list[bool]:
    """Per item of build_items (same pairs, same order): is the question a BASE_TEMPLATES template, by name or by
    content? The rule of inference.D3Engine._use_base for the templates."""
    from .d3train import select_pairs
    from .taxonomy import load_taxonomy

    universal = load_taxonomy().universal
    templates = [universal[n] for n in BASE_TEMPLATES if n in universal]
    return [name in BASE_TEMPLATES or ex.questions[name] in templates
            for ex, name in select_pairs(examples, source, None)]


def run_eval(a) -> None:
    import mlx.core as mx

    from .d3train import build_items
    from .jsonl import read_examples
    from .llm import console

    # MLX keeps freed buffers in its cache for reuse; with prompts of many lengths that cache grows to about the
    # size of the model (30 GB peak at 4-bit before this limit). A small cache keeps 6-bit and 8-bit within 36 GB.
    mx.set_cache_limit(int(a.cache_gb * 2**30))
    out = Path(a.out)
    preds = Path(a.preds) if a.preds else out.with_suffix(".preds.jsonl")
    preds.parent.mkdir(parents=True, exist_ok=True)
    # the lock comes first: a second run on the same file stops before it loads the model
    with locked_append(preds) as f:  # resumable: items already predicted are not run again
        decider = MLXDecider.load(a.mlx, adapter=a.adapter, causal=a.causal)
        examples = read_examples(a.data)
        source = a.label_source or ("gold" if any("gold" in ex.labels for ex in examples) else "final")
        items = build_items(examples, decider, source, None)
        flags = base_template_flags(examples, source)
        if len(flags) != len(items):
            raise SystemExit(f"{len(flags)} base template flags for {len(items)} items")
        index = list(range(len(items)))
        if a.limit and a.limit < len(items):
            index = sorted(random.Random(a.seed).sample(index, a.limit))
        t0, tokens, n = time.time(), 0, 0
        done = resume_rows(f, preds, run_trace(a, source), items, index)
        todo = [i for i in index if i not in done]
        console.print(f"{len(index)} items, {len(done)} done, {len(todo)} to run ({decider.model.__class__.__module__})")
        for batch in batches([len(items[i].ids) for i in todo], a.token_budget):
            sel = [todo[j] for j in batch]
            logits = decider.logits([items[i].ids for i in sel], [items[i].count for i in sel], base=a.base)
            for row, i in enumerate(sel):
                it = items[i]
                r = {"i": i, "name": it.name, "label": it.label, "probs": _softmax(logits[row, : it.count].tolist()),
                     "base_template": flags[i]}
                done[i] = r
                f.write(json.dumps(r) + "\n")
            f.flush()
            tokens += sum(len(items[i].ids) for i in sel)
            n += len(sel)
            dt = time.time() - t0
            if n and (n % 200 < len(sel) or n == len(todo)):
                eta = dt / n * (len(todo) - n)
                console.print(f"{n}/{len(todo)}  {n / dt:.2f} q/s  {tokens / dt:.0f} tok/s  eta {eta / 60:.0f} min")
    rows = [done[i] for i in index]
    report = {"model": ("pusula-d3" if a.adapter and not a.base else "d3-base") + "-mlx", "mlx": shown_path(a.mlx),
              "adapter": shown_path(a.adapter), "base_answer": a.base, "data": a.data, "reference": source,
              "limit": a.limit or None, "causal": a.causal, **score(rows)}
    if n:
        report["speed"] = {"questions_per_s": round(n / (time.time() - t0), 3),
                           "prompt_tokens_per_s": round(tokens / (time.time() - t0), 1)}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out)
    console.print(f"EVAL_DONE {out}")
    console.print_json(data=report)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m pusula.mlx_d3", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="quantize the d3 text backbone to MLX")
    c.add_argument("--d3", required=True, help="local vllm-sr/d3 snapshot directory")
    c.add_argument("--out", required=True)
    c.add_argument("--bits", type=int, default=4, choices=[4, 5, 6, 8, 16])
    c.add_argument("--group-size", type=int, default=64)
    e = sub.add_parser("eval", help="score a benchmark file (resumable, per-item predictions)")
    e.add_argument("--mlx", required=True, help="directory written by `convert`")
    e.add_argument("--adapter", default=None, help="Pusula adapter directory (LoRA + readout)")
    e.add_argument("--base", action="store_true", help="answer with the adapter off (base Decision 3.0)")
    e.add_argument("--data", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--preds", default=None, help="per-item predictions (default: <out>.preds.jsonl)")
    e.add_argument("--label-source", default=None)
    e.add_argument("--limit", type=int, default=0, help="a fixed random subset of this many questions")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--token-budget", type=int, default=4096, help="padded tokens per forward pass")
    e.add_argument("--cache-gb", type=float, default=2.0, help="MLX buffer cache limit in GB (lower = less memory)")
    e.add_argument("--causal", action="store_true", help="debugging: causal full attention (not how d3 runs)")
    m = sub.add_parser("combine", help="the own_base setup from an adapter run and a base run (no model)")
    m.add_argument("--adapter-preds", required=True, help=".preds.jsonl of the run with the adapter on")
    m.add_argument("--base-preds", required=True, help=".preds.jsonl of the base d3 run")
    m.add_argument("--base-questions", nargs="+", default=["eksik_bilgi"],
                   help="question types answered by base d3 (default: eksik_bilgi, as PUSULA_OWN_BASE=templates in "
                        "v2.1; the v2.0 default and the published 92.0 used ozel_soru eksik_bilgi)")
    m.add_argument("--out", required=True, help="report .json; the predictions go next to it as .preds.jsonl")
    k = sub.add_parser("compare", help="paired comparison of two runs on the same questions (markdown table)")
    k.add_argument("a", help=".preds.jsonl of run A (reference)")
    k.add_argument("b", help=".preds.jsonl of run B")
    k.add_argument("--a-label", default="A")
    k.add_argument("--b-label", default="B")
    k.add_argument("--confusion", nargs="*", default=[], help="question types to print a confusion matrix for")
    k.add_argument("--json", default=None, help="also write the numbers to this file")
    a = p.parse_args(argv)
    if a.cmd == "convert":
        print(convert(a.d3, a.out, a.bits, a.group_size))
    elif a.cmd == "combine":
        run_combine(a)
    elif a.cmd == "compare":
        run_compare(a)
    else:
        run_eval(a)


if __name__ == "__main__":
    main()
