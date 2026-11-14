"""Validation spread over the training processes gives the same numbers as one process (12 Nov 2026)."""
import json
import random

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from pusula.d3train import Item, evaluate  # noqa: E402


class _Decider:
    """Logits depend only on each question's own tokens, as in the real model with its attention mask."""

    def logits(self, ids, counts):
        rows = []
        for x, _c in zip(ids, counts):
            g = torch.Generator().manual_seed(sum(v * (i + 1) for i, v in enumerate(x)) % (2**31))
            rows.append(torch.randn(4, generator=g))
        return torch.stack(rows)


def _items() -> list[Item]:
    rng = random.Random(5)
    out = []
    for k in range(57):
        n = rng.randint(3, 40)
        out.append(Item(name=f"q{k % 5}", qtype="choice", ids=[rng.randint(1, 900) for _ in range(n)],
                        label=rng.randint(0, 3), count=4))
    return out


def _worker(rank: int, world: int, port: int, out: str) -> None:
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    m = evaluate(_Decider(), _items(), 64, rank, world, dist)
    with open(f"{out}.{rank}", "w") as f:
        json.dump(m, f)
    dist.destroy_process_group()


@pytest.mark.parametrize("world,port", [(3, 29681), (4, 29682)])
def test_spread_validation_matches_one_process(tmp_path, world, port):
    one = evaluate(_Decider(), _items(), 64)
    out = str(tmp_path / "r")
    mp.spawn(_worker, args=(world, port, out), nprocs=world, join=True)
    rows = [json.load(open(f"{out}.{r}")) for r in range(world)]
    assert one["n"] == 57
    assert all(r == one for r in rows)
