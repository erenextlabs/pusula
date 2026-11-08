"""Multi-GPU training must train one model: a fresh LoRA is random per process, so rank 0's values are spread."""
import json

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from pusula.d3train import params_digest, params_in_sync, sync_params  # noqa: E402


def _worker(rank: int, world: int, port: int, out: str, spread: bool) -> None:
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    torch.manual_seed(1000 + rank)  # another start per process, as an unseeded fresh LoRA
    params = [torch.nn.Parameter(torch.randn(4, 3)), torch.nn.Parameter(torch.randn(2).to(torch.bfloat16))]
    before = params_in_sync(dist, params, "cpu")
    if spread:
        sync_params(dist, params)
    after = params_in_sync(dist, params, "cpu")
    with open(f"{out}.{rank}", "w") as f:
        json.dump({"before": before, "after": after, "digest": params_digest(params)}, f)
    dist.destroy_process_group()


def _run(tmp_path, port: int, spread: bool) -> list[dict]:
    out = str(tmp_path / "r")
    mp.spawn(_worker, args=(3, port, out, spread), nprocs=3, join=True)
    return [json.load(open(f"{out}.{r}")) for r in range(3)]


def test_ranks_differ_until_rank_0_values_are_spread(tmp_path):
    rows = _run(tmp_path, 29671, True)
    assert not any(r["before"] for r in rows)
    assert all(r["after"] for r in rows)
    assert len({r["digest"] for r in rows}) == 1


def test_without_spreading_the_check_reports_a_mismatch(tmp_path):
    rows = _run(tmp_path, 29672, False)
    assert not any(r["after"] for r in rows)
    assert len({r["digest"] for r in rows}) == 3


def test_digest_sees_one_changed_value():
    a = [torch.ones(3)]
    b = [torch.ones(3)]
    assert params_digest(a) == params_digest(b)
    b[0][1] = 1.0000001
    assert params_digest(a) != params_digest(b)
