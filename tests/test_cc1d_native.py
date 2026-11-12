"""The causal-conv1d build for the local GPU keeps only that architecture, and refuses anything unexpected."""
import hashlib
import io
import tarfile

import pytest

from scripts import cc1d_native as c

SETUP = '''
    if not SKIP_CUDA_BUILD:
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_75,code=sm_75")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")
        if bare_metal_version >= Version("12.8"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_100,code=sm_100")
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_120,code=sm_120")
        nvcc = cc_flag
'''


def test_keeps_only_the_wanted_architecture():
    out = c.restrict(SETUP, ["120"])
    assert out.count('cc_flag.append("-gencode")') == 1
    assert 'arch=compute_120,code=sm_120' in out and "compute_80" not in out
    compile(out.replace("    if", "if", 1).replace("\n    ", "\n"), "setup.py", "exec")  # still valid Python


def test_two_machine_architectures():
    out = c.restrict(SETUP, ["80", "120"])
    assert out.count('cc_flag.append("-gencode")') == 2


@pytest.mark.parametrize("keep", [[], ["89"], ["120", "89"]])
def test_refuses_unknown_or_empty(keep):
    with pytest.raises(ValueError):
        c.restrict(SETUP, keep)


def _tgz(tmp_path, text):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        data = text.encode()
        info = tarfile.TarInfo(f"{c.NAME}/setup.py")
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    p = tmp_path / "src.tar.gz"
    p.write_bytes(buf.getvalue())
    return p


def test_wrong_sha256_falls_back(tmp_path, capsys):
    src = _tgz(tmp_path, SETUP)
    assert c.main([str(tmp_path / "w"), "--arch", "120", "--url", src.as_uri()]) == 1
    assert "sha256" in capsys.readouterr().err


def test_pinned_sha256_builds_the_tree(tmp_path, monkeypatch, capsys):
    src = _tgz(tmp_path, SETUP)
    monkeypatch.setattr(c, "SHA256", hashlib.sha256(src.read_bytes()).hexdigest())
    assert c.main([str(tmp_path / "w"), "--arch", "120", "--url", src.as_uri()]) == 0
    tree = capsys.readouterr().out.strip()
    assert tree.endswith(c.NAME)
    assert "compute_75" not in (tmp_path / "w" / c.NAME / "setup.py").read_text()
