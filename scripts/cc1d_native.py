"""Prepare a causal-conv1d 1.7.0 source tree that builds only for the GPUs of this machine.

The release setup.py compiles for nine GPU architectures with CUDA 13 (sm_75 ... sm_121); on 12 Nov 2026 that took
37 minutes on a rented machine while the paid GPUs waited. A build for the local architecture is the same code for
that GPU. The source comes from PyPI and is checked against a pinned sha256.

    python scripts/cc1d_native.py <empty work dir> [--arch 120 ...]   # prints the source dir to pip-install

Exit 1 (and nothing printed) when anything does not match: the caller then falls back to the full release build.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = ("https://files.pythonhosted.org/packages/f4/d1/eba27735f31bd5527d39a3c7f693e6ded7b844cab275db719a15ad37b6cd/"
       "causal_conv1d-1.7.0.tar.gz")
SHA256 = "3202758494eaa7b597ce1c282dfa188889506bfcb92cad3c407d26736bfcd32b"
NAME = "causal_conv1d-1.7.0"
PAIR = re.compile(r'^(?P<ind>[ \t]*)cc_flag\.append\("-gencode"\)\n(?P=ind)cc_flag\.append\("arch=compute_(?P<a>\d+),code=sm_(?P=a)"\)\n',
                  re.MULTILINE)


def local_archs() -> list[str]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=60, check=True).stdout
    return sorted({line.strip().replace(".", "") for line in out.splitlines() if line.strip()})


def restrict(text: str, keep: list[str]) -> str:
    """Drop every -gencode pair whose architecture is not in `keep`; each kept one must be in the release list."""
    found = [m.group("a") for m in PAIR.finditer(text)]
    if len(found) < 2 or any(a not in found for a in keep) or not keep:
        raise ValueError(f"architecture list not as expected: release {found}, wanted {keep}")
    return PAIR.sub(lambda m: m.group(0) if m.group("a") in keep
                    else f'{m.group("ind")}pass  # sm_{m.group("a")} left out: not in this machine\n', text)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("work")
    p.add_argument("--arch", nargs="*", default=None, help="compute capabilities without the dot (default: nvidia-smi)")
    p.add_argument("--url", default=URL)
    a = p.parse_args(argv)
    try:
        keep = a.arch or local_archs()
        work = Path(a.work)
        work.mkdir(parents=True, exist_ok=True)
        tgz = work / f"{NAME}.tar.gz"
        urllib.request.urlretrieve(a.url, tgz)
        if hashlib.sha256(tgz.read_bytes()).hexdigest() != SHA256:
            raise ValueError("sha256 does not match the pinned release")
        with tarfile.open(tgz) as t:
            t.extractall(work, filter="data")
        setup = work / NAME / "setup.py"
        setup.write_text(restrict(setup.read_text(), keep))
    except Exception as e:  # noqa: BLE001 - any failure means: use the full build
        print(f"cc1d_native: {e}", file=sys.stderr)
        return 1
    print(work / NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
