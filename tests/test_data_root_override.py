import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_companion_root_override(tmp_path):
    (tmp_path / "output" / "intermediate").mkdir(parents=True)
    env = dict(os.environ, FUSION_COMPANION_ROOT=str(tmp_path), PYTHONDONTWRITEBYTECODE="1")
    code = (
        f"import sys; sys.path.insert(0, {str(REPO)!r}); import fusion.data as d; "
        "print(d.AUTHORSHIP_ROOT); print(d._INTERMEDIATE)"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], env=env, capture_output=True,
                         text=True, check=True).stdout.splitlines()
    assert out[0] == str(tmp_path)
    assert out[1] == str(tmp_path / "output" / "intermediate")
