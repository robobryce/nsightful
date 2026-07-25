"""Opt-in GPU end-to-end tests for the real Nsight profiler wrappers."""

import importlib.util
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nsightful import parse_ncu_csv

pytestmark = pytest.mark.gpu


def _require_gpu_e2e(profiler: str) -> None:
    if os.environ.get("NSIGHTFUL_RUN_GPU_TESTS") != "1":
        pytest.skip("set NSIGHTFUL_RUN_GPU_TESTS=1 to run real profiler tests")
    if shutil.which(profiler) is None:
        pytest.skip(f"{profiler} is not installed")
    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy is not installed")


def _run_profiled_ipython(profiler: str, code: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["NSIGHTFUL_USE_WIDGETS"] = "0"
    profiler_args = ["--trace=cuda,nvtx,osrt"] if profiler == "nsys" else []
    return subprocess.run(
        [sys.executable, "-m", "nsightful.ipython", profiler, *profiler_args, "--", "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=180,
    )


def test_ncu_wrapper_profiles_cell_in_existing_namespace(tmp_path: Path) -> None:
    _require_gpu_e2e("ncu")
    report = tmp_path / "cell.ncu-rep"
    code = f"""
import os
from pathlib import Path
import cupy as cp

pid = os.getpid()
x = cp.arange(100_000, dtype=cp.int64)
cp.cuda.runtime.deviceSynchronize()
get_ipython().run_cell_magic(
    "ncu", "-o {report} --no-display", "x *= 2"
)
assert os.getpid() == pid
assert int(x.sum()) == 9_999_900_000
assert Path({str(report)!r}).is_file()
print("NSIGHTFUL_NCU_E2E_OK")
"""

    result = _run_profiled_ipython("ncu", code)

    assert "NSIGHTFUL_NCU_E2E_OK" in result.stdout
    csv_result = subprocess.run(
        ["ncu", f"--import={report}", "--csv"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    parsed = parse_ncu_csv(csv_result.stdout.splitlines())
    assert any("multiply" in kernel.lower() for kernel in parsed)


def test_nsys_wrapper_profiles_cell_in_existing_namespace(tmp_path: Path) -> None:
    _require_gpu_e2e("nsys")
    # Nsight Compute can hold the GPU's profiler lock briefly after its process exits.
    time.sleep(1)
    report = tmp_path / "cell.nsys-rep"
    sqlite_report = tmp_path / "cell.sqlite"
    code = f"""
import os
from pathlib import Path
import cupy as cp

pid = os.getpid()
warm = cp.ones(1, dtype=cp.int64)
warm *= 2
warm += 1
x = cp.arange(100_000, dtype=cp.int64)
cp.cuda.runtime.deviceSynchronize()
get_ipython().run_cell_magic(
    "nsys", "-o {report} --no-display", "x *= 2\\nx += 1"
)
assert os.getpid() == pid
assert int(x.sum()) == 10_000_000_000
assert Path({str(report)!r}).is_file()
assert Path({str(sqlite_report)!r}).is_file()
print("NSIGHTFUL_NSYS_E2E_OK")
"""

    result = _run_profiled_ipython("nsys", code)

    assert "NSIGHTFUL_NSYS_E2E_OK" in result.stdout
    connection = sqlite3.connect(sqlite_report)
    try:
        count = connection.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
    finally:
        connection.close()
    assert count is not None and count[0] >= 2
