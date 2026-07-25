"""IPython cell magics for kernels launched by Nsightful profiler wrappers."""

import atexit
import ctypes
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from IPython.core.error import UsageError
from IPython.core.magic import Magics, cell_magic, magics_class

from .ipython import (
    NCU_COMMAND_ENV,
    NCU_REPORT_ENV,
    NCU_TEMP_DIR_ENV,
    NSYS_COMMAND_ENV,
    NSYS_SESSION_ENV,
    PROFILER_ENV,
)

_INJECTION_ENV_PREFIXES = (
    "CASK_",
    "CUDA_INJECTION",
    "NSYS_",
    "NSYSDK_",
    "NVTX_INJECTION",
    "QUADD_",
)
_INJECTION_ENV_NAMES = {
    "__GL_CONSTANT_FRAME_RATE_HINT",
    "HOOK_FILE",
    "LD_PRELOAD",
    "NV_COMPUTE_PROFILER_INJECTION_PATH",
}
_NCU_OWNED_OPTIONS = {
    "-f",
    "-i",
    "-o",
    "--csv",
    "--export",
    "--force-overwrite",
    "--import",
    "--nvtx-include",
}
_NCU_DISPLAY_OPTIONS = {
    "--page",
    "--print-details",
    "--print-fp",
    "--print-kernel-base",
    "--print-metric-attribution",
    "--print-metric-instances",
    "--print-metric-name",
    "--print-nvtx-rename",
    "--print-rule-details",
    "--print-source",
    "--print-summary",
    "--print-units",
    "--resolve-source-file",
}
_NSYS_OWNED_OPTIONS = {
    "-c",
    "-f",
    "-o",
    "--after-collection-start",
    "--capture-range",
    "--capture-range-end",
    "--export",
    "--force-overwrite",
    "--output",
    "--session",
    "--session-new",
}
_NSYS_LAUNCH_OPTIONS = {"-t", "--trace"}
_NSYS_START_TIMEOUT_SECONDS = 30.0


def _option_name(argument: str) -> str:
    for short_option in ("-c", "-o", "-s", "-t"):
        if argument.startswith(short_option) and len(argument) > len(short_option):
            return short_option
    return argument.split("=", 1)[0]


def _parse_magic_arguments(line: str, profiler: str) -> Tuple[Path, bool, List[str]]:
    """Parse Nsightful-owned options and leave profiler-specific options untouched."""
    try:
        arguments = shlex.split(line)
    except ValueError as error:
        raise UsageError(str(error)) from error

    output: Optional[str] = None
    display = True
    profiler_args: List[str] = []
    index = 0
    passthrough = False
    while index < len(arguments):
        argument = arguments[index]
        if passthrough:
            profiler_args.append(argument)
        elif argument == "--":
            passthrough = True
        elif argument in ("-o", "--output"):
            index += 1
            if index == len(arguments):
                raise UsageError(f"{argument} requires a path")
            output = arguments[index]
        elif argument.startswith("--output="):
            output = argument.split("=", 1)[1]
        elif argument.startswith("-o") and len(argument) > 2:
            output = argument[2:]
        elif argument == "--no-display":
            display = False
        elif argument in ("-h", "--help"):
            raise UsageError(
                f"%%{profiler} [-o REPORT] [--no-display] [-- {profiler} report options]"
            )
        else:
            profiler_args.append(argument)
        index += 1

    owned = _NCU_OWNED_OPTIONS if profiler == "ncu" else _NSYS_OWNED_OPTIONS
    conflicts = sorted({_option_name(arg) for arg in profiler_args} & owned)
    if conflicts:
        raise UsageError(
            f"%%{profiler} controls these options: {', '.join(conflicts)}; "
            "use -o/--output to name the report"
        )
    if profiler == "ncu":
        unsupported = sorted(
            {
                _option_name(argument)
                for argument in profiler_args
                if argument.startswith("-") and _option_name(argument) not in _NCU_DISPLAY_OPTIONS
            }
        )
        if unsupported:
            raise UsageError(
                "%%ncu accepts only report display options after --; "
                f"pass collection options to nsightful-ncu instead: {', '.join(unsupported)}"
            )
    else:
        launch_options = sorted(
            {
                _option_name(argument)
                for argument in profiler_args
                if _option_name(argument) in _NSYS_LAUNCH_OPTIONS
            }
        )
        if launch_options:
            raise UsageError(
                "%%nsys cannot change application-scope options; "
                f"pass them to nsightful-nsys instead: {', '.join(launch_options)}"
            )

    suffix = ".ncu-rep" if profiler == "ncu" else ".nsys-rep"
    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = f"{profiler}-report-{timestamp}-{uuid.uuid4().hex[:8]}{suffix}"
    candidate = Path(output).expanduser()
    if not output or candidate.name in ("", ".", ".."):
        raise UsageError("-o/--output must name a report file")
    report = candidate.resolve()
    if report.is_dir():
        raise UsageError("-o/--output must name a report file, not a directory")
    if not report.name.endswith(suffix):
        report = report.with_name(report.name + suffix)
    report.parent.mkdir(parents=True, exist_ok=True)
    return report, display, profiler_args


def _clean_profiler_environment() -> Dict[str, str]:
    """Remove injection settings before running a nested Nsight control command."""
    environment = dict(os.environ)
    for name in list(environment):
        if name == "NSYS_CONFIG_DIRECTIVES":
            continue
        if name in _INJECTION_ENV_NAMES or name.startswith(_INJECTION_ENV_PREFIXES):
            environment.pop(name, None)
    return environment


def _run_profiler_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        env=_clean_profiler_environment(),
    )


def _check_profiler_command(
    result: subprocess.CompletedProcess[str], command: Sequence[str]
) -> None:
    if result.returncode == 0:
        return
    detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
    raise UsageError(f"{' '.join(command[:2])} failed: {detail}")


def _nsys_ready_callback(marker: Path) -> str:
    """Build a callback that marks when an interactive collection is armed."""
    script = f"from pathlib import Path; Path({str(marker)!r}).touch()"
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"


def _nsys_supports_ready_callback(nsys: str) -> bool:
    """Return whether ``nsys start`` supports the collection-ready callback."""
    result = _run_profiler_command([nsys, "start", "--help"])
    output = f"{result.stdout}\n{result.stderr}"
    return result.returncode == 0 and "--after-collection-start" in output


def _wait_for_nsys_collection(marker: Path) -> None:
    """Wait until Nsight Systems confirms that collection has started."""
    deadline = time.monotonic() + _NSYS_START_TIMEOUT_SECONDS
    while not marker.is_file():
        if time.monotonic() >= deadline:
            raise UsageError("nsys collection did not start before the timeout")
        time.sleep(0.01)


def _synchronize_current_cuda_context() -> None:
    """Synchronize the current CUDA context without depending on a CUDA Python package."""
    try:
        driver = ctypes.CDLL("libcuda.so.1")
        context = ctypes.c_void_p()
        driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        driver.cuCtxGetCurrent.restype = ctypes.c_int
        driver.cuCtxSynchronize.argtypes = []
        driver.cuCtxSynchronize.restype = ctypes.c_int
        if driver.cuCtxGetCurrent(ctypes.byref(context)) == 0 and context.value:
            driver.cuCtxSynchronize()
    except (AttributeError, OSError):
        pass


def _cell_succeeded(result: Any) -> bool:
    return bool(getattr(result, "success", True))


def _propagate_cell_error(result: Any) -> None:
    """Make a nested ``run_cell`` failure fail the outer magic execution request."""
    if _cell_succeeded(result):
        return
    error = getattr(result, "error_before_exec", None) or getattr(result, "error_in_exec", None)
    if isinstance(error, BaseException):
        raise error
    raise RuntimeError("profiled cell execution failed")


def _cleanup_ncu_temp_dir(path: str) -> None:
    """Remove the live aggregate report after a normal kernel shutdown."""
    directory = Path(path)
    if directory.name.startswith("nsightful-ncu-"):
        shutil.rmtree(directory, ignore_errors=True)


@magics_class
class NCUMagics(Magics):
    """Cell magic registered by the Nsight Compute wrapper."""

    @cell_magic
    def ncu(self, line: str, cell: str) -> Any:
        """Profile one cell with Nsight Compute and display its report."""
        if os.environ.get(PROFILER_ENV) != "ncu":
            raise UsageError("%%ncu requires a kernel launched with nsightful-ncu")
        report, display, report_args = _parse_magic_arguments(line, "ncu")
        ncu = os.environ.get(NCU_COMMAND_ENV)
        base_report = os.environ.get(NCU_REPORT_ENV)
        if not ncu or not base_report:
            raise UsageError("the nsightful-ncu wrapper environment is incomplete")

        try:
            import nvtx
        except ImportError as error:
            raise UsageError(
                '%%ncu requires nvtx; install it with pip install "nsightful[notebook]"'
            ) from error

        if self.shell is None:
            raise RuntimeError("%%ncu requires an active IPython shell")
        range_name = f"cell-{uuid.uuid4().hex}"
        range_id = nvtx.start_range(message=range_name, domain="Nsightful")
        try:
            result = self.shell.run_cell(cell)
            _synchronize_current_cuda_context()
        finally:
            nvtx.end_range(range_id)

        if not _cell_succeeded(result):
            print("[ncu] Cell failed; no report was exported.")
            _propagate_cell_error(result)
        if not Path(base_report).is_file():
            print("[ncu] No CUDA kernels were captured; no report was created.")
            return None

        export_command = [
            ncu,
            f"--import={base_report}",
            f"--nvtx-include=Nsightful@{range_name}",
            f"--export={report}",
            "--force-overwrite",
        ]
        export_result = _run_profiler_command(export_command)
        export_output = f"{export_result.stdout}\n{export_result.stderr}"
        if export_result.returncode != 0 and "No results were matched to export" in export_output:
            print("[ncu] No CUDA kernels were captured; no report was created.")
            return None
        _check_profiler_command(export_result, export_command)
        print(f"[ncu] Report: {report}")

        if display:
            csv_command = [ncu, f"--import={report}", "--csv", *report_args]
            csv_result = _run_profiler_command(csv_command)
            _check_profiler_command(csv_result, csv_command)
            from .notebook import display_ncu_csv_in_notebook

            display_ncu_csv_in_notebook(csv_result.stdout.splitlines())
        return None


@magics_class
class NSYSMagics(Magics):
    """Cell magic registered by the Nsight Systems wrapper."""

    @cell_magic
    def nsys(self, line: str, cell: str) -> Any:
        """Profile one cell with Nsight Systems and display its report."""
        if os.environ.get(PROFILER_ENV) != "nsys":
            raise UsageError("%%nsys requires a kernel launched with nsightful-nsys")
        report, display, start_args = _parse_magic_arguments(line, "nsys")
        nsys = os.environ.get(NSYS_COMMAND_ENV)
        session = os.environ.get(NSYS_SESSION_ENV)
        if not nsys or not session:
            raise UsageError("the nsightful-nsys wrapper environment is incomplete")
        if self.shell is None:
            raise RuntimeError("%%nsys requires an active IPython shell")

        start_defaults = []
        start_option_names = {_option_name(argument) for argument in start_args}
        if "--sample" not in start_option_names and "-s" not in start_option_names:
            start_defaults.append("--sample=none")
        if "--cpuctxsw" not in start_option_names:
            start_defaults.append("--cpuctxsw=none")
        supports_ready_callback = _nsys_supports_ready_callback(nsys)
        ready_context = (
            tempfile.TemporaryDirectory(prefix="nsightful-nsys-ready-")
            if supports_ready_callback
            else nullcontext(None)
        )
        with ready_context as ready_dir:
            ready_marker = Path(ready_dir) / "ready" if ready_dir is not None else None
            callback_args = (
                [f"--after-collection-start={_nsys_ready_callback(ready_marker)}"]
                if ready_marker is not None
                else []
            )
            start_command = [
                nsys,
                "start",
                f"--session={session}",
                *start_defaults,
                f"--output={str(report).replace('%', '%%')}",
                "--force-overwrite=true",
                "--export=sqlite",
                *callback_args,
                *start_args,
            ]
            _check_profiler_command(_run_profiler_command(start_command), start_command)

            result: Any = None
            cell_error: Optional[BaseException] = None
            try:
                if ready_marker is not None:
                    _wait_for_nsys_collection(ready_marker)
                result = self.shell.run_cell(cell)
                if not _cell_succeeded(result):
                    nested_error = getattr(result, "error_before_exec", None) or getattr(
                        result, "error_in_exec", None
                    )
                    cell_error = (
                        nested_error
                        if isinstance(nested_error, BaseException)
                        else RuntimeError("profiled cell execution failed")
                    )
                _synchronize_current_cuda_context()
            except BaseException as error:
                cell_error = error
                raise
            finally:
                stop_command = [nsys, "stop", f"--session={session}"]
                stop_result = _run_profiler_command(stop_command)
                if stop_result.returncode != 0 and cell_error is not None:
                    detail = (
                        stop_result.stderr.strip() or stop_result.stdout.strip() or "unknown error"
                    )
                    print(
                        f"[nsys] Failed to stop profiler after cell error: {detail}",
                        file=sys.stderr,
                    )
                else:
                    _check_profiler_command(stop_result, stop_command)
                if "--stats" in start_option_names and stop_result.returncode == 0:
                    if stop_result.stdout.strip():
                        print(stop_result.stdout.rstrip())
                    if stop_result.stderr.strip():
                        print(stop_result.stderr.rstrip(), file=sys.stderr)

        _propagate_cell_error(result)
        print(f"[nsys] Report: {report}")
        sqlite_report = report.with_suffix(".sqlite")
        if display:
            if not sqlite_report.is_file():
                raise UsageError(f"nsys did not create the expected SQLite export: {sqlite_report}")
            from .notebook import display_nsys_sqlite_file_in_notebook

            display_nsys_sqlite_file_in_notebook(str(sqlite_report))
        return None


def load_ipython_extension(ipython: Any) -> None:
    """Register only the magic matching the active profiler wrapper."""
    profiler = os.environ.get(PROFILER_ENV)
    if profiler == "ncu":
        temp_dir = os.environ.get(NCU_TEMP_DIR_ENV)
        if temp_dir:
            atexit.register(_cleanup_ncu_temp_dir, temp_dir)
        ipython.register_magics(NCUMagics)
    elif profiler == "nsys":
        ipython.register_magics(NSYSMagics)
    else:
        raise RuntimeError(
            "nsightful.magics must be loaded by the nsightful-ncu or nsightful-nsys wrapper"
        )
