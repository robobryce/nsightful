"""CPU-only tests for the Nsightful IPython cell magics."""

import ctypes
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
from IPython.core.error import UsageError

import nsightful.magics as magics
from nsightful.ipython import (
    NCU_COMMAND_ENV,
    NCU_REPORT_ENV,
    NCU_TEMP_DIR_ENV,
    NSYS_COMMAND_ENV,
    NSYS_SESSION_ENV,
    PROFILER_ENV,
)


def _result(
    returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["profiler"], returncode, stdout, stderr)


def _shell(result=None, side_effect=None):
    shell = SimpleNamespace(run_cell=Mock(return_value=result, side_effect=side_effect))
    return shell


def _install_fake_nvtx(monkeypatch):
    range_id = object()
    nvtx = SimpleNamespace(start_range=Mock(return_value=range_id), end_range=Mock())
    monkeypatch.setitem(sys.modules, "nvtx", nvtx)
    return nvtx, range_id


@pytest.fixture(autouse=True)
def _clear_wrapper_environment(monkeypatch):
    for name in (
        PROFILER_ENV,
        NCU_COMMAND_ENV,
        NCU_REPORT_ENV,
        NCU_TEMP_DIR_ENV,
        NSYS_COMMAND_ENV,
        NSYS_SESSION_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


class TestParseMagicArguments:
    @pytest.mark.parametrize(("profiler", "suffix"), (("ncu", ".ncu-rep"), ("nsys", ".nsys-rep")))
    def test_output_path_display_and_passthrough_options(
        self, monkeypatch, tmp_path, profiler, suffix
    ):
        monkeypatch.chdir(tmp_path)

        report, display, profiler_args = magics._parse_magic_arguments(
            "--output 'reports/my profile' --no-display -- --page raw --print-units base",
            profiler,
        )

        assert report == (tmp_path / f"reports/my profile{suffix}").resolve()
        assert report.parent.is_dir()
        assert display is False
        assert profiler_args == ["--page", "raw", "--print-units", "base"]

    def test_attached_short_output_path(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)

        report, _, _ = magics._parse_magic_arguments("-oreports/profile", "ncu")

        assert report == (tmp_path / "reports/profile.ncu-rep").resolve()

    @pytest.mark.parametrize(
        ("profiler", "argument", "option"),
        (
            ("ncu", "--export=somewhere", "--export"),
            ("ncu", "-- --csv", "--csv"),
            ("nsys", "--session other", "--session"),
            ("nsys", "--force-overwrite=true", "--force-overwrite"),
            ("ncu", "-- --set full", "--set"),
            ("nsys", "--trace=cuda,nvtx", "--trace"),
            ("nsys", "-tcuda", "-t"),
            ("nsys", "--capture-range=nvtx", "--capture-range"),
            ("nsys", "-ccudaProfilerApi", "-c"),
            ("nsys", "--after-collection-start=true", "--after-collection-start"),
        ),
    )
    def test_rejects_options_owned_by_magic(self, profiler, argument, option):
        with pytest.raises(UsageError) as error:
            magics._parse_magic_arguments(argument, profiler)

        assert option in str(error.value)

    @pytest.mark.parametrize(
        ("line", "message"),
        (
            ("-o", "requires a path"),
            ("--output", "requires a path"),
            ("--output=", "must name a report file"),
            ("-o /", "must name a report file"),
            ("'unterminated", "No closing quotation"),
            ("--help", "%%ncu"),
        ),
    )
    def test_reports_invalid_magic_arguments(self, line, message):
        with pytest.raises(UsageError, match=message):
            magics._parse_magic_arguments(line, "ncu")

    def test_generates_unique_default_report_name(self, monkeypatch, tmp_path):
        class FakeDateTime:
            @classmethod
            def now(cls):
                return SimpleNamespace(strftime=Mock(return_value="20260725-010203"))

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(magics, "datetime", FakeDateTime)
        monkeypatch.setattr(
            magics.uuid, "uuid4", Mock(return_value=SimpleNamespace(hex="abcdef1234567890"))
        )

        report, display, profiler_args = magics._parse_magic_arguments("", "ncu")

        assert report == (tmp_path / "ncu-report-20260725-010203-abcdef12.ncu-rep")
        assert display is True
        assert profiler_args == []


class TestProfilerHelpers:
    def test_clean_environment_removes_injection_variables(self, monkeypatch):
        injected = {
            "NSYS_LAUNCH": "1",
            "QUADD_INJECTION": "1",
            "NSYSDK_LIB": "/tmp/nsys",
            "CASK_INJECTION": "1",
            "CUDA_INJECTION32_PATH": "/tmp/32",
            "CUDA_INJECTION64_PATH": "/tmp/64",
            "CUDA_INJECTION_PATH": "/tmp/cuda",
            "NVTX_INJECTION64_PATH": "/tmp/nvtx",
            "NV_COMPUTE_PROFILER_INJECTION_PATH": "/tmp/ncu",
            "HOOK_FILE": "/tmp/hook",
            "__GL_CONSTANT_FRAME_RATE_HINT": "1",
            "LD_PRELOAD": "/tmp/injection.so",
        }
        for name, value in injected.items():
            monkeypatch.setenv(name, value)
        monkeypatch.setenv("LD_LIBRARY_PATH", "/keep")
        monkeypatch.setenv("NSYS_CONFIG_DIRECTIVES", "UseDebuginfod=false")
        monkeypatch.setenv(PROFILER_ENV, "ncu")

        environment = magics._clean_profiler_environment()

        assert injected.keys().isdisjoint(environment)
        assert environment["LD_LIBRARY_PATH"] == "/keep"
        assert environment["NSYS_CONFIG_DIRECTIVES"] == "UseDebuginfod=false"
        assert environment[PROFILER_ENV] == "ncu"

    def test_run_profiler_command_captures_output_with_clean_environment(self, monkeypatch):
        clean_environment = {"PATH": "/clean"}
        completed = _result(stdout="profile output")
        run = Mock(return_value=completed)
        monkeypatch.setattr(
            magics, "_clean_profiler_environment", Mock(return_value=clean_environment)
        )
        monkeypatch.setattr(magics.subprocess, "run", run)

        assert magics._run_profiler_command(("ncu", "--version")) is completed
        run.assert_called_once_with(
            ["ncu", "--version"],
            check=False,
            capture_output=True,
            text=True,
            env=clean_environment,
        )

    @pytest.mark.parametrize(
        ("stdout", "stderr", "detail"),
        (
            ("stdout detail\n", "", "stdout detail"),
            ("ignored", "stderr detail\n", "stderr detail"),
            ("", "", "unknown error"),
        ),
    )
    def test_check_profiler_command_raises_useful_error(self, stdout, stderr, detail):
        with pytest.raises(UsageError, match=detail):
            magics._check_profiler_command(
                _result(returncode=7, stdout=stdout, stderr=stderr),
                ["nsys", "start", "--trace=cuda"],
            )

    def test_check_profiler_command_accepts_success(self):
        magics._check_profiler_command(_result(), ["ncu", "--import=report"])

    def test_wait_for_nsys_collection_accepts_ready_marker(self, monkeypatch, tmp_path):
        marker = tmp_path / "ready"
        marker.touch()
        sleep = Mock()
        monkeypatch.setattr(magics.time, "sleep", sleep)

        magics._wait_for_nsys_collection(marker)

        sleep.assert_not_called()

    def test_wait_for_nsys_collection_times_out(self, monkeypatch, tmp_path):
        monkeypatch.setattr(magics, "_NSYS_START_TIMEOUT_SECONDS", 0)

        with pytest.raises(UsageError, match="did not start"):
            magics._wait_for_nsys_collection(tmp_path / "missing")

    def test_synchronize_current_cuda_context(self, monkeypatch):
        def get_current(context_pointer):
            context_pointer._obj.value = 1234
            return 0

        driver = SimpleNamespace(
            cuCtxGetCurrent=Mock(side_effect=get_current),
            cuCtxSynchronize=Mock(return_value=0),
        )
        monkeypatch.setattr(magics.ctypes, "CDLL", Mock(return_value=driver))

        magics._synchronize_current_cuda_context()

        driver.cuCtxGetCurrent.assert_called_once()
        assert driver.cuCtxGetCurrent.argtypes == [ctypes.POINTER(ctypes.c_void_p)]
        assert driver.cuCtxGetCurrent.restype is ctypes.c_int
        driver.cuCtxSynchronize.assert_called_once_with()

    @pytest.mark.parametrize("error", (OSError("no CUDA"), AttributeError("old driver")))
    def test_synchronize_tolerates_unavailable_cuda_driver(self, monkeypatch, error):
        monkeypatch.setattr(magics.ctypes, "CDLL", Mock(side_effect=error))

        magics._synchronize_current_cuda_context()

    def test_cleanup_ncu_temp_dir_only_removes_owned_directory(self, monkeypatch, tmp_path):
        remove = Mock()
        monkeypatch.setattr(magics.shutil, "rmtree", remove)

        magics._cleanup_ncu_temp_dir(str(tmp_path / "nsightful-ncu-session"))
        magics._cleanup_ncu_temp_dir(str(tmp_path / "user-data"))

        remove.assert_called_once_with(tmp_path / "nsightful-ncu-session", ignore_errors=True)


class TestNCUMagic:
    @pytest.mark.parametrize("profiler", (None, "nsys"))
    def test_requires_ncu_wrapper(self, monkeypatch, profiler):
        if profiler is not None:
            monkeypatch.setenv(PROFILER_ENV, profiler)
        shell = _shell()

        with pytest.raises(UsageError, match="kernel launched with nsightful-ncu"):
            magics.NCUMagics(shell=shell).ncu("", "x = 1")

        shell.run_cell.assert_not_called()

    @pytest.mark.parametrize(
        ("command", "report"), ((None, "/tmp/base.ncu-rep"), ("/usr/bin/ncu", None))
    )
    def test_requires_complete_wrapper_environment(self, monkeypatch, command, report):
        monkeypatch.setenv(PROFILER_ENV, "ncu")
        if command:
            monkeypatch.setenv(NCU_COMMAND_ENV, command)
        if report:
            monkeypatch.setenv(NCU_REPORT_ENV, report)

        with pytest.raises(UsageError, match="wrapper environment is incomplete"):
            magics.NCUMagics(shell=_shell()).ncu("", "x = 1")

    def test_requires_nvtx(self, monkeypatch, tmp_path):
        monkeypatch.setenv(PROFILER_ENV, "ncu")
        monkeypatch.setenv(NCU_COMMAND_ENV, "/opt/ncu")
        monkeypatch.setenv(NCU_REPORT_ENV, str(tmp_path / "base.ncu-rep"))
        monkeypatch.setitem(sys.modules, "nvtx", None)

        with pytest.raises(UsageError, match="requires nvtx"):
            magics.NCUMagics(shell=_shell()).ncu("", "x = 1")

    def test_profiles_exports_and_displays_successful_cell(self, monkeypatch, tmp_path, capsys):
        base_report = tmp_path / "base.ncu-rep"
        base_report.touch()
        output = tmp_path / "reports" / "chosen report"
        report = output.with_name("chosen report.ncu-rep")
        monkeypatch.setenv(PROFILER_ENV, "ncu")
        monkeypatch.setenv(NCU_COMMAND_ENV, "/opt/ncu")
        monkeypatch.setenv(NCU_REPORT_ENV, str(base_report))
        monkeypatch.setattr(
            magics.uuid, "uuid4", Mock(return_value=SimpleNamespace(hex="0123456789abcdef"))
        )
        nvtx, range_id = _install_fake_nvtx(monkeypatch)
        run_profiler = Mock(side_effect=[_result(), _result(stdout="heading\nrow one\nrow two\n")])
        synchronize = Mock()
        display = Mock()
        monkeypatch.setattr(magics, "_run_profiler_command", run_profiler)
        monkeypatch.setattr(magics, "_synchronize_current_cuda_context", synchronize)
        monkeypatch.setattr("nsightful.notebook.display_ncu_csv_in_notebook", display)
        cell_result = SimpleNamespace(success=True)
        shell = _shell(cell_result)

        returned = magics.NCUMagics(shell=shell).ncu(
            f"-o '{output}' -- --page raw --print-units base", "launch_kernel()"
        )

        assert returned is None
        shell.run_cell.assert_called_once_with("launch_kernel()")
        nvtx.start_range.assert_called_once_with(
            message="cell-0123456789abcdef", domain="Nsightful"
        )
        nvtx.end_range.assert_called_once_with(range_id)
        synchronize.assert_called_once_with()
        assert run_profiler.call_args_list == [
            call(
                [
                    "/opt/ncu",
                    f"--import={base_report}",
                    "--nvtx-include=Nsightful@cell-0123456789abcdef",
                    f"--export={report}",
                    "--force-overwrite",
                ]
            ),
            call(
                [
                    "/opt/ncu",
                    f"--import={report}",
                    "--csv",
                    "--page",
                    "raw",
                    "--print-units",
                    "base",
                ]
            ),
        ]
        display.assert_called_once_with(["heading", "row one", "row two"])
        assert f"[ncu] Report: {report}" in capsys.readouterr().out

    def test_no_display_only_exports_report(self, monkeypatch, tmp_path):
        base_report = tmp_path / "base.ncu-rep"
        base_report.touch()
        report = tmp_path / "cell.ncu-rep"
        monkeypatch.setenv(PROFILER_ENV, "ncu")
        monkeypatch.setenv(NCU_COMMAND_ENV, "ncu")
        monkeypatch.setenv(NCU_REPORT_ENV, str(base_report))
        _install_fake_nvtx(monkeypatch)
        run_profiler = Mock(return_value=_result())
        display = Mock()
        monkeypatch.setattr(magics, "_run_profiler_command", run_profiler)
        monkeypatch.setattr(magics, "_synchronize_current_cuda_context", Mock())
        monkeypatch.setattr("nsightful.notebook.display_ncu_csv_in_notebook", display)

        magics.NCUMagics(shell=_shell(SimpleNamespace(success=True))).ncu(
            f"-o {report} --no-display", "launch_kernel()"
        )

        assert run_profiler.call_count == 1
        assert f"--export={report}" in run_profiler.call_args.args[0]
        display.assert_not_called()

    def test_failed_cell_does_not_export(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv(PROFILER_ENV, "ncu")
        monkeypatch.setenv(NCU_COMMAND_ENV, "ncu")
        monkeypatch.setenv(NCU_REPORT_ENV, str(tmp_path / "base.ncu-rep"))
        _install_fake_nvtx(monkeypatch)
        run_profiler = Mock()
        synchronize = Mock()
        monkeypatch.setattr(magics, "_run_profiler_command", run_profiler)
        monkeypatch.setattr(magics, "_synchronize_current_cuda_context", synchronize)
        cell_error = ValueError("cell failed")
        cell_result = SimpleNamespace(
            success=False, error_before_exec=None, error_in_exec=cell_error
        )

        with pytest.raises(ValueError, match="cell failed"):
            magics.NCUMagics(shell=_shell(cell_result)).ncu("", "raise Exception")

        synchronize.assert_called_once_with()
        run_profiler.assert_not_called()
        assert "Cell failed; no report was exported" in capsys.readouterr().out

    def test_export_failure_is_reported_and_skips_display(self, monkeypatch, tmp_path):
        monkeypatch.setenv(PROFILER_ENV, "ncu")
        monkeypatch.setenv(NCU_COMMAND_ENV, "ncu")
        base_report = tmp_path / "base.ncu-rep"
        base_report.touch()
        monkeypatch.setenv(NCU_REPORT_ENV, str(base_report))
        _install_fake_nvtx(monkeypatch)
        monkeypatch.setattr(
            magics, "_run_profiler_command", Mock(return_value=_result(1, stderr="bad export"))
        )
        monkeypatch.setattr(magics, "_synchronize_current_cuda_context", Mock())
        display = Mock()
        monkeypatch.setattr("nsightful.notebook.display_ncu_csv_in_notebook", display)

        with pytest.raises(UsageError, match="bad export"):
            magics.NCUMagics(shell=_shell(SimpleNamespace(success=True))).ncu("", "launch_kernel()")

        display.assert_not_called()

    def test_no_captured_kernels_is_a_clear_nonfatal_result(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv(PROFILER_ENV, "ncu")
        monkeypatch.setenv(NCU_COMMAND_ENV, "ncu")
        base_report = tmp_path / "base.ncu-rep"
        base_report.touch()
        monkeypatch.setenv(NCU_REPORT_ENV, str(base_report))
        _install_fake_nvtx(monkeypatch)
        monkeypatch.setattr(
            magics,
            "_run_profiler_command",
            Mock(return_value=_result(1, stdout="No results were matched to export")),
        )
        monkeypatch.setattr(magics, "_synchronize_current_cuda_context", Mock())
        cell_result = SimpleNamespace(success=True)

        returned = magics.NCUMagics(shell=_shell(cell_result)).ncu(
            f"-o {tmp_path / 'empty'}", "answer = 42"
        )

        assert returned is None
        assert "No CUDA kernels were captured" in capsys.readouterr().out

    def test_first_cpu_only_cell_handles_missing_base_report(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv(PROFILER_ENV, "ncu")
        monkeypatch.setenv(NCU_COMMAND_ENV, "ncu")
        monkeypatch.setenv(NCU_REPORT_ENV, str(tmp_path / "missing-base.ncu-rep"))
        _install_fake_nvtx(monkeypatch)
        run_profiler = Mock()
        monkeypatch.setattr(magics, "_run_profiler_command", run_profiler)
        monkeypatch.setattr(magics, "_synchronize_current_cuda_context", Mock())

        returned = magics.NCUMagics(shell=_shell(SimpleNamespace(success=True))).ncu(
            "", "answer = 42"
        )

        assert returned is None
        run_profiler.assert_not_called()
        assert "No CUDA kernels were captured" in capsys.readouterr().out


class TestNSYSMagic:
    @pytest.fixture(autouse=True)
    def _collection_ready(self, monkeypatch):
        monkeypatch.setattr(magics, "_wait_for_nsys_collection", Mock())

    @pytest.mark.parametrize("profiler", (None, "ncu"))
    def test_requires_nsys_wrapper(self, monkeypatch, profiler):
        if profiler is not None:
            monkeypatch.setenv(PROFILER_ENV, profiler)
        shell = _shell()

        with pytest.raises(UsageError, match="kernel launched with nsightful-nsys"):
            magics.NSYSMagics(shell=shell).nsys("", "x = 1")

        shell.run_cell.assert_not_called()

    @pytest.mark.parametrize(("command", "session"), ((None, "session-1"), ("/usr/bin/nsys", None)))
    def test_requires_complete_wrapper_environment(self, monkeypatch, command, session):
        monkeypatch.setenv(PROFILER_ENV, "nsys")
        if command:
            monkeypatch.setenv(NSYS_COMMAND_ENV, command)
        if session:
            monkeypatch.setenv(NSYS_SESSION_ENV, session)

        with pytest.raises(UsageError, match="wrapper environment is incomplete"):
            magics.NSYSMagics(shell=_shell()).nsys("", "x = 1")

    def test_profiles_stops_and_displays_successful_cell(self, monkeypatch, tmp_path, capsys):
        output = tmp_path / "reports" / "timeline"
        report = output.with_name("timeline.nsys-rep")
        sqlite_report = report.with_suffix(".sqlite")
        monkeypatch.setenv(PROFILER_ENV, "nsys")
        monkeypatch.setenv(NSYS_COMMAND_ENV, "/opt/nsys")
        monkeypatch.setenv(NSYS_SESSION_ENV, "session-42")

        def run(command):
            if command[1] == "start":
                sqlite_report.touch()
            return _result()

        run_profiler = Mock(side_effect=run)
        synchronize = Mock()
        display = Mock()
        monkeypatch.setattr(magics, "_run_profiler_command", run_profiler)
        monkeypatch.setattr(magics, "_synchronize_current_cuda_context", synchronize)
        monkeypatch.setattr("nsightful.notebook.display_nsys_sqlite_file_in_notebook", display)
        cell_result = SimpleNamespace(success=True)
        shell = _shell(cell_result)

        returned = magics.NSYSMagics(shell=shell).nsys(
            f"--output '{output}' --sample process-tree --cpuctxsw=process-tree",
            "launch_kernel()",
        )

        assert returned is None
        shell.run_cell.assert_called_once_with("launch_kernel()")
        synchronize.assert_called_once_with()
        assert run_profiler.call_count == 2
        start_command = run_profiler.call_args_list[0].args[0]
        callback = next(
            argument
            for argument in start_command
            if argument.startswith("--after-collection-start=")
        )
        assert start_command == [
            "/opt/nsys",
            "start",
            "--session=session-42",
            f"--output={report}",
            "--force-overwrite=true",
            "--export=sqlite",
            callback,
            "--sample",
            "process-tree",
            "--cpuctxsw=process-tree",
        ]
        assert run_profiler.call_args_list[1] == call(["/opt/nsys", "stop", "--session=session-42"])
        display.assert_called_once_with(str(sqlite_report))
        assert f"[nsys] Report: {report}" in capsys.readouterr().out

    def test_no_display_does_not_require_sqlite_export(self, monkeypatch, tmp_path):
        report = tmp_path / "timeline.nsys-rep"
        monkeypatch.setenv(PROFILER_ENV, "nsys")
        monkeypatch.setenv(NSYS_COMMAND_ENV, "nsys")
        monkeypatch.setenv(NSYS_SESSION_ENV, "session-1")
        run_profiler = Mock(return_value=_result())
        display = Mock()
        monkeypatch.setattr(magics, "_run_profiler_command", run_profiler)
        monkeypatch.setattr(magics, "_synchronize_current_cuda_context", Mock())
        monkeypatch.setattr("nsightful.notebook.display_nsys_sqlite_file_in_notebook", display)

        magics.NSYSMagics(shell=_shell(SimpleNamespace(success=True))).nsys(
            f"-o {report} --no-display", "launch_kernel()"
        )

        assert run_profiler.call_count == 2
        display.assert_not_called()

    def test_escapes_percent_in_nsys_output_path(self, monkeypatch, tmp_path):
        report = tmp_path / "timeline-%n.nsys-rep"
        monkeypatch.setenv(PROFILER_ENV, "nsys")
        monkeypatch.setenv(NSYS_COMMAND_ENV, "nsys")
        monkeypatch.setenv(NSYS_SESSION_ENV, "session-1")
        run_profiler = Mock(return_value=_result())
        monkeypatch.setattr(magics, "_run_profiler_command", run_profiler)
        monkeypatch.setattr(magics, "_synchronize_current_cuda_context", Mock())

        magics.NSYSMagics(shell=_shell(SimpleNamespace(success=True))).nsys(
            f"-o {report} --no-display", "launch_kernel()"
        )

        start_command = run_profiler.call_args_list[0].args[0]
        assert f"--output={str(report).replace('%', '%%')}" in start_command

    def test_short_sample_option_replaces_only_sample_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv(PROFILER_ENV, "nsys")
        monkeypatch.setenv(NSYS_COMMAND_ENV, "nsys")
        monkeypatch.setenv(NSYS_SESSION_ENV, "session-1")
        run_profiler = Mock(return_value=_result())
        monkeypatch.setattr(magics, "_run_profiler_command", run_profiler)
        monkeypatch.setattr(magics, "_synchronize_current_cuda_context", Mock())

        magics.NSYSMagics(shell=_shell(SimpleNamespace(success=True))).nsys(
            f"-o {tmp_path / 'trace'} --no-display -s process-tree", "launch_kernel()"
        )

        start_command = run_profiler.call_args_list[0].args[0]
        assert "--sample=none" not in start_command
        assert "--cpuctxsw=none" in start_command
        assert start_command[-2:] == ["-s", "process-tree"]

    def test_attached_short_sample_option_replaces_sample_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv(PROFILER_ENV, "nsys")
        monkeypatch.setenv(NSYS_COMMAND_ENV, "nsys")
        monkeypatch.setenv(NSYS_SESSION_ENV, "session-1")
        run_profiler = Mock(return_value=_result())
        monkeypatch.setattr(magics, "_run_profiler_command", run_profiler)
        monkeypatch.setattr(magics, "_synchronize_current_cuda_context", Mock())

        magics.NSYSMagics(shell=_shell(SimpleNamespace(success=True))).nsys(
            f"-o {tmp_path / 'trace'} --no-display -sprocess-tree", "launch_kernel()"
        )

        start_command = run_profiler.call_args_list[0].args[0]
        assert "--sample=none" not in start_command
        assert "-sprocess-tree" in start_command

    def test_stats_output_from_stop_is_displayed(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv(PROFILER_ENV, "nsys")
        monkeypatch.setenv(NSYS_COMMAND_ENV, "nsys")
        monkeypatch.setenv(NSYS_SESSION_ENV, "session-1")
        run_profiler = Mock(
            side_effect=[_result(), _result(stdout="CUDA API Summary\n", stderr="stats note\n")]
        )
        monkeypatch.setattr(magics, "_run_profiler_command", run_profiler)
        monkeypatch.setattr(magics, "_synchronize_current_cuda_context", Mock())

        magics.NSYSMagics(shell=_shell(SimpleNamespace(success=True))).nsys(
            f"-o {tmp_path / 'trace'} --no-display --stats=true", "launch_kernel()"
        )

        captured = capsys.readouterr()
        assert "CUDA API Summary" in captured.out
        assert "stats note" in captured.err

    def test_stops_profiler_when_cell_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv(PROFILER_ENV, "nsys")
        monkeypatch.setenv(NSYS_COMMAND_ENV, "nsys")
        monkeypatch.setenv(NSYS_SESSION_ENV, "session-1")
        run_profiler = Mock(return_value=_result())
        synchronize = Mock()
        monkeypatch.setattr(magics, "_run_profiler_command", run_profiler)
        monkeypatch.setattr(magics, "_synchronize_current_cuda_context", synchronize)
        shell = _shell(side_effect=RuntimeError("cell exploded"))

        with pytest.raises(RuntimeError, match="cell exploded"):
            magics.NSYSMagics(shell=shell).nsys(
                f"-o {tmp_path / 'trace'} --no-display", "explode()"
            )

        assert run_profiler.call_count == 2
        assert run_profiler.call_args_list[-1] == call(["nsys", "stop", "--session=session-1"])
        synchronize.assert_not_called()

    def test_failed_execution_result_is_propagated_after_stop(self, monkeypatch, tmp_path):
        monkeypatch.setenv(PROFILER_ENV, "nsys")
        monkeypatch.setenv(NSYS_COMMAND_ENV, "nsys")
        monkeypatch.setenv(NSYS_SESSION_ENV, "session-1")
        run_profiler = Mock(return_value=_result())
        monkeypatch.setattr(magics, "_run_profiler_command", run_profiler)
        monkeypatch.setattr(magics, "_synchronize_current_cuda_context", Mock())
        cell_result = SimpleNamespace(
            success=False,
            error_before_exec=None,
            error_in_exec=ValueError("cell failed"),
        )

        with pytest.raises(ValueError, match="cell failed"):
            magics.NSYSMagics(shell=_shell(cell_result)).nsys(
                f"-o {tmp_path / 'trace'} --no-display", "raise ValueError('cell failed')"
            )

        assert run_profiler.call_count == 2

    def test_stop_failure_does_not_mask_failed_execution_result(
        self, monkeypatch, tmp_path, capsys
    ):
        monkeypatch.setenv(PROFILER_ENV, "nsys")
        monkeypatch.setenv(NSYS_COMMAND_ENV, "nsys")
        monkeypatch.setenv(NSYS_SESSION_ENV, "session-1")
        run_profiler = Mock(side_effect=[_result(), _result(1, stderr="stop also failed")])
        monkeypatch.setattr(magics, "_run_profiler_command", run_profiler)
        monkeypatch.setattr(magics, "_synchronize_current_cuda_context", Mock())
        cell_result = SimpleNamespace(
            success=False,
            error_before_exec=None,
            error_in_exec=ValueError("cell failed"),
        )

        with pytest.raises(ValueError, match="cell failed"):
            magics.NSYSMagics(shell=_shell(cell_result)).nsys(
                f"-o {tmp_path / 'trace'} --no-display", "raise ValueError('cell failed')"
            )

        assert "stop also failed" in capsys.readouterr().err

    def test_stop_failure_does_not_mask_cell_error(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv(PROFILER_ENV, "nsys")
        monkeypatch.setenv(NSYS_COMMAND_ENV, "nsys")
        monkeypatch.setenv(NSYS_SESSION_ENV, "session-1")
        run_profiler = Mock(side_effect=[_result(), _result(1, stderr="stop also failed")])
        monkeypatch.setattr(magics, "_run_profiler_command", run_profiler)
        shell = _shell(side_effect=RuntimeError("cell exploded"))

        with pytest.raises(RuntimeError, match="cell exploded"):
            magics.NSYSMagics(shell=shell).nsys(
                f"-o {tmp_path / 'trace'} --no-display", "explode()"
            )

        assert "stop also failed" in capsys.readouterr().err

    def test_start_failure_does_not_run_cell_or_stop(self, monkeypatch):
        monkeypatch.setenv(PROFILER_ENV, "nsys")
        monkeypatch.setenv(NSYS_COMMAND_ENV, "nsys")
        monkeypatch.setenv(NSYS_SESSION_ENV, "session-1")
        run_profiler = Mock(return_value=_result(1, stderr="start failed"))
        monkeypatch.setattr(magics, "_run_profiler_command", run_profiler)
        shell = _shell()

        with pytest.raises(UsageError, match="start failed"):
            magics.NSYSMagics(shell=shell).nsys("--no-display", "launch_kernel()")

        shell.run_cell.assert_not_called()
        run_profiler.assert_called_once()

    def test_stop_failure_is_reported(self, monkeypatch):
        monkeypatch.setenv(PROFILER_ENV, "nsys")
        monkeypatch.setenv(NSYS_COMMAND_ENV, "nsys")
        monkeypatch.setenv(NSYS_SESSION_ENV, "session-1")
        run_profiler = Mock(side_effect=[_result(), _result(1, stderr="stop failed")])
        monkeypatch.setattr(magics, "_run_profiler_command", run_profiler)
        monkeypatch.setattr(magics, "_synchronize_current_cuda_context", Mock())

        with pytest.raises(UsageError, match="stop failed"):
            magics.NSYSMagics(shell=_shell(SimpleNamespace(success=True))).nsys(
                "--no-display", "launch_kernel()"
            )

    def test_missing_sqlite_export_is_reported_after_stop(self, monkeypatch, tmp_path):
        monkeypatch.setenv(PROFILER_ENV, "nsys")
        monkeypatch.setenv(NSYS_COMMAND_ENV, "nsys")
        monkeypatch.setenv(NSYS_SESSION_ENV, "session-1")
        run_profiler = Mock(return_value=_result())
        monkeypatch.setattr(magics, "_run_profiler_command", run_profiler)
        monkeypatch.setattr(magics, "_synchronize_current_cuda_context", Mock())

        with pytest.raises(UsageError, match="expected SQLite export"):
            magics.NSYSMagics(shell=_shell(SimpleNamespace(success=True))).nsys(
                f"-o {tmp_path / 'missing'}", "launch_kernel()"
            )

        assert run_profiler.call_count == 2


class TestExtensionRegistration:
    @pytest.mark.parametrize(
        ("profiler", "magic_class"), (("ncu", magics.NCUMagics), ("nsys", magics.NSYSMagics))
    )
    def test_registers_only_active_profiler_magic(self, monkeypatch, profiler, magic_class):
        monkeypatch.setenv(PROFILER_ENV, profiler)
        ipython = SimpleNamespace(register_magics=Mock())

        magics.load_ipython_extension(ipython)

        ipython.register_magics.assert_called_once_with(magic_class)

    def test_extension_rejects_unwrapped_ipython(self):
        ipython = SimpleNamespace(register_magics=Mock())

        with pytest.raises(RuntimeError, match="must be loaded"):
            magics.load_ipython_extension(ipython)

        ipython.register_magics.assert_not_called()

    def test_ncu_extension_registers_temp_report_cleanup(self, monkeypatch, tmp_path):
        temp_dir = tmp_path / "nsightful-ncu-session"
        monkeypatch.setenv(PROFILER_ENV, "ncu")
        monkeypatch.setenv(NCU_TEMP_DIR_ENV, str(temp_dir))
        register = Mock()
        monkeypatch.setattr(magics.atexit, "register", register)
        ipython = SimpleNamespace(register_magics=Mock())

        magics.load_ipython_extension(ipython)

        register.assert_called_once_with(magics._cleanup_ncu_temp_dir, str(temp_dir))
