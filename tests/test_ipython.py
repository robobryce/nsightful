"""CPU-only tests for the Nsight IPython wrappers."""

import builtins
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nsightful import ipython


@pytest.mark.parametrize(
    ("arguments", "expected_profiler", "expected_ipython"),
    [
        ([], [], []),
        (["--no-banner"], [], ["--no-banner"]),
        (
            ["--set", "full", "--", "kernel", "-f", "connection.json"],
            ["--set", "full"],
            ["kernel", "-f", "connection.json"],
        ),
        (["--", "--quick"], [], ["--quick"]),
        (
            ["--trace=cuda", "--", "kernel", "--", "tail"],
            ["--trace=cuda"],
            ["kernel", "--", "tail"],
        ),
    ],
)
def test_split_wrapper_arguments(arguments, expected_profiler, expected_ipython):
    assert ipython.split_wrapper_arguments(arguments) == (expected_profiler, expected_ipython)


def test_build_ipython_command_for_terminal(monkeypatch):
    monkeypatch.setattr(ipython.sys, "executable", "/venv/bin/python")

    assert ipython.build_ipython_command(["--no-banner", "-i", "script.py"]) == [
        "/venv/bin/python",
        "-m",
        "IPython",
        "--ext=nsightful.magics",
        "--InteractiveShellApp.reraise_ipython_extension_failures=True",
        "--no-banner",
        "-i",
        "script.py",
    ]


def test_build_ipython_command_for_kernel(monkeypatch):
    monkeypatch.setattr(ipython.sys, "executable", "/venv/bin/python")

    assert ipython.build_ipython_command(["kernel", "-f", "connection.json"]) == [
        "/venv/bin/python",
        "-m",
        "ipykernel_launcher",
        "--IPKernelApp.extra_extensions=nsightful.magics",
        "--IPKernelApp.reraise_ipython_extension_failures=True",
        "-f",
        "connection.json",
    ]


def test_build_ncu_profiler_command_and_environment(monkeypatch, tmp_path):
    profiler_dir = tmp_path / "ncu-work"
    which = Mock(return_value="/opt/nsight-compute/ncu")
    mkdtemp = Mock(return_value=str(profiler_dir))
    monkeypatch.setattr(ipython.shutil, "which", which)
    monkeypatch.setattr(ipython.tempfile, "mkdtemp", mkdtemp)
    monkeypatch.setattr(ipython.sys, "executable", "/venv/bin/python")
    original_environment = {"KEEP_ME": "yes", ipython.PROFILER_ENV: "nsys"}

    command, environment = ipython.build_profiler_command(
        "ncu",
        ["--set", "full", "--clock-control=none"],
        ["--no-banner"],
        original_environment,
    )

    report = profiler_dir / "kernel.ncu-rep"
    assert command == [
        "/opt/nsight-compute/ncu",
        "--target-processes=application-only",
        "--forward-signals",
        "--force-overwrite",
        f"--export={report}",
        "--nvtx",
        "--nvtx-include=regex:Nsightful@.*",
        "--set",
        "full",
        "--clock-control=none",
        "/venv/bin/python",
        "-m",
        "IPython",
        "--ext=nsightful.magics",
        "--InteractiveShellApp.reraise_ipython_extension_failures=True",
        "--no-banner",
    ]
    assert environment == {
        "KEEP_ME": "yes",
        ipython.PROFILER_ENV: "ncu",
        ipython.NCU_COMMAND_ENV: "/opt/nsight-compute/ncu",
        ipython.NCU_REPORT_ENV: str(report),
        ipython.NCU_TEMP_DIR_ENV: str(profiler_dir),
    }
    assert original_environment == {"KEEP_ME": "yes", ipython.PROFILER_ENV: "nsys"}
    which.assert_called_once_with("ncu")
    mkdtemp.assert_called_once_with(prefix="nsightful-ncu-")


def test_build_nsys_profiler_command_and_environment(monkeypatch):
    which = Mock(return_value="/opt/nsight-systems/nsys")
    monkeypatch.setattr(ipython.shutil, "which", which)
    monkeypatch.setattr(ipython.os, "getpid", lambda: 4321)
    monkeypatch.setattr(
        ipython.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="0123456789abcdef0123456789abcdef"),
    )
    monkeypatch.setattr(ipython.sys, "executable", "/venv/bin/python")
    original_environment = {"KEEP_ME": "yes"}

    command, environment = ipython.build_profiler_command(
        "nsys",
        ["--trace=cuda,nvtx"],
        ["kernel", "-f", "connection.json"],
        original_environment,
    )

    session = "nsightful-4321-0123456789ab"
    assert command == [
        "/opt/nsight-systems/nsys",
        "launch",
        f"--session={session}",
        "--trace=cuda,nvtx",
        "/venv/bin/python",
        "-m",
        "ipykernel_launcher",
        "--IPKernelApp.extra_extensions=nsightful.magics",
        "--IPKernelApp.reraise_ipython_extension_failures=True",
        "-f",
        "connection.json",
    ]
    assert environment == {
        "KEEP_ME": "yes",
        ipython.PROFILER_ENV: "nsys",
        ipython.NSYS_COMMAND_ENV: "/opt/nsight-systems/nsys",
        ipython.NSYS_SESSION_ENV: session,
    }
    assert original_environment == {"KEEP_ME": "yes"}
    which.assert_called_once_with("nsys")


def test_build_profiler_command_copies_process_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(ipython.shutil, "which", lambda _profiler: "/usr/bin/ncu")
    monkeypatch.setattr(ipython.tempfile, "mkdtemp", lambda **_kwargs: str(tmp_path))
    monkeypatch.setenv("NSIGHTFUL_TEST_SENTINEL", "present")

    _, environment = ipython.build_profiler_command("ncu", [], [])

    assert environment["NSIGHTFUL_TEST_SENTINEL"] == "present"
    environment["NSIGHTFUL_TEST_SENTINEL"] = "changed"
    assert ipython.os.environ["NSIGHTFUL_TEST_SENTINEL"] == "present"


@pytest.mark.parametrize(
    ("profiler", "arguments", "conflicts"),
    [
        ("ncu", ["--export=mine", "--force-overwrite"], "--export, --force-overwrite"),
        ("ncu", ["-o", "mine"], "-o"),
        ("ncu", ["--target-processes", "all"], "--target-processes"),
        ("ncu", ["--replay-mode=application"], "--replay-mode"),
        ("ncu", ["-c", "1"], "-c"),
        ("ncu", ["-c1"], "-c"),
        ("ncu", ["-o/tmp/report"], "-o"),
        ("nsys", ["-xfalse"], "-x"),
        ("nsys", ["--session=my-session"], "--session"),
        ("nsys", ["--session-new", "my-session"], "--session-new"),
    ],
)
def test_build_profiler_command_rejects_owned_options_before_tool_lookup(
    monkeypatch, profiler, arguments, conflicts
):
    which = Mock(side_effect=AssertionError("tool lookup should not run"))
    monkeypatch.setattr(ipython.shutil, "which", which)

    with pytest.raises(
        ValueError,
        match=rf"nsightful controls these {profiler} options: {conflicts}",
    ):
        ipython.build_profiler_command(profiler, arguments, [])

    which.assert_not_called()


def test_build_profiler_command_rejects_unknown_profiler(monkeypatch):
    which = Mock(side_effect=AssertionError("tool lookup should not run"))
    monkeypatch.setattr(ipython.shutil, "which", which)

    with pytest.raises(ValueError, match="unsupported profiler: nvprof"):
        ipython.build_profiler_command("nvprof", [], [])

    which.assert_not_called()


@pytest.mark.parametrize(
    ("profiler", "product"),
    [("ncu", "NVIDIA Nsight Compute"), ("nsys", "NVIDIA Nsight Systems")],
)
def test_build_profiler_command_reports_missing_tool(monkeypatch, profiler, product):
    monkeypatch.setattr(ipython.shutil, "which", lambda _profiler: None)

    with pytest.raises(FileNotFoundError, match=rf"{profiler} was not found.*{product}"):
        ipython.build_profiler_command(profiler, [], [])


def test_run_ipython_replaces_process_with_built_command(monkeypatch):
    command = ["/usr/bin/ncu", "--flag", "/venv/bin/python", "-m", "IPython"]
    environment = {"PROFILE": "ncu"}
    build = Mock(return_value=(command, environment))

    class ProcessReplaced(Exception):
        pass

    execvpe = Mock(side_effect=ProcessReplaced)
    validate = Mock()
    monkeypatch.setattr(ipython, "build_profiler_command", build)
    monkeypatch.setattr(ipython, "validate_profiler_version", validate)
    monkeypatch.setattr(ipython.os, "execvpe", execvpe)

    with pytest.raises(ProcessReplaced):
        ipython.run_ipython("ncu", ["--set", "full", "--", "kernel", "-f", "connection.json"])

    build.assert_called_once_with("ncu", ["--set", "full"], ["kernel", "-f", "connection.json"])
    validate.assert_called_once_with("ncu", command[0])
    execvpe.assert_called_once_with(command[0], command, environment)


def test_run_ipython_fails_if_exec_unexpectedly_returns(monkeypatch):
    monkeypatch.setattr(
        ipython,
        "build_profiler_command",
        lambda *_args: (["/usr/bin/nsys", "launch"], {"PROFILE": "nsys"}),
    )
    monkeypatch.setattr(ipython.os, "execvpe", Mock(return_value=None))
    monkeypatch.setattr(ipython, "validate_profiler_version", Mock())

    with pytest.raises(AssertionError, match="os.execvpe unexpectedly returned"):
        ipython.run_ipython("nsys", [])


@pytest.mark.parametrize(
    ("profiler", "output", "expected"),
    [
        ("ncu", "NVIDIA Nsight Compute Version 2024.3.1.0", (2024, 3, 1)),
        ("nsys", "NVIDIA Nsight Systems version 2026.1.3.243", (2026, 1, 3)),
    ],
)
def test_validate_profiler_version(monkeypatch, profiler, output, expected):
    run = Mock(return_value=ipython.subprocess.CompletedProcess([], 0, output, ""))
    monkeypatch.setattr(ipython.subprocess, "run", run)

    assert ipython.validate_profiler_version(profiler, f"/opt/{profiler}") == expected
    run.assert_called_once_with(
        [f"/opt/{profiler}", "--version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize(
    ("profiler", "output", "message"),
    [
        ("ncu", "Version 2024.2.0", "ncu 2024.2.0 is too old"),
        ("nsys", "NVIDIA Nsight Systems version 2024.6.2", "nsys 2024.6.2 is too old"),
        ("nsys", "not a version", "could not determine nsys version"),
    ],
)
def test_validate_profiler_version_rejects_unsupported_output(
    monkeypatch, profiler, output, message
):
    monkeypatch.setattr(
        ipython.subprocess,
        "run",
        Mock(return_value=ipython.subprocess.CompletedProcess([], 0, output, "")),
    )

    with pytest.raises(RuntimeError, match=message):
        ipython.validate_profiler_version(profiler, f"/opt/{profiler}")


def test_validate_profiler_version_reports_execution_error(monkeypatch):
    monkeypatch.setattr(
        ipython.subprocess,
        "run",
        Mock(side_effect=ipython.subprocess.TimeoutExpired("ncu", 10)),
    )

    with pytest.raises(RuntimeError, match="could not determine ncu version"):
        ipython.validate_profiler_version("ncu", "/opt/ncu")


def _recording_kernel_spec_manager(records, destination):
    class RecordingKernelSpecManager:
        def install_kernel_spec(self, source_dir, **kwargs):
            source = Path(source_dir)
            records.append(
                {
                    "source": source,
                    "spec": json.loads((source / "kernel.json").read_text()),
                    "kwargs": kwargs,
                }
            )
            return str(destination)

    return RecordingKernelSpecManager


@pytest.mark.parametrize(
    ("profiler", "display_name"),
    [("ncu", "Python 3 (Nsight Compute)"), ("nsys", "Python 3 (Nsight Systems)")],
)
def test_install_kernel_default_spec_and_user_location(
    monkeypatch, tmp_path, capsys, profiler, display_name
):
    import jupyter_client.kernelspec

    records = []
    destination = tmp_path / "installed" / profiler
    manager = _recording_kernel_spec_manager(records, destination)
    monkeypatch.setattr(jupyter_client.kernelspec, "KernelSpecManager", manager)
    monkeypatch.setattr(ipython.sys, "executable", "/venv/bin/python")

    result = ipython.install_kernel(profiler, [])

    assert result == str(destination)
    assert len(records) == 1
    record = records[0]
    assert record["source"].name.startswith("nsightful-kernelspec-")
    assert not record["source"].exists()
    assert record["kwargs"] == {
        "kernel_name": f"nsightful-{profiler}",
        "user": True,
        "prefix": None,
        "replace": True,
    }
    assert record["spec"] == {
        "argv": [
            "/venv/bin/python",
            "-m",
            "nsightful.ipython",
            profiler,
            "--",
            "kernel",
            "-f",
            "{connection_file}",
        ],
        "display_name": display_name,
        "language": "python",
        "metadata": {"debugger": True, "nsightful_profiler": profiler},
    }
    assert capsys.readouterr().out == (f"Installed {display_name!r} kernelspec at {destination}\n")


def test_install_kernel_custom_spec_profiler_args_and_prefix(monkeypatch, tmp_path):
    import jupyter_client.kernelspec

    records = []
    prefix = tmp_path / "jupyter-prefix"
    destination = prefix / "share" / "jupyter" / "kernels" / "gpu-profile"
    manager = _recording_kernel_spec_manager(records, destination)
    monkeypatch.setattr(jupyter_client.kernelspec, "KernelSpecManager", manager)
    monkeypatch.setattr(ipython.sys, "executable", "/fresh-venv/bin/python")

    result = ipython.install_kernel(
        "ncu",
        [
            "--name",
            "gpu-profile",
            "--display-name",
            "GPU profile",
            "--prefix",
            str(prefix),
            "--profiler-args=--set 'full set' --clock-control=none",
        ],
    )

    assert result == str(destination)
    assert records[0]["kwargs"] == {
        "kernel_name": "gpu-profile",
        "user": False,
        "prefix": str(prefix),
        "replace": True,
    }
    assert records[0]["spec"]["argv"] == [
        "/fresh-venv/bin/python",
        "-m",
        "nsightful.ipython",
        "ncu",
        "--set",
        "full set",
        "--clock-control=none",
        "--",
        "kernel",
        "-f",
        "{connection_file}",
    ]
    assert records[0]["spec"]["display_name"] == "GPU profile"


def test_install_kernel_sys_prefix_location(monkeypatch, tmp_path):
    import jupyter_client.kernelspec

    records = []
    destination = tmp_path / "installed"
    manager = _recording_kernel_spec_manager(records, destination)
    monkeypatch.setattr(jupyter_client.kernelspec, "KernelSpecManager", manager)
    monkeypatch.setattr(ipython.sys, "prefix", "/fresh-venv")

    ipython.install_kernel("nsys", ["--sys-prefix"])

    assert records[0]["kwargs"] == {
        "kernel_name": "nsightful-nsys",
        "user": False,
        "prefix": "/fresh-venv",
        "replace": True,
    }


@pytest.mark.parametrize(
    ("profiler", "profiler_args", "conflict"),
    [
        ("ncu", "--export=mine", "--export"),
        ("nsys", "--session mine", "--session"),
    ],
)
def test_install_kernel_rejects_owned_profiler_arguments(capsys, profiler, profiler_args, conflict):
    with pytest.raises(SystemExit) as error:
        ipython.install_kernel(profiler, [f"--profiler-args={profiler_args}"])

    assert error.value.code == 2
    assert f"nsightful controls these {profiler} options: {conflict}" in capsys.readouterr().err


def test_install_kernel_reports_malformed_profiler_arguments(capsys):
    with pytest.raises(SystemExit) as error:
        ipython.install_kernel("ncu", ["--profiler-args='unterminated"])

    assert error.value.code == 2
    assert "No closing quotation" in capsys.readouterr().err


def test_install_kernel_reports_missing_notebook_dependency(monkeypatch):
    real_import = builtins.__import__

    def import_without_jupyter_client(name, *args, **kwargs):
        if name == "jupyter_client.kernelspec":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_jupyter_client)

    with pytest.raises(RuntimeError, match=r'pip install "nsightful\[notebook\]"'):
        ipython.install_kernel("ncu", [])


@pytest.mark.parametrize(
    ("entrypoint", "profiler"),
    [(ipython.main_ncu, "ncu"), (ipython.main_nsys, "nsys")],
)
def test_console_entrypoints_forward_arguments(monkeypatch, entrypoint, profiler):
    dispatch = Mock()
    monkeypatch.setattr(ipython, "_main_for_profiler", dispatch)
    monkeypatch.setattr(ipython.sys, "argv", [f"nsightful-{profiler}", "--flag", "value"])

    entrypoint()

    dispatch.assert_called_once_with(profiler, ["--flag", "value"])


def test_module_entrypoint_selects_profiler(monkeypatch):
    dispatch = Mock()
    monkeypatch.setattr(ipython, "_main_for_profiler", dispatch)
    monkeypatch.setattr(
        ipython.sys,
        "argv",
        ["python", "nsys", "--trace=cuda", "--", "kernel", "-f", "connection.json"],
    )

    ipython.main()

    dispatch.assert_called_once_with(
        "nsys", ["--trace=cuda", "--", "kernel", "-f", "connection.json"]
    )


def test_profiler_dispatches_install_without_exec(monkeypatch):
    install = Mock()
    run = Mock()
    monkeypatch.setattr(ipython, "install_kernel", install)
    monkeypatch.setattr(ipython, "run_ipython", run)

    ipython._main_for_profiler("ncu", ["install", "--name", "custom"])

    install.assert_called_once_with("ncu", ["--name", "custom"])
    run.assert_not_called()


def test_profiler_dispatches_wrapper_execution(monkeypatch):
    install = Mock()
    run = Mock()
    monkeypatch.setattr(ipython, "install_kernel", install)
    monkeypatch.setattr(ipython, "run_ipython", run)

    ipython._main_for_profiler("nsys", ["--trace=cuda", "--", "kernel"])

    run.assert_called_once_with("nsys", ["--trace=cuda", "--", "kernel"])
    install.assert_not_called()
