"""Launch IPython under NVIDIA Nsight Compute or Nsight Systems."""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, NoReturn, Optional, Sequence, Tuple

PROFILER_ENV = "NSIGHTFUL_PROFILER"
NCU_COMMAND_ENV = "NSIGHTFUL_NCU_COMMAND"
NCU_REPORT_ENV = "NSIGHTFUL_NCU_REPORT"
NCU_TEMP_DIR_ENV = "NSIGHTFUL_NCU_TEMP_DIR"
NSYS_COMMAND_ENV = "NSIGHTFUL_NSYS_COMMAND"
NSYS_SESSION_ENV = "NSIGHTFUL_NSYS_SESSION"

_PROFILERS = ("ncu", "nsys")
_MINIMUM_VERSIONS = {"ncu": (2024, 3, 0), "nsys": (2024, 1, 1)}
_NCU_OWNED_OPTIONS = {
    "-c",
    "-f",
    "-o",
    "-s",
    "--app-replay-buffer",
    "--app-replay-match",
    "--app-replay-mode",
    "--export",
    "--force-overwrite",
    "--forward-signals",
    "--kill",
    "--launch-count",
    "--launch-skip",
    "--launch-skip-before-match",
    "--mode",
    "--nvtx",
    "--nvtx-exclude",
    "--nvtx-include",
    "--nvtx-push-pop-scope",
    "--profile-from-start",
    "--range-replay-options",
    "--replay-mode",
    "--target-processes",
}
_NSYS_OWNED_OPTIONS = {
    "-x",
    "--session",
    "--session-new",
    "--stop-on-exit",
}


def _option_name(argument: str) -> str:
    """Return an option without its ``=value`` suffix."""
    for short_option in ("-c", "-o", "-s", "-x"):
        if argument.startswith(short_option) and len(argument) > len(short_option):
            return short_option
    return argument.split("=", 1)[0]


def split_wrapper_arguments(arguments: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Split profiler arguments from IPython arguments at ``--``.

    With no separator, all arguments are treated as IPython arguments. This makes
    ``nsightful-ncu`` behave like an IPython executable for the common no-profiler-option case.
    """
    args = list(arguments)
    if "--" not in args:
        return [], args
    separator = args.index("--")
    return args[:separator], args[separator + 1 :]


def build_ipython_command(arguments: Sequence[str]) -> List[str]:
    """Build the Python command used as the profiler target."""
    args = list(arguments)
    if args and args[0] == "kernel":
        return [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "--IPKernelApp.extra_extensions=nsightful.magics",
            "--IPKernelApp.reraise_ipython_extension_failures=True",
            *args[1:],
        ]
    return [
        sys.executable,
        "-m",
        "IPython",
        "--ext=nsightful.magics",
        "--InteractiveShellApp.reraise_ipython_extension_failures=True",
        *args,
    ]


def _validate_profiler_arguments(profiler: str, arguments: Sequence[str]) -> None:
    owned = _NCU_OWNED_OPTIONS if profiler == "ncu" else _NSYS_OWNED_OPTIONS
    conflicts = sorted({_option_name(arg) for arg in arguments} & owned)
    if conflicts:
        joined = ", ".join(conflicts)
        raise ValueError(f"nsightful controls these {profiler} options: {joined}")


def build_profiler_command(
    profiler: str,
    profiler_arguments: Sequence[str],
    ipython_arguments: Sequence[str],
    environment: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """Build a profiler command and the target environment.

    Command construction is kept separate from ``exec`` so the wrapper can be unit tested.
    """
    if profiler not in _PROFILERS:
        raise ValueError(f"unsupported profiler: {profiler}")
    _validate_profiler_arguments(profiler, profiler_arguments)

    executable = shutil.which(profiler)
    if executable is None:
        raise FileNotFoundError(
            f"{profiler} was not found on PATH; install NVIDIA Nsight "
            f"{'Compute' if profiler == 'ncu' else 'Systems'}"
        )

    env = dict(os.environ if environment is None else environment)
    env[PROFILER_ENV] = profiler
    target = build_ipython_command(ipython_arguments)

    if profiler == "ncu":
        report_dir = Path(tempfile.mkdtemp(prefix="nsightful-ncu-"))
        base_report = report_dir / "kernel.ncu-rep"
        env[NCU_COMMAND_ENV] = executable
        env[NCU_REPORT_ENV] = str(base_report)
        env[NCU_TEMP_DIR_ENV] = str(report_dir)
        command = [
            executable,
            "--target-processes=application-only",
            "--forward-signals",
            "--force-overwrite",
            f"--export={base_report}",
            "--nvtx",
            "--nvtx-include=regex:Nsightful@.*",
            *profiler_arguments,
            *target,
        ]
    else:
        session = f"nsightful-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        env[NSYS_COMMAND_ENV] = executable
        env[NSYS_SESSION_ENV] = session
        command = [
            executable,
            "launch",
            f"--session={session}",
            *profiler_arguments,
            *target,
        ]

    return command, env


def validate_profiler_version(profiler: str, executable: str) -> Tuple[int, int, int]:
    """Validate that a profiler supports the long-lived kernel control model."""
    try:
        result = subprocess.run(
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"could not determine {profiler} version: {error}") from error
    output = f"{result.stdout}\n{result.stderr}"
    match = re.search(r"\b(20\d{2})\.(\d+)(?:\.(\d+))?", output)
    if result.returncode != 0 or match is None:
        detail = output.strip() or f"exit status {result.returncode}"
        raise RuntimeError(f"could not determine {profiler} version: {detail}")
    version = (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3) or 0),
    )
    minimum = _MINIMUM_VERSIONS[profiler]
    if version < minimum:
        found = ".".join(str(part) for part in version)
        required = ".".join(str(part) for part in minimum)
        raise RuntimeError(f"{profiler} {found} is too old; nsightful requires {required} or newer")
    return version


def run_ipython(profiler: str, arguments: Sequence[str]) -> NoReturn:
    """Replace the current process with IPython wrapped by ``profiler``."""
    profiler_args, ipython_args = split_wrapper_arguments(arguments)
    command, environment = build_profiler_command(profiler, profiler_args, ipython_args)
    try:
        validate_profiler_version(profiler, command[0])
        os.execvpe(command[0], command, environment)
    except BaseException:
        temp_dir = environment.get(NCU_TEMP_DIR_ENV)
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    temp_dir = environment.get(NCU_TEMP_DIR_ENV)
    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)
    raise AssertionError("os.execvpe unexpectedly returned")


def _kernel_install_parser(profiler: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"nsightful-{profiler} install",
        description=f"Install a Jupyter kernel that runs IPython under {profiler}.",
    )
    parser.add_argument("--name", default=f"nsightful-{profiler}", help="Kernel identifier")
    default_display = "Nsight Compute" if profiler == "ncu" else "Nsight Systems"
    parser.add_argument(
        "--display-name",
        default=f"Python 3 ({default_display})",
        help="Name shown in Jupyter",
    )
    location = parser.add_mutually_exclusive_group()
    location.add_argument("--user", action="store_true", help="Install for the current user")
    location.add_argument("--sys-prefix", action="store_true", help="Install into sys.prefix")
    location.add_argument("--prefix", help="Install into an explicit prefix")
    parser.add_argument(
        "--profiler-args",
        default="",
        metavar="ARGS",
        help=f"Arguments passed to {profiler}, as one shell-quoted string",
    )
    return parser


def install_kernel(profiler: str, arguments: Sequence[str]) -> str:
    """Install a venv-bound Jupyter kernelspec for one profiler."""
    parser = _kernel_install_parser(profiler)
    args = parser.parse_args(list(arguments))
    try:
        profiler_args = shlex.split(args.profiler_args)
    except ValueError as error:
        parser.error(str(error))
    try:
        _validate_profiler_arguments(profiler, profiler_args)
    except ValueError as error:
        parser.error(str(error))

    try:
        from jupyter_client.kernelspec import KernelSpecManager
    except ImportError as error:
        raise RuntimeError(
            'kernel installation requires the notebook extra: pip install "nsightful[notebook]"'
        ) from error

    kernel_argv = [
        sys.executable,
        "-m",
        "nsightful.ipython",
        profiler,
        *profiler_args,
        "--",
        "kernel",
        "-f",
        "{connection_file}",
    ]
    kernel_spec = {
        "argv": kernel_argv,
        "display_name": args.display_name,
        "language": "python",
        "metadata": {"debugger": True, "nsightful_profiler": profiler},
    }

    with tempfile.TemporaryDirectory(prefix="nsightful-kernelspec-") as temp_dir:
        spec_dir = Path(temp_dir)
        (spec_dir / "kernel.json").write_text(json.dumps(kernel_spec, indent=2) + "\n")
        manager = KernelSpecManager()
        destination = manager.install_kernel_spec(
            str(spec_dir),
            kernel_name=args.name,
            user=args.user or (not args.sys_prefix and args.prefix is None),
            prefix=sys.prefix if args.sys_prefix else args.prefix,
            replace=True,
        )
    print(f"Installed {args.display_name!r} kernelspec at {destination}")
    return destination


def _main_for_profiler(profiler: str, arguments: Sequence[str]) -> None:
    if arguments and arguments[0] == "install":
        install_kernel(profiler, arguments[1:])
        return
    run_ipython(profiler, arguments)


def main_ncu() -> None:
    """Entry point for ``nsightful-ncu``."""
    _main_for_profiler("ncu", sys.argv[1:])


def main_nsys() -> None:
    """Entry point for ``nsightful-nsys``."""
    _main_for_profiler("nsys", sys.argv[1:])


def main() -> None:
    """Module entry point used by installed kernelspecs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiler", choices=_PROFILERS)
    args, remaining = parser.parse_known_args()
    _main_for_profiler(args.profiler, remaining)


if __name__ == "__main__":
    main()
