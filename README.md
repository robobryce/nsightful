# Nsightful

[![Apache 2.0 with LLVM exceptions](https://img.shields.io/badge/license-Apache%202.0%20with%20LLVM%20exceptions-blue.svg)](LICENSE)
[![Tests](https://github.com/brycelelbach/nsightful/actions/workflows/test.yml/badge.svg)](https://github.com/brycelelbach/nsightful/actions/workflows/test.yml)

A Python package for converting [NVIDIA Nsight Compute (NCU)](https://developer.nvidia.com/nsight-compute)
and [NVIDIA Nsight Systems (NSYS)](https://developer.nvidia.com/nsight-systems) profile reports to
other formats and elegantly displaying them in Jupyter notebooks and other web-based no-install
tools.

In a nutshell, `nsightful` contains:
- Nsight Compute CSV -> Python dicts, Markdown.
- Nsight Systems SQLite -> [Google Chrome Trace Event Format](https://perfetto.dev/docs/getting-started/other-formats#chrome-json-format) JSON.
- Command line tools for converting Nsight Compute and Nsight Systems reports to those formats.
- Jupyter notebook widget for displaying Nsight Compute and Nsight Systems reports.
- IPython wrappers and `%%ncu` / `%%nsys` cell magics for profiling notebook cells in place.

## Installation

If you just need the command line tool:

```bash
pip install git+https://github.com/brycelelbach/nsightful.git
```

If you want to use the Jupyter notebook widgets or cell profilers:

```bash
pip install "nsightful[notebook] @ git+https://github.com/brycelelbach/nsightful.git"
```

## Quick Start

### Profiling Notebook Cells

Nsightful can install Jupyter kernels that start IPython under Nsight Compute or Nsight Systems.
The matching cell magic is ready as soon as you select the kernel; there is no enable command and
no kernel restart.

```bash
# Install both kernels for the current user.
nsightful-ncu install --user
nsightful-nsys install --user
```

Select either **Python 3 (Nsight Compute)** or **Python 3 (Nsight Systems)** in Jupyter. A notebook
uses one profiler at a time: the Nsight Compute kernel provides `%%ncu`, while the Nsight Systems
kernel provides `%%nsys`. Variables and imports from earlier cells remain available in the
profiled cell.

```python
import cupy as cp

x = cp.arange(1_000_000)
```

With the Nsight Compute kernel:

```python
%%ncu -o multiply.ncu-rep
x *= 2
```

With the Nsight Systems kernel:

```python
%%nsys -o multiply.nsys-rep
x *= 2
```

Both magics save the native report and display it in the notebook. `%%nsys` also exports the
corresponding SQLite file. Pass `--no-display` to save without rendering. Arguments after `--` on
`%%ncu` are passed to the report import command; other arguments on `%%nsys` are passed to
`nsys start`.

Profiler collection options must be chosen when the wrapper starts. They can be stored in a
custom kernelspec at installation time:

```bash
nsightful-ncu install --user '--profiler-args=--set full --clock-control none'
nsightful-nsys install --user '--profiler-args=--trace=cuda,nvtx,osrt'
```

The wrappers can also launch a terminal IPython directly. Arguments before `--` go to the
profiler, and arguments after it go to IPython:

```bash
nsightful-ncu --set full --
nsightful-nsys --trace=cuda,nvtx,osrt --
```

### Nsight Compute (NCU)

#### Generating NCU CSV Data

First, you need to generate NCU CSV data from your CUDA application:

```bash
# Profile your application
ncu --set full -o myreport ./myapplication

# Export to CSV
ncu --import myreport.ncu-rep --csv > myreport.csv

# Convert CSV to Markdown (output to a file)
nsightful myreport.csv -o myreport.md
```

#### NCU Python Conversion API

```python
import nsightful

# Convert CSV file to Markdown string
with open('myreport.csv', 'r') as f:
    markdown_content = nsightful.convert_ncu_csv_to_flat_markdown(f)
    print(markdown_content)

# Parse structured data for custom processing
with open('myreport.csv', 'r') as f:
    ncu_data = nsightful.parse_ncu_csv(f)
    # ncu_data is a nested dictionary with kernel -> section -> metrics/rules
```

#### NCU Jupyter Notebook Widget

Nsightful provides a function to display NCU data in a Jupyter notebook widget, reminiscent of the
Nsight Compute GUI.
This widget has:
- A kernel selector dropdown.
- A summary tab showing all Nsight recommendations and advisories.
- One detailed tab for each Nsight compute section.

If you have an existing report file or use the Nsight JupyterLab extension to run a cell under the
profile and collect a report, then you can display it with nsightful:

```python
!pip install "nsightful[notebook] @ git+https://github.com/brycelelbach/nsightful.git"
```

```python
import nsightful

nsightful.display_ncu_csv_file_in_notebook('myreport.csv')
```

or

```python
import nsightful

with open('myreport.csv', 'r') as f:
    nsightful.display_ncu_csv_in_notebook(f)
```

To collect a report from code already loaded in the notebook, use the Nsight Compute kernel and
`%%ncu` as shown above. This avoids writing a self-contained script or restarting the kernel.

### Nsight Systems (NSYS)

#### Generating NSYS SQLite Data

First, you need to generate NSYS SQLite data from your CUDA application:

```bash
# Profile your application
nsys profile -o myreport ./myapplication

# Convert SQLite to Chrome Trace JSON (output to a file)
nsightful myreport.sqlite -o myreport.json
```

#### NSYS Python Conversion API

```python
import nsightful

# Convert SQLite file to Chrome Trace JSON string
with open('myreport.sqlite', 'rb') as f:
    json_content = nsightful.convert_nsys_sqlite_to_chrome_trace_json(f)
    print(json_content)

# Parse structured data for custom processing
with open('myreport.sqlite', 'rb') as f:
    nsys_data = nsightful.parse_nsys_sqlite(f)
    # nsys_data is a structured dictionary with trace events
```

#### NSYS Jupyter Notebook Widget

Nsightful provides a function to display NSYS data in [Perfetto](https://ui.perfetto.dev/), a visual profiler with a interactive timeline view.

If you have an existing report file or use the Nsight JupyterLab extension to run a cell under the profile and collect a report, then you can display it with Nsightful:

```python
!pip install "nsightful[notebook] @ git+https://github.com/brycelelbach/nsightful.git"
```

```python
import nsightful

nsightful.display_nsys_sqlite_file_in_notebook('myreport.sqlite')
```

or

```python
import nsightful

with open('myreport.sqlite', 'rb') as f:
    nsightful.display_nsys_sqlite_in_notebook(f)
```

To collect a timeline from code already loaded in the notebook, use the Nsight Systems kernel and
`%%nsys` as shown above. Nsightful synchronizes the active CUDA context before stopping capture so
asynchronous GPU work remains inside the report.

## Example Output

### NCU to Flat Markdown

```markdown
# mykernel

## Speed Of Light

| Metric Name | Metric Unit | Metric Value |
|-------------|-------------|--------------|
| DRAM Frequency | cycle/nsecond | 1.215 |
| SM Frequency | cycle/nsecond | 1.410 |

🔧 **OPTIMIZATION**: This kernel achieves 45% of the theoretical maximum DRAM bandwidth...

## Memory Workload

| Metric Name | Metric Unit | Metric Value |
|-------------|-------------|--------------|
| Memory Throughput | Gbyte/second | 256.7 |
| Memory Utilization | % | 18.2 |

⚠️ **WARNING**: Memory bandwidth utilization is low. Consider increasing arithmetic intensity...
```

## Nsight JupyterLab Extension

If you are using a Jupyter environment where you are able to install JupyterLab extensions, check
out [the Nsight JupyterLab extension](https://pypi.org/project/jupyterlab-nvidia-nsight/).
Nsightful's custom kernels provide cell profiling without a JupyterLab extension or an explicit
enable-and-restart cycle. The extension remains useful when you want the full Nsight GUIs embedded
in JupyterLab.

The Nsight JupyterLab extension allows you to do two things:

- Profile a cell with Nsight Compute or Nsight Systems. We recommend using this with Nsightful's
  Jupyter widgets when possible.
- Run the Nsight GUIs within the notebook. This is a more full-featured alternative to Nsightful's
  Jupyter widgets.

## Requirements

- Python 3.10+
- NVIDIA Nsight Compute 2024.3+ (`ncu`) and/or Nsight Systems 2026.1.1+ (`nsys`) on `PATH` for
  cell profiling
- For Jupyter notebook features: install the `notebook` extra

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/brycelelbach/nsightful.git
cd nsightful

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev,notebook]"

# Set up pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src/ --cov-report=html

# Run tests for specific Python versions (requires tox)
pip install tox
tox
```

### Code Quality Checks

This project uses `pre-commit` to automatically run code quality checks. The hooks include:
- **Black**: Code formatting.
- **Flake8**: Linting for syntax errors and code quality issues.
- **MyPy**: Static type checking

To run all pre-commit hooks on all files:

```bash
# Run hooks on all files
pre-commit run --all-files

# Run all hooks on staged files only
pre-commit run

# Run a specific hook
pre-commit run black
pre-commit run flake8
pre-commit run mypy
```

## License

This project is licensed under the BSD-3-Clause License. See [LICENSE](LICENSE) for details.
