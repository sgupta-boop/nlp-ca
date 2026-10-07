"""Build a notebook from a list of (kind, source) cells, then execute it in place.

    from scripts.nbtools import build
    build("notebooks/01_columns.ipynb", [("md", "# Title"), ("code", "print(1)")])
"""
import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SETUP = """import sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_colwidth', 80)
pd.set_option('display.width', 200)
FIG = ROOT / 'reports' / 'figures'
FIG.mkdir(parents=True, exist_ok=True)"""


def build(path: str, cells: list[tuple[str, str]], execute: bool = True) -> None:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.cells = [nbformat.v4.new_code_cell(SETUP)]
    for kind, src in cells:
        nb.cells.append(nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src))
    out = ROOT / path
    nbformat.write(nb, out)
    if execute:
        subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                        "--inplace", "--ExecutePreprocessor.timeout=3600", str(out)], check=True)
