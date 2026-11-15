"""Build the Jupyter notebooks of the two worked examples from the Python scripts.

Each script is written in the percent format: a line "# %%" starts a code cell and a line
"# %% [markdown]" starts a text cell whose lines are comments. Run from this folder:

    python build_notebooks.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["example1_acoustic_link.py", "example2_cross_segment.py"]


def cells_of(text):
    cells, kind, lines = [], None, []
    for line in text.splitlines():
        if line.startswith("# %%"):
            if kind is not None:
                cells.append((kind, lines))
            kind, lines = ("markdown" if "[markdown]" in line else "code"), []
        elif kind is not None:
            lines.append(line)
    cells.append((kind, lines))
    return cells


def to_notebook(text, name):
    out = []
    for kind, lines in cells_of(text):
        if kind == "markdown":
            lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines]
        while lines and not lines[-1].strip():
            lines.pop()
        while lines and not lines[0].strip():
            lines.pop(0)
        source = [l + "\n" for l in lines[:-1]] + lines[-1:]
        cell = {"cell_type": kind, "metadata": {}, "source": source}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        out.append(cell)
    return {"cells": out,
            "metadata": {"colab": {"name": name, "provenance": []},
                         "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    (HERE / "notebooks").mkdir(exist_ok=True)
    for script in SCRIPTS:
        name = script.replace(".py", ".ipynb")
        nb = to_notebook((HERE / script).read_text(encoding="utf-8"), name)
        (HERE / "notebooks" / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("written", HERE / "notebooks" / name, f"({len(nb['cells'])} cells)")
