"""Reproduce every simulation result and figure of the article.

    python run_all.py            # runs all examples, then makes the figures
    python run_all.py --figures  # only remakes the figures from results/*.json

Example 1 (Section II-F): about 11 minutes on a 2-thread CPU.
Example 2 (Section VII-E): about 7 minutes on a 2-thread CPU.
Seeds are fixed inside each script, so the JSON files and figures are reproducible
up to small floating-point differences between PyTorch versions and CPUs.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLES = ["example1_acoustic_link.py", "example2_spoofing.py"]


def run(script):
    print(f"--- running {script}", flush=True)
    subprocess.run([sys.executable, str(HERE / script)], check=True, cwd=HERE)


if __name__ == "__main__":
    if "--figures" not in sys.argv:
        for s in EXAMPLES:
            run(s)
    run("make_figures.py")
    print("--- done: results in code/results/, figures in figures/")
