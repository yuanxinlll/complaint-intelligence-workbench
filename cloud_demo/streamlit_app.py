"""Hosted synthetic-data demo. The local FT01 entrypoint is unchanged."""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CIW_PUBLIC_DEMO"] = "1"
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "2")
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
