"""Package submission/ into a Codabench code-submission archive.

    python tools/build_submission.py --out submission.zip [--check]

The archive holds run.py, model_def.py, metadata.yaml, assets/ (configs and the three weight files)
and a copy of the installed timm package, because the evaluation machine has no internet access.
--check extracts the archive and runs `import run; run.load_model()` in an isolated interpreter.
"""
import argparse
import os
import subprocess
import sys
import tempfile
import zipfile

import timm

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SUB = os.path.join(ROOT, "submission")
WEIGHTS = ["weights.pt", "weights2.pt", "weights3.pt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="submission.zip")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    missing = [w for w in WEIGHTS if not os.path.exists(os.path.join(SUB, "assets", w))]
    if missing:
        sys.exit(f"missing submission/assets/{missing}; run scripts/download_weights.sh first")
    if timm.__version__ != "1.0.9":
        print(f"warning: timm {timm.__version__} installed, the weights were trained with timm 1.0.9")
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(SUB):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                full = os.path.join(root, f)
                z.write(full, os.path.relpath(full, SUB))
        tdir = os.path.dirname(timm.__file__)
        for root, dirs, files in os.walk(tdir):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                full = os.path.join(root, f)
                z.write(full, os.path.join("timm", os.path.relpath(full, tdir)))
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.0f} MB)")
    if a.check:
        with tempfile.TemporaryDirectory() as td:
            zipfile.ZipFile(a.out).extractall(td)
            code = (f"import sys; sys.path.insert(0, {td!r}); import run; run.load_model(); "
                    "assert run._state['model'] is not None and len(run._state['members']) == 2; print('archive OK')")
            subprocess.run([sys.executable, "-I", "-c", code], cwd=td, check=True)


if __name__ == "__main__":
    main()
