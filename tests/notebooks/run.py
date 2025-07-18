# This is tested via GitHub Actions (within the Jupyter container), but we
# don't want pytest to run it directly, hence the strange name.

import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOKS_TO_SKIP = {
    "Running the Evaluator with Mods.ipynb",  # requires private annotators
}


def test_notebooks(notebooks_dir):
    notebooks = [
        f
        for f in os.listdir(notebooks_dir)
        if f.endswith(".ipynb") and f not in NOTEBOOKS_TO_SKIP
    ]
    if not notebooks:
        print(f"No notebooks found in {notebooks_dir}")
        return

    for notebook in notebooks:
        notebook_path = os.path.join(notebooks_dir, notebook)
        print(f"Testing notebook: {notebook_path}")
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)
            ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
            try:
                ep.preprocess(nb, {"metadata": {"path": notebooks_dir}})
                print(f"Notebook {notebook} executed successfully.")
            except Exception as e:
                print(f"Error executing notebook {notebook}: {e}")
                raise


if __name__ == "__main__":
    notebooks_dir = "/app/flightpaths"
    test_notebooks(notebooks_dir)
