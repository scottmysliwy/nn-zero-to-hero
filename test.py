import nbformat
from nbconvert import PythonExporter
import glob
import os

# Get a list of all notebook files in the current directory
notebook_files = glob.glob("*.ipynb")

# Instantiate the exporter. We use the python exporter here
python_exporter = PythonExporter()

# Loop through notebook files and convert them
for nb_file in notebook_files:
    # Read the notebook file
    with open(nb_file) as f:
        nb_contents = f.read()

    # Convert using the PythonExporter
    notebook = nbformat.reads(nb_contents, as_version=4)
    body, _ = python_exporter.from_notebook_node(notebook)

    # Write to .py file
    with open(os.path.splitext(nb_file)[0] + ".py", "w") as f:
        f.write(body)
