# Carpalx Keyboard Optimizer

**Carpalx** is a keyboard layout optimizer that uses simulated annealing to find layouts that minimize typing effort. This repository contains a Python port of the original Perl implementation, provided as a Jupyter Notebook for easy experimentation in Google Colab.

## Usage

### Jupyter Notebook (Google Colab)

The primary artifact is `carpalx.ipynb`. This notebook is self-contained and includes:

*   **Configuration**: All effort models, weights, and parameters are embedded.
*   **Logic**: The core optimization algorithms (Simulated Annealing) ported to Python.
*   **Visualization**: Tools to visualize keyboard layouts.

**To use:**
1.  Open `carpalx.ipynb` in Google Colab or a local Jupyter environment.
2.  Run the cells.
3.  Upload your own corpus text file (e.g., `corpus.txt`) when prompted or ensure it exists in the working directory.
4.  The notebook will output the optimized layout and effort statistics.

### Python Script

A standalone Python script `carpalx.py` is also provided.

```bash
python3 carpalx.py -conf etc/carpalx.conf
```

Note: The script relies on configuration files in `etc/`.

## Legacy Perl Implementation

The original Perl implementation has been moved to the `legacy/` directory.

*   `legacy/bin/`: Original Perl scripts (`carpalx`, `generatetriads`, etc.)

## Original Documentation

See http://mkweb.bcgsc.ca/carpalx for the original project documentation and theory behind the effort models.

## License

See original license information in source files.
