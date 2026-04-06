# Carpalx Keyboard Optimizer

**Carpalx** is a keyboard layout optimizer that uses simulated annealing and Late Acceptance Hill Climbing (LAHC) to minimize typing effort. This repository offers three implementations: a standalone web application, a Google Colab notebook, and a Python script.

## Implementations

### 1. Web Application (`carpalx.html`)

A standalone HTML-JS tool for instant optimization in your browser.

*   **Keyboard Visualization**: Real-time layout rendering.
*   **File-Based Analysis**: Upload `.txt` corpora for precise effort calculation and optimization training.
*   **Optimization**: Run simulated annealing or **Late Acceptance Hill Climbing (LAHC)** directly in the browser with configurable penalties and row constraints.

**To use:** Open `carpalx.html` in any modern web browser. Upload a text file to begin analysis or optimization.

### 2. Google Colab Notebook (`carpalx.ipynb`)

A self-contained Python-based notebook for advanced experimentation.

*   **Self-Contained**: All configuration models and logic are embedded.
*   **Custom Corpora**: Upload and analyze your own text files.
*   **Rich Visualization**: Uses Matplotlib for layout plots.

**To use:** Open `carpalx.ipynb` in Google Colab or a local Jupyter environment.

### 3. Python Script (`carpalx.py`)

The core Python port of the original Carpalx logic.

```bash
python3 carpalx.py -conf etc/carpalx.conf
```

## Legacy Perl Implementation

The original Perl implementation is available in the `legacy/` directory for reference.

## Original Documentation

See [mkweb.bcgsc.ca/carpalx](http://mkweb.bcgsc.ca/carpalx) for the original project documentation and theory behind the effort models.

## License

GNU General Public License. See source files for details.
