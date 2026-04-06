import json
import os

def update_ipynb(filepath):
    if not os.path.exists(filepath):
        print(f"File {filepath} not found")
        return

    with open(filepath, 'r') as f:
        nb = json.load(f)

    with open('carpalx.py', 'r') as f:
        py_code = f.read()

    new_cells = []

    # 0. Header
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Carpalx Keyboard Optimizer\n",
            "\n",
            "This notebook is a Python port of the [Carpalx](http://mkweb.bcgsc.ca/carpalx/) keyboard layout optimizer.\n",
            "\n",
            "## Instructions\n",
            "\n",
            "1.  **Upload your corpus**: Upload your `.txt` corpus file to the Colab environment.\n",
            "2.  **Configure**: You can modify the configuration in the \"Configuration\" section if needed.\n",
            "3.  **Run**: Execute the cells to run the optimization."
        ]
    })

    # 1. Imports
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "import re\n",
            "import random\n",
            "import math\n",
            "import copy\n",
            "import glob\n",
            "import time\n",
            "import argparse\n",
            "from collections import defaultdict\n",
            "import pickle\n",
            "\n",
            "# For visualization\n",
            "try:\n",
            "    import matplotlib.pyplot as plt\n",
            "    import matplotlib.patches as patches\n",
            "    from matplotlib.font_manager import FontProperties\n",
            "except ImportError:\n",
            "    plt = None\n",
            "\n",
            "print(\"Libraries imported successfully.\")"
        ]
    })

    # 2. Virtual FS Markdown
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Configuration Files (Embedded)\n",
            "\n",
            "The following cell defines the necessary configuration files directly in the notebook."
        ]
    })

    # 3. Virtual FS Code
    virtual_fs_code = [
        "# --- Virtual File System for Configs ---\n",
        "VIRTUAL_FILES = {}\n",
        "\n",
        "VIRTUAL_FILES['etc/carpalx.conf'] = \"\"\"\n",
        "action = loadkeyboard,loadtriads,optimize,quit\n",
        "corpus  = corpus.txt\n",
        "mode    = english\n",
        "triads_overlap  = yes\n",
        "triads_min_freq = 1\n",
        "keyboard_input  = keyboards/qwerty.conf\n",
        "keyboard_output = optimized_keyboard.conf\n",
        "<effort_model>\n",
        "<<include effort/01.conf>>\n",
        "</effort_model>\n",
        "<annealing>\n",
        "action     = minimize\n",
        "iterations = 1000\n",
        "t0         = 10\n",
        "p0         = 1\n",
        "k          = 10\n",
        "minswaps   = 1\n",
        "maxswaps   = 3\n",
        "onestep    = no\n",
        "mode       = lahc\n",
        "history_size = 500\n",
        "</annealing>\n",
        "<<include mask/letters.conf>>\n",
        "<<include modes/mode.conf>>\n",
        "\"\"\"\n",
        "\n",
        "VIRTUAL_FILES['etc/keyboards/qwerty.conf'] = \"\"\"\n",
        "<keyboard>\n",
        "<row 1>\n",
        "keys    = `~ 1! 2@ 3# 4$ 5% 6^ 7& 8* 9( 0) -_ =+\n",
        "fingers =  0  0  1   2  3  3  6  6 7   8  9  9  9\n",
        "</row>\n",
        "<row 2>\n",
        "keys    = q w e r t y u i o p [{ ]} \\\\\n",
        "fingers = 0 1 2 3 3 6 6 7 8 9  9  9  9\n",
        "</row>\n",
        "<row 3>\n",
        "keys    = a s d f g h j k l ;: '\\\"\n",
        "fingers = 0 1 2 3 3 6 6 7 8  9  9\n",
        "</row>\n",
        "<row 4>\n",
        "keys    = z x c v b n m ,< .> /?\n",
        "fingers = 0 1 2 3 3 6 6  7  8  9\n",
        "</row>\n",
        "</keyboard>\n",
        "\"\"\"\n",
        "\n",
        "VIRTUAL_FILES['etc/effort/01.conf'] = \"\"\"\n",
        "<k_param>\n",
        "<<include k/01.conf>>\n",
        "</k_param>\n",
        "<weight_param>\n",
        "<<include effort/weight/01.conf>>\n",
        "</weight_param>\n",
        "<path_cost>\n",
        "<<include effort/path/01.conf>>\n",
        "</path_cost>\n",
        "<finger_distance>\n",
        "<<include effort/base/01.conf>>\n",
        "</finger_distance>\n",
        "\"\"\"\n",
        "\n",
        "VIRTUAL_FILES['etc/effort/base/01.conf'] = \"\"\"\n",
        "<row 1>\n",
        "effort = 5 4 4 4 4 4 4.5 4 4 4 4 4.5 5.5\n",
        "</row>\n",
        "<row 2>\n",
        "effort = 2 2 2 2 2.5 3 2 2 2 2 2.5 4 6\n",
        "</row>\n",
        "<row 3>\n",
        "effort = 0 0 0 0 2 2 0 0 0 0 2\n",
        "</row>\n",
        "<row 4>\n",
        "effort = 2 2 2 2 3.5 2 2 2 2 2\n",
        "</row>\n",
        "\"\"\"\n",
        "\n",
        "VIRTUAL_FILES['etc/effort/k/01.conf'] = \"\"\"\n",
        "kb = 0.3555\n",
        "kp = 0.6423\n",
        "ks = 0.4268\n",
        "k1 = 1\n",
        "k2 = 0.367\n",
        "k3 = 0.235\n",
        "\"\"\"\n",
        "\n",
        "VIRTUAL_FILES['etc/effort/weight/01.conf'] = \"\"\"\n",
        "<penalties>\n",
        "shift       = 0\n",
        "default     = 0\n",
        "path_offset = 0\n",
        "<weight>\n",
        "hand   = 1\n",
        "row    = 1.3088\n",
        "finger = 2.5948\n",
        "</weight>\n",
        "<row>\n",
        "0 = 1.5\n",
        "1 = 0.5\n",
        "2 = 0\n",
        "3 = 1\n",
        "</row>\n",
        "<hand>\n",
        "left = 0\n",
        "right = 0\n",
        "</hand>\n",
        "<finger>\n",
        "left =  1 0.5 0 0 0\n",
        "right = 0 0 0 0.5 1\n",
        "</finger>\n",
        "</penalties>\n",
        "\"\"\"\n",
        "\n",
        "VIRTUAL_FILES['etc/mask/letters.conf'] = \"\"\"\n",
        "<mask_row 1>\n",
        "mask = 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
        "</mask_row>\n",
        "<mask_row 2>\n",
        "mask = 1 1 1 1 1 1 1 1 1 1 0 0 0\n",
        "</mask_row>\n",
        "<mask_row 3>\n",
        "mask = 1 1 1 1 1 1 1 1 1 0 0\n",
        "</mask_row>\n",
        "<mask_row 4>\n",
        "mask = 1 1 1 1 1 1 1 0 0 0\n",
        "</mask_row>\n",
        "\"\"\"\n",
        "\n",
        "VIRTUAL_FILES['etc/modes/mode.conf'] = \"\"\"\n",
        "<mode_def english>\n",
        "force_case      = lc\n",
        "reject_char_rx  = [\\\\W_0-9]\n",
        "accept_repeats  = no\n",
        "accept_line_rx  = \\\\w\n",
        "</mode_def>\n",
        "\"\"\"\n",
        "\n",
        "VIRTUAL_FILES['etc/effort/path/01.conf'] = \"\"\"\n",
        "000 = 0\n",
        "001 = 0.3\n",
        "002 = 0.6\n",
        "003 = 0.9\n",
        "006 = 1.8\n",
        "010 = 0.3\n",
        "011 = 0.6\n",
        "012 = 0.9\n",
        "013 = 1.2\n",
        "016 = 2.1\n",
        "020 = 0.6\n",
        "021 = 0.9\n",
        "022 = 1.2\n",
        "023 = 1.5\n",
        "026 = 2.4\n",
        "030 = 0.9\n",
        "032 = 1.5\n",
        "033 = 1.8\n",
        "036 = 2.7\n",
        "040 = 1.2\n",
        "042 = 1.8\n",
        "043 = 2.1\n",
        "046 = 3\n",
        "050 = 1.5\n",
        "052 = 2.1\n",
        "053 = 2.4\n",
        "056 = 3.3\n",
        "060 = 1.8\n",
        "062 = 2.4\n",
        "063 = 2.7\n",
        "066 = 3.6\n",
        "070 = 2.1\n",
        "072 = 2.7\n",
        "073 = 3\n",
        "076 = 3.9\n",
        "102 = 1.6\n",
        "103 = 1.9\n",
        "104 = 2.2\n",
        "112 = 1.9\n",
        "113 = 2.2\n",
        "114 = 2.5\n",
        "122 = 2.2\n",
        "123 = 2.5\n",
        "124 = 2.8\n",
        "132 = 2.5\n",
        "133 = 2.8\n",
        "134 = 3.1\n",
        "142 = 2.8\n",
        "143 = 3.1\n",
        "144 = 3.4\n",
        "152 = 3.1\n",
        "153 = 3.4\n",
        "154 = 3.7\n",
        "162 = 3.4\n",
        "163 = 3.7\n",
        "164 = 4\n",
        "172 = 3.7\n",
        "173 = 4\n",
        "174 = 4.3\n",
        "200 = 2\n",
        "201 = 2.3\n",
        "202 = 2.6\n",
        "203 = 2.9\n",
        "204 = 3.2\n",
        "205 = 3.5\n",
        "206 = 3.8\n",
        "210 = 2.3\n",
        "211 = 2.6\n",
        "212 = 2.9\n",
        "213 = 3.2\n",
        "214 = 3.5\n",
        "215 = 3.8\n",
        "216 = 4.1\n",
        "217 = 4.4\n",
        "220 = 2.6\n",
        "221 = 2.9\n",
        "222 = 3.2\n",
        "223 = 3.5\n",
        "224 = 3.8\n",
        "225 = 4.1\n",
        "226 = 4.4\n",
        "227 = 4.7\n",
        "230 = 2.9\n",
        "232 = 3.5\n",
        "233 = 3.8\n",
        "234 = 4.1\n",
        "235 = 4.4\n",
        "236 = 4.7\n",
        "237 = 5\n",
        "240 = 3.2\n",
        "242 = 3.8\n",
        "243 = 4.1\n",
        "244 = 4.4\n",
        "246 = 5\n",
        "247 = 5.3\n",
        "250 = 3.5\n",
        "252 = 4.1\n",
        "253 = 4.4\n",
        "254 = 4.7\n",
        "256 = 5.3\n",
        "257 = 5.6\n",
        "260 = 3.8\n",
        "262 = 4.4\n",
        "263 = 4.7\n",
        "264 = 5\n",
        "266 = 5.6\n",
        "267 = 5.9\n",
        "270 = 4.1\n",
        "272 = 4.7\n",
        "273 = 5\n",
        "274 = 5.3\n",
        "275 = 5.6\n",
        "276 = 5.9\n",
        "277 = 6.2\n",
        "\"\"\"\n",
        "\n",
        "def setup_virtual_files():\n",
        "    for path, content in VIRTUAL_FILES.items():\n",
        "        dir_name = os.path.dirname(path)\n",
        "        if dir_name:\n",
        "            os.makedirs(dir_name, exist_ok=True)\n",
        "        with open(path, 'w', encoding='utf-8') as f:\n",
        "            f.write(content)\n",
        "    print(\"Configuration files created.\")\n",
        "\n",
        "setup_virtual_files()"
    ]
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": virtual_fs_code
    })

    # 4. Core Logic Markdown
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Core Logic\n",
            "\n",
            "This cell contains the main Carpalx optimization logic ported to Python."
        ]
    })

    # 5. Core Logic Code
    # Filter out the __main__ part from carpalx.py for the notebook
    core_lines = []
    lines = py_code.split('\n')
    skip = False
    for line in lines:
        if "if __name__ == '__main__':" in line:
            skip = True
        if not skip:
            core_lines.append(line + '\n')

    # Inject plot method into Keyboard class
    final_core_lines = []
    for line in core_lines:
        final_core_lines.append(line)
        if 'def swap_keys(self, k1_coord, k2_coord):' in line:
             # Find end of swap_keys method to inject plot method
             pass

    # Simple injection after Keyboard class definition for now (at the end of class)
    # Actually carpalx.py structure:
    # class Keyboard:
    #   ...
    #   def swap_keys(self, k1_coord, k2_coord):
    #     ...

    plot_method = [
        "    def plot(self, title=\"Keyboard Layout\"):\n",
        "        if not plt: return\n",
        "        fig, ax = plt.subplots(figsize=(12, 5))\n",
        "        ax.set_xlim(0, 15)\n",
        "        ax.set_ylim(-5, 0)\n",
        "        ax.set_aspect('equal')\n",
        "        ax.axis('off')\n",
        "        ax.set_title(title)\n",
        "\n",
        "        for row in self.keys:\n",
        "            for k in row:\n",
        "                r, c = k['row'], k['col']\n",
        "                x = c + (r * 0.5)\n",
        "                y = -r\n",
        "                \n",
        "                rect = patches.Rectangle((x, y-0.9), 0.9, 0.9, linewidth=1, edgecolor='black', facecolor='white')\n",
        "                ax.add_patch(rect)\n",
        "                ax.text(x + 0.45, y - 0.45, k['uc'], ha='center', va='center', fontsize=10, fontweight='bold')\n",
        "        plt.show()\n"
    ]

    # Find where to inject plot_method in core_lines
    injected_lines = []
    for line in core_lines:
        injected_lines.append(line)
        if "self.map[key2['uc']] = key2" in line:
             injected_lines.extend(plot_method)

    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": injected_lines
    })

    # Update Carpalx.run to call plot
    for cell in new_cells:
        if cell['cell_type'] == 'code' and any('class Carpalx:' in l for l in cell['source']):
            new_run = []
            skip_old_run = False
            current_class = None
            for line in cell['source']:
                if line.startswith('class '):
                    current_class = line.split('(')[0].split(':')[0].strip().split()[-1]

                if 'def run(self):' in line and current_class == 'Carpalx':
                    new_run.append(line)
                    new_run.append("        print(f\"Loading keyboard from {self.config['keyboard_input']}\")\n")
                    new_run.append("        self.keyboard = Keyboard(self.config['keyboard_input'], self.config)\n")
                    new_run.append("        print(f\"Loading triads from {self.config['corpus']}\")\n")
                    new_run.append("        self.triads = Corpus(self.config['corpus'], self.config).triads\n")
                    new_run.append("        if not self.triads: return\n")
                    new_run.append("        self.keyboard.plot(\"Initial Keyboard\")\n")
                    new_run.append("        self.optimize()\n")
                    new_run.append("        self.keyboard.plot(\"Optimized Keyboard\")\n")
                    skip_old_run = True
                elif skip_old_run and (line.startswith('    def ') or line.startswith('class ')):
                    skip_old_run = False
                    new_run.append(line)
                elif not skip_old_run:
                    new_run.append(line)
            cell['source'] = new_run

    # 6. Execution Markdown
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Run Optimization\n",
            "\n",
            "Upload a `corpus.txt` file or run the cell below to create a sample one."
        ]
    })

    # 7. Execution Code
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "if not os.path.exists('corpus.txt'):\n",
            "    print(\"Creating sample corpus.txt...\")\n",
            "    with open('corpus.txt', 'w') as f:\n",
            "        f.write(\"the quick brown fox jumps over the lazy dog \" * 100)\n",
            "\n",
            "app = Carpalx('etc/carpalx.conf')\n",
            "app.run()"
        ]
    })

    nb['cells'] = new_cells
    nb['nbformat'] = 4
    nb['nbformat_minor'] = 4

    with open(filepath, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"Updated {filepath}")

if __name__ == '__main__':
    update_ipynb('carpalx.ipynb')
