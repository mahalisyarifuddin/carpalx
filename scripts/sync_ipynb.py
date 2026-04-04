import json

def update_ipynb(filepath):
    with open(filepath, 'r') as f:
        nb = json.load(f)

    with open('carpalx.py', 'r') as f:
        py_code = f.read()

    # Simple replacement of the core code cell.
    # Assumes the core code cell is the 5th cell (index 4) based on previous read.
    nb['cells'][4]['source'] = [line + '\n' for line in py_code.split('\n')]

    # Update virtual files (step 3) to include reject_repeats
    setup_cell = nb['cells'][3]['source']
    new_setup = []
    for line in setup_cell:
        if 'accept_line_rx  = \\\\w' in line:
            new_setup.append(line)
            new_setup.append('accept_repeats  = no\n')
        else:
            new_setup.append(line)
    nb['cells'][3]['source'] = new_setup

    with open(filepath, 'w') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    update_ipynb('carpalx.ipynb')
