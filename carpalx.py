import os
import sys
import re
import random
import math
import copy
import glob
import time
import argparse
from collections import defaultdict
import pickle

# For visualization (optional in script, but needed for notebook)
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.font_manager import FontProperties
except ImportError:
    plt = None

# --- Configuration Parser ---

class Config:
    def __init__(self):
        self.data = {}
        self.config_dir = ""
        self.search_paths = []

    def load(self, filepath):
        abs_path = os.path.abspath(filepath)
        self.config_dir = os.path.dirname(abs_path)
        # Add default search paths
        self.search_paths = [
            self.config_dir,
            os.path.join(self.config_dir, 'etc'),
            os.path.join(os.path.dirname(self.config_dir), 'etc'),
            os.getcwd()
        ]
        self.data = self._parse_file(abs_path)
        self._post_process(self.data)

    def _resolve_path(self, path, current_file_dir=None):
        # If absolute, check if exists
        if os.path.isabs(path):
            if os.path.exists(path):
                return path
            return path # Return as is if not found?

        # Try relative to current file
        if current_file_dir:
            p = os.path.join(current_file_dir, path)
            if os.path.exists(p):
                return p

        # Try relative to config dir
        p = os.path.join(self.config_dir, path)
        if os.path.exists(p):
            return p

        # Try search paths
        for sp in self.search_paths:
            p = os.path.join(sp, path)
            if os.path.exists(p):
                return p

        # If still not found, return the path relative to config dir (default behavior)
        return os.path.join(self.config_dir, path)

    def _parse_file(self, filepath):
        # Note: filepath passed here should ideally be resolvable.
        # If it's an include, we resolved it before calling.

        if not os.path.exists(filepath):
            # One last try resolving it?
            filepath = self._resolve_path(filepath)
            if not os.path.exists(filepath):
                print(f"Warning: Config file not found: {filepath}")
                return {}

        config = {}
        stack = [config]

        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if '#' in line:
                line = line.split('#', 1)[0].strip()
            if not line:
                continue

            match = re.match(r'^<<include\s+(.+)>>$', line)
            if match:
                include_path = match.group(1).strip()
                current_dir = os.path.dirname(filepath)
                # Resolve include path using robust logic
                full_include_path = self._resolve_path(include_path, current_dir)

                included_conf = self._parse_file(full_include_path)
                self._merge_dict(stack[-1], included_conf)
                continue

            match = re.match(r'^<(\w+)(?:\s+(.+))?>$', line)
            if match:
                block_name = match.group(1)
                block_arg = match.group(2)
                new_block = {}
                if block_arg:
                    if block_name not in stack[-1]:
                        stack[-1][block_name] = {}
                    stack[-1][block_name][block_arg] = new_block
                else:
                    stack[-1][block_name] = new_block
                stack.append(new_block)
                continue

            match = re.match(r'^</(\w+)>$', line)
            if match:
                if len(stack) > 1:
                    stack.pop()
                continue

            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                stack[-1][key] = value
            else:
                stack[-1][line] = 1

        return config

    def _merge_dict(self, dest, source):
        for k, v in source.items():
            if isinstance(v, dict) and k in dest and isinstance(dest[k], dict):
                self._merge_dict(dest[k], v)
            else:
                dest[k] = v

    def _post_process(self, data):
        def substitute(value, root_data):
            if isinstance(value, str):
                if "rand(26)" in value:
                     return "".join([chr(97 + random.randint(0, 25)) for _ in range(6)])

                def repl(match):
                    expr = match.group(1)
                    if expr.startswith("$CONF{") and expr.endswith("}"):
                        path = expr[6:-1].split("}{")
                        curr = root_data
                        try:
                            for p in path:
                                curr = curr.get(p)
                            return str(curr)
                        except:
                            return match.group(0)
                    return match.group(0)

                new_value = re.sub(r'__([^_]+)__', repl, value)
                return new_value
            elif isinstance(value, dict):
                for k, v in value.items():
                    value[k] = substitute(v, root_data)
                return value
            return value

        if 'runid' in data and isinstance(data['runid'], str) and 'rand' in data['runid']:
             data['runid'] = "".join([chr(97 + random.randint(0, 25)) for _ in range(6)])

        self.data = substitute(data, self.data)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def __getitem__(self, key):
        return self.data[key]

def dclone(obj):
    return copy.deepcopy(obj)

def get_timestamp():
    return time.time()

class Carpalx:
    def __init__(self, conf_file):
        self.conf = Config()
        self.conf.load(conf_file)
        self.config = self.conf.data
        if 'effort_model' not in self.config:
            self.config['effort_model'] = {}
        self.keyboard = None
        self.triads = None

    def run(self):
        actions = self.config.get('action', '').split(',')
        for action in actions:
            action = action.strip()
            if not action: continue
            if action == 'loadkeyboard':
                self.load_keyboard()
            elif action == 'loadtriads':
                self.load_triads()
            elif action == 'optimize':
                self.optimize()
            elif action == 'reporteffort':
                self.report_effort()
            elif action == 'quit' or action == 'exit':
                break
            else:
                print(f"Unknown action: {action}")

    def load_keyboard(self):
        print(f"Loading keyboard from {self.config['keyboard_input']}")
        self.keyboard = Keyboard(self.config['keyboard_input'], self.config)

    def load_triads(self):
        print(f"Loading triads from {self.config['corpus']}")
        self.triads = Corpus(self.config['corpus'], self.config).triads

    def optimize(self):
        print("Optimizing keyboard...")
        optimizer = SimulatedAnnealing(self.keyboard, self.triads, self.config)
        self.keyboard = optimizer.run()
        if 'keyboard_output' in self.config:
            out_file = self.config['keyboard_output']
            print(f"Saving optimized keyboard to {out_file}")
            self.keyboard.save(out_file)

    def report_effort(self):
        print("Reporting effort...")
        effort = self.keyboard.calculate_effort(self.triads)
        print(f"Total Effort: {effort}")

class Keyboard:
    def __init__(self, layout_file, config):
        self.config = config
        self.layout_file = layout_file
        self.keys = []
        self.map = {}
        self._load_layout(layout_file)
        self._load_effort_model()

    def _resolve_path(self, path):
        # Attempt to resolve layout file path
        if os.path.exists(path): return path

        # Check relative to config dir
        conf_dir = self.config.get('config_dir_path_XXX', '') # Config object does not store it in data dict, but we need it.
        # Since Config parser is external, we might not have easy access to config_dir here unless we passed it.
        # But we can use Config object again if we want.

        # Try searching in standard locations
        search_paths = ['etc', 'etc/keyboards', 'keyboards']
        for sp in search_paths:
             p = os.path.join(sp, path)
             if os.path.exists(p): return p

        # Also try relative to ../etc if running from bin?
        return path

    def _load_layout(self, layout_file):
        layout_conf = Config()
        # Load config to get search paths logic
        layout_conf.load(self._resolve_path(layout_file))
        data = layout_conf.data
        if 'keyboard' not in data or 'row' not in data['keyboard']:
            raise ValueError("Invalid keyboard layout file")

        rows = data['keyboard']['row']
        sorted_rows = sorted(rows.keys(), key=lambda x: int(x))
        self.keys = []
        for r_idx in sorted_rows:
            row_data = rows[r_idx]
            keys_list = row_data['keys'].split()
            fingers_list = row_data['fingers'].split()
            row_objs = []
            col_idx = 0
            for k, f in zip(keys_list, fingers_list):
                if len(k) == 1: lc, uc = k, k.upper()
                elif len(k) == 2: lc, uc = k[0], k[1]
                else: lc, uc = k[0], k[1]
                finger = int(f)
                hand = 1 if finger > 4 else 0
                key_obj = {'row': int(r_idx) - 1, 'col': col_idx, 'lc': lc, 'uc': uc, 'finger': finger, 'hand': hand, 'effort': {}}
                row_objs.append(key_obj)
                self.map[lc] = key_obj
                self.map[uc] = key_obj
                col_idx += 1
            self.keys.append(row_objs)

    def _load_effort_model(self):
        em = self.config['effort_model']
        if 'finger_distance' not in em or 'row' not in em['finger_distance']:
            print("Warning: No finger distance (base effort) defined.")
            return
        fd_rows = em['finger_distance']['row']
        for r in range(len(self.keys)):
            r_key = str(r + 1)
            if r_key in fd_rows:
                efforts = [float(x) for x in fd_rows[r_key]['effort'].split()]
                for c, key in enumerate(self.keys[r]):
                    if c < len(efforts):
                        base_effort = efforts[c]
                        key['effort']['base'] = base_effort
                        penalties = em['weight_param']['penalties']
                        w_hand = float(penalties['weight']['hand'])
                        w_row = float(penalties['weight']['row'])
                        w_finger = float(penalties['weight']['finger'])
                        base_penalty = float(penalties['default'])
                        h_str = 'right' if key['hand'] == 1 else 'left'
                        p_hand = float(penalties['hand'].get(h_str, 0))
                        p_row = float(penalties['row'].get(r_key, 0))
                        f_str = 'left' if key['hand'] == 0 else 'right'
                        f_vals = [float(x) for x in penalties['finger'][f_str].split()]
                        f_idx = key['finger'] if key['hand'] == 0 else key['finger'] - 5
                        p_finger = f_vals[f_idx] if f_idx < len(f_vals) else 0
                        total_penalty = base_penalty + w_hand * p_hand + w_row * p_row + w_finger * p_finger
                        key['effort']['penalty'] = total_penalty
                        k_param = em['k_param']
                        kb = float(k_param['kb'])
                        kp = float(k_param['kp'])
                        key['effort']['total'] = kb * base_effort + kp * total_penalty
                    else:
                        print(f"Warning: No effort defined for key at {r},{c}")

    def calculate_effort(self, triads):
        total_effort = 0
        total_triads = 0
        k_param = self.config['effort_model']['k_param']
        k1 = float(k_param['k1'])
        k2 = float(k_param['k2'])
        k3 = float(k_param['k3'])
        kb = float(k_param['kb'])
        kp = float(k_param['kp'])
        ks = float(k_param['ks'])
        path_cost_conf = self.config['effort_model'].get('path_cost', {})

        for triad, freq in triads.items():
            c1, c2, c3 = triad
            if c1 not in self.map or c2 not in self.map or c3 not in self.map: continue
            k1_obj = self.map[c1]
            k2_obj = self.map[c2]
            k3_obj = self.map[c3]
            be1 = k1_obj['effort']['base']
            be2 = k2_obj['effort']['base']
            be3 = k3_obj['effort']['base']
            pe1 = k1_obj['effort']['penalty']
            pe2 = k2_obj['effort']['penalty']
            pe3 = k3_obj['effort']['penalty']
            term_base = k1*be1 * (1 + k2*be2 * (1 + k3*be3))
            term_penalty = k1*pe1 * (1 + k2*pe2 * (1 + k3*pe3))
            triad_effort = kb * term_base + kp * term_penalty
            if ks != 0:
                path_effort = self._calculate_path_effort(k1_obj, k2_obj, k3_obj, path_cost_conf)
                triad_effort += ks * path_effort
            total_effort += triad_effort * freq
            total_triads += freq
        return total_effort / total_triads if total_triads > 0 else 0

    def _calculate_path_effort(self, k1, k2, k3, path_cost_conf):
        h1, h2, h3 = k1['hand'], k2['hand'], k3['hand']
        r1, r2, r3 = k1['row'], k2['row'], k3['row']
        f1, f2, f3 = k1['finger'], k2['finger'], k3['finger']
        hand_flag = 0
        if h1 == h3:
            if h2 == h3: hand_flag = 2
            else: hand_flag = 1
        finger_flag = 3
        if f1 > f2:
            if f2 > f3: finger_flag = 0
            elif f2 == f3: finger_flag = 1 if k2['lc'] == k3['lc'] else 6
            elif f3 == f1: finger_flag = 4
            elif f1 > f3 and f3 > f2: finger_flag = 2
        elif f1 < f2:
            if f2 < f3: finger_flag = 0
            elif f2 == f3: finger_flag = 1 if k2['lc'] == k3['lc'] else 6
            elif f3 == f1: finger_flag = 4
            elif f1 < f3 and f3 < f2: finger_flag = 2
        elif f1 == f2:
            if f2 < f3 or f3 < f1: finger_flag = 1 if k1['lc'] == k2['lc'] else 6
            elif f2 == f3:
                if k1['lc'] != k2['lc'] and k2['lc'] != k3['lc'] and k1['lc'] != k3['lc']: finger_flag = 7
                else: finger_flag = 5
        row_flag = 5
        dr = sorted([abs(r1-r2), abs(r1-r3), abs(r2-r3)], reverse=True)
        drmax = dr[0]
        if r1 < r2:
            if r3 == r2: row_flag = 1
            elif r2 < r3: row_flag = 4
            elif drmax == 1: row_flag = 3
            else: row_flag = 7
        elif r1 > r2:
            if r3 == r2: row_flag = 2
            elif r2 > r3: row_flag = 6
            elif drmax == 1: row_flag = 3
            else: row_flag = 7
        else:
            if r2 > r3: row_flag = 2
            elif r2 < r3: row_flag = 1
            else: row_flag = 0
        path_key = f"{hand_flag}{row_flag}{finger_flag}"
        cost_str = path_cost_conf.get(path_key, "0")
        if '#' in str(cost_str): cost_str = str(cost_str).split('#')[0]
        return float(cost_str)

    def save(self, filepath):
        with open(filepath, 'w') as f:
            f.write("<keyboard>\n")
            for r_idx, row in enumerate(self.keys):
                keys = []
                fingers = []
                for k in row:
                    if k['lc'] == k['uc'].lower(): ks = k['lc']
                    else: ks = k['lc'] + k['uc']
                    keys.append(ks)
                    fingers.append(str(k['finger']))
                f.write(f"<row {r_idx+1}>\n")
                f.write(f"keys = {' '.join(keys)}\n")
                f.write(f"fingers = {' '.join(fingers)}\n")
                f.write("</row>\n")
            f.write("</keyboard>\n")

    def swap_keys(self, k1_coord, k2_coord):
        r1, c1 = k1_coord
        r2, c2 = k2_coord
        key1 = self.keys[r1][c1]
        key2 = self.keys[r2][c2]
        key1['lc'], key2['lc'] = key2['lc'], key1['lc']
        key1['uc'], key2['uc'] = key2['uc'], key1['uc']
        self.map[key1['lc']] = key1
        self.map[key1['uc']] = key1
        self.map[key2['lc']] = key2
        self.map[key2['uc']] = key2

class Corpus:
    def __init__(self, filepath, config):
        self.config = config
        self.triads = defaultdict(int)
        self._load(filepath)

    def _load(self, filepath):
        # Resolve path
        if not os.path.exists(filepath):
            # Try to find it
            if os.path.exists(os.path.join('corpus', os.path.basename(filepath))):
                filepath = os.path.join('corpus', os.path.basename(filepath))
            elif os.path.exists(os.path.join('..', 'corpus', os.path.basename(filepath))):
                 filepath = os.path.join('..', 'corpus', os.path.basename(filepath))

        if not os.path.exists(filepath):
            print(f"Warning: Corpus file not found: {filepath}")
            return
        mode_name = self.config.get('mode', 'english')
        mode = self.config.get('mode_def', {}).get(mode_name, {})
        force_case = mode.get('force_case', 'no')
        reject_char_rx = mode.get('reject_char_rx')
        accept_line_rx = mode.get('accept_line_rx')
        triads_overlap = self.config.get('triads_overlap') in ['yes', '1', 1, True]
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if accept_line_rx and not re.search(accept_line_rx, line): continue
                if force_case == 'lc': line = line.lower()
                elif force_case == 'uc': line = line.upper()
                if reject_char_rx: line = re.sub(reject_char_rx, '', line)
                line = re.sub(r'\s', '', line)
                for i in range(len(line) - 2):
                    triad = line[i:i+3]
                    self.triads[triad] += 1
        min_freq = int(self.config.get('triads_min_freq', 0))
        if min_freq > 0:
            self.triads = {k: v for k, v in self.triads.items() if v >= min_freq}

class SimulatedAnnealing:
    def __init__(self, keyboard, triads, config):
        self.keyboard = keyboard
        self.triads = triads
        self.config = config
        self.params = config.get('annealing', {})
        self.iterations = int(self.params.get('iterations', 1000))
        self.t0 = float(self.params.get('t0', 10))
        self.k = float(self.params.get('k', 10))
        self.p0 = float(self.params.get('p0', 1))
        self.relocatable = self._get_relocatable_keys()

    def _get_relocatable_keys(self):
        reloc = []
        mask_conf = self.config.get('mask_row', {})
        for r_idx, row_data in mask_conf.items():
            r = int(r_idx) - 1
            mask = [int(x) for x in row_data['mask'].split()]
            for c, m in enumerate(mask):
                if m == 1: reloc.append((r, c))
        return reloc

    def run(self):
        current_effort = self.keyboard.calculate_effort(self.triads)
        print(f"Initial Effort: {current_effort}")
        best_keyboard = copy.deepcopy(self.keyboard)
        best_effort = current_effort
        for i in range(1, self.iterations + 1):
            if not self.relocatable: break
            k1 = random.choice(self.relocatable)
            k2 = random.choice(self.relocatable)
            while k1 == k2: k2 = random.choice(self.relocatable)
            self.keyboard.swap_keys(k1, k2)
            new_effort = self.keyboard.calculate_effort(self.triads)
            deffort = new_effort - current_effort
            t = self.t0 * math.exp(-i * self.k / self.iterations)
            accept = False
            if deffort < 0: accept = True
            else:
                p = self.p0 * math.exp(-deffort / t) if t > 0 else 0
                if random.random() < p: accept = True
            if accept:
                current_effort = new_effort
                if current_effort < best_effort:
                    best_effort = current_effort
                    best_keyboard = copy.deepcopy(self.keyboard)
                    print(f"Iter {i}: New Best Effort: {best_effort:.4f}")
            else:
                self.keyboard.swap_keys(k1, k2)
            if i % 100 == 0:
                print(f"Iter {i}, Temp {t:.4f}, Effort {current_effort:.4f}")
        return best_keyboard

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Carpalx Keyboard Optimizer (Python Port)')
    parser.add_argument('-conf', dest='configfile', help='Configuration file', required=False)
    args = parser.parse_args()
    conf_file = args.configfile if args.configfile else 'etc/carpalx.conf'
    if not os.path.exists(conf_file):
        if os.path.exists(os.path.join('etc', 'carpalx.conf')):
            conf_file = os.path.join('etc', 'carpalx.conf')
        else:
            print("Configuration file not found.")
            sys.exit(1)
    app = Carpalx(conf_file)
    app.run()
