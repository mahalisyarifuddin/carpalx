import json
import re

def main():
    path_costs = {}
    with open('etc/effort/path/01.conf', 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Match lines like "000 = 0 # comments"
            match = re.match(r'^(\d{3})\s*=\s*([\d\.]+)', line)
            if match:
                key = match.group(1)
                value = float(match.group(2))
                path_costs[key] = value

    print(json.dumps(path_costs, indent=2))

if __name__ == '__main__':
    main()
